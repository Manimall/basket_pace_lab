package enrich

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"

	"basket_pace_lab/etl_scout/internal/model"
	"basket_pace_lab/etl_scout/internal/sofascore"
	"basket_pace_lab/etl_scout/internal/storage"
)

func quietLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// fakeFetcher returns canned results keyed by event id.
type fakeFetcher struct{ byID map[int]sofascore.Result }

func (f fakeFetcher) FetchMany(_ context.Context, ids []int) []sofascore.Result {
	out := make([]sofascore.Result, 0, len(ids))
	for _, id := range ids {
		if r, ok := f.byID[id]; ok {
			out = append(out, r)
		}
	}
	return out
}

// fakeSaver records saved rows and can be told to fail.
type fakeSaver struct {
	saved int
	err   error
}

func (s *fakeSaver) SaveAdvancedStats(_ context.Context, _, _ int64, _ model.AdvancedStats) error {
	if s.err != nil {
		return s.err
	}
	s.saved++
	return nil
}

// goodStats is a complete box score (home 89-73) that yields valid metrics.
func goodStats() *model.StatisticsResponse {
	item := func(name, h, a string) model.StatisticItem {
		return model.StatisticItem{Name: name, Home: h, Away: a}
	}
	return &model.StatisticsResponse{
		Statistics: []model.PeriodStatistics{{
			Period: "ALL",
			Groups: []model.StatisticGroup{{Items: []model.StatisticItem{
				item("Points", "89", "73"),
				item("Field Goals", "33/73", "27/68"),
				item("3-Pointers", "9/24", "7/21"),
				item("Free Throws", "14/18", "12/17"),
				item("Offensive Rebounds", "11", "8"),
				item("Defensive Rebounds", "28", "24"),
				item("Turnovers", "10", "14"),
			}}},
		}},
	}
}

func ref(id int64, eventID int) storage.MatchRef {
	return storage.MatchRef{ID: id, EventID: eventID, HomeTeamID: id*10 + 1, AwayTeamID: id*10 + 2}
}

func TestRunEnrichesGoodMatch(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{
		100: {EventID: 100, Stats: goodStats()},
	}}
	saver := &fakeSaver{}
	enr := New(fetcher, saver, quietLogger())

	sum := enr.Run(context.Background(), []storage.MatchRef{ref(1, 100)})

	if sum.Enriched != 1 || sum.Skipped != 0 || sum.Failed != 0 {
		t.Fatalf("unexpected summary: %+v", sum)
	}
	if saver.saved != 2 { // home + away rows
		t.Errorf("expected 2 saved rows, got %d", saver.saved)
	}
}

func TestRunSkipsNoStatistics(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{
		100: {EventID: 100, Err: sofascore.ErrNoStatistics},
	}}
	saver := &fakeSaver{}
	sum := New(fetcher, saver, quietLogger()).Run(context.Background(), []storage.MatchRef{ref(1, 100)})

	if sum.Skipped != 1 || sum.Failed != 0 || sum.Enriched != 0 {
		t.Fatalf("expected one soft skip, got %+v", sum)
	}
	if saver.saved != 0 {
		t.Errorf("nothing should be saved on skip, got %d", saver.saved)
	}
}

func TestRunSkipsIncompleteBoxScore(t *testing.T) {
	// All-zero shooting → possessions 0 → ComputeAdvanced rejects → skip.
	empty := &model.StatisticsResponse{
		Statistics: []model.PeriodStatistics{{
			Period: "ALL",
			Groups: []model.StatisticGroup{{Items: []model.StatisticItem{
				{Name: "Points", Home: "0", Away: "0"},
				{Name: "Field Goals", Home: "0/0", Away: "0/0"},
				{Name: "3-Pointers", Home: "0/0", Away: "0/0"},
				{Name: "Free Throws", Home: "0/0", Away: "0/0"},
				{Name: "Offensive Rebounds", Home: "0", Away: "0"},
				{Name: "Defensive Rebounds", Home: "0", Away: "0"},
				{Name: "Turnovers", Home: "0", Away: "0"},
			}}},
		}},
	}
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{100: {EventID: 100, Stats: empty}}}
	sum := New(fetcher, &fakeSaver{}, quietLogger()).Run(context.Background(), []storage.MatchRef{ref(1, 100)})
	if sum.Skipped != 1 || sum.Enriched != 0 {
		t.Fatalf("expected skip on incomplete box score, got %+v", sum)
	}
}

func TestRunCountsFetchErrorAsFailed(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{
		100: {EventID: 100, Err: errors.New("connection reset")},
	}}
	sum := New(fetcher, &fakeSaver{}, quietLogger()).Run(context.Background(), []storage.MatchRef{ref(1, 100)})
	if sum.Failed != 1 || sum.Skipped != 0 {
		t.Fatalf("expected one failure, got %+v", sum)
	}
}

func TestRunCountsSaveErrorAsFailed(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{100: {EventID: 100, Stats: goodStats()}}}
	saver := &fakeSaver{err: errors.New("db down")}
	sum := New(fetcher, saver, quietLogger()).Run(context.Background(), []storage.MatchRef{ref(1, 100)})
	if sum.Failed != 1 || sum.Enriched != 0 {
		t.Fatalf("expected one failure on save error, got %+v", sum)
	}
}

func TestRunMixedBatch(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{
		100: {EventID: 100, Stats: goodStats()},
		200: {EventID: 200, Err: sofascore.ErrNoStatistics},
		300: {EventID: 300, Err: errors.New("boom")},
	}}
	refs := []storage.MatchRef{ref(1, 100), ref(2, 200), ref(3, 300)}
	sum := New(fetcher, &fakeSaver{}, quietLogger()).Run(context.Background(), refs)

	if sum.Enriched != 1 || sum.Skipped != 1 || sum.Failed != 1 {
		t.Fatalf("unexpected mixed summary: %+v", sum)
	}
}
