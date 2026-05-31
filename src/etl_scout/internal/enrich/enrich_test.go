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

// fakeFetcher streams canned results in id order, honouring the stop signal.
type fakeFetcher struct{ byID map[int]sofascore.Result }

func (f fakeFetcher) FetchManyFunc(_ context.Context, ids []int, handle func(sofascore.Result) bool) {
	for _, id := range ids {
		r, ok := f.byID[id]
		if !ok {
			continue
		}
		if !handle(r) {
			return
		}
	}
}

// defaults for tests: high breaker threshold, progress log effectively off.
const (
	testMaxConsecFail = 5
	testProgressEvery = 100
)

func newEnricher(f Fetcher, s Saver) *Enricher {
	return New(f, s, quietLogger(), testMaxConsecFail, testProgressEvery)
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

// goodStats is a complete box score that yields valid metrics. Labels match the
// live Sofascore endpoint (no "Points" item; final scores come from MatchRef).
func goodStats() *model.StatisticsResponse {
	item := func(name, h, a string) model.StatisticItem {
		return model.StatisticItem{Name: name, Home: h, Away: a}
	}
	return &model.StatisticsResponse{
		Statistics: []model.PeriodStatistics{{
			Period: "ALL",
			Groups: []model.StatisticGroup{{Items: []model.StatisticItem{
				item("Field goals", "33/73", "27/68"),
				item("3 pointers", "9/24", "7/21"),
				item("Free throws", "14/18", "12/17"),
				item("Offensive rebounds", "11", "8"),
				item("Defensive rebounds", "28", "24"),
				item("Turnovers", "10", "14"),
			}}},
		}},
	}
}

func ref(id int64, eventID int) storage.MatchRef {
	return storage.MatchRef{
		ID: id, EventID: eventID,
		HomeTeamID: id*10 + 1, AwayTeamID: id*10 + 2,
		HomeScore: 89, AwayScore: 73,
	}
}

func TestRunEnrichesGoodMatch(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{
		100: {EventID: 100, Stats: goodStats()},
	}}
	saver := &fakeSaver{}
	enr := newEnricher(fetcher, saver)

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
	sum := newEnricher(fetcher, saver).Run(context.Background(), []storage.MatchRef{ref(1, 100)})

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
	sum := newEnricher(fetcher, &fakeSaver{}).Run(context.Background(), []storage.MatchRef{ref(1, 100)})
	if sum.Skipped != 1 || sum.Enriched != 0 {
		t.Fatalf("expected skip on incomplete box score, got %+v", sum)
	}
}

func TestRunCountsFetchErrorAsFailed(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{
		100: {EventID: 100, Err: errors.New("connection reset")},
	}}
	sum := newEnricher(fetcher, &fakeSaver{}).Run(context.Background(), []storage.MatchRef{ref(1, 100)})
	if sum.Failed != 1 || sum.Skipped != 0 {
		t.Fatalf("expected one failure, got %+v", sum)
	}
}

func TestRunCountsSaveErrorAsFailed(t *testing.T) {
	fetcher := fakeFetcher{byID: map[int]sofascore.Result{100: {EventID: 100, Stats: goodStats()}}}
	saver := &fakeSaver{err: errors.New("db down")}
	sum := newEnricher(fetcher, saver).Run(context.Background(), []storage.MatchRef{ref(1, 100)})
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
	sum := newEnricher(fetcher, &fakeSaver{}).Run(context.Background(), refs)

	if sum.Enriched != 1 || sum.Skipped != 1 || sum.Failed != 1 {
		t.Fatalf("unexpected mixed summary: %+v", sum)
	}
}

func TestRunCircuitBreakerAbortsOnConsecutiveFailures(t *testing.T) {
	// 10 matches, all hard fetch failures; breaker threshold 3 must stop early.
	byID := make(map[int]sofascore.Result, 10)
	refs := make([]storage.MatchRef, 0, 10)
	for i := 1; i <= 10; i++ {
		byID[i] = sofascore.Result{EventID: i, Err: errors.New("403 throttled")}
		refs = append(refs, ref(int64(i), i))
	}
	const breaker = 3
	enr := New(fakeFetcher{byID: byID}, &fakeSaver{}, quietLogger(), breaker, testProgressEvery)

	sum := enr.Run(context.Background(), refs)

	if !sum.Aborted {
		t.Errorf("expected circuit breaker to abort the run")
	}
	if sum.Failed != breaker {
		t.Errorf("expected exactly %d failures before abort, got %d", breaker, sum.Failed)
	}
}

func TestRunBreakerNotTrippedWhenSkipsResetCounter(t *testing.T) {
	// fail, skip, fail, skip, fail … never reaches 2 consecutive fails.
	byID := make(map[int]sofascore.Result, 6)
	refs := make([]storage.MatchRef, 0, 6)
	for i := 1; i <= 6; i++ {
		if i%2 == 1 {
			byID[i] = sofascore.Result{EventID: i, Err: errors.New("network blip")}
		} else {
			byID[i] = sofascore.Result{EventID: i, Err: sofascore.ErrNoStatistics}
		}
		refs = append(refs, ref(int64(i), i))
	}
	enr := New(fakeFetcher{byID: byID}, &fakeSaver{}, quietLogger(), 2, testProgressEvery)

	sum := enr.Run(context.Background(), refs)

	if sum.Aborted {
		t.Errorf("breaker should not trip when skips reset the counter: %+v", sum)
	}
	if sum.Failed != 3 || sum.Skipped != 3 {
		t.Errorf("expected 3 failed + 3 skipped, got %+v", sum)
	}
}
