package sofascore

import (
	"testing"

	"basket_pace_lab/etl_scout/internal/model"
)

func allPeriodResponse(items []model.StatisticItem) *model.StatisticsResponse {
	return &model.StatisticsResponse{
		Statistics: []model.PeriodStatistics{
			{
				Period: periodAll,
				Groups: []model.StatisticGroup{{GroupName: "Scoring", Items: items}},
			},
		},
	}
}

// fixtureItems mirror a live ALL period. Labels use real Sofascore casing and
// punctuation on purpose ("Field goals", "3 pointers", hyphenated variants) so
// the test exercises the mapper's lowercase + hyphen→space normalisation. Note
// there is no "Points" item — the live statistics endpoint omits it.
func fixtureItems() []model.StatisticItem {
	return []model.StatisticItem{
		{Name: "Field goals", Home: "33/73", Away: "27/68"},
		{Name: "3-Pointers", Home: "9/24", Away: "7/21"}, // hyphen → must still match "3 pointers"
		{Name: "Free throws", Home: "14/18", Away: "12/17"},
		{Name: "Offensive rebounds", Home: "11", Away: "8"},
		{Name: "Defensive rebounds", Home: "28", Away: "24"},
		{Name: "Turnovers", Home: "10", Away: "14"},
	}
}

func TestParseMatchStatistics(t *testing.T) {
	got, err := ParseMatchStatistics(allPeriodResponse(fixtureItems()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Home.FieldGoals != (model.ShotStat{Made: 33, Attempted: 73}) {
		t.Errorf("home FG: got %+v", got.Home.FieldGoals)
	}
	if got.Home.ThreePointers != (model.ShotStat{Made: 9, Attempted: 24}) {
		t.Errorf("home 3P (hyphen label): got %+v", got.Home.ThreePointers)
	}
	if got.Away.Turnovers != 14 || got.Away.OffensiveRebounds != 8 {
		t.Errorf("away counts: TOV %d OREB %d", got.Away.Turnovers, got.Away.OffensiveRebounds)
	}
}

func TestParseMatchStatisticsMissingAllPeriod(t *testing.T) {
	resp := &model.StatisticsResponse{
		Statistics: []model.PeriodStatistics{{Period: "1ST"}},
	}
	if _, err := ParseMatchStatistics(resp); err == nil {
		t.Fatal("expected error when ALL period is absent")
	}
}

func TestParseMatchStatisticsMalformedMetric(t *testing.T) {
	items := fixtureItems()
	items[0].Home = "" // break Field goals
	if _, err := ParseMatchStatistics(allPeriodResponse(items)); err == nil {
		t.Fatal("expected error for malformed Field goals string")
	}
}

func TestParseMatchStatisticsNilResponse(t *testing.T) {
	if _, err := ParseMatchStatistics(nil); err == nil {
		t.Fatal("expected error for nil response")
	}
}
