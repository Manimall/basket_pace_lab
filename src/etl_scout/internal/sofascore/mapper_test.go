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

// fixtureItems mirror event 12571063's ALL period (home 89, away 73).
func fixtureItems() []model.StatisticItem {
	return []model.StatisticItem{
		{Name: statPoints, Home: "89", Away: "73"},
		{Name: statFieldGoal, Home: "33/73", Away: "27/68"},
		{Name: statThrees, Home: "9/24", Away: "7/21"},
		{Name: statFreeThrow, Home: "14/18", Away: "12/17"},
		{Name: statOffReb, Home: "11", Away: "8"},
		{Name: statDefReb, Home: "28", Away: "24"},
		{Name: statTurnover, Home: "10", Away: "14"},
	}
}

func TestParseMatchStatistics(t *testing.T) {
	got, err := ParseMatchStatistics(allPeriodResponse(fixtureItems()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.HomePoints != 89 || got.AwayPoints != 73 {
		t.Errorf("points: got %d-%d, want 89-73", got.HomePoints, got.AwayPoints)
	}
	if got.Home.FieldGoals != (model.ShotStat{Made: 33, Attempted: 73}) {
		t.Errorf("home FG: got %+v", got.Home.FieldGoals)
	}
	if got.Home.ThreePointers != (model.ShotStat{Made: 9, Attempted: 24}) {
		t.Errorf("home 3P: got %+v", got.Home.ThreePointers)
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
	items[1].Home = "" // break Field Goals
	if _, err := ParseMatchStatistics(allPeriodResponse(items)); err == nil {
		t.Fatal("expected error for malformed Field Goals string")
	}
}

func TestParseMatchStatisticsNilResponse(t *testing.T) {
	if _, err := ParseMatchStatistics(nil); err == nil {
		t.Fatal("expected error for nil response")
	}
}
