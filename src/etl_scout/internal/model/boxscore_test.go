package model

import (
	"math"
	"testing"
)

const epsilon = 1e-6

func approxEqual(t *testing.T, label string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > epsilon {
		t.Errorf("%s: got %.6f, want %.6f", label, got, want)
	}
}

// fixtureHome mirrors the real Sofascore fixture (event 12571063, home side):
// FG 33/73, 3P 9/24, FT 14/18, OREB 11, DREB 28, TOV 10. Final 89-73.
func fixtureHome() TeamBoxScore {
	return TeamBoxScore{
		FieldGoals:        ShotStat{Made: 33, Attempted: 73},
		ThreePointers:     ShotStat{Made: 9, Attempted: 24},
		FreeThrows:        ShotStat{Made: 14, Attempted: 18},
		Turnovers:         10,
		OffensiveRebounds: 11,
		DefensiveRebounds: 28,
	}
}

func TestPossessions(t *testing.T) {
	// 73 + 0.44*18 + 10 - 11 = 79.92
	approxEqual(t, "possessions", fixtureHome().Possessions(), 79.92)
}

func TestComputeAdvanced(t *testing.T) {
	const (
		homePoints = 89
		awayPoints = 73
	)
	adv, err := fixtureHome().ComputeAdvanced(homePoints, awayPoints)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	approxEqual(t, "possessions", adv.Possessions, 79.92)
	approxEqual(t, "true_pace", adv.TruePace, 79.92)
	// ORtg = 89 / 79.92 * 100
	approxEqual(t, "ortg", adv.OffensiveRating, float64(homePoints)/79.92*100.0)
	// DRtg = 73 / 79.92 * 100
	approxEqual(t, "drtg", adv.DefensiveRating, float64(awayPoints)/79.92*100.0)

	if adv.Turnovers != 10 {
		t.Errorf("turnovers: got %d, want 10", adv.Turnovers)
	}
	if adv.ThreePointersMade != 9 || adv.ThreePointersAttempted != 24 {
		t.Errorf("threes: got %d/%d, want 9/24", adv.ThreePointersMade, adv.ThreePointersAttempted)
	}
}

func TestComputeAdvancedSpotChecksRatings(t *testing.T) {
	// Independent round-number check: 100 possessions, 110 scored, 95 allowed.
	box := TeamBoxScore{
		FieldGoals: ShotStat{Made: 40, Attempted: 90}, // FGA 90
		FreeThrows: ShotStat{Made: 10, Attempted: 25}, // 0.44*25 = 11
		Turnovers:  9,                                 // +9
		// OREB 10 → 90 + 11 + 9 - 10 = 100 possessions
		OffensiveRebounds: 10,
	}
	adv, err := box.ComputeAdvanced(110, 95)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	approxEqual(t, "possessions", adv.Possessions, 100.0)
	approxEqual(t, "ortg", adv.OffensiveRating, 110.0)
	approxEqual(t, "drtg", adv.DefensiveRating, 95.0)
}

func TestComputeAdvancedRejectsNonPositivePossessions(t *testing.T) {
	// FGA 0, FTA 0, TOV 0, OREB 5 → possessions = -5.
	box := TeamBoxScore{OffensiveRebounds: 5}
	if _, err := box.ComputeAdvanced(50, 50); err == nil {
		t.Fatal("expected error for non-positive possessions")
	}
}
