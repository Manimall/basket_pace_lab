// Package model holds the domain box-score structures the scout produces.
//
// Two layers live here on purpose:
//   - the wire structs (StatisticsResponse …) mirror Sofascore's JSON exactly,
//     so decoding stays a dumb, allocation-free mapping;
//   - the domain structs (TeamBoxScore …) are provider-agnostic and are what
//     the rest of the pipeline (and the DB writer) consume.
package model

import "fmt"

const (
	// FTAToPossessionFactor is Dean Oliver's coefficient: the fraction of a
	// possession consumed by a single free-throw attempt.
	FTAToPossessionFactor = 0.44
	// ratingScale normalises offensive/defensive ratings to points per 100
	// possessions, the standard basketball efficiency unit.
	ratingScale = 100.0
)

// ── Sofascore wire format ──────────────────────────────────────────────────
// Endpoint: GET /event/{id}/statistics
// Shape: { "statistics": [ { "period": "ALL", "groups": [ { "statisticsItems":
//         [ { "name": "Field goals", "home": "40/85", "away": "38/80" } ] } ] } ] }

// StatisticsResponse is the top-level statistics payload.
type StatisticsResponse struct {
	Statistics []PeriodStatistics `json:"statistics"`
}

// PeriodStatistics groups all stat items for one period ("ALL", "1Q", "OT"…).
type PeriodStatistics struct {
	Period string           `json:"period"`
	Groups []StatisticGroup `json:"groups"`
}

// StatisticGroup is Sofascore's UI grouping (Scoring, Rebounds, …).
type StatisticGroup struct {
	GroupName string          `json:"groupName"`
	Items     []StatisticItem `json:"statisticsItems"`
}

// StatisticItem is a single labelled metric. Home/Away are display strings,
// e.g. "40/85 (47%)" for shooting or "13" for turnovers.
type StatisticItem struct {
	Name string `json:"name"`
	Home string `json:"home"`
	Away string `json:"away"`
}

// ── Domain model ───────────────────────────────────────────────────────────

// ShotStat is a made/attempted pair (field goals, three-pointers, free throws).
type ShotStat struct {
	Made      int `json:"made"`
	Attempted int `json:"attempted"`
}

// TeamBoxScore is one team's box score for a single period. Counting stats are
// the raw inputs to the Four Factors / possession model computed downstream.
type TeamBoxScore struct {
	FieldGoals        ShotStat `json:"field_goals"`    // FGM / FGA
	ThreePointers     ShotStat `json:"three_pointers"` // 3PM / 3PA
	FreeThrows        ShotStat `json:"free_throws"`    // FTM / FTA
	Turnovers         int      `json:"turnovers"`
	OffensiveRebounds int      `json:"offensive_rebounds"`
	DefensiveRebounds int      `json:"defensive_rebounds"`
}

// PeriodBoxScore pairs home/away box scores for one period of one match.
type PeriodBoxScore struct {
	Period string       `json:"period"` // "1Q".."4Q", "OT", "ALL"
	Home   TeamBoxScore `json:"home"`
	Away   TeamBoxScore `json:"away"`
}

// MatchBoxScore is the fully-parsed box score for a single Sofascore event.
type MatchBoxScore struct {
	EventID int              `json:"event_id"`
	Periods []PeriodBoxScore `json:"periods"`
}

// ── Advanced metrics ───────────────────────────────────────────────────────

// AdvancedStats are the derived efficiency metrics persisted per team per
// match. TruePace mirrors Possessions (raw possessions, not minute-normalised).
type AdvancedStats struct {
	Possessions            float64 `json:"possessions"`
	TruePace               float64 `json:"true_pace"`
	Turnovers              int     `json:"turnovers"`
	ThreePointersMade      int     `json:"three_pointers_made"`
	ThreePointersAttempted int     `json:"three_pointers_attempted"`
	OffensiveRating        float64 `json:"offensive_rating"`
	DefensiveRating        float64 `json:"defensive_rating"`
}

// Possessions estimates the possessions a team used:
//
//	FGA + 0.44*FTA + Turnovers - OffensiveRebounds
func (b TeamBoxScore) Possessions() float64 {
	return float64(b.FieldGoals.Attempted) +
		FTAToPossessionFactor*float64(b.FreeThrows.Attempted) +
		float64(b.Turnovers) -
		float64(b.OffensiveRebounds)
}

// ComputeAdvanced derives a team's advanced stats from its own box score and
// the final points each side scored. pointsScored is the team's own total;
// pointsAllowed is the opponent's. DRtg uses the opponent's points over THIS
// team's possessions, per the project's pace-symmetric convention.
//
// Returns an error when possessions are non-positive — a rating would then
// divide by zero or go negative, signalling an incomplete box score.
func (b TeamBoxScore) ComputeAdvanced(pointsScored, pointsAllowed int) (AdvancedStats, error) {
	poss := b.Possessions()
	if poss <= 0 {
		return AdvancedStats{}, fmt.Errorf("non-positive possessions (%.2f): incomplete box score", poss)
	}
	return AdvancedStats{
		Possessions:            poss,
		TruePace:               poss,
		Turnovers:              b.Turnovers,
		ThreePointersMade:      b.ThreePointers.Made,
		ThreePointersAttempted: b.ThreePointers.Attempted,
		OffensiveRating:        float64(pointsScored) / poss * ratingScale,
		DefensiveRating:        float64(pointsAllowed) / poss * ratingScale,
	}, nil
}
