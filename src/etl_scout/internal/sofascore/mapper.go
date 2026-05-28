package sofascore

import (
	"fmt"

	"basket_pace_lab/etl_scout/internal/model"
)

// Sofascore period + metric labels (verified against a live statistics payload).
const (
	periodAll = "ALL" // match-level aggregate period

	statPoints    = "Points"
	statFieldGoal = "Field Goals"
	statThrees    = "3-Pointers"
	statFreeThrow = "Free Throws"
	statOffReb    = "Offensive Rebounds"
	statDefReb    = "Defensive Rebounds"
	statTurnover  = "Turnovers"
)

// MatchStatistics is the parsed, match-level box score for both teams.
type MatchStatistics struct {
	Home       model.TeamBoxScore
	Away       model.TeamBoxScore
	HomePoints int
	AwayPoints int
}

// ParseMatchStatistics converts a Sofascore statistics payload into per-team
// match box scores plus each side's final points, reading the "ALL" period.
//
// It returns a wrapped error (never a partial result) when the "ALL" period is
// missing or any required metric is absent/malformed, so the caller can log a
// single clear slog.Error and skip the event.
func ParseMatchStatistics(resp *model.StatisticsResponse) (MatchStatistics, error) {
	if resp == nil {
		return MatchStatistics{}, fmt.Errorf("nil statistics response")
	}

	home, away, ok := flattenAllPeriod(resp)
	if !ok {
		return MatchStatistics{}, fmt.Errorf("statistics payload has no %q period", periodAll)
	}

	homeBox, homePts, err := buildTeamBoxScore(home)
	if err != nil {
		return MatchStatistics{}, fmt.Errorf("home box score: %w", err)
	}
	awayBox, awayPts, err := buildTeamBoxScore(away)
	if err != nil {
		return MatchStatistics{}, fmt.Errorf("away box score: %w", err)
	}

	return MatchStatistics{
		Home:       homeBox,
		Away:       awayBox,
		HomePoints: homePts,
		AwayPoints: awayPts,
	}, nil
}

// flattenAllPeriod collapses the "ALL" period's groups into two name→value
// maps (home, away). ok is false when no "ALL" period exists.
func flattenAllPeriod(resp *model.StatisticsResponse) (home, away map[string]string, ok bool) {
	for _, p := range resp.Statistics {
		if p.Period != periodAll {
			continue
		}
		home = make(map[string]string)
		away = make(map[string]string)
		for _, g := range p.Groups {
			for _, it := range g.Items {
				home[it.Name] = it.Home
				away[it.Name] = it.Away
			}
		}
		return home, away, true
	}
	return nil, nil, false
}

// buildTeamBoxScore parses one team's flattened metric map into a TeamBoxScore
// and the team's points. Every required metric must be present and valid.
func buildTeamBoxScore(stats map[string]string) (model.TeamBoxScore, int, error) {
	fieldGoals, err := model.ParseShotStat(stats[statFieldGoal])
	if err != nil {
		return model.TeamBoxScore{}, 0, fmt.Errorf("%s: %w", statFieldGoal, err)
	}
	threes, err := model.ParseShotStat(stats[statThrees])
	if err != nil {
		return model.TeamBoxScore{}, 0, fmt.Errorf("%s: %w", statThrees, err)
	}
	freeThrows, err := model.ParseShotStat(stats[statFreeThrow])
	if err != nil {
		return model.TeamBoxScore{}, 0, fmt.Errorf("%s: %w", statFreeThrow, err)
	}
	turnovers, err := model.ParseCount(stats[statTurnover])
	if err != nil {
		return model.TeamBoxScore{}, 0, fmt.Errorf("%s: %w", statTurnover, err)
	}
	offReb, err := model.ParseCount(stats[statOffReb])
	if err != nil {
		return model.TeamBoxScore{}, 0, fmt.Errorf("%s: %w", statOffReb, err)
	}
	defReb, err := model.ParseCount(stats[statDefReb])
	if err != nil {
		return model.TeamBoxScore{}, 0, fmt.Errorf("%s: %w", statDefReb, err)
	}
	points, err := model.ParseCount(stats[statPoints])
	if err != nil {
		return model.TeamBoxScore{}, 0, fmt.Errorf("%s: %w", statPoints, err)
	}

	return model.TeamBoxScore{
		FieldGoals:        fieldGoals,
		ThreePointers:     threes,
		FreeThrows:        freeThrows,
		Turnovers:         turnovers,
		OffensiveRebounds: offReb,
		DefensiveRebounds: defReb,
	}, points, nil
}
