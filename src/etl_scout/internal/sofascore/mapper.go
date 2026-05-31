package sofascore

import (
	"fmt"
	"strings"

	"basket_pace_lab/etl_scout/internal/model"
)

// Sofascore metric labels, normalised to lowercase with hyphens as spaces.
// Sofascore's casing and punctuation are not stable ("Free Throws" vs
// "Free throws", "3-Pointers" vs "3 pointers"), so lookups normalise the same
// way — the casing defence the Python parser also applies. NOTE: the statistics
// endpoint carries no "points" item; final scores come from the DB instead.
const (
	periodAll = "ALL" // match-level aggregate period

	statFieldGoal = "field goals"
	statThrees    = "3 pointers"
	statFreeThrow = "free throws"
	statOffReb    = "offensive rebounds"
	statDefReb    = "defensive rebounds"
	statTurnover  = "turnovers"
)

// MatchStatistics is the parsed, match-level box score for both teams. Final
// points are intentionally absent — Sofascore's statistics endpoint omits them,
// so the caller supplies scores from the database.
type MatchStatistics struct {
	Home model.TeamBoxScore
	Away model.TeamBoxScore
}

// ParseMatchStatistics converts a Sofascore statistics payload into per-team
// match box scores, reading the "ALL" period.
//
// It returns a wrapped error (never a partial result) when the "ALL" period is
// missing or any required metric is absent/malformed, so the caller can log a
// single clear slog.Warn and skip the event.
func ParseMatchStatistics(resp *model.StatisticsResponse) (MatchStatistics, error) {
	if resp == nil {
		return MatchStatistics{}, fmt.Errorf("nil statistics response")
	}

	home, away, ok := flattenAllPeriod(resp)
	if !ok {
		return MatchStatistics{}, fmt.Errorf("statistics payload has no %q period", periodAll)
	}

	homeBox, err := buildTeamBoxScore(home)
	if err != nil {
		return MatchStatistics{}, fmt.Errorf("home box score: %w", err)
	}
	awayBox, err := buildTeamBoxScore(away)
	if err != nil {
		return MatchStatistics{}, fmt.Errorf("away box score: %w", err)
	}

	return MatchStatistics{Home: homeBox, Away: awayBox}, nil
}

// flattenAllPeriod collapses the "ALL" period's groups into two name→value
// maps (home, away). Keys are lowercased with hyphens turned into spaces so
// label drift ("3-Pointers" / "3 pointers") still matches. ok is false when no
// "ALL" period exists.
func flattenAllPeriod(resp *model.StatisticsResponse) (home, away map[string]string, ok bool) {
	for _, p := range resp.Statistics {
		if p.Period != periodAll {
			continue
		}
		home = make(map[string]string)
		away = make(map[string]string)
		for _, g := range p.Groups {
			for _, it := range g.Items {
				key := normaliseLabel(it.Name)
				home[key] = it.Home
				away[key] = it.Away
			}
		}
		return home, away, true
	}
	return nil, nil, false
}

func normaliseLabel(name string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(name)), "-", " ")
}

// buildTeamBoxScore parses one team's flattened metric map into a TeamBoxScore.
// Every required metric must be present and valid.
func buildTeamBoxScore(stats map[string]string) (model.TeamBoxScore, error) {
	fieldGoals, err := model.ParseShotStat(stats[statFieldGoal])
	if err != nil {
		return model.TeamBoxScore{}, fmt.Errorf("%s: %w", statFieldGoal, err)
	}
	threes, err := model.ParseShotStat(stats[statThrees])
	if err != nil {
		return model.TeamBoxScore{}, fmt.Errorf("%s: %w", statThrees, err)
	}
	freeThrows, err := model.ParseShotStat(stats[statFreeThrow])
	if err != nil {
		return model.TeamBoxScore{}, fmt.Errorf("%s: %w", statFreeThrow, err)
	}
	turnovers, err := model.ParseCount(stats[statTurnover])
	if err != nil {
		return model.TeamBoxScore{}, fmt.Errorf("%s: %w", statTurnover, err)
	}
	offReb, err := model.ParseCount(stats[statOffReb])
	if err != nil {
		return model.TeamBoxScore{}, fmt.Errorf("%s: %w", statOffReb, err)
	}
	defReb, err := model.ParseCount(stats[statDefReb])
	if err != nil {
		return model.TeamBoxScore{}, fmt.Errorf("%s: %w", statDefReb, err)
	}

	return model.TeamBoxScore{
		FieldGoals:        fieldGoals,
		ThreePointers:     threes,
		FreeThrows:        freeThrows,
		Turnovers:         turnovers,
		OffensiveRebounds: offReb,
		DefensiveRebounds: defReb,
	}, nil
}
