// Package enrich orchestrates the box-score enrichment pipeline:
// concurrent fetch → parse → advanced-metric calc → upsert.
package enrich

import (
	"context"
	"errors"
	"log/slog"

	"basket_pace_lab/etl_scout/internal/model"
	"basket_pace_lab/etl_scout/internal/sofascore"
	"basket_pace_lab/etl_scout/internal/storage"
)

// Fetcher concurrently downloads statistics for many events (the worker pool).
type Fetcher interface {
	FetchMany(ctx context.Context, eventIDs []int) []sofascore.Result
}

// Saver persists one team's advanced stats for a match.
type Saver interface {
	SaveAdvancedStats(ctx context.Context, matchID, teamID int64, adv model.AdvancedStats) error
}

// Summary is the end-of-run tally returned to the caller for the final log.
type Summary struct {
	Enriched int // both team rows saved
	Skipped  int // no/incomplete box score on Sofascore (soft skip)
	Failed   int // fetch or DB error (hard failure)
}

// Enricher ties the fetcher and saver together. It owns no state beyond its
// collaborators, so it is cheap to construct per run.
type Enricher struct {
	fetcher Fetcher
	saver   Saver
	log     *slog.Logger
}

// New builds an Enricher.
func New(fetcher Fetcher, saver Saver, log *slog.Logger) *Enricher {
	return &Enricher{fetcher: fetcher, saver: saver, log: log}
}

type outcome int

const (
	outcomeEnriched outcome = iota
	outcomeSkipped
	outcomeFailed
)

// Run fetches all referenced events concurrently, then parses, computes, and
// upserts each result. Per-match problems are logged and tallied; they never
// abort the whole run.
func (e *Enricher) Run(ctx context.Context, refs []storage.MatchRef) Summary {
	byEvent := make(map[int]storage.MatchRef, len(refs))
	ids := make([]int, 0, len(refs))
	for _, r := range refs {
		byEvent[r.EventID] = r
		ids = append(ids, r.EventID)
	}

	var sum Summary
	for _, res := range e.fetcher.FetchMany(ctx, ids) {
		ref, ok := byEvent[res.EventID]
		if !ok {
			e.log.Warn("result for unknown event id", "event_id", res.EventID)
			continue
		}
		switch e.process(ctx, ref, res) {
		case outcomeEnriched:
			sum.Enriched++
		case outcomeSkipped:
			sum.Skipped++
		case outcomeFailed:
			sum.Failed++
		}
	}
	return sum
}

// process handles a single fetched event: classify fetch errors, parse the box
// score, compute both teams' advanced metrics, and upsert. Missing/incomplete
// box scores are a soft skip (slog.Warn); fetch/DB errors are failures.
func (e *Enricher) process(ctx context.Context, ref storage.MatchRef, res sofascore.Result) outcome {
	if res.Err != nil {
		if errors.Is(res.Err, sofascore.ErrNoStatistics) {
			e.log.Warn("no box score on sofascore; skipping",
				"match_id", ref.ID, "event_id", ref.EventID)
			return outcomeSkipped
		}
		e.log.Warn("fetch failed; skipping",
			"match_id", ref.ID, "event_id", ref.EventID, "error", res.Err)
		return outcomeFailed
	}

	stats, err := sofascore.ParseMatchStatistics(res.Stats)
	if err != nil {
		e.log.Warn("empty/unparseable box score; skipping",
			"match_id", ref.ID, "event_id", ref.EventID, "reason", err)
		return outcomeSkipped
	}

	homeAdv, err := stats.Home.ComputeAdvanced(stats.HomePoints, stats.AwayPoints)
	if err != nil {
		e.log.Warn("incomplete home box score; skipping",
			"match_id", ref.ID, "event_id", ref.EventID, "reason", err)
		return outcomeSkipped
	}
	awayAdv, err := stats.Away.ComputeAdvanced(stats.AwayPoints, stats.HomePoints)
	if err != nil {
		e.log.Warn("incomplete away box score; skipping",
			"match_id", ref.ID, "event_id", ref.EventID, "reason", err)
		return outcomeSkipped
	}

	if err := e.saver.SaveAdvancedStats(ctx, ref.ID, ref.HomeTeamID, homeAdv); err != nil {
		return outcomeFailed
	}
	if err := e.saver.SaveAdvancedStats(ctx, ref.ID, ref.AwayTeamID, awayAdv); err != nil {
		return outcomeFailed
	}
	return outcomeEnriched
}
