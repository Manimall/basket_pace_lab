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

// Fetcher streams statistics for many events to a handler (the worker pool).
type Fetcher interface {
	FetchManyFunc(ctx context.Context, eventIDs []int, handle func(sofascore.Result) bool)
}

// Saver persists one team's advanced stats for a match.
type Saver interface {
	SaveAdvancedStats(ctx context.Context, matchID, teamID int64, adv model.AdvancedStats) error
}

// Summary is the end-of-run tally returned to the caller for the final log.
type Summary struct {
	Enriched int  // both team rows saved
	Skipped  int  // no/incomplete box score on Sofascore (soft skip)
	Failed   int  // fetch or DB error (hard failure)
	Aborted  bool // circuit breaker tripped before all matches were processed
}

// Enricher ties the fetcher and saver together. It owns no state beyond its
// collaborators and tuning knobs, so it is cheap to construct per run.
type Enricher struct {
	fetcher       Fetcher
	saver         Saver
	log           *slog.Logger
	maxConsecFail int // circuit breaker threshold
	progressEvery int // progress log cadence
}

// New builds an Enricher. maxConsecFail trips the circuit breaker (a sustained
// block, e.g. an IP throttle); progressEvery sets the progress-log cadence.
func New(fetcher Fetcher, saver Saver, log *slog.Logger, maxConsecFail, progressEvery int) *Enricher {
	return &Enricher{
		fetcher:       fetcher,
		saver:         saver,
		log:           log,
		maxConsecFail: maxConsecFail,
		progressEvery: progressEvery,
	}
}

type outcome int

const (
	outcomeEnriched outcome = iota
	outcomeSkipped
	outcomeFailed
)

// Run streams each referenced event through fetch → parse → compute → upsert,
// persisting progress incrementally. A run-ending block (many consecutive
// failures) trips the circuit breaker and stops early; already-saved matches
// stay, so a re-run resumes via the NOT EXISTS guard. Soft skips (no/incomplete
// box score) reset the consecutive counter — they mean the server answered.
func (e *Enricher) Run(ctx context.Context, refs []storage.MatchRef) Summary {
	byEvent := make(map[int]storage.MatchRef, len(refs))
	ids := make([]int, 0, len(refs))
	for _, r := range refs {
		byEvent[r.EventID] = r
		ids = append(ids, r.EventID)
	}

	var sum Summary
	consecutiveFails := 0

	e.fetcher.FetchManyFunc(ctx, ids, func(res sofascore.Result) bool {
		ref, ok := byEvent[res.EventID]
		if !ok {
			e.log.Warn("result for unknown event id", "event_id", res.EventID)
			return true
		}

		switch e.process(ctx, ref, res) {
		case outcomeEnriched:
			sum.Enriched++
			consecutiveFails = 0
		case outcomeSkipped:
			sum.Skipped++
			consecutiveFails = 0
		case outcomeFailed:
			sum.Failed++
			consecutiveFails++
		}

		if processed := sum.Enriched + sum.Skipped + sum.Failed; processed%e.progressEvery == 0 {
			e.log.Info("progress",
				"processed", processed, "enriched", sum.Enriched,
				"skipped", sum.Skipped, "failed", sum.Failed)
		}

		if consecutiveFails >= e.maxConsecFail {
			e.log.Error("circuit breaker tripped; aborting (progress saved, re-run to resume)",
				"consecutive_failures", consecutiveFails)
			sum.Aborted = true
			return false
		}
		return true
	})
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

	// Points come from the DB (statistics endpoint omits them): home ORtg uses
	// home points scored vs away points allowed, and vice versa for the away side.
	homeAdv, err := stats.Home.ComputeAdvanced(ref.HomeScore, ref.AwayScore)
	if err != nil {
		e.log.Warn("incomplete home box score; skipping",
			"match_id", ref.ID, "event_id", ref.EventID, "reason", err)
		return outcomeSkipped
	}
	awayAdv, err := stats.Away.ComputeAdvanced(ref.AwayScore, ref.HomeScore)
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
