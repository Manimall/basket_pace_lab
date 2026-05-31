// Command scout enriches the matches database with box-score advanced metrics.
//
// It selects finished matches in the configured leagues that still lack a row
// in team_match_advanced, fetches each one's Sofascore box score concurrently,
// computes Possessions / ORtg / DRtg / 3PM / 3PA, and upserts the results.
//
// With -loop the process becomes a resilient daemon: on a circuit-breaker abort
// (sustained 403 / IP throttle) it sleeps for the configured cooldown and then
// resumes (the NOT EXISTS guard makes each pass pick up where the last stopped),
// instead of exiting. SIGINT/SIGTERM cancels everything, including a cooldown.
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"basket_pace_lab/etl_scout/internal/config"
	"basket_pace_lab/etl_scout/internal/ctxutil"
	"basket_pace_lab/etl_scout/internal/enrich"
	"basket_pace_lab/etl_scout/internal/sofascore"
	"basket_pace_lab/etl_scout/internal/storage"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	limitFlag := flag.Int("limit", -1, "override max matches to enrich this run (-1 = use config, 0 = no limit)")
	loopFlag := flag.Bool("loop", false, "run as a daemon: on circuit-breaker abort, cooldown and resume instead of exiting")
	flag.Parse()

	if err := run(logger, *limitFlag, *loopFlag); err != nil {
		logger.Error("scout failed", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger, limitFlag int, loop bool) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	// Cancel everything (in-flight fetches AND cooldown sleeps) on SIGINT/SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool, err := storage.New(ctx, cfg.DB, logger)
	if err != nil {
		return err
	}
	defer pool.Close()
	logger.Info("connected to postgres", "host", cfg.DB.Host, "db", cfg.DB.Name)

	client, err := sofascore.NewClient(cfg.Sofascore, logger)
	if err != nil {
		return err
	}
	defer client.Close()

	limit := cfg.Scout.MatchLimit
	if limitFlag >= 0 {
		limit = limitFlag
	}
	enricher := enrich.New(client, pool, logger, cfg.Scout.MaxConsecutiveFails, cfg.Scout.ProgressEvery)

	for iteration := 1; ; iteration++ {
		refs, err := pool.FetchMatchesNeedingAdvanced(ctx, cfg.Scout.Leagues, cfg.Scout.SeasonStart, limit)
		if err != nil {
			return err
		}
		if len(refs) == 0 {
			logger.Info("nothing to enrich", "leagues", cfg.Scout.Leagues, "season_start", cfg.Scout.SeasonStart)
			return nil
		}

		logger.Info("starting enrichment",
			"iteration", iteration, "matches", len(refs),
			"leagues", cfg.Scout.Leagues, "season_start", cfg.Scout.SeasonStart,
			"workers", cfg.Sofascore.Workers, "rps", cfg.Sofascore.RequestsPerSec, "loop", loop)

		start := time.Now()
		summary := enricher.Run(ctx, refs)
		logger.Info("enrichment complete",
			"iteration", iteration,
			"enriched", summary.Enriched, "skipped", summary.Skipped, "failed", summary.Failed,
			"total", len(refs), "aborted", summary.Aborted,
			"elapsed_sec", time.Since(start).Round(time.Millisecond).Seconds())

		if ctx.Err() != nil {
			logger.Info("context cancelled; stopping")
			return nil
		}
		// Stop on a single-shot run, or once a pass completes without a block
		// (everything reachable is enriched; remaining skips have no box score).
		if !loop || !summary.Aborted {
			return nil
		}

		// -loop + circuit-breaker abort (IP throttle): deep cooldown, then resume.
		// The next pass re-queries (NOT EXISTS) and a fresh Run resets the counter.
		logger.Warn("circuit breaker aborted; cooling down before resume",
			"cooldown", cfg.Scout.LoopCooldown)
		if err := ctxutil.Sleep(ctx, cfg.Scout.LoopCooldown); err != nil {
			logger.Info("context cancelled during cooldown; stopping")
			return nil
		}
	}
}
