// Command scout enriches the matches database with box-score advanced metrics.
//
// It selects finished matches in the configured leagues that still lack a row
// in team_match_advanced, fetches each one's Sofascore box score concurrently,
// computes Possessions / ORtg / DRtg / 3PM / 3PA, and upserts the results.
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
	"basket_pace_lab/etl_scout/internal/enrich"
	"basket_pace_lab/etl_scout/internal/sofascore"
	"basket_pace_lab/etl_scout/internal/storage"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	limitFlag := flag.Int("limit", -1, "override max matches to enrich this run (-1 = use config, 0 = no limit)")
	flag.Parse()

	if err := run(logger, *limitFlag); err != nil {
		logger.Error("scout failed", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger, limitFlag int) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	// Cancel the whole run on SIGINT/SIGTERM for clean shutdown.
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

	refs, err := pool.FetchMatchesNeedingAdvanced(ctx, cfg.Scout.Leagues, cfg.Scout.SeasonStart, limit)
	if err != nil {
		return err
	}
	if len(refs) == 0 {
		logger.Info("nothing to enrich", "leagues", cfg.Scout.Leagues, "season_start", cfg.Scout.SeasonStart)
		return nil
	}

	logger.Info("starting enrichment",
		"matches", len(refs), "leagues", cfg.Scout.Leagues, "season_start", cfg.Scout.SeasonStart,
		"workers", cfg.Sofascore.Workers, "rps", cfg.Sofascore.RequestsPerSec)

	start := time.Now()
	enricher := enrich.New(client, pool, logger, cfg.Scout.MaxConsecutiveFails, cfg.Scout.ProgressEvery)
	summary := enricher.Run(ctx, refs)
	elapsed := time.Since(start)

	logger.Info("enrichment complete",
		"enriched", summary.Enriched,
		"skipped", summary.Skipped,
		"failed", summary.Failed,
		"total", len(refs),
		"aborted", summary.Aborted,
		"elapsed_sec", elapsed.Round(time.Millisecond).Seconds())
	return nil
}
