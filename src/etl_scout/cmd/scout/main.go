// Command scout is the etl_scout entrypoint: it loads config, connects to
// PostgreSQL, and concurrently pulls box-score statistics from Sofascore.
//
// This iteration wires the boilerplate end-to-end. Event IDs are passed via
// -events for a smoke run; persistence of parsed box scores lands in a later
// iteration.
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"basket_pace_lab/etl_scout/internal/config"
	"basket_pace_lab/etl_scout/internal/sofascore"
	"basket_pace_lab/etl_scout/internal/storage"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	eventsRaw := flag.String("events", "", "comma-separated Sofascore event IDs to fetch")
	flag.Parse()

	if err := run(logger, *eventsRaw); err != nil {
		logger.Error("scout failed", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger, eventsRaw string) error {
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

	client := sofascore.NewClient(cfg.Sofascore, logger)
	defer client.Close()

	eventIDs, err := parseEventIDs(eventsRaw)
	if err != nil {
		return err
	}
	if len(eventIDs) == 0 {
		logger.Info("no -events provided; boilerplate is wired, nothing to fetch")
		return nil
	}

	logger.Info("fetching box scores",
		"events", len(eventIDs), "workers", cfg.Sofascore.Workers,
		"rps", cfg.Sofascore.RequestsPerSec)

	results := client.FetchMany(ctx, eventIDs)
	ok, failed := 0, 0
	for _, r := range results {
		if r.Err != nil {
			failed++
			logger.Warn("fetch failed", "event_id", r.EventID, "error", r.Err)
			continue
		}
		ok++
		logger.Info("fetched", "event_id", r.EventID, "periods", len(r.Stats.Statistics))
	}
	logger.Info("done", "ok", ok, "failed", failed)
	return nil
}

func parseEventIDs(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		id, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}
