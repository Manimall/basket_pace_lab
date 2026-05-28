// Package storage owns the PostgreSQL connection pool shared by the scout and
// the writers that persist parsed box-score metrics.
package storage

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"basket_pace_lab/etl_scout/internal/config"
	"basket_pace_lab/etl_scout/internal/model"
)

// execer is the slice of pgx we depend on. Both *pgxpool.Pool and a test fake
// satisfy it, so SaveAdvancedStats is unit-testable without a live database.
type execer interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
}

// Pool wraps a pgx connection pool. It is safe for concurrent use by the
// worker pool — pgxpool hands out one connection per acquire.
type Pool struct {
	*pgxpool.Pool
	log *slog.Logger
}

// New opens a pgx pool against the configured database and verifies
// connectivity with a Ping before returning. The caller owns Close().
func New(ctx context.Context, cfg config.DBConfig, log *slog.Logger) (*Pool, error) {
	pool, err := pgxpool.New(ctx, cfg.DSN())
	if err != nil {
		return nil, fmt.Errorf("open pgx pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping database %s:%s/%s: %w",
			cfg.Host, cfg.Port, cfg.Name, err)
	}
	return &Pool{Pool: pool, log: log}, nil
}

const upsertAdvancedSQL = `
INSERT INTO team_match_advanced (
    match_id, team_id, possessions, true_pace, turnovers,
    three_pointers_made, three_pointers_attempted,
    offensive_rating, defensive_rating
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (match_id, team_id) DO UPDATE SET
    possessions              = EXCLUDED.possessions,
    true_pace                = EXCLUDED.true_pace,
    turnovers                = EXCLUDED.turnovers,
    three_pointers_made      = EXCLUDED.three_pointers_made,
    three_pointers_attempted = EXCLUDED.three_pointers_attempted,
    offensive_rating         = EXCLUDED.offensive_rating,
    defensive_rating         = EXCLUDED.defensive_rating,
    updated_at               = now()`

// SaveAdvancedStats upserts one team's advanced metrics for a single match into
// team_match_advanced (insert on first sight, update on re-scrape). Failures are
// logged with full context via slog.Error and returned for the caller to handle.
func (p *Pool) SaveAdvancedStats(ctx context.Context, matchID, teamID int64, adv model.AdvancedStats) error {
	if err := upsertAdvanced(ctx, p.Pool, matchID, teamID, adv); err != nil {
		p.log.Error("save advanced stats failed",
			"match_id", matchID, "team_id", teamID, "error", err)
		return err
	}
	p.log.Debug("saved advanced stats", "match_id", matchID, "team_id", teamID)
	return nil
}

// upsertAdvanced is the execer-driven core, separated so tests can inject a
// fake and assert the SQL/argument mapping without a real database.
func upsertAdvanced(ctx context.Context, ex execer, matchID, teamID int64, adv model.AdvancedStats) error {
	_, err := ex.Exec(ctx, upsertAdvancedSQL,
		matchID, teamID, adv.Possessions, adv.TruePace, adv.Turnovers,
		adv.ThreePointersMade, adv.ThreePointersAttempted,
		adv.OffensiveRating, adv.DefensiveRating)
	if err != nil {
		return fmt.Errorf("upsert advanced stats (match %d, team %d): %w", matchID, teamID, err)
	}
	return nil
}
