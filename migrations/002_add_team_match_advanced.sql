-- Migration 002: per-team, per-match advanced (Four Factors) stats.
--
-- Populated by the Go etl_scout service (src/etl_scout) from box-score feeds
-- (Sofascore) so that leagues whose per-quarter box score is otherwise NULL
-- (EuroLeague, ABA, …) gain possessions / pace / efficiency metrics.
--
-- Team-oriented (one row per team per match) on purpose: ORtg/DRtg are
-- match-level aggregates and map cleanly to (match_id, team_id), unlike the
-- home/away period rows in quarter_stats.
--
-- Run via psql (supports multiple statements):
--   psql $DATABASE_URL -f migrations/002_add_team_match_advanced.sql

CREATE TABLE IF NOT EXISTS team_match_advanced (
    match_id                 BIGINT  NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    team_id                  BIGINT  NOT NULL REFERENCES teams(id)   ON DELETE RESTRICT,
    possessions              DOUBLE PRECISION,
    true_pace                DOUBLE PRECISION,
    turnovers                SMALLINT,
    three_pointers_made      SMALLINT,
    three_pointers_attempted SMALLINT,
    offensive_rating         DOUBLE PRECISION,
    defensive_rating         DOUBLE PRECISION,
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, team_id)
);

CREATE INDEX IF NOT EXISTS ix_team_match_advanced_team ON team_match_advanced (team_id);

COMMENT ON TABLE  team_match_advanced                    IS 'Per-team, per-match advanced metrics from box-score feeds; fills NULL gaps for non-NBA leagues.';
COMMENT ON COLUMN team_match_advanced.possessions        IS 'FGA + 0.44*FTA + Turnovers - OffReb.';
COMMENT ON COLUMN team_match_advanced.true_pace          IS 'Raw possessions (not minute-normalised); mirrors possessions.';
COMMENT ON COLUMN team_match_advanced.offensive_rating   IS 'Points scored / possessions * 100.';
COMMENT ON COLUMN team_match_advanced.defensive_rating   IS 'Opponent points / team possessions * 100.';
