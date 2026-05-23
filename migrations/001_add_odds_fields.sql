-- Migration 001: Add bookmaker odds fields to matches table
-- Run via psql (supports multiple statements):
--   psql $DATABASE_URL -f migrations/001_add_odds_fields.sql
--
-- If applying via asyncpg/SQLAlchemy, execute each ALTER TABLE separately.

ALTER TABLE matches ADD COLUMN IF NOT EXISTS total_line_open       FLOAT;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS total_line_source     VARCHAR(32);
ALTER TABLE matches ADD COLUMN IF NOT EXISTS total_line_scraped_at TIMESTAMPTZ;

COMMENT ON COLUMN matches.total_line         IS 'Closing Over/Under line (primary ML feature)';
COMMENT ON COLUMN matches.total_line_open    IS 'Opening Over/Under line (line movement = open - close)';
COMMENT ON COLUMN matches.total_line_source  IS 'Bookmaker: pinnacle | bet365 | 1xbet | …';
COMMENT ON COLUMN matches.total_line_scraped_at IS 'UTC timestamp when line was scraped from Flashscore';
