#!/usr/bin/env bash
# auto_grid_trigger.sh
#
# Polls the enrichment queue every 2 minutes.
# Fires grid_search_leagues.py automatically when pending drops to 0
# for all 6 catchup leagues (or after a stall timeout).
#
# Usage:
#   nohup ./scripts/auto_grid_trigger.sh > grid_trigger.log 2>&1 &

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENV_FILE="$PROJECT_ROOT/.env"
GRID_SCRIPT="$SCRIPT_DIR/grid_search_leagues.py"
CSV_OUT="$PROJECT_ROOT/results/final_grid_search.csv"

TARGET_LEAGUES="LegaA,ABA,Israel,PBA_PhilCup,PBA_CommCup,PBA_GovCup"
POLL_INTERVAL=120          # seconds between checks
STALL_TIMEOUT=5400         # 90 min: if no progress in this window → give up and run anyway
LOG_PREFIX="[auto_grid_trigger]"

# ── Load DB credentials from .env ─────────────────────────────────────────────

if [[ ! -f "$ENV_FILE" ]]; then
  echo "$LOG_PREFIX ERROR: .env not found at $ENV_FILE" >&2
  exit 1
fi
set -a && source "$ENV_FILE" && set +a

PSQL_CMD="psql -h ${DB_HOST:-localhost} -p ${DB_PORT:-5434} \
  -U ${DB_USER:-postgres} -d ${DB_NAME:-basket_pace}"

# ── SQL: total pending for the 6 catchup leagues ─────────────────────────────


# Fixed SQL league list (macOS-safe — no dynamic paste/sed needed)
_LEAGUES_SQL="'LegaA','ABA','Israel','PBA_PhilCup','PBA_CommCup','PBA_GovCup'"

pending_sql() {
  # Returns a single integer: sum of pending matches across the 6 leagues.
  PGPASSWORD="${DB_PASSWORD:-postgres}" $PSQL_CMD -t -A -c "
    SELECT COALESCE(SUM(cnt), 0)
    FROM (
      SELECT count(*) FILTER (
        WHERE m.external_id ~ '^[0-9]+\$'
          AND NOT EXISTS (
            SELECT 1 FROM team_match_advanced t WHERE t.match_id = m.id
          )
      ) AS cnt
      FROM matches m
      WHERE m.home_score_final IS NOT NULL
        AND COALESCE(m.tournament_name, 'NBA') IN ($_LEAGUES_SQL)
    ) sub;" 2>/dev/null || echo "-1"
}

per_league_sql() {
  PGPASSWORD="${DB_PASSWORD:-postgres}" $PSQL_CMD -c "
    SELECT
      COALESCE(m.tournament_name, 'NBA') AS league,
      count(*) FILTER (WHERE m.external_id ~ '^[0-9]+\$')              AS scoutable,
      count(*) FILTER (
        WHERE m.external_id ~ '^[0-9]+\$'
          AND EXISTS (SELECT 1 FROM team_match_advanced t WHERE t.match_id = m.id)
      )                                                                 AS enriched,
      count(*) FILTER (
        WHERE m.external_id ~ '^[0-9]+\$'
          AND NOT EXISTS (SELECT 1 FROM team_match_advanced t WHERE t.match_id = m.id)
      )                                                                 AS pending
    FROM matches m
    WHERE m.home_score_final IS NOT NULL
      AND COALESCE(m.tournament_name, 'NBA') IN ($_LEAGUES_SQL)
    GROUP BY 1
    ORDER BY pending DESC;" 2>/dev/null || echo "(DB query failed)"
}

# ── Main loop ─────────────────────────────────────────────────────────────────

echo "$LOG_PREFIX Started $(date '+%Y-%m-%d %H:%M:%S')"
echo "$LOG_PREFIX Target leagues : $TARGET_LEAGUES"
echo "$LOG_PREFIX Poll interval  : ${POLL_INTERVAL}s"
echo "$LOG_PREFIX Stall timeout  : ${STALL_TIMEOUT}s"
echo "$LOG_PREFIX CSV output     : $CSV_OUT"
echo "──────────────────────────────────────────────────────────────────────────"

last_pending=-1
stall_seconds=0
iteration=0

while true; do
  iteration=$((iteration + 1))
  now=$(date '+%Y-%m-%d %H:%M:%S')

  pending=$(pending_sql)

  if [[ "$pending" == "-1" ]]; then
    echo "$LOG_PREFIX [$now] #$iteration DB query failed — will retry in ${POLL_INTERVAL}s"
    sleep "$POLL_INTERVAL"
    continue
  fi

  # Progress check for stall detection
  if [[ "$last_pending" -ge 0 && "$pending" -eq "$last_pending" ]]; then
    stall_seconds=$((stall_seconds + POLL_INTERVAL))
  else
    stall_seconds=0
  fi

  echo "$LOG_PREFIX [$now] #$iteration pending=$pending  stall=${stall_seconds}s"

  # Print per-league breakdown every 5th poll
  if (( iteration % 5 == 1 )); then
    echo "── Per-league status ──"
    per_league_sql
    echo "──────────────────────────────────────────────────────────────────────────"
  fi

  last_pending=$pending

  # ── Exit conditions ──────────────────────────────────────────────────────

  if [[ "$pending" -eq 0 ]]; then
    echo "$LOG_PREFIX [$now] All pending = 0. Launching Grid Search!"
    break
  fi

  if [[ "$stall_seconds" -ge "$STALL_TIMEOUT" ]]; then
    echo "$LOG_PREFIX [$now] Stall timeout (${STALL_TIMEOUT}s) — no progress for ${stall_seconds}s."
    echo "$LOG_PREFIX Running Grid Search anyway on current data."
    break
  fi

  sleep "$POLL_INTERVAL"
done

# ── Launch Grid Search ────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════════════════"
echo "$LOG_PREFIX $(date '+%Y-%m-%d %H:%M:%S') Launching: python $GRID_SCRIPT --csv $CSV_OUT"
echo "══════════════════════════════════════════════════════════════════════════"

mkdir -p "$(dirname "$CSV_OUT")"
cd "$PROJECT_ROOT"
python "$GRID_SCRIPT" --csv "$CSV_OUT"
exit_code=$?

echo "══════════════════════════════════════════════════════════════════════════"
if [[ $exit_code -eq 0 ]]; then
  echo "$LOG_PREFIX Grid Search DONE. Results: $CSV_OUT"
else
  echo "$LOG_PREFIX Grid Search FAILED (exit $exit_code). Check output above." >&2
fi
echo "$LOG_PREFIX Finished $(date '+%Y-%m-%d %H:%M:%S')"
