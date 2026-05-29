"""Raw SQL for the per-league feature pipeline loader.

Kept separate from :mod:`src.features.score_features` so the loader/feature
logic stays focused (and under the module line limit). Both queries are
season-gated on ``:current_season_start`` so every consumer sees the same
single-season cohort (past seasons are excluded at the source).
"""
from __future__ import annotations

# Matches + box-score advanced (team_match_advanced) pivoted to home/away via
# two LEFT JOINs keyed on (match_id, team_id). The box columns are NULL for any
# match/league the Go scout has not backfilled.
SQL_MATCHES: str = """
    SELECT
        m.id           AS match_id,
        m.scheduled_at,
        m.home_team_id,
        m.away_team_id,
        m.home_score_final,
        m.away_score_final,
        m.season_type::text AS season_type,
        COALESCE(m.tournament_name, 'NBA') AS league,
        m.total_line,
        m.total_line_open,
        ha.possessions              AS home_possessions,
        aa.possessions              AS away_possessions,
        ha.turnovers                AS home_turnovers,
        aa.turnovers                AS away_turnovers,
        ha.three_pointers_made      AS home_three_pointers_made,
        aa.three_pointers_made      AS away_three_pointers_made,
        ha.three_pointers_attempted AS home_three_pointers_attempted,
        aa.three_pointers_attempted AS away_three_pointers_attempted,
        ha.offensive_rating         AS home_offensive_rating,
        aa.offensive_rating         AS away_offensive_rating,
        ha.defensive_rating         AS home_defensive_rating,
        aa.defensive_rating         AS away_defensive_rating
    FROM matches m
    LEFT JOIN team_match_advanced ha
           ON ha.match_id = m.id AND ha.team_id = m.home_team_id
    LEFT JOIN team_match_advanced aa
           ON aa.match_id = m.id AND aa.team_id = m.away_team_id
    WHERE m.home_score_final IS NOT NULL
      AND m.away_score_final IS NOT NULL
      AND m.has_quarter_breakdown = TRUE
      AND m.scheduled_at >= :current_season_start
    ORDER BY m.scheduled_at
"""

# Per-quarter (Q1-Q4 regulation only) scores + box-score on the same season gate.
SQL_QS: str = """
    SELECT qs.match_id, qs.period_number,
           qs.home_score,        qs.away_score,
           qs.home_pace,         qs.away_pace,
           qs.home_possessions,  qs.away_possessions,
           qs.home_turnovers,    qs.away_turnovers
    FROM quarter_stats qs
    JOIN matches m ON m.id = qs.match_id
    WHERE qs.period_type::text = 'QUARTER'
      AND qs.period_number IN (1, 2, 3, 4)
      AND m.scheduled_at >= :current_season_start
    ORDER BY qs.match_id, qs.period_number
"""
