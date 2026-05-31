-- Мониторинг очереди обогащения Go-скаута.
-- Запускать: psql -h localhost -p 5434 -U postgres -d basket_pace -f scripts/enrichment_progress.sql
-- Или одной строкой:
--   PGPASSWORD=postgres psql -h localhost -p 5434 -U postgres -d basket_pace \
--     -c "$(cat scripts/enrichment_progress.sql)"

SELECT
    COALESCE(m.tournament_name, 'NBA')                                   AS league,

    -- Всего матчей с числовым Sofascore ID (видны скауту)
    count(*) FILTER (WHERE m.external_id ~ '^[0-9]+$')                  AS total_scoutable,

    -- Уже обогащены (есть строка в team_match_advanced)
    count(*) FILTER (
        WHERE m.external_id ~ '^[0-9]+$'
          AND EXISTS (SELECT 1 FROM team_match_advanced t WHERE t.match_id = m.id)
    )                                                                    AS enriched,

    -- Остаток в очереди
    count(*) FILTER (
        WHERE m.external_id ~ '^[0-9]+$'
          AND NOT EXISTS (SELECT 1 FROM team_match_advanced t WHERE t.match_id = m.id)
    )                                                                    AS pending,

    -- Прогресс в процентах
    round(
        100.0 * count(*) FILTER (
            WHERE m.external_id ~ '^[0-9]+$'
              AND EXISTS (SELECT 1 FROM team_match_advanced t WHERE t.match_id = m.id)
        ) / NULLIF(
            count(*) FILTER (WHERE m.external_id ~ '^[0-9]+$'), 0
        ), 1
    )                                                                    AS pct_done,

    -- Матчи с Flashscore ID — скаут их не видит (нужен sofascore_id_finder)
    count(*) FILTER (WHERE m.external_id LIKE 'fs_%')                   AS flashscore_only

FROM matches m
WHERE m.home_score_final IS NOT NULL
  AND m.away_score_final IS NOT NULL
GROUP BY 1
ORDER BY pending DESC, total_scoutable DESC;
