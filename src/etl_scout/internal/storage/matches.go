package storage

import (
	"context"
	"fmt"
)

// MatchRef is the minimal match identity the enricher needs: our internal id,
// the Sofascore event id to fetch, both team ids, and the final scores (the
// statistics endpoint omits points, so ORtg/DRtg use these DB-sourced totals).
type MatchRef struct {
	ID         int64 // matches.id (FK target for team_match_advanced)
	EventID    int   // Sofascore event id (matches.external_id, numeric)
	HomeTeamID int64
	AwayTeamID int64
	HomeScore  int
	AwayScore  int
}

// matchesNeedingAdvancedSQL selects finished matches for the given leagues that
// still have NO row in team_match_advanced (idempotency guard against re-scrape).
// Leagues are matched via the project convention COALESCE(tournament,'NBA').
// external_id is constrained to numeric so the ::bigint cast is always safe.
const matchesNeedingAdvancedSQL = `
SELECT m.id, m.external_id::bigint, m.home_team_id, m.away_team_id,
       m.home_score_final, m.away_score_final
FROM matches m
WHERE COALESCE(m.tournament_name, 'NBA') = ANY($1)
  AND m.home_score_final IS NOT NULL
  AND m.away_score_final IS NOT NULL
  AND m.external_id ~ '^[0-9]+$'
  AND NOT EXISTS (
        SELECT 1 FROM team_match_advanced t WHERE t.match_id = m.id
  )
ORDER BY m.scheduled_at DESC
LIMIT CASE WHEN $2 <= 0 THEN NULL ELSE $2 END`

// FetchMatchesNeedingAdvanced returns matches from the given leagues that lack
// advanced stats. A limit <= 0 means "no limit".
func (p *Pool) FetchMatchesNeedingAdvanced(ctx context.Context, leagues []string, limit int) ([]MatchRef, error) {
	rows, err := p.Query(ctx, matchesNeedingAdvancedSQL, leagues, limit)
	if err != nil {
		return nil, fmt.Errorf("query matches needing advanced stats: %w", err)
	}
	defer rows.Close()

	var refs []MatchRef
	for rows.Next() {
		var r MatchRef
		if err := rows.Scan(&r.ID, &r.EventID, &r.HomeTeamID, &r.AwayTeamID, &r.HomeScore, &r.AwayScore); err != nil {
			return nil, fmt.Errorf("scan match ref: %w", err)
		}
		refs = append(refs, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate match refs: %w", err)
	}
	return refs, nil
}
