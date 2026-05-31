"""Pure parsers for Flashscore Over/Under GraphQL responses.

I/O-free: given captured ``OVER_UNDER`` API payloads, pick the sharpest total
line (the bookmaker whose Over/Under payout odds are the most symmetric). Kept
separate from the Playwright scraping in ``odds.py`` so the math is unit-testable
and each module stays under the 250-line limit.
"""
from __future__ import annotations

_WORST_ASYMMETRY: float = float("inf")


def parse_best_opportunity(opportunities: list[dict]) -> tuple[float | None, float]:
    """Pick the handicap with the most symmetric Over/Under odds.

    Args:
        opportunities: One bookmaker's O/U opportunities (one per line).

    Returns:
        ``(handicap, asymmetry_score)``. ``handicap`` is None when no
        opportunity parsed; lower ``asymmetry_score`` = sharper line.
    """
    best_handicap: float | None = None
    best_score:    float        = _WORST_ASYMMETRY
    for opp in opportunities:
        try:
            over_v   = float(opp["over"]["value"])
            under_v  = float(opp["under"]["value"])
            handicap = float(opp["handicap"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        score = abs(over_v - under_v) / (over_v + under_v)
        if score < best_score:
            best_score    = score
            best_handicap = handicap
    return best_handicap, best_score


def extract_total_line(ou_responses: list[dict]) -> tuple[float | None, str]:
    """Select the sharpest total line across all captured bookmaker responses.

    Args:
        ou_responses: Captured ``OVER_UNDER`` GraphQL response payloads.

    Returns:
        ``(total_close, bookmaker_id_str)``; ``(None, "")`` when nothing parsed.
    """
    candidates: list[tuple[float, float, str]] = []  # (asymmetry, handicap, bm_id)
    for resp in ou_responses:
        bm_data = (resp.get("data") or {}).get("findPrematchOddsForBookmaker")
        if not bm_data:
            continue
        bm_id = str(bm_data.get("bookmakerId", ""))
        handicap, score = parse_best_opportunity(bm_data.get("opportunities") or [])
        if handicap is not None:
            candidates.append((score, handicap, bm_id))
    if not candidates:
        return None, ""
    candidates.sort()  # smallest asymmetry first
    _, handicap, bm_id = candidates[0]
    return handicap, bm_id
