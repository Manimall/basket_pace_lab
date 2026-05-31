"""Tests for the pure Flashscore O/U parsers in odds_parse.py."""
from __future__ import annotations

from src.data_collection.flashscore.odds_parse import (
    extract_total_line,
    parse_best_opportunity,
)


def _opp(over: float, under: float, handicap: float) -> dict:
    return {
        "over":     {"value": over},
        "under":    {"value": under},
        "handicap": {"value": handicap},
    }


def test_parse_best_opportunity_picks_most_symmetric() -> None:
    opps = [_opp(1.50, 2.50, 160.5), _opp(1.90, 1.95, 165.5)]  # 2nd is symmetric
    handicap, score = parse_best_opportunity(opps)
    assert handicap == 165.5
    assert score < 0.05


def test_parse_best_opportunity_skips_malformed() -> None:
    opps = [{"over": {"value": "x"}}, _opp(1.90, 1.90, 150.0)]
    handicap, _ = parse_best_opportunity(opps)
    assert handicap == 150.0


def test_parse_best_opportunity_empty_returns_none() -> None:
    handicap, score = parse_best_opportunity([])
    assert handicap is None
    assert score == float("inf")


def _bm_response(bm_id: str, opps: list[dict]) -> dict:
    return {"data": {"findPrematchOddsForBookmaker": {
        "bookmakerId": bm_id, "opportunities": opps,
    }}}


def test_extract_total_line_picks_sharpest_bookmaker() -> None:
    responses = [
        _bm_response("16", [_opp(1.50, 2.50, 160.5)]),  # asymmetric
        _bm_response("18", [_opp(1.91, 1.91, 165.5)]),  # sharpest
    ]
    line, bm = extract_total_line(responses)
    assert line == 165.5
    assert bm == "18"


def test_extract_total_line_empty_returns_none() -> None:
    assert extract_total_line([]) == (None, "")


def test_extract_total_line_ignores_responses_without_bookmaker_data() -> None:
    assert extract_total_line([{"data": {}}, {"errors": ["x"]}]) == (None, "")
