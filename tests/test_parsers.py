"""
Tests for team name normalization and similarity matching.

Covers:
  - _norm: diacritics removal, lowercasing, stop-word removal, aliases
  - _sim: token overlap, containment boost, edge cases
  - _NAME_ALIASES: known alias mappings round-trip through _norm
"""
from __future__ import annotations

from src.data_collection.flashscore_enricher import _NAME_ALIASES, _norm, _sim


# ── _norm ──────────────────────────────────────────────────────────────────────


def test_norm_lowercases():
    assert _norm("BOSTON CELTICS") == _norm("Boston Celtics")


def test_norm_strips_diacritics():
    assert _norm("Žalgiris") == _norm("Zalgiris")
    assert _norm("Fenerbahçe") == _norm("Fenerbahce")


def test_norm_removes_stop_words():
    result = _norm("Telekom Baskets Bonn")
    # "telekom" and "baskets" are stop words; only "bonn" should survive
    assert "telekom" not in result
    assert "bonn" in result


def test_norm_removes_digits_standalone():
    # standalone digits should be stripped, e.g. "BC 49 Berlin"
    result = _norm("BC 49 Berlin")
    assert "49" not in result


def test_norm_removes_sponsor_tokens():
    result = _norm("EA7 Emporio Armani Milano")
    # alias maps this → "olimpia"; stop words remove ea7/emporio/armani/milano too
    assert "ea7" not in result
    assert "emporio" not in result


def test_norm_applies_alias_before_stop_words():
    # "barca basket" → alias → "barcelona", then stop-word "basket" is already gone
    result = _norm("Barça Basket")
    assert "barcelona" in result


def test_norm_strips_punctuation():
    result = _norm("Partizan Mozzart-Bet")
    assert "-" not in result


# ── _NAME_ALIASES ─────────────────────────────────────────────────────────────


def test_aliases_are_lowercase_keys():
    for k in _NAME_ALIASES:
        assert k == k.lower(), f"Alias key not lowercase: {k!r}"


def test_known_alias_ea7():
    result = _norm("EA7 Emporio Armani Milano")
    # after alias replacement → "olimpia" survives stop-word pass
    assert "olimpia" in result


def test_known_alias_barca():
    result = _norm("Barca Basket")
    assert "barcelona" in result


def test_known_alias_nanjing():
    # "nanjing monkey kings" → "nanjing tongxi" via alias
    result = _norm("Nanjing Monkey Kings")
    assert "nanjing" in result
    assert "tongxi" in result


def test_known_alias_zhejiang():
    result = _norm("Zhejiang Golden Bulls")
    assert "zhejiang" in result
    assert "guangsha" in result


# ── _sim ───────────────────────────────────────────────────────────────────────


def test_sim_identical_strings():
    assert _sim("boston celtics", "boston celtics") == 1.0


def test_sim_no_overlap():
    score = _sim("boston celtics", "miami heat")
    assert score == 0.0


def test_sim_partial_overlap():
    score = _sim("boston celtics", "boston bruins")
    assert 0.0 < score < 1.0


def test_sim_empty_string_returns_zero():
    assert _sim("", "boston celtics") == 0.0
    assert _sim("boston celtics", "") == 0.0


def test_sim_containment_boost_applied():
    # "bonn" tokens are a subset of "bonn rhein sieg" — containment boost kicks in
    short = "bonn"
    long_  = "bonn rhein sieg"
    score_contained = _sim(short, long_)
    # purely token overlap would be 1/3 ≈ 0.33; boost should push above 0.5
    assert score_contained > 0.5, f"Containment boost not applied: {score_contained}"


def test_sim_symmetric():
    a, b = "fenerbahce", "fenerbahce beko"
    assert abs(_sim(a, b) - _sim(b, a)) < 1e-9


def test_sim_range_zero_to_one():
    for a, b in [("abc", "def"), ("abc", "abc"), ("abc def", "abc")]:
        s = _sim(a, b)
        assert 0.0 <= s <= 1.0, f"_sim({a!r}, {b!r}) = {s} out of [0,1]"
