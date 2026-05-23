"""
Team name normalisation and fuzzy similarity for Flashscore matching.

_norm   — strips diacritics, stop words, sponsor tokens; applies aliases
_sim    — token overlap + containment boost (handles city-name subsets)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

_STOP = re.compile(
    r"\b(basketball|club|bc|bk|fc|sk|ak|kk|as|ss|bball|city|team|"
    r"real|istanbul|moscow|milan|milano|munchen|munich|london|"
    r"koszykowki|baloncesto|pallacanestro|basket|baskets|sport|brose|s|"
    r"telekom|ewe|fraport|skyliners|mhp|mlp|ratiopharm|"
    r"fitness|first|gladiators|towers|seawolves|riesen|lowen|lions|"
    r"academics|bv|rasta|"
    r"ea7|emporio|armani|ldlc|beko|meridianbet|mozzart|bet)\b",
    re.IGNORECASE,
)
_PUNCT       = re.compile(r"[^a-z0-9 ]")
_DIGITS_ONLY = re.compile(r"\b\d+\b")

_SPLIT_COMPOUNDS = [
    ("sunrockers", "sun rockers"),
    ("neozone",    "neo zone"),
]

_NAME_ALIASES: dict[str, str] = {
    "ea7 emporio armani milano": "olimpia",
    "barca basket":              "barcelona",
    "barca":                     "barcelona",
    "crvena zvezda":             "crvena zvezda meridianbet",
    "partizan":                  "partizan mozzart bet",
    "nanjing monkey kings":      "nanjing tongxi",
    "zhejiang golden bulls":     "zhejiang guangsha",
}


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    if name in _NAME_ALIASES:
        name = _NAME_ALIASES[name]
    for compound, expanded in _SPLIT_COMPOUNDS:
        name = name.replace(compound, expanded)
    name = _STOP.sub(" ", name)
    name = _DIGITS_ONLY.sub(" ", name)
    name = _PUNCT.sub(" ", name)
    return " ".join(name.split())


def _sim(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(len(ta), len(tb))
    big, small = (ta, tb) if len(ta) >= len(tb) else (tb, ta)
    if small and small.issubset(big):
        containment = 0.5 + 0.25 * (len(small) / len(big))
        overlap = max(overlap, containment)
    return overlap


QScore = tuple[int | None, int | None]


@dataclass
class FsMatch:
    """One match entry from the Flashscore results index."""
    fs_id:      str
    date_str:   str
    match_date: date | None
    home_raw:   str
    away_raw:   str
    home_norm:  str
    away_norm:  str
    q_scores:   list[QScore] = field(default_factory=list)


@dataclass
class DbMatch:
    match_id:   int
    ext_id:     str
    league:     str
    match_date: date
    home_team:  str
    away_team:  str
