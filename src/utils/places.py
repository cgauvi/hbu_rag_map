"""
places.py — Which municipality a place name written with an address means.

Adresses Québec prints the *amalgamated* municipality on every point:
``Montréal`` for Villeray, ``Québec`` for Sillery and for Limoilou alike,
``Saguenay`` for Jonquière. People don't write it that way. They write the town
that existed before the 2002 mergers, the arrondissement, or the quartier:
"1234 chemin Saint-Louis, Sillery", "rue Racine, Chicoutimi", "Montcalm".

**A name can mean more than one place, so it resolves to a ranked list.** The
gazetteer, ``places.csv`` beside this module, holds one row per meaning:
*Montcalm* is a quartier of Québec (weight 0.7) and a municipality in the
Laurentides (0.3); *Mont-Royal* is the demerged town first and the Montréal
mountain second; *Westmount* is only Westmount, which is a city again since
2006 and not part of Montréal. `resolve` returns every meaning with its
likelihood, and the address tool combines that with where the street and
number actually exist. Nothing is out of scope: a municipality whose
addresses are not loaded is still a candidate, and when it is the likelier
reading the tool says so rather than picking the loaded one.

Names are compared folded: case, accents, hyphens, "St"/"Ste", a leading
"Ville de" or article, a trailing province or postal code. A name that is not
in the gazetteer is matched fuzzily against it ("Montcam" is Montcalm), and
failing that is taken as a municipality of its own.
"""

from __future__ import annotations

import csv
import difflib
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

GAZETTEER_PATH = Path(__file__).with_name("places.csv")

#: How much a fuzzy hit is worth next to an exact one of the same similarity
#: - "Montcam" is probably Montcalm, but the user did not write it.
FUZZY_PENALTY = 0.8
FUZZY_CUTOFF = 0.8

#: How much a later comma segment that names the same municipality lifts a
#: candidate: "Montcalm, Québec" is the quartier more surely than "Montcalm".
CONTEXT_BOOST = 3.0

_PUNCTUATION_RE = re.compile(r"[-‐–—'’.,/()]")
_SAINT_RE = re.compile(r"\bst\b")
_SAINTE_RE = re.compile(r"\bste\b")

#: What a person writes around a place name and the layer does not print.
_PREFIX_RE = re.compile(
    r"^(?:ville de|ville d|city of|town of|municipalite de|borough of"
    r"|arrondissement(?: de| du| des)?|arr|secteur(?: de| du)?"
    r"|quartier(?: de| du)?)\s+"
)
_ARTICLE_RE = re.compile(r"^(?:l|le|la|les)\s+")
_POSTAL_RE = re.compile(r"\b[a-z]\d[a-z] ?\d[a-z]\d\b")
_PROVINCE_RE = re.compile(r"(?:\s|^)(?:qc|pq|quebec|province de quebec|canada)$")


def fold(text: str | None) -> str:
    """A place name lower-cased, unaccented, punctuation to single spaces.

    The same fold `queries._MUNICIPALITY_KEY_SQL` applies to the stored
    municipality, so ``fold("Québec")`` is what that column compares equal to.
    """
    text = (text or "").replace("œ", "oe").replace("Œ", "oe").replace("æ", "ae")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"\s+", " ", _PUNCTUATION_RE.sub(" ", text)).strip()
    return _SAINTE_RE.sub("sainte", _SAINT_RE.sub("saint", text))


def _strip(text: str | None, *, articles: bool = True) -> str:
    """`fold`, less a postal code, a trailing province and a leading "Ville de"."""
    key = _PREFIX_RE.sub("", _POSTAL_RE.sub(" ", fold(text)).strip())
    # "Sillery (Québec)" is Sillery; "Québec" alone is the city.
    while True:
        shorter = _PROVINCE_RE.sub("", key).strip()
        if not shorter or shorter == key:
            break
        key = shorter
    if key in {"qc", "pq", "canada", "province de quebec"}:
        return ""
    if articles:
        key = _ARTICLE_RE.sub("", key)
    return key.strip()


@dataclass(frozen=True)
class Candidate:
    """One municipality a place name may mean, and how likely it is.

    ``score`` is normalised over the candidates `resolve` returns, so it reads
    as a share: 0.7 is "seven times in ten this name means this". ``kind`` is
    what the name is in that municipality (city, arrondissement,
    former_municipality, quartier, municipality, or unknown for a name the
    gazetteer does not hold) and ``note`` what the gazetteer says about it.
    """

    municipality: str
    score: float
    kind: str
    name: str
    note: str = ""
    fuzzy: bool = False

    @property
    def key(self) -> str:
        """The municipality as the address query compares it."""
        return fold(self.municipality)

    def describe(self) -> str:
        """"Montcalm, a quartier of Québec (La Cité-Limoilou)" and the like."""
        kind = self.kind.replace("_", " ")
        if self.kind in {"city", "municipality", "unknown"}:
            text = self.municipality
            if self.note and self.kind == "municipality":
                text += f" ({self.note})"
        else:
            article = "an" if kind[0] in "aeiou" else "a"
            text = f"{self.name}, {article} {kind} of {self.municipality}"
            if self.note:
                text += f" ({self.note})"
        return text


@dataclass(frozen=True)
class _Entry:
    name: str
    municipality: str
    kind: str
    weight: float
    note: str


@lru_cache(maxsize=1)
def gazetteer() -> dict[str, tuple[_Entry, ...]]:
    """``places.csv``, keyed by folded name. Lines starting with # are comments."""
    with GAZETTEER_PATH.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    table: dict[str, list[_Entry]] = {}
    for row in csv.DictReader(lines):
        entry = _Entry(
            name=row["name"].strip(),
            municipality=row["municipality"].strip(),
            kind=row["kind"].strip(),
            weight=float(row["weight"] or 1),
            note=(row.get("note") or "").strip(),
        )
        table.setdefault(_strip(entry.name), []).append(entry)
    return {key: tuple(entries) for key, entries in table.items()}


def _raw_candidates(place: str | None) -> list[Candidate]:
    """Every meaning of ``place``, unnormalised: exact, else fuzzy, else itself."""
    key = _strip(place)
    if not key:
        return []
    table = gazetteer()
    hits = [(entry, 1.0, False) for entry in table.get(key, ())]
    if not hits:
        for close in difflib.get_close_matches(key, table, n=3, cutoff=FUZZY_CUTOFF):
            similarity = difflib.SequenceMatcher(None, key, close).ratio()
            hits += [(entry, similarity * FUZZY_PENALTY, True) for entry in table[close]]
    if not hits:
        name = (place or "").strip()
        own = _strip(place, articles=False)
        return [Candidate(municipality=own, score=1.0, kind="unknown", name=name)]
    return [
        Candidate(
            municipality=entry.municipality,
            score=entry.weight * factor,
            kind=entry.kind,
            name=entry.name,
            note=entry.note,
            fuzzy=fuzzy,
        )
        for entry, factor, fuzzy in hits
    ]


def resolve(place: str | None, context: tuple[str, ...] | list[str] = ()) -> list[Candidate]:
    """The municipalities ``place`` may mean, likeliest first, scores summing to 1.

    ``context`` is whatever else was written with the address - the later
    comma segments of "rue X, Montcalm, Québec" - and a candidate whose
    municipality a context segment also names is lifted by `CONTEXT_BOOST`.
    Two meanings in the same municipality ("Plateau" is a quartier of Montréal
    and one of Québec) are summed under the likelier one.
    """
    raw = _raw_candidates(place)
    if not raw:
        return []
    named = {c.key for segment in context for c in _raw_candidates(segment)}
    best: dict[str, Candidate] = {}
    for cand in raw:
        score = cand.score * (CONTEXT_BOOST if cand.key in named else 1.0)
        held = best.get(cand.key)
        if held is None:
            best[cand.key] = replace(cand, score=score)
        else:
            keep = held if held.score >= score else cand
            best[cand.key] = replace(keep, score=held.score + score)
    total = sum(c.score for c in best.values()) or 1.0
    return sorted(
        (replace(c, score=c.score / total) for c in best.values()),
        key=lambda c: (-c.score, c.municipality),
    )


def is_known(place: str | None) -> bool:
    """Whether ``place`` names something in the gazetteer, exactly or nearly."""
    return any(c.kind != "unknown" for c in resolve(place))


def city_for(place: str | None) -> str | None:
    """The likeliest municipality ``place`` means, or None when it names nothing known."""
    candidates = [c for c in resolve(place) if c.kind != "unknown"]
    return candidates[0].municipality if candidates else None


def places_from_address(text: str | None) -> list[str]:
    """The place names written after the street in a one-string address.

    ``"7430 rue Lajeunesse, Montréal (Québec) H2R 2H8"`` gives ``["Montréal
    (Québec) H2R 2H8"]``, which `resolve` reads as Montréal. Segments that are
    only a province or a postal code are dropped. The first is the place and
    the rest are its context: in "Saint-Augustin-de-Desmaures, Québec" the
    town is the place, and "Québec" can only lift a candidate the town already
    had, never replace it.
    """
    return [
        segment.strip()
        for segment in (text or "").split(",")[1:]
        if _strip(segment, articles=False)
    ]
