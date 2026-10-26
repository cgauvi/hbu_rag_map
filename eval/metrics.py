"""Scoring: what a retrieval ranked, what an answer stated, and how to add it up.

Retrieval is scored at the **document** level. A chunk is relevant when its
document is one the case names (``expect.doc_ids``), or - for the corpus-wide
questions that are labelled by vocabulary rather than by document - when its
text holds one of the case's words. Several chunks of one document are each
relevant; that is what nDCG is for, and why recall@k here means "the governing
document appeared in the first k passages" rather than a count of chunks.

Answers are scored on what the database can state: a number the grid gives,
a code the zone carries, a resolution the lot is subject to. Those checks are
deterministic and free, and they are deliberately narrow - a value stated
*near* a unit word, not anywhere in the prose - because a storey count also
looks like a lot number's last digit. Everything the database cannot state is
the judge's (``judge.py``), reported as its own column rather than folded in.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence

KS = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def doc_id_of(hit: dict) -> str | None:
    """A hit's document, from its ``doc_id`` or from its chunk id.

    ``rag.search_at_lot`` returns no ``doc_id`` column; its chunk ids are
    ``<doc_id>:<nnnn>``.
    """
    if hit.get("doc_id"):
        return str(hit["doc_id"])
    chunk = str(hit.get("chunk_id") or "")
    return chunk.split(":", 1)[0] if ":" in chunk else (chunk or None)


#: Typographic dashes and spaces a model writes into a code it copied: gpt-oss
#: prints "H02‑086" with a non-breaking hyphen, and lot numbers arrive with
#: narrow no-break spaces. Folded to ASCII before anything is compared.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_SPACES = dict.fromkeys(map(ord, "    "), " ")


def fold(text: str) -> str:
    """Lowercase, accents stripped, dashes and spaces made plain - the fold
    the corpus's tsvector uses, plus the typography a model adds."""
    text = unicodedata.normalize("NFKD", (text or "").translate(_DASHES).translate(_SPACES))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def relevance(hits: Sequence[dict], expect: dict) -> list[int]:
    """1 per hit that answers the case, in rank order."""
    want_docs = {str(d) for d in (expect.get("doc_ids") or [])}
    want_words = [fold(w) for w in (expect.get("passages_contain_any") or [])]
    out = []
    for hit in hits:
        doc = doc_id_of(hit)
        if want_docs and doc in want_docs:
            out.append(1)
            continue
        if want_words:
            text = fold(hit.get("chunk_text") or hit.get("text") or "")
            if any(w in text for w in want_words):
                out.append(1)
                continue
        out.append(0)
    return out


def first_rank(rels: Sequence[int]) -> int | None:
    for rank, rel in enumerate(rels, 1):
        if rel:
            return rank
    return None


def hit_at(rels: Sequence[int], k: int) -> float:
    rank = first_rank(rels)
    return 1.0 if rank is not None and rank <= k else 0.0


def mrr(rels: Sequence[int]) -> float:
    rank = first_rank(rels)
    return 1.0 / rank if rank else 0.0


def ndcg_at(rels: Sequence[int], k: int, *, total_relevant: int | None = None) -> float:
    """Binary nDCG@k. ``total_relevant`` caps the ideal when it is known
    (the number of chunks the expected documents have); otherwise the ideal
    is every retrieved relevant chunk ranked first."""
    gains = [1.0 / math.log2(i + 2) for i, rel in enumerate(rels[:k]) if rel]
    dcg = sum(gains)
    ideal_count = total_relevant if total_relevant is not None else sum(rels)
    ideal_count = min(ideal_count, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg else 0.0


def retrieval_scores(hits: Sequence[dict], expect: dict, *, total_relevant: int | None = None) -> dict:
    """The retrieval columns for one case. Empty when the case expects nothing."""
    if not (expect.get("doc_ids") or expect.get("passages_contain_any")):
        return {}
    rels = relevance(hits, expect)
    scores = {f"hit@{k}": hit_at(rels, k) for k in KS}
    scores["mrr"] = mrr(rels)
    scores["ndcg@10"] = ndcg_at(rels, 10, total_relevant=total_relevant)
    scores["first_rank"] = first_rank(rels)
    scores["retrieved"] = len(hits)
    return scores


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

#: Words a storey count, a height or a coverage is stated beside. French and
#: English both, since the prompt asks for English with the terms of art kept.
UNIT_WORDS = {
    "storeys_max": ("storey", "storeys", "stories", "floor", "floors", "etage", "etages", "level", "levels"),
    "storeys_min": ("storey", "storeys", "stories", "floor", "floors", "etage", "etages", "level", "levels"),
    "height_max_m": ("m", "metre", "metres", "meter", "meters", "height", "hauteur"),
    "coverage_max_pct": ("%", "percent", "per cent", "coverage", "implantation"),
    "coverage_min_pct": ("%", "percent", "per cent", "coverage", "implantation"),
    "cos_max": ("cos", "floor area ratio", "far", "ratio"),
}

_NUMBER = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)(?![\d])")


def _as_number(token: str) -> float | None:
    """"3", "3.0", "2,5" - the spellings a model gives a grid value."""
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def _unit_near(around: str, words: tuple[str, ...]) -> bool:
    """A unit word as a whole word - "m" must not match the m in "maximum"."""
    for w in words:
        if w.isalnum():
            if re.search(rf"\b{re.escape(w)}\b", around):
                return True
        elif w in around:
            return True
    return False


def value_stated(answer: str, key: str, value: float, *, window: int = 60) -> bool:
    """Whether ``answer`` states ``value`` within ``window`` chars of a unit word.

    Near, not anywhere: "3" appears in most lot numbers, and a coverage of
    "60" in many a zone code. The window is generous because the prose says
    things like "a maximum of 3 storeys" and "3 étages (storeys)". Compared
    as numbers, because "2.0 storeys" is the same statement as "2 storeys".
    """
    text = fold(answer)
    words = UNIT_WORDS.get(key, ())
    for match in _NUMBER.finditer(text):
        stated = _as_number(match.group(1))
        if stated is None or abs(stated - float(value)) > 1e-9:
            continue
        lo, hi = max(0, match.start() - window), min(len(text), match.end() + window)
        if _unit_near(text[lo:hi], words):
            return True
    return False


def mentions_any(answer: str, candidates: Iterable[str]) -> bool:
    text = fold(answer)
    return any(fold(c) in text for c in candidates if c)


def mentions_all(answer: str, candidates: Iterable[str]) -> bool:
    text = fold(answer)
    return all(fold(c) in text for c in candidates if c)


#: An answer that declines, the way the prompt asks it to on a question the
#: corpus does not cover. Checked with the judge's ``declines`` beside it.
DECLINE_PHRASES = (
    "not covered", "does not cover", "do not cover", "doesn't cover", "not in the", "nothing in the",
    "no information", "any information", "don't have", "do not have", "doesn't have", "does not have",
    "cannot answer", "can't answer", "unable to answer", "cannot help", "can't help", "not able to",
    "not something", "outside", "not a zoning", "isn't a zoning", "is not about zoning",
    "nothing about", "says nothing", "nothing on", "not a zoning matter", "isn't a zoning matter",
    "no record", "no passage", "any passage", "not return", "not find", "found no", "no document",
    "no mention", "no data", "any data", "cannot provide", "can't provide", "unable to provide",
    "not available", "beyond", "not address", "does not address", "not loaded", "isn't loaded",
    "in the data that", "the data i have", "the loaded data", "corpus does not", "corpus doesn't",
)


def declines(answer: str) -> bool:
    return mentions_any(answer, DECLINE_PHRASES)


def answer_checks(
    answer: str,
    expect: dict,
    *,
    cited_docs: Iterable[str] = (),
    cited_sources: Iterable[str] = (),
) -> dict:
    """Every deterministic check the case asks for, each its own column.

    ``answer_correct`` is their conjunction: a case passes when every check it
    carries passes. A case with no checks (judge-only) has no such column.

    ``cited_docs`` / ``cited_sources`` are the documents and source tables of
    the passages the turn retrieved *for the lot* (containment or proximity)
    - what a "no change here" case must not have found a resolution in. A
    borough-wide search that happens to surface somebody else's resolution
    is not that, so the caller leaves corpus-scope hits out.
    """
    checks: dict[str, float] = {}
    values = expect.get("values") or {}
    for key, value in values.items():
        checks[f"states_{key}"] = 1.0 if value_stated(answer, key, float(value)) else 0.0
    if expect.get("must_mention_any"):
        checks["mentions_any"] = 1.0 if mentions_any(answer, expect["must_mention_any"]) else 0.0
    if expect.get("must_mention_all"):
        checks["mentions_all"] = 1.0 if mentions_all(answer, expect["must_mention_all"]) else 0.0
    if expect.get("nothing"):
        checks["declines"] = 1.0 if declines(answer) else 0.0
    if expect.get("no_change"):
        # No resolution is retrieved and none is named: the agent may say "no
        # projet particulier applies", which is right, but must not produce
        # one. The judge's `asserts_past_change` is the reading of the prose.
        forbidden = expect.get("must_not_mention_any") or []
        cited = {str(d) for d in cited_docs}
        wrong_docs = cited & {str(d) for d in (expect.get("forbidden_doc_ids") or [])}
        wrong_sources = any(
            (expect.get("forbidden_source") or "PPCMOI") in str(s) for s in cited_sources
        )
        checks["no_invented_change"] = (
            0.0 if (wrong_docs or wrong_sources or mentions_any(answer, forbidden)) else 1.0
        )
    if checks:
        checks["answer_correct"] = 1.0 if all(v >= 1.0 for v in checks.values()) else 0.0
    return checks


# ---------------------------------------------------------------------------
# Adding up
# ---------------------------------------------------------------------------


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% interval on a rate - what separates a difference from noise."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def median(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def percentile(values: Sequence[float], pct: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)
