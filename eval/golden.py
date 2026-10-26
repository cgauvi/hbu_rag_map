"""The question set: loaded from the fixture, or generated from the database.

Every case carries a question, optionally the lot it is asked "here" about,
and ``expect``: what the database says the answer is. The labels are
*generated*, not written, because the mapping from a lot to the sheet that
governs it, the storey count that sheet states, and the resolution the lot
is subject to are all already in the database - writing them by hand would
only add a way to be wrong. What is written by hand is the question, the
category, and the choice of which lots make a fair sample.

The fixture is frozen on disk (``tests/fixtures/agent_eval.yaml``) so that a
run next month scores the same questions as a run today. Regenerate it
deliberately, with ``python -m eval cases --write``, and read the diff.

Categories
----------
storeys          "how many storeys" on a lot one zone covers; the grid states it
uses             "what uses are permitted" on such a lot; the grid lists them
coverage         "maximum lot coverage" on such a lot; the grid states it
ppcmoi           "what zoning change has occurred here" on a lot a projet
                 particulier resolution covers; that resolution is the answer
ppcmoi-negative  the same question on a lot no resolution covers; the answer
                 must not produce one
combo            storeys and past change in one question
corpus           a question with no lot, from the retrieval fixture, labelled
                 by vocabulary
off-corpus       a question the corpus does not cover; the right answer declines

Expect fields
-------------
zone                 the governing zone code
doc_ids              documents that answer; retrieval relevance
values               {storeys_max: 3, ...} numbers the answer must state
must_mention_any     one of these must appear in the answer
passages_contain_any one of these must appear in a relevant passage (corpus)
nothing              the right answer declines (off-corpus)
no_change            the answer must not assert a past change (ppcmoi-negative)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "agent_eval.yaml"
RETRIEVAL_FIXTURE = FIXTURE.with_name("retrieval_eval.yaml")

ZONE_SOURCE = "Reglement_urbanisme__VSP_REG_ZONE"
PPCMOI_SOURCE = "Reglement_urbanisme__VSP_REG_PPCMOI"
SNAPSHOT = "2026-09-01"

#: The corpus-wide questions carried over from the retrieval fixture: a mix
#: of French and English, exact terms, and one the corpus is silent on.
CORPUS_CASE_IDS = (
    "terme-hauteur-etages",
    "en-hauteur-etages",
    "en-marges-laterales",
    "exact-garderie",
    "exact-usage-class-c3",
    "cil-zone-code-uses",
    "terme-cos",
)
OFF_CORPUS_CASE_IDS = ("neg-snow-removal", "neg-library-hours", "neg-dog-licence")


@dataclass
class Case:
    id: str
    category: str
    question: str
    lang: str = "en"
    neighborhood: str | None = None
    lot: dict | None = None  # lot_number, neighborhood, lon, lat
    expect: dict = field(default_factory=dict)
    source: str = ""
    notes: str = ""

    @property
    def tags(self) -> list[str]:
        return [self.category, self.neighborhood or "corpus", self.lang]

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"id": self.id, "category": self.category, "question": self.question, "lang": self.lang}
        if self.neighborhood:
            out["neighborhood"] = self.neighborhood
        if self.lot:
            out["lot"] = self.lot
        out["expect"] = self.expect
        if self.source:
            out["source"] = self.source
        if self.notes:
            out["notes"] = self.notes
        return out


def load(path: Path = FIXTURE) -> list[Case]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = []
    for row in data.get("cases", []):
        cases.append(
            Case(
                id=row["id"],
                category=row["category"],
                question=row["question"],
                lang=row.get("lang", "en"),
                neighborhood=row.get("neighborhood"),
                lot=row.get("lot"),
                expect=row.get("expect") or {},
                source=row.get("source", ""),
                notes=row.get("notes", ""),
            )
        )
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in " + str(path))
    return cases


def select(cases: list[Case], *, only: list[str] | None = None, categories: list[str] | None = None, limit: int | None = None) -> list[Case]:
    chosen = cases
    if only:
        wanted = set(only)
        chosen = [c for c in chosen if c.id in wanted]
    if categories:
        wanted = set(categories)
        chosen = [c for c in chosen if c.category in wanted]
    if limit:
        chosen = chosen[:limit]
    return chosen


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

#: Deterministic sampling: rows are ordered by a hash of their key and this
#: seed, so "random" lots are the same lots every time the generator runs.
SEED = "agent-eval-2026-10"


def _order(column: str) -> str:
    return f"md5({column}::text || '{SEED}')"


def _num(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _resolution_tokens(text: str) -> list[str]:
    """The ways an answer can name a resolution.

    A projet particulier carries several identifiers at once - the council
    resolution ("CA17 140107"), the file ("PP15-14012"), an older city
    council number ("CO9902770"), a by-law number ("97-040") - and an answer
    that quotes any one of them has named it. All of them are lifted out of
    whatever text is given (the title, or the document's own chunks).
    """
    tokens: list[str] = []
    text = text or ""
    for match in re.finditer(r"\b(CA\d{2})\s+(\d{6})\b", text):
        tokens += [f"{match.group(1)} {match.group(2)}", match.group(2)]
    for match in re.finditer(r"\b(PP\d{2}-\d{5})\b", text):
        tokens.append(match.group(1))
    for match in re.finditer(r"\b(CO\d{2}-?\d{5})\b", text):
        tokens.append(match.group(1))
    for match in re.finditer(r"#\s*([\d-]{3,})", text):
        tokens.append(match.group(1))
    for match in re.finditer(r"\b(\d{2}-\d{3})\b", text):
        tokens.append(match.group(1))
    seen: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    return seen


def _doc_tokens(cur, doc_ids: list[str], titles: list[str], description: str) -> list[str]:
    """Identifiers for a resolution, from its title and its chunks' text."""
    cur.execute("SELECT text FROM rag.chunks WHERE doc_id = ANY(%s) ORDER BY chunk_id", (list(doc_ids),))
    texts = [t for (t,) in cur.fetchall()]
    tokens: list[str] = []
    for source in [*(titles or []), description or "", *texts]:
        for t in _resolution_tokens(source or ""):
            if t not in tokens:
                tokens.append(t)
    # An identifier that many documents carry names nothing: "01-283" is the
    # borough's zoning by-law and is printed on every grid sheet, so an answer
    # quoting it has not found this resolution.
    specific: list[str] = []
    for t in tokens:
        cur.execute(
            "SELECT count(DISTINCT doc_id) FROM rag.chunks WHERE neighborhood = 'VSMPE' AND text LIKE %s",
            (f"%{t}%",),
        )
        if cur.fetchone()[0] <= 3:
            specific.append(t)
    return specific


def _zone_doc(cur, neighborhood: str, source_table: str, zone: str) -> str | None:
    cur.execute(
        "SELECT doc_id FROM rag.chunks WHERE neighborhood = %s AND source_table = %s "
        "  AND feature_ids ? %s ORDER BY chunk_id LIMIT 1",
        (neighborhood, source_table, zone),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _vsmpe_lots(cur, *, limit: int, min_storeys: float = 2) -> list[dict]:
    """VSMPE lots one zone covers (>= 90%), with the zone's stated norms."""
    cur.execute(
        f"""
        SELECT lf.lot_number, lf.feature_id AS zone, lf.pct_of_lot,
               f.attributes->>'ETAGE_MIN'    AS storeys_min,
               f.attributes->>'ETAGE_MAX'    AS storeys_max,
               f.attributes->>'TAUX_IMP_MAX' AS coverage_max_pct,
               f.attributes->>'TAUX_IMP_MIN' AS coverage_min_pct,
               f.attributes->>'USAGE'        AS usages,
               f.attributes->>'LIEN_GRILLE'  AS url,
               ST_X(ST_PointOnSurface(l.geom)) AS lon,
               ST_Y(ST_PointOnSurface(l.geom)) AS lat
          FROM silver.lot_features lf
          JOIN rag.features f ON f.feature_uid = lf.feature_uid
          JOIN rag.lots l ON l.lot_uid = lf.lot_uid
         WHERE lf.neighborhood = 'VSMPE' AND lf.scrape_date = %s
           AND f.source_table = %s
           AND lf.pct_of_lot >= 90
           AND (f.attributes->>'ETAGE_MAX')::float >= %s
           AND lf.lot_area_m2 BETWEEN 200 AND 2000
           AND NOT EXISTS (
               SELECT 1 FROM rag.features p
                WHERE p.source_table = %s AND p.neighborhood = 'VSMPE'
                  AND p.scrape_date = %s AND ST_Intersects(p.geom, l.geom))
         ORDER BY {_order('lf.lot_number')}
         LIMIT %s
        """,
        (SNAPSHOT, ZONE_SOURCE, min_storeys, PPCMOI_SOURCE, SNAPSHOT, limit),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _cil_lots(cur, *, limit: int) -> list[dict]:
    """CIL lots in a zone whose parsed grid has one column with a storey max."""
    cur.execute(
        f"""
        WITH one AS (
            SELECT feature_id, min(floors_max) AS storeys_max, min(floors_min) AS storeys_min,
                   min(height_max_m) AS height_max_m, min(doc_id) AS doc_id
              FROM silver.zoning_grid_columns
             WHERE neighborhood = 'CIL'
             GROUP BY feature_id
            HAVING count(*) = 1 AND count(floors_max) = 1
        )
        SELECT lf.lot_number, lf.feature_id AS zone, lf.pct_of_lot,
               one.storeys_min, one.storeys_max, one.height_max_m, one.doc_id,
               ST_X(ST_PointOnSurface(l.geom)) AS lon,
               ST_Y(ST_PointOnSurface(l.geom)) AS lat
          FROM silver.lot_features lf
          JOIN one ON one.feature_id = lf.feature_id
          JOIN rag.lots l ON l.lot_uid = lf.lot_uid
         WHERE lf.neighborhood = 'CIL' AND lf.pct_of_lot >= 90
           AND lf.lot_area_m2 BETWEEN 200 AND 2000
         ORDER BY {_order('lf.lot_number')}
         LIMIT %s
        """,
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _ppcmoi_lots(cur, *, limit: int) -> list[dict]:
    """VSMPE lots a projet-particulier polygon covers (>= 90%), whose
    resolution is in the corpus - one row per lot, every doc for the polygon."""
    cur.execute(
        f"""
        WITH covered AS (
            SELECT DISTINCT ON (l.lot_number)
                   l.lot_number, f.feature_id, f.attributes->>'DESCRIPTION' AS description,
                   ST_X(ST_PointOnSurface(l.geom)) AS lon, ST_Y(ST_PointOnSurface(l.geom)) AS lat
              FROM rag.features f
              JOIN rag.lots l
                ON l.neighborhood = 'VSMPE' AND l.scrape_date = %s AND ST_Intersects(f.geom, l.geom)
             WHERE f.source_table = %s AND f.neighborhood = 'VSMPE' AND f.scrape_date = %s
               AND ST_Area(ST_Intersection(f.geom, l.geom)::geography) > 0.9 * ST_Area(l.geom::geography)
               AND ST_Area(l.geom::geography) BETWEEN 150 AND 5000
               AND EXISTS (
                   SELECT 1 FROM rag.chunks c
                   CROSS JOIN jsonb_array_elements_text(c.feature_ids) fid
                    WHERE c.source_table = f.source_table AND fid.value = f.feature_id)
             ORDER BY l.lot_number, f.feature_id
        )
        SELECT c.lot_number, c.feature_id, c.description, c.lon, c.lat,
               array_agg(DISTINCT ch.doc_id) AS doc_ids,
               array_agg(DISTINCT ch.title)  AS titles,
               count(ch.chunk_id)            AS chunks
          FROM covered c
          JOIN rag.chunks ch ON ch.source_table = %s
           AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(ch.feature_ids) fid WHERE fid.value = c.feature_id)
         GROUP BY c.lot_number, c.feature_id, c.description, c.lon, c.lat
         ORDER BY {_order('c.lot_number')}
         LIMIT %s
        """,
        (SNAPSHOT, PPCMOI_SOURCE, SNAPSHOT, PPCMOI_SOURCE, limit),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _lot_dict(row: dict, neighborhood: str) -> dict:
    return {
        "lot_number": row["lot_number"],
        "neighborhood": neighborhood,
        "lon": round(float(row["lon"]), 7),
        "lat": round(float(row["lat"]), 7),
    }


def _slug(lot_number: str) -> str:
    return lot_number.replace(" ", "")


def generate(conn, *, counts: dict[str, int] | None = None) -> list[Case]:
    """Every case, from the database. ``conn`` is a psycopg connection."""
    counts = {
        "storeys_vsmpe": 5, "storeys_cil": 3, "uses_vsmpe": 3, "uses_cil": 2,
        "coverage": 3, "ppcmoi": 6, "ppcmoi_negative": 4, "combo": 2,
        **(counts or {}),
    }
    cases: list[Case] = []
    with conn.cursor() as cur:
        # -- VSMPE, one zone covering, no projet particulier anywhere on it --
        need = counts["storeys_vsmpe"] + counts["uses_vsmpe"] + counts["coverage"] + counts["ppcmoi_negative"]
        vsmpe = _vsmpe_lots(cur, limit=need)
        pool = list(vsmpe)

        for i in range(counts["storeys_vsmpe"]):
            row = pool.pop(0)
            doc = _zone_doc(cur, "VSMPE", ZONE_SOURCE, row["zone"])
            values = {"storeys_max": _num(row["storeys_max"])}
            if _num(row["storeys_min"]):
                values["storeys_min"] = _num(row["storeys_min"])
            french = i == 1
            cases.append(Case(
                id=f"vsmpe-storeys-{_slug(row['lot_number'])}",
                category="storeys",
                question=("Combien d'étages le règlement permet-il sur ce lot, au minimum et au maximum?"
                          if french else
                          "What does the by-law permit on this lot in terms of number of storeys, minimum and maximum?"),
                lang="fr" if french else "en",
                neighborhood="VSMPE",
                lot=_lot_dict(row, "VSMPE"),
                expect={"zone": row["zone"], "doc_ids": [doc] if doc else [], "values": values,
                        "must_mention_any": [row["zone"]]},
                source="generated: rag.features ETAGE_MIN/ETAGE_MAX via silver.lot_features",
            ))

        for _ in range(counts["uses_vsmpe"]):
            row = pool.pop(0)
            doc = _zone_doc(cur, "VSMPE", ZONE_SOURCE, row["zone"])
            codes = [c.strip() for c in (row["usages"] or "").split(";") if c.strip()]
            cases.append(Case(
                id=f"vsmpe-uses-{_slug(row['lot_number'])}",
                category="uses",
                question="What uses (usages) are permitted on this lot?",
                neighborhood="VSMPE",
                lot=_lot_dict(row, "VSMPE"),
                expect={"zone": row["zone"], "doc_ids": [doc] if doc else [],
                        "must_mention_any": [row["zone"]], "must_mention_all": codes[:1],
                        "usage_codes": codes},
                source="generated: rag.features USAGE via silver.lot_features",
            ))

        for _ in range(counts["coverage"]):
            row = pool.pop(0)
            doc = _zone_doc(cur, "VSMPE", ZONE_SOURCE, row["zone"])
            values = {"coverage_max_pct": _num(row["coverage_max_pct"])}
            cases.append(Case(
                id=f"vsmpe-coverage-{_slug(row['lot_number'])}",
                category="coverage",
                question="What is the maximum lot coverage (taux d'implantation) permitted on this lot?",
                neighborhood="VSMPE",
                lot=_lot_dict(row, "VSMPE"),
                expect={"zone": row["zone"], "doc_ids": [doc] if doc else [], "values": values,
                        "must_mention_any": [row["zone"]]},
                source="generated: rag.features TAUX_IMP_MAX via silver.lot_features",
            ))

        for _ in range(counts["ppcmoi_negative"]):
            row = pool.pop(0)
            cases.append(Case(
                id=f"vsmpe-nochange-{_slug(row['lot_number'])}",
                category="ppcmoi-negative",
                question="What zoning change has occurred in the past here? Has a projet particulier or an amendment been adopted for this lot?",
                neighborhood="VSMPE",
                lot=_lot_dict(row, "VSMPE"),
                expect={"zone": row["zone"], "no_change": True, "forbidden_source": "PPCMOI"},
                source="generated: no PPCMOI polygon intersects the lot",
                notes="No projet particulier covers this lot; the right answer says so.",
            ))

        # -- CIL: the parsed grid states the storeys --------------------------
        cil = _cil_lots(cur, limit=counts["storeys_cil"] + counts["uses_cil"])
        for i, row in enumerate(cil):
            doc = row["doc_id"] or _zone_doc(cur, "CIL", "Zonage__ZONAGE_EN_VIGUEUR", row["zone"])
            if i < counts["storeys_cil"]:
                values = {"storeys_max": _num(row["storeys_max"])}
                if _num(row["height_max_m"]):
                    values["height_max_m"] = _num(row["height_max_m"])
                cases.append(Case(
                    id=f"cil-storeys-{_slug(row['lot_number'])}",
                    category="storeys",
                    question="What does the by-law permit on this lot in terms of number of storeys and height?",
                    neighborhood="CIL",
                    lot=_lot_dict(row, "CIL"),
                    expect={"zone": row["zone"], "doc_ids": [doc] if doc else [], "values": values,
                            "must_mention_any": [row["zone"]]},
                    source="generated: silver.zoning_grid_columns floors_max/height_max_m",
                ))
            else:
                cases.append(Case(
                    id=f"cil-uses-{_slug(row['lot_number'])}",
                    category="uses",
                    question="What uses are permitted on this lot, and under what conditions?",
                    neighborhood="CIL",
                    lot=_lot_dict(row, "CIL"),
                    expect={"zone": row["zone"], "doc_ids": [doc] if doc else [],
                            "must_mention_any": [row["zone"]], "passages_contain_any": ["usage"]},
                    source="generated: the zone's grid sheet",
                ))

        # -- PPCMOI: a resolution covers the lot ------------------------------
        # One lot per polygon: two lots under the same resolution would score
        # the same document twice.
        ppcmoi: list[dict] = []
        seen_features: set[str] = set()
        for row in _ppcmoi_lots(cur, limit=(counts["ppcmoi"] + counts["combo"]) * 3):
            if row["feature_id"] in seen_features:
                continue
            seen_features.add(row["feature_id"])
            ppcmoi.append(row)
            if len(ppcmoi) == counts["ppcmoi"] + counts["combo"]:
                break
        for i, row in enumerate(ppcmoi):
            tokens = _doc_tokens(cur, list(row["doc_ids"]), list(row["titles"] or []), row["description"] or "")
            lot = _lot_dict(row, "VSMPE")
            if i < counts["ppcmoi"]:
                french = i == 1
                cases.append(Case(
                    id=f"vsmpe-ppcmoi-{_slug(row['lot_number'])}",
                    category="ppcmoi",
                    question=("Quel changement de zonage a eu lieu ici dans le passé? Un projet particulier ou une modification a-t-il été adopté pour ce lot?"
                              if french else
                              "What zoning change has occurred in the past here? Has a projet particulier or an amendment been adopted for this lot?"),
                    lang="fr" if french else "en",
                    neighborhood="VSMPE",
                    lot=lot,
                    expect={"doc_ids": list(row["doc_ids"]), "must_mention_any": tokens,
                            "ppcmoi_feature_id": row["feature_id"], "relevant_chunks": int(row["chunks"])},
                    source="generated: PPCMOI polygon covering the lot, its resolution in rag.chunks",
                    notes=f"Resolution: {row['description']}",
                ))
            else:
                # The combo: this lot's zone storeys plus its resolution.
                cur.execute(
                    """
                    SELECT lf.feature_id, f.attributes->>'ETAGE_MAX', f.attributes->>'ETAGE_MIN'
                      FROM silver.lot_features lf JOIN rag.features f ON f.feature_uid = lf.feature_uid
                     WHERE lf.lot_number = %s AND lf.neighborhood = 'VSMPE' AND f.source_table = %s
                     ORDER BY lf.pct_of_lot DESC LIMIT 1
                    """,
                    (row["lot_number"], ZONE_SOURCE),
                )
                zone = cur.fetchone()
                expect: dict[str, Any] = {"doc_ids": list(row["doc_ids"]), "must_mention_any": tokens,
                                          "ppcmoi_feature_id": row["feature_id"]}
                if zone and _num(zone[1]):
                    expect["zone"] = zone[0]
                    expect["values"] = {"storeys_max": _num(zone[1])}
                cases.append(Case(
                    id=f"vsmpe-combo-{_slug(row['lot_number'])}",
                    category="combo",
                    question="How many storeys does the by-law permit on this lot, and has the zoning for it ever been changed by a projet particulier?",
                    neighborhood="VSMPE",
                    lot=lot,
                    expect=expect,
                    source="generated: zone ETAGE_MAX + the PPCMOI resolution covering the lot",
                    notes=f"Resolution: {row['description']}",
                ))

    # -- corpus-wide and off-corpus, from the retrieval fixture ---------------
    retrieval = {q["id"]: q for q in (yaml.safe_load(RETRIEVAL_FIXTURE.read_text(encoding="utf-8")) or {}).get("questions", [])}
    for qid in CORPUS_CASE_IDS:
        q = retrieval[qid]
        text = q.get("question_fr") or q.get("question_en")
        expect = {}
        if qid == "terme-cos":
            # The retrieval fixture keeps this one *because* no chunk holds
            # the term: the grids state density as logements/ha. Through the
            # agent the right answer is that the corpus does not say.
            expect["nothing"] = True
        else:
            if q.get("must_contain_any"):
                expect["passages_contain_any"] = list(q["must_contain_any"])
            if q.get("expect_feature_id"):
                expect["zone"] = q["expect_feature_id"]
                expect["must_mention_any"] = [q["expect_feature_id"]]
        cases.append(Case(
            id=f"corpus-{qid}",
            category="corpus",
            question=text,
            lang="fr" if q.get("question_fr") else "en",
            neighborhood=q.get("neighborhood"),
            expect=expect,
            source=f"fixture: retrieval_eval.yaml#{qid}",
            notes="Asked with no lot selected." + (" The corpus does not use this term; a good answer says so." if qid == "terme-cos" else ""),
        ))
    anchor = next((c for c in cases if c.category == "storeys" and c.neighborhood == "VSMPE"), None)
    for qid in OFF_CORPUS_CASE_IDS:
        q = retrieval[qid]
        cases.append(Case(
            id=f"offcorpus-{qid}",
            category="off-corpus",
            question="For the selected lot: " + (q.get("question_fr") or q.get("question_en")),
            lang="fr" if q.get("question_fr") else "en",
            neighborhood=anchor.neighborhood if anchor else None,
            lot=anchor.lot if anchor else None,
            expect={"nothing": True},
            source=f"fixture: retrieval_eval.yaml#{qid}",
            notes="Asked with a lot selected; the corpus holds nothing on it and the answer should say so.",
        ))
    return cases


HEADER = """# The agent eval set: one question per row, through the real assistant.
#
# GENERATED by `python -m eval cases --write` from the local database on
# {date}; every `expect` is what the database states (see eval/golden.py for
# the queries). Frozen here so a run next month scores the same questions.
# Regenerate deliberately and read the diff.
#
# Run it:   python -m eval run --config open           (HuggingFace, no $ cap)
#           python -m eval run --config claude-bge     (Anthropic, capped)
#           python -m eval compare                     (every config + report)
#
# {summary}
"""


def dump(cases: list[Case], path: Path = FIXTURE) -> None:
    from datetime import date

    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    summary = f"{len(cases)} cases: " + ", ".join(f"{k} {v}" for k, v in by_cat.items())
    body = yaml.safe_dump({"cases": [c.to_dict() for c in cases]}, allow_unicode=True, sort_keys=False, width=100)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(date=date.today().isoformat(), summary=summary) + "\n" + body, encoding="utf-8")


def fingerprint(cases: list[Case]) -> str:
    """A short hash of the set, written into every result row."""
    blob = json.dumps([c.to_dict() for c in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
