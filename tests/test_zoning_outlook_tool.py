"""The zoning-change outlook tool, with the query layer stubbed.

What is checked: a missing function reads as "not loaded" and names the
files and the make target; the lot comes from the number given, the address
or the selection; the outlook's tier, rate and nudges are printed as what
they are; every amendment listed is filed in the citation ledger under the
number the model prints; an already-allowed target says so; a question adds
passages scoped to the lot's zones and its peers.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import ToolException

from src.tools import rag_tools
from src.utils import neighborhoods, queries, state


def _caps(**flags):
    return queries.Capabilities(**{"postgis": True, "pgvector": True, "chunks": True, "lots": True, **flags})


def _amendment(**over):
    base = {
        "amendment_id": "R.C.A.1V.Q. 549",
        "kind": "zoning_amendment",
        "title": "Augmentation du nombre de logements au 355, boulevard René-Lévesque Ouest",
        "subject_zone_codes": ["14040Hb"],
        "bylaw_numbers": ["R.C.A.1V.Q. 549"],
        "gpd_numbers": ["GT2025-233", "CA1-2025-0215"],
        "max_dwellings_before": 8,
        "max_dwellings_after": 10,
        "project_dwellings": 10,
        "raises_cap": True,
        "stage": "adopted",
        "outcome": "approved",
        "first_seen": "2025-06-16",
        "decision_date": "2025-07-07",
        "council_opinion": "favorable",
        "votes": {"A": 0, "B": 0, "C": 9},
        "subject_addresses": ["355, boulevard René-Lévesque Ouest"],
        "seats": 8,
        "excerpt": "Augmenter le nombre maximal de logements de 8 à 10.",
        "url": "https://gpddocs.ville.quebec.qc.ca/gpdblob/GT2025-233.pdf",
        "citations": [{"doc_id": "gt233", "source_kind": "gpd", "url": "https://gpddocs.ville.quebec.qc.ca/gpdblob/GT2025-233.pdf"}],
    }
    base.update(over)
    return base


def _outlook(**over):
    base = {
        "lot_number": "1 304 793",
        "neighborhood": "CIL",
        "target_dwellings": 12,
        "window_years": 5,
        "since": "2021-10-26",
        "primary_zone": "14040Hb",
        "current_max_dwellings": 10.0,
        "cap_unstated": False,
        "already_allowed": False,
        "zones": [
            {
                "feature_id": "14040Hb", "neighborhood": "CIL", "is_primary": True, "pct_of_lot": 100.0,
                "zone_dominance": "Hb", "usage_habitation": "H1", "max_dwellings": 10.0,
                "dwelling_density_max_per_ha": None, "height_max_m": 13.0, "floors_max": None,
                "implantation_mode": "I", "heritage_sector": None, "piia_sector": None,
                "grid_url": "https://carte.ville.quebec.qc.ca/GrillesZonage/HandlerZonage.ashx?14040Hb",
                "bylaw_status": [{"bylaw_number": "R.C.A.1V.Q. 549", "in_force_date": "2025-07-10", "excerpt": "R.C.A.1V.Q. 549En vigueur le 2025-07-10"}],
                "amendments": [_amendment()],
                "site_activity": {"count": 2, "by_outcome": {"approved": 2}, "by_works": {"new_construction": 2},
                                  "recent": [{"meeting_date": "2025-03-03", "works_kind": "new_construction", "outcome": "approved", "address": "12 rue X"}]},
            }
        ],
        "peers": [
            {"feature_id": "19009Hb", "neighborhood": "CIL", "distance": 0.0, "same_borough": True, "zone_dominance": "Hb",
             "usage_habitation": "H1", "max_dwellings": 10.0, "amendments": [], "raises_approved": 0},
            {"feature_id": "12019Hb", "neighborhood": "CIL", "distance": 0.2, "same_borough": True, "zone_dominance": "Hb",
             "usage_habitation": "H1", "max_dwellings": 8.0, "raises_approved": 1, "newest_raise": "2024-02-01",
             "amendments": [_amendment(amendment_id="R.C.A.1V.Q. 600", subject_zone_codes=["12019Hb"], bylaw_numbers=["R.C.A.1V.Q. 600"],
                                       decision_date="2024-02-01", max_dwellings_before=6, max_dwellings_after=8,
                                       url="https://gpddocs.ville.quebec.qc.ca/gpdblob/GT2024-010.pdf")]},
        ],
        "score": {
            "peer_count": 2, "peers_with_approved_raise": 1, "weighted_peers": 0.54, "precedent_rate": 0.27,
            "half_life_years": 3.0,
            "modifiers": [
                {"name": "own_zone_raise_approved", "effect": 0.15, "count": 1},
                {"name": "target_beyond_precedents", "effect": -0.1, "median_after": 10},
                {"name": "grid_amended_recently", "effect": 0.05, "in_force_date": "2025-07-10"},
            ],
            "own_zone": {"approved": 1, "refused": 0, "in_progress": 0},
            "median_after_among_precedents": 10,
            "value": 0.37, "tier": "high", "tiers": {"moderate_from": 0.1, "high_from": 0.3},
        },
    }
    base.update(over)
    return base


def _invoke(**kwargs):
    return rag_tools.zoning_change_outlook.invoke(kwargs)


@pytest.fixture
def outlook_db(monkeypatch, lot_row):
    calls: dict[str, dict] = {}
    answer: dict = {"outlook": _outlook()}

    def outlook(lot_number, target_dwellings, **kwargs):
        calls["outlook"] = {"lot_number": lot_number, "target": target_dwellings, **kwargs}
        return answer["outlook"]

    def search(embedding, **kwargs):
        calls["search"] = kwargs
        return [
            {"chunk_id": "c1", "doc_id": "d1", "url": "https://example.test/sheet.pdf", "title": "14040Hb",
             "source_table": "Zonage__ZONAGE_EN_VIGUEUR", "neighborhood": "CIL", "scrape_date": "2026-09-01",
             "chunk_text": "Hauteur maximale 13 m.", "similarity": 0.7}
        ]

    monkeypatch.setattr(queries, "capabilities", lambda: _caps(council_items=True, council_search=True, zoning_outlook=True))
    monkeypatch.setattr(queries, "zoning_change_outlook", outlook)
    monkeypatch.setattr(queries, "search_corpus", search)
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(rag_tools, "embed_query", lambda q: [0.0] * 1024)
    state.set_selected_lot(None, None, None, None)
    calls["answer"] = answer
    return calls


def test_a_missing_function_names_the_files_and_the_target(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(council_items=True))
    with pytest.raises(ToolException) as excinfo:
        _invoke(target_dwellings=10, lot_number="1 304 793")
    assert "sql/038, 039 and 041" in str(excinfo.value)
    assert "make zoning-amendments" in str(excinfo.value)


def test_the_target_is_required_and_positive(outlook_db):
    with pytest.raises(ToolException):
        _invoke(target_dwellings=0, lot_number="1 304 793")
    assert "outlook" not in outlook_db


def test_the_lot_number_given_is_the_lot_asked_about(outlook_db):
    text = _invoke(target_dwellings=12, lot_number="1 304 793", years=5)
    assert outlook_db["outlook"] == {"lot_number": "1 304 793", "target": 12, "years": 5, "peer_count": 30}
    assert "Zoning-change outlook for lot 1 304 793" in text
    assert "Zone 14040Hb (primary, 100% of the lot): class Hb, groups H1, cap 10 dwellings/building" in text
    assert "grid in force 2025-07-10 (R.C.A.1V.Q. 549)" in text
    assert "caps a building at 10 dwellings; the target of 12 needs an amendment" in text


def test_the_score_is_printed_as_a_rate_with_its_nudges(outlook_db):
    text = _invoke(target_dwellings=12, lot_number="1 304 793")
    assert "Outlook: high (0.37). Precedent rate 0.27: 1 of 2 peer zones" in text
    assert "+0.15: a cap raise was approved in this zone (1)" in text
    assert "-0.10: the target is beyond what precedents reached (median cap after a raise: 10)" in text
    assert "+0.05: the zone's grid was amended in the last two years (2025-07-10)" in text
    assert "1 cap raise(s) approved, 0 refused, 0 in progress" in text
    assert "not a probability" in text


def test_every_amendment_is_numbered_and_filed_for_citation(outlook_db):
    text = _invoke(target_dwellings=12, lot_number="1 304 793")
    assert "Amendments on record in the lot's zone(s): 1" in text
    assert "[1] · 2025-07-07 · adopted (approved) · cap 8 → 10 dwellings · raises the cap · project 10 · R.C.A.1V.Q. 549 · council favorable" in text
    assert "Peer zones: 2 (1 with an amendment on record, 1 with none: 19009Hb)" in text
    assert "Peer 12019Hb (" + neighborhoods.label("CIL") + "; distance 0.2; groups H1; cap 8):" in text
    assert "[2] · 2024-02-01 · adopted (approved) · cap 6 → 8 dwellings" in text
    assert "CUCQ decisions in the primary zone over the window: 2 (approved 2; new construction 2)" in text
    ledger = state.citations()
    urls = [hit.get("url") for hit in ledger.values()]
    assert "https://gpddocs.ville.quebec.qc.ca/gpdblob/GT2025-233.pdf" in urls
    assert "https://gpddocs.ville.quebec.qc.ca/gpdblob/GT2024-010.pdf" in urls


def test_an_already_allowed_target_says_so(outlook_db):
    outlook_db["answer"]["outlook"] = _outlook(target_dwellings=8, already_allowed=True,
                                               score={**_outlook()["score"], "tier": "already_allowed"})
    text = _invoke(target_dwellings=8, lot_number="1 304 793")
    assert "already allows 10 dwellings per building in the primary zone, at or above the target of 8" in text
    assert "Outlook: already allowed" in text


def test_the_selected_lot_is_the_default(outlook_db):
    state.set_selected_lot("1 304 793", -71.23, 46.80, "CIL")
    _invoke(target_dwellings=12)
    assert outlook_db["outlook"]["lot_number"] == "1 304 793"


def test_no_lot_at_all_asks_for_one(outlook_db):
    with pytest.raises(ToolException) as excinfo:
        _invoke(target_dwellings=12)
    assert "No lot given" in str(excinfo.value)


def test_an_unknown_lot_points_at_the_lot_tools(outlook_db):
    outlook_db["answer"]["outlook"] = {"error": "lot not found in silver.lot_zone_pieces"}
    with pytest.raises(ToolException) as excinfo:
        _invoke(target_dwellings=12, lot_number="9 999 999")
    assert "find_lot_by_address" in str(excinfo.value)


def test_a_question_adds_passages_about_the_zones_and_their_peers(outlook_db):
    text = _invoke(target_dwellings=12, lot_number="1 304 793", question="hauteur maximale")
    assert outlook_db["search"]["zones"][:3] == ["14040Hb", "19009Hb", "12019Hb"]
    assert outlook_db["search"]["query_text"] == "hauteur maximale"
    assert "Hauteur maximale 13 m." in text
    assert "[3]" in text
