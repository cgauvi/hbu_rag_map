"""The two conseils de quartier tools, with the query layer stubbed.

What is checked: a missing table reads as "not loaded" and names the file
that creates it; the filters a caller writes in plain words reach the query
as the vocabulary; a place is resolved from an address, from coordinates or
from the selection; every item listed is filed in the citation ledger so
the [n] the model prints is one the pane can expand; and nothing found says
so without inventing a decision.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.tools import ToolException

from src.tools import rag_tools
from src.utils import queries, state


def _caps(**flags):
    return queries.Capabilities(**{"postgis": True, "pgvector": True, "chunks": True, "lots": True, **flags})


def _invoke(tool, **kwargs):
    return tool.invoke(kwargs)


def _item(**over):
    base = {
        "neighborhood": "CIL",
        "scrape_date": "2026-09-01",
        "doc_id": "abc123",
        "item_index": 4,
        "source_kind": "gpd",
        "item_kind": "demolition",
        "title": "Demande de démolition au 439, rue Jeanne-d’Arc",
        "council_name": "Montcalm",
        "meeting_date": date(2025, 6, 16),
        "decision": "refused",
        "outcome": "refused",
        "decision_date": date(2025, 7, 7),
        "item_date": date(2025, 7, 7),
        "council_opinion": "unfavorable",
        "council_opinion_excerpt": "Le conseil de quartier s’oppose à la démolition",
        "document_number": "CA1-2025-0300",
        "url": "https://gpddocs.ville.quebec.qc.ca/gpdblob/CA1-2025-0300.pdf",
        "subject_addresses": ["439, rue Jeanne-d’Arc"],
        "lot_numbers": ["5342215"],
        "subject_zone_codes": [],
        "project_dwellings": None,
        "max_dwellings_before": None,
        "max_dwellings_after": None,
        "citations": {"url": "https://gpddocs.ville.quebec.qc.ca/gpdblob/CA1-2025-0300.pdf", "chunk_ids": ["abc123:0000"]},
        "excerpt": "Il est résolu de refuser la demande de démolition du bâtiment sis au 439, rue Jeanne-d’Arc.",
        "distance_m": 42.0,
        "site_kind": "address",
        "site_key": "439, rue Jeanne-d’Arc",
        "site_lot_number": "5 342 215",
        "is_subject": True,
    }
    base.update(over)
    return base


@pytest.fixture
def council_db(monkeypatch, lot_row):
    """A database holding one refused demolition near the selected point."""
    calls: dict[str, dict] = {}

    def items_near(lon, lat, **kwargs):
        calls["items"] = {"lon": lon, "lat": lat, **kwargs}
        return [_item()]

    def search(embedding, **kwargs):
        calls["search"] = kwargs
        return [
            {
                "chunk_id": "abc123:0000", "doc_id": "abc123", "url": _item()["url"],
                "title": "Résolution CA1-2025-0300", "source_table": "council_gpd",
                "neighborhood": "CIL", "scrape_date": "2026-09-01",
                "chunk_text": "Il est résolu de refuser la demande de démolition.",
                "similarity": 0.81, "distance_m": 42.0, "item_index": 0,
                "item_kind": "demolition", "outcome": "refused", "item_date": date(2025, 7, 7),
            }
        ]

    monkeypatch.setattr(queries, "capabilities", lambda: _caps(council_items=True, council_search=True))
    monkeypatch.setattr(queries, "council_items_near", items_near)
    monkeypatch.setattr(queries, "search_council_chunks", search)
    monkeypatch.setattr(rag_tools, "embed_query", lambda q: [0.0] * 1024)
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    return calls


# ---------------------------------------------------------------------------
# Missing data reads as missing data
# ---------------------------------------------------------------------------


def test_missing_sites_name_the_sql_file_and_the_make_target(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(council_items=False))
    with pytest.raises(ToolException, match="035_silver_council_item_sites.sql") as exc:
        _invoke(rag_tools.council_decisions_near, address="439 rue Jeanne-d'Arc, Québec")
    assert "make council-minutes" in str(exc.value)


def test_a_question_needs_the_search_function(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(council_items=True, council_search=False))
    with pytest.raises(ToolException, match="036_council_search.sql"):
        _invoke(rag_tools.council_decisions_near, question="démolition", lat=46.8, lon=-71.2)
    with pytest.raises(ToolException, match="036_council_search.sql"):
        _invoke(rag_tools.search_council_minutes, question="démolition")


# ---------------------------------------------------------------------------
# The filters reach the query as the vocabulary
# ---------------------------------------------------------------------------


def test_plain_words_become_kinds_and_outcomes(council_db):
    out = _invoke(
        rag_tools.council_decisions_near,
        lat=46.80413, lon=-71.23612, item_kinds="demolitions, zoning", outcome="decided",
        since="2025-01-01", until="2025-12-31", radius_m=300,
    )
    call = council_db["items"]
    assert call["item_kinds"] == ["demolition", "zoning_amendment"]
    assert call["outcomes"] == ["approved", "refused"]
    assert call["since"] == date(2025, 1, 1) and call["until"] == date(2025, 12, 31)
    assert call["radius_m"] == 300.0
    assert "within 300 m" in out and "kind demolition/zoning amendment" in out
    assert "search" not in council_db, "no question, no embedding"


def test_last_months_sets_since_when_no_date_is_given(council_db):
    _invoke(rag_tools.council_decisions_near, lat=46.8, lon=-71.2, last_months=12)
    since = council_db["items"]["since"]
    assert since is not None
    assert 330 <= (date.today() - since).days <= 400


def test_an_unknown_kind_or_outcome_is_refused_with_the_vocabulary(council_db):
    with pytest.raises(ToolException, match="minor_variance"):
        _invoke(rag_tools.council_decisions_near, lat=46.8, lon=-71.2, item_kinds="teardown")
    with pytest.raises(ToolException, match="decided"):
        _invoke(rag_tools.council_decisions_near, lat=46.8, lon=-71.2, outcome="maybe")
    with pytest.raises(ToolException, match="ISO date"):
        _invoke(rag_tools.council_decisions_near, lat=46.8, lon=-71.2, since="last year")


# ---------------------------------------------------------------------------
# The place
# ---------------------------------------------------------------------------


def test_an_address_is_resolved_to_its_lot_and_selected(council_db, monkeypatch, lot_row):
    asked: dict = {}

    def lots_by_address(street, civic, **kwargs):
        asked.update({"street": street, "civic": civic, **kwargs})
        return [{"lot_number": "5 342 215", "neighborhood": "CIL"}]

    monkeypatch.setattr(queries, "lots_by_address", lots_by_address)
    out = _invoke(rag_tools.council_decisions_near, address="439 rue Jeanne-d'Arc, Montcalm, Québec")
    assert asked["civic"] == 439 and "jeanne" in asked["street"].lower()
    # "Montcalm, Québec" is read as the municipalities it may mean.
    assert asked["municipalities"] and "quebec" in asked["municipalities"]
    assert council_db["items"]["lon"] == lot_row["lon"] and council_db["items"]["lat"] == lot_row["lat"]
    assert state.get_selected_lot()["lot_number"] == lot_row["lot_number"]
    assert "439 rue Jeanne-d'Arc" in out


def test_an_address_that_matches_nothing_points_at_the_address_tool(council_db, monkeypatch):
    monkeypatch.setattr(queries, "lots_by_address", lambda *_a, **_k: [])
    with pytest.raises(ToolException, match="find_lot_by_address"):
        _invoke(rag_tools.council_decisions_near, address="9999 rue Inexistante, Québec")


def test_the_selected_lot_is_the_default_place(council_db):
    state.set_selected_lot("5 342 215", -71.23612, 46.80413, "CIL")
    out = _invoke(rag_tools.council_decisions_near)
    assert council_db["items"]["lon"] == -71.23612
    assert "lot 5 342 215" in out


def test_no_place_at_all_asks_for_one(council_db):
    with pytest.raises(ToolException, match="No place given"):
        _invoke(rag_tools.council_decisions_near)


# ---------------------------------------------------------------------------
# What the model is handed
# ---------------------------------------------------------------------------


def test_every_item_is_numbered_and_filed_for_citation(council_db):
    out = _invoke(rag_tools.council_decisions_near, lat=46.80413, lon=-71.23612)
    assert "[1]" in out
    assert "2025-07-07 · demolition · refused · council unfavorable · resolution/sommaire CA1-2025-0300" in out
    assert "42 m away (about 439, rue Jeanne-d’Arc)" in out
    assert "lots: 5342215" in out
    assert _item()["url"] in out
    ledger = state.citations()
    assert ledger[1]["url"] == _item()["url"]
    assert ledger[1]["scope"] == "council" and ledger[1]["outcome"] == "refused"
    assert ledger[1]["chunk_text"].startswith("Il est résolu de refuser")


def test_a_question_adds_the_passages_under_the_same_numbering(council_db):
    out = _invoke(rag_tools.council_decisions_near, question="démolition résidentielle", lat=46.8, lon=-71.2, radius_m=500)
    search = council_db["search"]
    assert search["lon"] == -71.2 and search["radius_m"] == 500.0
    assert "[1]" in out and "[2]" in out
    assert "Passages from the minutes" in out
    assert "Résolution CA1-2025-0300" in out and "similarity 0.810" in out
    assert set(state.citations()) == {1, 2}


def test_nothing_found_says_so_without_inventing_a_decision(council_db, monkeypatch):
    monkeypatch.setattr(queries, "council_items_near", lambda *_a, **_k: [])
    out = _invoke(rag_tools.council_decisions_near, lat=46.8, lon=-71.2, outcome="approved", item_kinds="demolition")
    assert "No planning item has a site in that radius with kind demolition, outcome approved" in out
    assert "'no decision read' means" in out
    assert not state.citations()


def test_search_council_minutes_is_the_corpus_with_no_place(council_db):
    out = _invoke(rag_tools.search_council_minutes, question="hébergement touristique", neighborhood="CIL", outcome="refused")
    search = council_db["search"]
    assert search["neighborhood"] == "CIL" and search["outcomes"] == ["refused"]
    assert "lon" not in search
    assert "Council minutes search in La Cité-Limoilou, Québec" in out
    assert "[1]" in out and state.citations()[1]["scope"] == "council"
