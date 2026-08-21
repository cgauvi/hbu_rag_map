"""The agent's tools, with the query layer stubbed.

Two things are being checked. The first is that a tool tells the model
something it can act on — a missing table has to read as "this is not loaded",
not as a Postgres error, or the model retries it three times. The second is
that finding a lot actually moves the map: the tool's return string is only
half its job.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import ToolException

from src.tools import map_tools, parcel_tools, rag_tools
from src.utils import queries, state


def _caps(**flags):
    return queries.Capabilities(**{"postgis": True, "pgvector": True, **flags})


def _invoke(tool, **kwargs):
    """Call through the StructuredTool wrapper, as the agent does."""
    return tool.invoke(kwargs)


# ---------------------------------------------------------------------------
# Missing data reads as missing data
# ---------------------------------------------------------------------------


def test_a_missing_table_names_what_is_absent(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=False))
    with pytest.raises(ToolException, match="not loaded"):
        _invoke(parcel_tools.find_lot, lot_number="2 170 935")


def test_a_missing_corpus_names_the_asset_that_creates_it(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(chunks=False))
    with pytest.raises(ToolException, match="document_index"):
        _invoke(rag_tools.search_regulations, question="hauteur maximale")


def test_a_missing_search_function_points_at_the_right_sql_file(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(chunks=True, lots=True, search_at_lot=False)
    )
    state.set_selected_lot("2 170 935", -73.6, 45.5)
    with pytest.raises(ToolException, match="003_spatial_search.sql"):
        _invoke(rag_tools.regulations_at_lot, question="hauteur")


# ---------------------------------------------------------------------------
# find_lot
# ---------------------------------------------------------------------------


def test_find_lot_selects_and_frames_it(monkeypatch, lot_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)

    answer = _invoke(parcel_tools.find_lot, lot_number="2170935")

    assert "2 170 935" in answer
    command = state.take_map_command()
    assert command["select_lot"] == "2 170 935"
    assert command["fit_bounds"] is not None
    assert state.get_selected_lot()["lot_number"] == "2 170 935"


def test_find_lot_on_an_unknown_number_says_so(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: None)

    with pytest.raises(ToolException, match="No lot numbered"):
        _invoke(parcel_tools.find_lot, lot_number="9999999")


def test_describe_selected_lot_with_nothing_selected(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True))
    assert "No lot is selected" in _invoke(parcel_tools.describe_selected_lot)


def test_describe_selected_lot_reports_the_click(monkeypatch, lot_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    state.set_selected_lot("2 170 935", -73.6195, 45.5405, "VSMPE")

    answer = _invoke(parcel_tools.describe_selected_lot)

    assert "2 170 935" in answer and "268 m²" in answer


# ---------------------------------------------------------------------------
# zoning
# ---------------------------------------------------------------------------


def test_zoning_for_lot_reports_the_grid_and_the_pdf(monkeypatch, lot_row, zone_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, features=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(queries, "zoning_for_lot", lambda *_a, **_k: [zone_row])

    answer = _invoke(parcel_tools.zoning_for_lot, lot_number="2 170 935")

    assert "C01-001" in answer
    assert "Usages autorisés: C.4;H" in answer
    assert "Hauteur max (m): 23" in answer
    assert "C01-001.pdf" in answer


def test_zoning_for_lot_uses_the_selection_when_given_nothing(monkeypatch, lot_row, zone_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, features=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(queries, "zoning_for_lot", lambda *_a, **_k: [zone_row])
    state.set_selected_lot("2 170 935", -73.6195, 45.5405)

    assert "C01-001" in _invoke(parcel_tools.zoning_for_lot)


def test_a_lot_straddling_two_zones_reports_both(monkeypatch, lot_row, zone_row):
    other = {**zone_row, "zone": "H02-004", "overlap_m2": 90.0, "lot_area_m2": 267.9}
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, features=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(queries, "zoning_for_lot", lambda *_a, **_k: [zone_row, other])

    answer = _invoke(parcel_tools.zoning_for_lot, lot_number="2 170 935")

    assert "C01-001" in answer and "H02-004" in answer
    # The share is what tells the model the first is not the whole answer.
    assert "%" in answer


def test_no_zoning_polygon_is_reported_not_raised(monkeypatch, lot_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, features=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(queries, "zoning_for_lot", lambda *_a, **_k: [])

    assert "No zoning polygon" in _invoke(parcel_tools.zoning_for_lot, lot_number="2 170 935")


# ---------------------------------------------------------------------------
# buildings
# ---------------------------------------------------------------------------


def test_buildings_on_lot_reports_coverage(monkeypatch, lot_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, buildings=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(
        queries, "buildings_on_lot",
        lambda *_a, **_k: [{"building_uid": 1, "area_m2": 140.0, "overlap_m2": 130.0,
                            "pct_of_building": 92.9}],
    )

    answer = _invoke(parcel_tools.buildings_on_lot, lot_number="2 170 935")

    assert "1 footprint" in answer
    # 130 / 267.9 ≈ 49%
    assert "49% of the lot" in answer


def test_a_vacant_lot_reads_as_vacant(monkeypatch, lot_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, buildings=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(queries, "buildings_on_lot", lambda *_a, **_k: [])

    assert "vacant" in _invoke(parcel_tools.buildings_on_lot, lot_number="2 170 935")


# ---------------------------------------------------------------------------
# list_lots
# ---------------------------------------------------------------------------


def test_list_lots_needs_a_viewport(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True))
    state.set_viewport(None, None, None)
    with pytest.raises(ToolException, match="viewport"):
        _invoke(parcel_tools.list_lots)


def test_list_lots_sorts_biggest_first(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True))
    state.set_viewport((-73.7, 45.5, -73.6, 45.6), 17, (45.55, -73.65))

    def fake(*_a, **_k):
        return queries.FeatureSet(
            features=[
                {"properties": {"lot_number": "A", "area_m2": 300.0}},
                {"properties": {"lot_number": "B", "area_m2": 900.0}},
            ],
            layer="lots",
        )

    monkeypatch.setattr(queries, "lots_in_bbox", fake)

    answer = _invoke(parcel_tools.list_lots, min_area_m2=100)

    assert answer.index("Lot B") < answer.index("Lot A")


# ---------------------------------------------------------------------------
# map control
# ---------------------------------------------------------------------------


def test_focus_map_rejects_swapped_coordinates():
    with pytest.raises(ToolException, match="swapped"):
        _invoke(map_tools.focus_map, lat=-73.6, lon=45.5)


def test_focus_map_clamps_the_zoom():
    _invoke(map_tools.focus_map, lat=45.55, lon=-73.62, zoom=42)
    assert state.take_map_command()["zoom"] == 19


def test_setting_a_layer_with_no_table_is_refused(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(buildings=False))
    with pytest.raises(ToolException, match="not loaded"):
        _invoke(map_tools.set_map_layers, buildings=True)


def test_turning_a_layer_off_needs_no_table(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(buildings=False))
    assert "buildings off" in _invoke(map_tools.set_map_layers, buildings=False)


def test_no_layer_named_is_an_error():
    with pytest.raises(ToolException, match="at least one"):
        _invoke(map_tools.set_map_layers)


def test_clearing_the_size_filter_says_so():
    answer = _invoke(map_tools.filter_lots_on_map)
    assert "cleared" in answer
    assert state.take_map_command()["filters"] == {
        "min_area_m2": None, "max_area_m2": None
    }


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_regulations_at_lot_records_what_the_pane_should_show(monkeypatch, lot_row):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(lots=True, chunks=True, search_at_lot=True),
    )
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(rag_tools, "embed_query", lambda *_a, **_k: [0.1, 0.2])
    monkeypatch.setattr(
        queries, "search_at_lot",
        lambda *_a, **_k: [
            {"chunk_id": "c1", "chunk_text": "hauteur maximale 23 m",
             "similarity": 0.87, "url": "http://x/C01-001.pdf",
             "source_table": "Reglement_urbanisme__VSP_REG_ZONE"}
        ],
    )

    answer = _invoke(rag_tools.regulations_at_lot, question="hauteur maximale",
                     lot_number="2 170 935")

    assert "[1]" in answer and "hauteur maximale 23 m" in answer
    assert state.RagBuffer["scope"] == "lot"
    assert state.RagBuffer["lot_number"] == "2 170 935"


def test_an_empty_retrieval_is_reported_honestly(monkeypatch, lot_row):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(lots=True, chunks=True, search_at_lot=True),
    )
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(rag_tools, "embed_query", lambda *_a, **_k: [0.1])
    monkeypatch.setattr(queries, "search_at_lot", lambda *_a, **_k: [])

    assert "Nothing in the corpus matched" in _invoke(
        rag_tools.regulations_at_lot, question="hauteur", lot_number="2 170 935"
    )


def test_match_count_is_capped(monkeypatch, lot_row):
    captured = {}
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(lots=True, chunks=True, search_at_lot=True),
    )
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(rag_tools, "embed_query", lambda *_a, **_k: [0.1])
    monkeypatch.setattr(
        queries, "search_at_lot",
        lambda *a, **k: captured.update(match_count=k["match_count"]) or [],
    )

    _invoke(rag_tools.regulations_at_lot, question="x", lot_number="2 170 935",
            match_count=100)

    assert captured["match_count"] == rag_tools.MAX_MATCHES


def test_regulations_near_falls_back_to_the_selected_lot(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(chunks=True, search_near=True)
    )
    monkeypatch.setattr(rag_tools, "embed_query", lambda *_a, **_k: [0.1])
    monkeypatch.setattr(
        queries, "search_near",
        lambda _e, lon, lat, **k: captured.update(lon=lon, lat=lat) or [],
    )
    state.set_selected_lot("2 170 935", -73.6195, 45.5405)

    _invoke(rag_tools.regulations_near, question="caractère du secteur")

    assert captured == {"lon": -73.6195, "lat": 45.5405}


def test_regulations_near_with_no_reference_point_asks_for_one(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(chunks=True, search_near=True)
    )
    state.clear_selected_lot()
    state.set_viewport(None, None, None)

    with pytest.raises(ToolException, match="click on the map"):
        _invoke(rag_tools.regulations_near, question="x")
