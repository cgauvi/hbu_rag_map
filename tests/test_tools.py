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
    assert "Permitted uses: C.4;H" in answer
    assert "Height max (m): 23" in answer
    assert "C01-001.pdf" in answer


def test_zoning_for_lot_uses_the_selection_when_given_nothing(monkeypatch, lot_row, zone_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, features=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(queries, "zoning_for_lot", lambda *_a, **_k: [zone_row])
    state.set_selected_lot("2 170 935", -73.6195, 45.5405)

    assert "C01-001" in _invoke(parcel_tools.zoning_for_lot)


def test_zoning_for_lot_asks_for_the_lots_own_snapshot(monkeypatch, lot_row, zone_row):
    """Not "whichever dates are loaded".

    The lot row came from one load of the cadastre and the zones that govern
    it are that load's. Asked across every date, the same zone comes back once
    per date and the answer reads as a lot straddling zones it does not.
    """
    captured = {}
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, features=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(
        queries, "zoning_for_lot",
        lambda *_a, **kwargs: captured.update(kwargs) or [zone_row],
    )

    _invoke(parcel_tools.zoning_for_lot, lot_number="2 170 935")

    assert captured["scrape_date"] == lot_row["scrape_date"]


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


def _coverage(**overrides):
    """A `queries.lot_coverage` reply, defaulting to the `lot_row` fixture's lot."""
    row = {
        "lot_number": "2 170 935",
        "neighborhood": "VSMPE",
        "scrape_date": "2026-08-20",
        "lot_area_m2": 267.9227,
        "num_footprints": 1,
        "covered_area_m2": 130.0,
    }
    row.update(overrides)
    row["coverage_pct"] = queries.coverage_pct(
        row["covered_area_m2"], row["lot_area_m2"]
    )
    return row


def test_buildings_on_lot_reports_coverage(monkeypatch, lot_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, buildings=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(
        queries, "buildings_on_lot",
        lambda *_a, **_k: [{"building_uid": 1, "area_m2": 140.0, "overlap_m2": 130.0,
                            "pct_of_building": 92.9}],
    )
    monkeypatch.setattr(queries, "lot_coverage", lambda *_a, **_k: _coverage())

    answer = _invoke(parcel_tools.buildings_on_lot, lot_number="2 170 935")

    assert "1 footprint" in answer
    # 130 / 267.9 ≈ 49%
    assert "49% of the lot" in answer
    # Ground, not floor: the two are different measurements in the same unit,
    # and the model quotes whichever word the tool used.
    assert "of its ground" in answer


def test_buildings_on_lot_reads_the_total_off_the_union_not_off_the_rows(
    monkeypatch, lot_row
):
    """Lot 2 165 628: one building, 164 m² of a 312 m² lot — not two and 329.

    The rows are what the reported bug looked like — the same footprint under
    two ids, one per snapshot loaded — and the tool must not add them up. The
    total comes from `lot_coverage`, which unions the clipped shapes within one
    snapshot, so the answer stays inside the lot however many rows arrive.
    """
    lot = dict(lot_row, lot_number="2 165 628", area_m2=311.6195007413626)
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, buildings=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot)
    monkeypatch.setattr(
        queries, "buildings_on_lot",
        lambda *_a, **_k: [
            {"building_uid": 63821, "area_m2": 1954.9, "overlap_m2": 164.486},
            {"building_uid": 23837, "area_m2": 1954.9, "overlap_m2": 164.486},
        ],
    )
    monkeypatch.setattr(
        queries, "lot_coverage",
        lambda *_a, **_k: _coverage(
            lot_number="2 165 628",
            lot_area_m2=311.6195007413626,
            covered_area_m2=164.48598719830625,
        ),
    )

    answer = _invoke(parcel_tools.buildings_on_lot, lot_number="2 165 628")

    assert "164 m² of its ground" in answer
    assert "53% of the lot" in answer
    assert "329" not in answer


def test_buildings_on_lot_asks_both_reads_for_the_lots_own_snapshot(
    monkeypatch, lot_row
):
    """A listing and a total from two different loads describe two lots."""
    dates = []
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, buildings=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(
        queries, "buildings_on_lot",
        lambda _n, scrape_date=None: dates.append(scrape_date) or [
            {"building_uid": 1, "area_m2": 140.0, "overlap_m2": 130.0}
        ],
    )
    monkeypatch.setattr(
        queries, "lot_coverage",
        lambda _n, scrape_date=None: dates.append(scrape_date) or _coverage(),
    )

    _invoke(parcel_tools.buildings_on_lot, lot_number="2 170 935")

    assert dates == [lot_row["scrape_date"], lot_row["scrape_date"]]


def test_buildings_on_lot_names_the_whole_footprint_and_the_part_on_this_lot(
    monkeypatch, lot_row
):
    """A footprint spanning several lots is bigger than any one of them.

    Lot 2 165 628 carries 164 m² of a 1,955 m² building, and printing the two
    numbers without saying which is which is how a building came to read as
    six times the lot it stands on.
    """
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, buildings=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(
        queries, "buildings_on_lot",
        lambda *_a, **_k: [{"building_uid": 1, "area_m2": 1954.9, "overlap_m2": 164.5}],
    )
    monkeypatch.setattr(
        queries, "lot_coverage", lambda *_a, **_k: _coverage(covered_area_m2=164.5)
    )

    answer = _invoke(parcel_tools.buildings_on_lot, lot_number="2 170 935")

    assert "1,955 m² footprint in all" in answer
    assert "164 m² of it on this lot" in answer


def test_a_vacant_lot_reads_as_vacant(monkeypatch, lot_row):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, buildings=True))
    monkeypatch.setattr(queries, "lot_by_number", lambda *_a, **_k: lot_row)
    monkeypatch.setattr(queries, "buildings_on_lot", lambda *_a, **_k: [])
    monkeypatch.setattr(
        queries, "lot_coverage",
        lambda *_a, **_k: _coverage(num_footprints=0, covered_area_m2=0.0),
    )

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


# ---------------------------------------------------------------------------
# find_lot_by_address
# ---------------------------------------------------------------------------


def _address_row(**overrides) -> dict:
    """One row shaped like queries.lots_by_address returns."""
    row = {
        "lot_number": "3 457 943", "neighborhood": "VSMPE", "scrape_date": "2026-09-01",
        "lot_uid": 42, "exact_name": True, "in_view": False,
        "num_addresses": 41, "num_civic_addresses": 1, "num_lot_addresses": 41,
        "num_pieces": 1, "civic_min": 7430, "civic_max": 7430,
        "civic_address": "7430 Rue Lajeunesse", "street_name": "Rue Lajeunesse",
        "municipality": "Montréal", "feature_id": "H02-132",
        "source_table": "Reglement_urbanisme__VSP_REG_ZONE", "piece_basis": "piece",
        "match_basis": "within", "snap_distance_m": None,
    }
    row.update(overrides)
    return row


def _address_stubs(monkeypatch, lot_row, rows, *, coverage=None):
    """The query layer behind the tool: what it was asked, and canned rows.

    ``rows`` is a list, or a function of the lookup's keyword arguments for a
    test that answers differently per city. ``captured`` holds the *first*
    lookup's arguments; ``captured["calls"]`` every lookup's.
    """
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(lots=True, lot_addresses=True))
    captured = {"calls": []}

    def fake(street, civic_number, **kwargs):
        call = {**kwargs, "street": street, "civic_number": civic_number}
        if not captured["calls"]:
            captured.update(call)
        captured["calls"].append(call)
        return [dict(r) for r in (rows(kwargs) if callable(rows) else rows)]

    monkeypatch.setattr(queries, "lots_by_address", fake)
    monkeypatch.setattr(
        queries, "lot_by_number",
        lambda number, **_k: {**lot_row, "lot_number": number},
    )
    monkeypatch.setattr(queries, "street_summary", lambda *_a, **_k: [])
    monkeypatch.setattr(queries, "nearest_addresses", lambda *_a, **_k: [])
    monkeypatch.setattr(queries, "similar_streets", lambda *_a, **_k: [])
    monkeypatch.setattr(queries, "doors_near", lambda *_a, **_k: [])
    monkeypatch.setattr(
        queries, "address_coverage",
        lambda: coverage if coverage is not None else [
            {"neighborhood": "VSMPE", "scrape_date": "2026-09-01",
             "num_addresses": 88414, "num_lots": 21385}
        ],
    )
    monkeypatch.setattr(queries, "neighborhoods", lambda *_a, **_k: ["CIL", "VSMPE"])
    return captured


#: Coverage that names its cities, so a reading of a place can be unloaded.
_THREE_CITIES = [
    {"neighborhood": n, "municipality": m, "scrape_date": "2026-09-01",
     "num_addresses": 1, "num_lots": 1}
    for n, m in (("VSMPE", "Montréal"), ("CIL", "Québec"), ("SSC", "Québec"),
                 ("SAG", "Saguenay"))
]


def test_addresses_missing_names_the_asset_that_fills_them(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(lots=True, lot_addresses=False)
    )
    with pytest.raises(ToolException, match="lot_addresses"):
        _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=7430)


def test_an_address_selects_and_frames_its_lot(monkeypatch, lot_row):
    captured = _address_stubs(monkeypatch, lot_row, [_address_row()])

    answer = _invoke(
        parcel_tools.find_lot_by_address, street="rue Lajeunesse", civic_number=7430
    )

    assert "7430 Rue Lajeunesse" in answer
    assert "lot 3 457 943" in answer and "H02-132" in answer
    assert "1 door(s) and 41 addressable unit(s)" in answer
    assert captured["civic_number"] == 7430 and captured["street"] == "rue Lajeunesse"
    command = state.take_map_command()
    assert command["select_lot"] == "3 457 943"
    assert command["fit_bounds"] is not None
    assert state.get_selected_lot()["lot_number"] == "3 457 943"


def test_the_whole_address_in_one_argument_is_taken_apart(monkeypatch, lot_row):
    captured = _address_stubs(monkeypatch, lot_row, [_address_row()])

    _invoke(parcel_tools.find_lot_by_address, street="7390 A Rue De Lanaudière")

    assert captured["civic_number"] == 7390
    assert captured["civic_suffix"] == "A"
    assert captured["street"] == "Rue De Lanaudière"


def test_no_place_sends_no_municipality(monkeypatch, lot_row):
    captured = _address_stubs(monkeypatch, lot_row, [_address_row()])
    _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=7430)
    assert captured["municipalities"] is None


@pytest.mark.parametrize(
    ("kwargs", "municipalities"),
    [
        ({"street": "chemin Saint-Louis", "civic_number": 1234, "city": "Sillery"},
         ["quebec"]),
        ({"street": "1234 chemin Saint-Louis, Cap-Rouge"}, ["quebec"]),
        ({"street": "rue Racine", "civic_number": 1, "city": "CHICOUTIMI"}, ["saguenay"]),
        ({"street": "7430 Lajeunesse, Montréal (Québec) H2R 2H8"}, ["montreal"]),
        ({"street": "rue X", "civic_number": 1, "city": "Montcalm"}, ["quebec", "montcalm"]),
        # A place handed over as a borough code is read as the place it is.
        ({"street": "Lajeunesse", "civic_number": 7430, "neighborhood": "Villeray"},
         ["montreal"]),
    ],
)
def test_the_place_is_looked_for_in_every_city_it_may_mean(
    monkeypatch, lot_row, kwargs, municipalities
):
    captured = _address_stubs(monkeypatch, lot_row, [_address_row()])
    _invoke(parcel_tools.find_lot_by_address, **kwargs)
    assert captured["municipalities"] == municipalities
    assert captured["neighborhood"] is None


def test_a_registered_borough_code_stays_a_code(monkeypatch, lot_row):
    captured = _address_stubs(monkeypatch, lot_row, [_address_row()])
    _invoke(
        parcel_tools.find_lot_by_address,
        street="Lajeunesse", civic_number=7430, neighborhood="VSMPE",
    )
    assert captured["neighborhood"] == "VSMPE" and captured["municipalities"] is None


def test_the_likelier_reading_of_a_place_is_selected_and_the_other_named(
    monkeypatch, lot_row
):
    """Montcalm is the Québec quartier seven times in ten; the town is not loaded."""
    _address_stubs(
        monkeypatch, lot_row,
        [_address_row(lot_number="1 315 000", municipality="Québec", neighborhood="CIL")],
        coverage=_THREE_CITIES,
    )

    answer = _invoke(
        parcel_tools.find_lot_by_address,
        street="avenue Cartier", civic_number=1000, city="Montcalm",
    )

    assert "stands on lot 1 315 000" in answer
    assert "70% Montcalm, a quartier of Québec" in answer
    assert "30% Montcalm (Laurentides)" in answer and "not loaded" in answer
    assert state.take_map_command()["select_lot"] == "1 315 000"


def test_an_unloaded_likelier_reading_is_proposed_not_selected(monkeypatch, lot_row):
    """Mont-Royal is the town before the mountain; a Montréal door is only a maybe."""
    _address_stubs(
        monkeypatch, lot_row,
        [_address_row(lot_number="2 000 001", municipality="Montréal")],
        coverage=_THREE_CITIES,
    )

    answer = _invoke(
        parcel_tools.find_lot_by_address,
        street="chemin Remembrance", civic_number=1, city="Mont-Royal",
    )

    assert "none is a clear choice" in answer and "2 000 001" in answer
    assert "Not checked, because their addresses are not loaded: Mont-Royal" in answer
    assert state.take_map_command() is None


def test_two_cities_with_the_address_are_ranked_by_the_place(monkeypatch, lot_row):
    """Plateau is Montréal's four times in five: clear enough to select."""
    rows = [
        _address_row(lot_number="1 000 002", municipality="Québec", neighborhood="SSC"),
        _address_row(lot_number="1 000 001", municipality="Montréal"),
    ]
    _address_stubs(monkeypatch, lot_row, rows, coverage=_THREE_CITIES)

    answer = _invoke(
        parcel_tools.find_lot_by_address, street="rue X", civic_number=1, city="Plateau"
    )

    assert "stands on lot 1 000 001" in answer
    assert "Also matched, not selected" in answer and "1 000 002" in answer


def test_an_even_split_between_two_cities_is_proposed_with_likelihoods(
    monkeypatch, lot_row
):
    rows = [
        _address_row(lot_number="1 000 001", municipality="Montréal"),
        _address_row(lot_number="1 000 002", municipality="Québec", neighborhood="CIL"),
    ]
    _address_stubs(monkeypatch, lot_row, rows, coverage=_THREE_CITIES)

    answer = _invoke(
        parcel_tools.find_lot_by_address, street="rue X", civic_number=1, city="Vieux-Port"
    )

    assert "none is a clear choice" in answer
    assert "50% likely" in answer and "Propose these to the user" in answer
    assert state.take_map_command() is None


def test_a_place_the_address_is_not_in_proposes_where_it_is(monkeypatch, lot_row):
    """Westmount is not loaded; the same door in Montréal is offered, not taken."""
    captured = _address_stubs(
        monkeypatch, lot_row,
        lambda kw: [] if kw["municipalities"] else [_address_row(municipality="Montréal")],
        coverage=_THREE_CITIES,
    )

    answer = _invoke(
        parcel_tools.find_lot_by_address,
        street="Lajeunesse", civic_number=7430, city="Westmount",
    )

    assert [c["municipalities"] for c in captured["calls"]] == [["westmount"], None]
    assert "No 7430 Lajeunesse, Westmount in any municipality" in answer
    assert "exist elsewhere" in answer and "3 457 943" in answer
    assert state.take_map_command() is None


def test_an_unknown_place_is_its_own_municipality_and_says_so(monkeypatch, lot_row):
    captured = _address_stubs(monkeypatch, lot_row, [])

    answer = _invoke(
        parcel_tools.find_lot_by_address,
        street="boulevard des Laurentides", civic_number=1, city="Blorpville",
    )

    assert captured["municipalities"] == ["blorpville"]
    assert "not a place this tool knows" in answer
    assert state.take_map_command() is None


def test_a_missing_number_proposes_the_nearest_doors(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(
        queries, "street_summary",
        lambda *_a, **_k: [{
            "street_name": "Rue Lajeunesse", "neighborhood": "VSMPE",
            "scrape_date": "2026-09-01", "exact_name": True, "num_lots": 312,
            "num_civic_addresses": 1090, "civic_min": 7000, "civic_max": 9199,
        }],
    )
    monkeypatch.setattr(
        queries, "nearest_addresses",
        lambda *_a, **_k: [
            {"civic_address": "7429 Rue Lajeunesse", "municipality": "Montréal",
             "lot_number": "3 457 940", "neighborhood": "VSMPE"},
            {"civic_address": "7433 Rue Lajeunesse", "municipality": "Montréal",
             "lot_number": "3 457 944", "neighborhood": "VSMPE"},
        ],
    )

    answer = _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=7431)

    assert "Nearest doors on that street" in answer
    assert "7429 Rue Lajeunesse" in answer and "7433 Rue Lajeunesse" in answer


def test_a_missing_street_proposes_the_ones_spelled_like_it(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])
    captured = {}

    def similar(street, **kwargs):
        captured.update(kwargs, street=street)
        return [{"street_name": "Rue Lajeunesse", "neighborhood": "VSMPE",
                 "municipality": "Montréal", "num_lots": 312,
                 "civic_min": 7000, "civic_max": 9199, "similarity": 0.94}]

    monkeypatch.setattr(queries, "similar_streets", similar)

    answer = _invoke(parcel_tools.find_lot_by_address, street="Lajeunese", civic_number=7430)

    assert "Streets spelled like it" in answer and "Rue Lajeunesse (Villeray–Saint-Michel–Parc-Extension, Montréal [VSMPE])" in answer
    assert captured["street"] == "Lajeunese"


def test_the_viewport_goes_with_the_lookup(monkeypatch, lot_row):
    """So the borough the user is looking at can break a tie between two."""
    captured = _address_stubs(monkeypatch, lot_row, [_address_row()])
    state.set_viewport((-73.7, 45.5, -73.6, 45.6), 17, (45.55, -73.65))

    _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=7430)

    assert captured["bounds"] == (-73.7, 45.5, -73.6, 45.6)


def test_an_exact_street_beats_one_that_merely_contains_it(monkeypatch, lot_row):
    """400 Jarry Est and 400 Jarry Ouest both exist; "Jarry Est" is not a question."""
    rows = [
        _address_row(lot_number="1 000 001", street_name="Rue Jarry Est",
                     civic_address="400 Rue Jarry Est", exact_name=True),
        _address_row(lot_number="1 000 002", street_name="Rue Jarry Ouest",
                     civic_address="400 Rue Jarry Ouest", exact_name=False),
    ]
    _address_stubs(monkeypatch, lot_row, rows)

    answer = _invoke(parcel_tools.find_lot_by_address, street="Jarry Est", civic_number=400)

    assert "stands on lot 1 000 001" in answer
    assert "Also matched, not selected" in answer and "1 000 002" in answer
    assert state.take_map_command()["select_lot"] == "1 000 001"


def test_the_same_street_in_two_boroughs_is_a_question_unless_one_is_in_view(
    monkeypatch, lot_row
):
    rows = [
        _address_row(lot_number="1 000 001", neighborhood="VSMPE", in_view=False),
        _address_row(lot_number="1 000 002", neighborhood="CIL", in_view=False),
    ]
    _address_stubs(monkeypatch, lot_row, rows)

    answer = _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=7430)

    assert "none is a clear choice" in answer
    assert "1 000 001" in answer and "1 000 002" in answer
    assert state.take_map_command() is None
    assert state.get_selected_lot()["lot_number"] is None

    rows[1]["in_view"] = True
    answer = _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=7430)

    assert "stands on lot 1 000 002" in answer
    assert state.take_map_command()["select_lot"] == "1 000 002"


def test_a_snapped_point_says_so(monkeypatch, lot_row):
    _address_stubs(
        monkeypatch, lot_row, [_address_row(match_basis="snapped", snap_distance_m=1.4)]
    )
    answer = _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=7430)
    assert "snapped to it (1.4 m)" in answer


def test_a_number_the_street_does_not_have_reports_the_streets_span(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(
        queries, "street_summary",
        lambda *_a, **_k: [{
            "street_name": "Rue Lajeunesse", "neighborhood": "VSMPE",
            "scrape_date": "2026-09-01", "exact_name": True, "num_lots": 312,
            "num_civic_addresses": 1090, "civic_min": 7000, "civic_max": 9199,
        }],
    )

    answer = _invoke(parcel_tools.find_lot_by_address, street="Lajeunesse", civic_number=99999)

    assert "No 99999 Lajeunesse" in answer
    assert "runs 7000–9199" in answer and "no door numbered 99999" in answer
    assert state.take_map_command() is None


def test_a_street_in_a_borough_without_addresses_says_which(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])

    answer = _invoke(
        parcel_tools.find_lot_by_address, street="Grande Allée", civic_number=1
    )

    assert "No street matching 'Grande Allée'" in answer
    assert "addresses loaded for Villeray–Saint-Michel–Parc-Extension, Montréal [VSMPE] (2026-09-01)" in answer
    assert "Lots are loaded for CIL but their addresses are not" in answer


def test_no_number_describes_the_street(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(
        queries, "street_summary",
        lambda *_a, **_k: [
            {"street_name": "Rue Jarry Est", "neighborhood": "VSMPE",
             "scrape_date": "2026-09-01", "exact_name": False, "num_lots": 400,
             "num_civic_addresses": 900, "civic_min": 1, "civic_max": 4999},
            {"street_name": "Rue Jarry Ouest", "neighborhood": "VSMPE",
             "scrape_date": "2026-09-01", "exact_name": False, "num_lots": 120,
             "num_civic_addresses": 300, "civic_min": 1, "civic_max": 899},
        ],
    )

    answer = _invoke(parcel_tools.find_lot_by_address, street="Jarry")

    assert "Rue Jarry Est" in answer and "Rue Jarry Ouest" in answer
    assert "400 lots" in answer and "Give a civic number" in answer


def test_no_street_at_all_is_an_error(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])
    with pytest.raises(ToolException, match="No street name"):
        _invoke(parcel_tools.find_lot_by_address, street="   ", civic_number=7430)


def _fraser(lot_number, number, **overrides):
    return _address_row(
        lot_number=lot_number, neighborhood="CIL", municipality="Québec",
        civic_address=f"{number} Rue Fraser", street_name="Rue Fraser",
        civic_min=number, civic_max=number, num_addresses=1, num_civic_addresses=1,
        num_lot_addresses=3, feature_id="14047Hb", **overrides,
    )


def test_several_doors_typed_together_are_pooled_on_their_lot(monkeypatch, lot_row):
    """A triplex's facade reads "189, 191, 193": one building, one lot."""
    captured = _address_stubs(
        monkeypatch, lot_row,
        lambda kw: [],
    )
    by_number = {n: [_fraser("1 302 447", n)] for n in (189, 191, 193)}
    monkeypatch.setattr(
        queries, "lots_by_address",
        lambda street, n, **kw: captured["calls"].append(n) or by_number.get(n, []),
    )

    answer = _invoke(
        parcel_tools.find_lot_by_address, street="189, 191, 193 Rue Fraser, Montcalm"
    )

    assert captured["calls"] == [189, 191, 193]
    assert "189, 191, 193 Rue Fraser, Québec stands on lot 1 302 447" in answer
    assert "3 door(s)" in answer
    assert state.take_map_command()["select_lot"] == "1 302 447"


def test_a_number_between_two_doors_of_one_lot_selects_it(monkeypatch, lot_row):
    """191 is not printed; 189 and 193 are, both on lot 1 302 447."""
    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(
        queries, "lots_by_address",
        lambda street, n, **kw: [_fraser("1 302 447", n)] if n in (189, 193) else [],
    )
    monkeypatch.setattr(
        queries, "doors_near",
        lambda *_a, **_k: [
            {"neighborhood": "CIL", "street_name": "Rue Fraser", "municipality": "Québec",
             "lot_number": lot, "civic_number": n, "exact_name": True}
            for lot, n in (("1 302 448", 187), ("1 302 447", 189),
                           ("1 302 447", 193), ("1 302 419", 197))
        ],
    )

    answer = _invoke(
        parcel_tools.find_lot_by_address, street="rue Fraser", civic_number=191,
        city="Montcalm",
    )

    assert answer.startswith("No 191 rue Fraser, Montcalm as typed.")
    assert "falls between 189 and 193 Rue Fraser" in answer
    assert "stands on lot 1 302 447" in answer
    assert state.take_map_command()["select_lot"] == "1 302 447"


def test_a_swapped_numbered_street_is_asked_not_selected(monkeypatch, lot_row):
    """4 1re Avenue found as 1 4e Avenue is a guess: the user confirms it first."""
    asked = []

    def lookup(street, n, **kw):
        asked.append((street, n))
        return [_address_row(lot_number="9 000 001", civic_address="3 4e Rue",
                             street_name="4e Rue")] if (street, n) == ("4e rue", 3) else []

    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(queries, "lots_by_address", lookup)

    answer = _invoke(parcel_tools.find_lot_by_address, street="4 3e rue")

    assert asked == [("3e rue", 4), ("4e rue", 3)]
    assert "number and the street's ordinal swapped: 3 4e rue" in answer
    assert "Ask the user whether they meant 3 4e Rue" in answer
    assert "lot 9 000 001" in answer and "stands on" not in answer
    assert state.take_map_command() is None
    assert state.get_selected_lot()["lot_number"] is None


def test_a_misspelt_street_is_proposed_under_its_likely_spelling(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(
        queries, "lots_by_address",
        lambda street, n, **kw: [_address_row()] if street == "Rue Lajeunesse" else [],
    )
    monkeypatch.setattr(
        queries, "similar_streets",
        lambda *_a, **_k: [{"street_name": "Rue Lajeunesse", "neighborhood": "VSMPE",
                            "municipality": "Montréal", "num_lots": 312,
                            "civic_min": 7000, "civic_max": 9199, "similarity": 0.94}],
    )

    answer = _invoke(parcel_tools.find_lot_by_address, street="Lajeunese", civic_number=7430)

    assert "'Lajeunese' was read as a misspelling of Rue Lajeunesse" in answer
    assert "nothing is selected" in answer and "3 457 943" in answer
    assert state.take_map_command() is None


def test_two_guesses_are_listed_for_the_user_to_pick(monkeypatch, lot_row):
    """1 4e Avenue and 1 A 4e Avenue are two lots: both go to the user."""
    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(
        queries, "lots_by_address",
        lambda street, n, **kw: [
            _address_row(lot_number="1 568 411", civic_address="1 4e Avenue"),
            _address_row(lot_number="1 568 409", civic_address="1 A 4e Avenue"),
        ] if (street, n) == ("4e avenue", 1) else [],
    )

    answer = _invoke(parcel_tools.find_lot_by_address, street="4 1re avenue")

    assert "which of these they meant" in answer
    assert "1 568 411" in answer and "1 568 409" in answer
    assert state.take_map_command() is None


def test_a_loose_reading_outside_the_place_is_only_proposed(monkeypatch, lot_row):
    """The bracket exists, but only in a city the place does not mean."""
    _address_stubs(
        monkeypatch, lot_row,
        lambda kw: [], coverage=_THREE_CITIES,
    )
    monkeypatch.setattr(
        queries, "lots_by_address",
        lambda street, n, **kw: (
            [_fraser("1 302 447", n)] if n in (189, 193) and not kw["municipalities"] else []
        ),
    )
    monkeypatch.setattr(
        queries, "doors_near",
        lambda *_a, municipalities=None, **_k: [] if municipalities else [
            {"neighborhood": "CIL", "street_name": "Rue Fraser", "municipality": "Québec",
             "lot_number": "1 302 447", "civic_number": n, "exact_name": True}
            for n in (189, 193)
        ],
    )

    answer = _invoke(
        parcel_tools.find_lot_by_address, street="rue Fraser", civic_number=191,
        city="Chicoutimi",
    )

    assert "exist elsewhere" in answer and "falls between 189 and 193" in answer
    assert state.take_map_command() is None


def test_data_status_reports_which_boroughs_have_addresses(monkeypatch, lot_row):
    _address_stubs(monkeypatch, lot_row, [])
    monkeypatch.setattr(queries, "scrape_dates", lambda *_a, **_k: [])

    answer = _invoke(parcel_tools.data_status)

    assert "addresses: Villeray–Saint-Michel–Parc-Extension, Montréal [VSMPE] (2026-09-01, 88,414 points on 21,385 lots)" in answer
