"""The Opportunities layer: the second axis of the shortlist table on the map.

`gold.lot_investment_opportunities` files every lot under a *site thesis* -
why the parcel is acquirable - and this is the layer, the per-lot read, the
two rankings, the colours and the tool that surface it. What these pin: the
join is on the cadastral number (the table has been found stranded on an old
`lot_uid` generation once already), the two screens reach the query, an
unknown thesis in a URL draws every thesis rather than none, and the legend
and the style read the same colours.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.tools import ToolException

from src.tools import map_tools, parcel_tools
from src.utils import basemap, queries, tiles


@pytest.fixture
def captured(monkeypatch):
    calls: list[tuple[str, object]] = []
    rows: list[dict] = []

    def fake_query(sql, params=None):
        calls.append((sql, params))
        return list(rows)

    monkeypatch.setattr(queries, "query", fake_query)
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(
            postgis=True, lots=True, redevelopment_gap=True,
            investment_opportunities=True,
        ),
    )
    return calls, rows


def opportunity_row(**overrides) -> dict:
    row = {
        "lot_uid": 4211,
        "lot_number": "2 214 147",
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 9, 1),
        "area_m2": 2694.0,
        "site_thesis": "teardown",
        "investment_thesis": "residential",
        "site_thesis_rank": 4,
        "is_top_site_opportunity": True,
        "site_yield_on_cost_pct": 5.37,
        "redevelopment_npv_gain_cad": 3_630_000.0,
        "existing_year_built": 1955,
        "existing_num_storeys": 2,
        "hbu_floors": 6,
        "storey_headroom": 4,
        "existing_dominant_use_description": "Immeuble commercial",
        "is_heritage_sector": False,
        "has_piia_review": True,
        "demolition_review_required": False,
        "improvement_added_storeys": 1,
        "improvement_floor_m2": 270.0,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# The viewport read
# ---------------------------------------------------------------------------


def test_opportunities_join_the_lots_on_the_cadastral_number(captured):
    calls, _ = captured
    queries.opportunities_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql = calls[0][0]
    assert "o.lot_number   = l.lot_number" in sql
    assert "o.lot_uid" not in sql.split("WHERE")[0].split("ON")[1]
    assert "lot_investment_opportunities o" in sql


def test_opportunities_screen_on_a_thesis_and_the_shortlist(captured):
    calls, _ = captured
    queries.opportunities_in_bbox(
        (-73.7, 45.5, -73.6, 45.6), site_thesis="brownfield", top_only=True
    )
    sql, params = calls[0]
    assert "o.site_thesis <> 'none'" in sql
    assert params["site_thesis"] == "brownfield"
    assert params["top_only"] is True
    assert params["limit"] == queries.DEFAULT_FEATURE_LIMIT + 1


def test_opportunity_features_carry_what_the_tooltip_and_the_colour_read(captured):
    _, rows = captured
    rows.append(opportunity_row())
    features = queries.opportunities_in_bbox((-73.7, 45.5, -73.6, 45.6))
    props = features.features[0]["properties"]
    for key in ("site_thesis", "site_thesis_rank", "is_top_site_opportunity",
                "site_yield_on_cost_pct", "has_piia_review"):
        assert key in props
    assert features.layer == "opportunities"


# ---------------------------------------------------------------------------
# The tile
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_scalar(monkeypatch):
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        queries, "scalar", lambda sql, params=None: calls.append((sql, params)) or b""
    )
    return calls


def test_the_opportunities_tile_joins_on_the_cadastral_number(captured_scalar):
    queries.mvt_tile("opportunities", 13, 2400, 2900)
    sql = captured_scalar[0][0]
    assert "o.lot_number   = l.lot_number" in sql
    assert "o.site_thesis <> 'none'" in sql


def test_the_two_screens_reach_the_tile(captured_scalar):
    queries.mvt_tile(
        "opportunities", 13, 2400, 2900, site_thesis="teardown", top_only=True
    )
    params = captured_scalar[0][1]
    assert params["site_thesis"] == "teardown"
    assert params["top_only"] is True


def test_the_screens_default_to_every_thesis(captured_scalar):
    queries.mvt_tile("opportunities", 13, 2400, 2900)
    params = captured_scalar[0][1]
    assert params["site_thesis"] is None
    assert params["top_only"] is False


def test_the_layer_has_a_detail_zoom_and_no_aggregate():
    assert "opportunities" in queries.MVT_LAYER_NAMES
    assert "opportunities" in queries.MVT_DETAIL_ZOOM
    assert "opportunities" not in queries.AGGREGATE_LAYERS
    assert not queries.serves_aggregate("opportunities", 5)


def test_a_tile_url_names_a_thesis_the_table_can_hold():
    assert tiles._tile_arguments({"site_thesis": ["teardown"]}) == {
        "site_thesis": "teardown"
    }
    assert tiles._tile_arguments({"site_thesis": ["TearDown"]}) == {
        "site_thesis": "teardown"
    }
    # A value the table cannot hold draws every thesis, not none.
    assert tiles._tile_arguments({"site_thesis": ["renamed"]}) == {}
    assert tiles._tile_arguments({"top_only": ["1"]}) == {"top_only": True}


def test_the_partition_probe_knows_the_layer():
    schema, table, asset = queries.MAP_PARTITION_TABLES["opportunities"]
    assert schema == queries.GOLD_SCHEMA
    assert table == asset == "lot_investment_opportunities"


# ---------------------------------------------------------------------------
# The per-lot read and the rankings
# ---------------------------------------------------------------------------


def test_lot_opportunity_returns_none_without_the_table(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: queries.Capabilities(postgis=True)
    )
    assert queries.lot_opportunity(4211) is None


def test_lot_opportunity_resolves_the_uid_through_the_cadastre(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, investment_opportunities=True),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.lot_opportunity(4211)
    sql = seen[0]
    assert "l.lot_number   = o.lot_number" in sql
    assert "WHERE l.lot_uid = %(lot_uid)s" in sql
    for column in (
        "o.site_thesis", "o.site_thesis_rank", "o.site_yield_on_cost_pct",
        "o.demolition_cost_cad", "o.remediation_cost_cad",
        "o.improvement_floor_m2", "o.is_heritage_sector", "o.has_piia_review",
        "o.screen_assumptions", "o.investment_thesis",
    ):
        assert column in sql


def test_top_site_opportunities_orders_on_the_rank_the_dataplatform_gave(monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, investment_opportunities=True),
    )
    monkeypatch.setattr(
        queries, "query", lambda sql, params=None: seen.append((sql, params)) or []
    )
    queries.top_site_opportunities(site_thesis="brownfield", limit=5)
    sql, params = seen[0]
    assert "ORDER BY o.site_thesis_rank ASC" in sql
    assert "o.site_thesis_rank IS NOT NULL" in sql
    assert params["site_thesis"] == "brownfield"
    assert params["limit"] == 5


def test_the_rankings_need_the_table(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: queries.Capabilities(postgis=True)
    )
    assert queries.top_site_opportunities() == []
    assert queries.site_thesis_totals() == []


def test_the_capability_is_probed_and_named(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "query_one",
        lambda sql, params=None: seen.append(sql) or {"investment_opportunities": True},
    )
    try:
        queries.capabilities()
    except TypeError:
        pass  # the stub answers one column; the SQL is what is under test
    assert "lot_investment_opportunities" in seen[0]
    caps = queries.Capabilities(postgis=True, lots=True, buildings=True, features=True)
    assert f"{queries.GOLD_SCHEMA}.lot_investment_opportunities" in caps.missing()
    assert f"{queries.GOLD_SCHEMA}.lot_investment_opportunities" not in caps.missing(
        include_advisory=False
    )


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------


def test_every_thesis_has_a_colour_and_a_legend_row():
    rows = dict(basemap.opportunities_legend_rows())
    for thesis in queries.SITE_THESES:
        assert basemap.SITE_THESIS_COLORS[thesis] in rows
    # One row per thesis, one for the shortlist's edge, one for the good
    # candidate's green edge.
    assert len(rows) == len(queries.SITE_THESES) + 2


def test_the_shortlist_gets_the_heavier_edge():
    top = basemap._opportunity_style(
        {"properties": {"site_thesis": "teardown", "is_top_site_opportunity": True}}
    )
    rest = basemap._opportunity_style(
        {"properties": {"site_thesis": "teardown", "is_top_site_opportunity": False}}
    )
    assert top["fillColor"] == rest["fillColor"] == basemap.SITE_THESIS_COLORS["teardown"]
    assert top["weight"] > rest["weight"]


def test_a_thesis_this_map_has_no_colour_for_is_still_drawn():
    style = basemap._opportunity_style({"properties": {"site_thesis": "renamed"}})
    assert style["fillColor"] == basemap._SITE_THESIS_OTHER_COLOR


def test_the_tile_style_reads_the_same_colours():
    js = basemap._style_js("opportunities")
    assert "site_thesis" in js
    assert basemap.SITE_THESIS_COLORS["brownfield"] in js
    assert "is_top_site_opportunity" in js


def test_the_layer_sits_with_the_shading_and_is_gated_at_its_detail_zoom():
    order = basemap.TILE_LAYER_ORDER
    assert order.index("opportunities") == order.index("capacity") + 1
    assert order == queries.MVT_LAYER_NAMES
    assert basemap.TILE_LAYER_MIN_ZOOM["opportunities"] == queries.MVT_DETAIL_ZOOM["opportunities"]
    assert basemap.TILE_LAYER_NAMES["opportunities"] == "Opportunities"
    assert basemap.DEFAULT_LAYERS["opportunities"] is False


def test_decorate_labels_the_site_thesis_the_way_the_tooltip_does():
    feature_set = queries._as_feature_set(
        [opportunity_row()], layer="opportunities", id_key="lot_number", limit=10
    )
    basemap.decorate(feature_set, "opportunities")
    props = feature_set.features[0]["properties"]
    assert props["site_label"] == "teardown · rank 4 (shortlist)"
    assert props["yield_label"].startswith("5.4% on cost · +$3,630,000 vs holding")
    assert props["standing_label"] == "built 1955 · 2 of 6 storeys · Immeuble commercial"
    assert props["flags_label"] == "PIIA review"


def test_decorate_says_when_a_filed_lot_does_not_pay():
    feature_set = queries._as_feature_set(
        [opportunity_row(site_thesis_rank=None, is_top_site_opportunity=False,
                         site_yield_on_cost_pct=None)],
        layer="opportunities", id_key="lot_number", limit=10,
    )
    basemap.decorate(feature_set, "opportunities")
    props = feature_set.features[0]["properties"]
    assert props["site_label"] == "teardown · does not pay"
    assert props["yield_label"] == "—"


def test_decorate_says_the_addition_on_an_improvement():
    feature_set = queries._as_feature_set(
        [opportunity_row(site_thesis="improvement", site_yield_on_cost_pct=6.2,
                         is_heritage_sector=True, has_piia_review=False)],
        layer="opportunities", id_key="lot_number", limit=10,
    )
    basemap.decorate(feature_set, "opportunities")
    props = feature_set.features[0]["properties"]
    assert props["yield_label"] == "6.2% on cost · +270 m², 1 storey"
    assert props["flags_label"] == "heritage sector"


def test_build_map_draws_the_layer_from_geojson():
    feature_set = queries._as_feature_set(
        [opportunity_row()], layer="opportunities", id_key="lot_number", limit=10
    )
    basemap.decorate(feature_set, "opportunities")
    fmap = basemap.build_map(opportunities=feature_set)
    names = [
        child.layer_name for child in fmap._children.values()
        if hasattr(child, "layer_name")
    ]
    assert any(name.startswith("Opportunities (1)") for name in names)


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


def _caps(**flags):
    return queries.Capabilities(**{"postgis": True, "pgvector": True, **flags})


def test_the_top_sites_tool_names_the_missing_table(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps())
    with pytest.raises(ToolException, match="investment_opportunities"):
        parcel_tools.top_site_opportunities.invoke({"site_thesis": "teardown"})


def test_the_top_sites_tool_refuses_a_thesis_the_table_cannot_hold(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    with pytest.raises(ToolException, match="Unknown site thesis"):
        parcel_tools.top_site_opportunities.invoke({"site_thesis": "greenfield"})


def test_the_top_sites_tool_reads_as_a_shortlist(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        queries, "top_site_opportunities",
        lambda **kwargs: [
            {
                "lot_number": "2 214 147", "lot_area_m2": 2694.0,
                "site_thesis": "teardown", "investment_thesis": "residential",
                "site_thesis_rank": 1, "num_ranked_in_site_thesis": 46,
                "site_yield_on_cost_pct": 5.37,
                "redevelopment_npv_gain_cad": 3_630_000.0,
                "existing_year_built": 1955, "existing_num_storeys": 2,
                "hbu_floors": 6,
                "existing_dominant_use_description": "Immeuble commercial",
                "grid_zone": "C04-083", "has_piia_review": True,
            }
        ],
    )
    text = parcel_tools.top_site_opportunities.invoke({"site_thesis": "teardown"})
    assert "Lot 2 214 147" in text
    assert "teardown rank 1 of 46" in text
    assert "5.4% on cost" in text
    assert "+$3,630,000 vs holding" in text
    assert "built 1955, 2/6 storeys" in text
    assert "flags: PIIA" in text


def test_lot_efficiency_says_why_the_site_is_acquirable(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(lots=True, redevelopment_gap=True, investment_opportunities=True),
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(
        queries, "lot_capacity",
        lambda uid, **kw: {
            "hbu_status": "solved", "existing_floor_area_m2": 270.0,
            "hbu_floor_area_m2": 2700.0, "used_pct": 10.0, "lot_area_m2": 2694.0,
            "existing_num_assessment_units": 1,
        },
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: {
            "site_thesis": "brownfield", "site_thesis_rank": 2,
            "num_ranked_in_site_thesis": 56, "is_top_site_opportunity": True,
            "site_yield_on_cost_pct": 4.8, "demolition_cost_cad": 67_500.0,
            "site_assessment_cost_cad": 12_000.0, "remediation_cost_cad": 404_100.0,
            "site_total_project_cost_cad": 5_000_000.0,
            "existing_year_built": 1960, "existing_num_storeys": 1, "hbu_floors": 3,
            "is_heritage_sector": False, "has_piia_review": False,
            "demolition_review_required": False,
        },
    )
    text = parcel_tools.lot_efficiency.invoke({"lot_number": "2 214 147"})
    assert "Site thesis: brownfield" in text
    assert "ranked 2 of 56 brownfield sites and on that thesis's shortlist" in text
    assert "remediation $404,100" in text
    assert "4.8% on $5,000,000 all in" in text


def test_the_sentence_for_a_lot_with_no_thesis_still_names_the_flags():
    text = parcel_tools._site_thesis_sentence(
        {"site_thesis": "none", "is_heritage_sector": True, "heritage_sector": "Oui"}
    )
    assert text.startswith("Site thesis: none")
    assert "secteur d'intérêt patrimonial" in text


def test_the_opportunities_layer_can_be_switched_from_the_chat(monkeypatch):
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(investment_opportunities=False))
    with pytest.raises(ToolException, match="not loaded"):
        map_tools.set_map_layers.invoke({"opportunities": True})
    monkeypatch.setattr(queries, "capabilities", lambda: _caps(investment_opportunities=True))
    assert "opportunities on" in map_tools.set_map_layers.invoke({"opportunities": True})


# ---------------------------------------------------------------------------
# The three futures
# ---------------------------------------------------------------------------


def test_lot_opportunity_carries_the_three_futures_for_both_panes(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, investment_opportunities=True),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.lot_opportunity(4211)
    sql = seen[0]
    for column in (
        "o.owner_hold_value_cad", "o.owner_enhance_value_cad", "o.owner_rebuild_value_cad",
        "o.owner_best_future", "o.acquisition_cost_cad", "o.buyer_npv_rebuild_cad",
        "o.buyer_yield_enhance_pct", "o.residual_price_rebuild_cad", "o.buyer_best_future",
        "o.enhance_added_storeys", "o.enhance_disruption_cad", "o.enhance_assumptions",
        "o.site_costs_cad",
    ):
        assert column in sql


def test_futures_totals_needs_the_table_and_groups_by_future(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: queries.Capabilities(postgis=True)
    )
    assert queries.futures_totals() == []
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, investment_opportunities=True),
    )
    monkeypatch.setattr(queries, "query", lambda sql, params=None: seen.append(sql) or [])
    queries.futures_totals(neighborhood="VSMPE")
    assert "GROUP BY f.future" in seen[0]
    assert "o.owner_best_future = f.future" in seen[0]


def _futures_row(**overrides) -> dict:
    row = {
        "lot_number": "2 214 147", "lot_area_m2": 2694.0, "hbu_status": "solved",
        "owner_hold_value_cad": 400_000.0, "owner_enhance_value_cad": 520_000.0,
        "owner_rebuild_value_cad": 864_000.0, "owner_gain_enhance_cad": 120_000.0,
        "owner_gain_rebuild_cad": 464_000.0, "owner_best_future": "rebuild",
        "acquisition_cost_cad": 450_000.0, "buyer_npv_hold_cad": -50_000.0,
        "buyer_npv_enhance_cad": 70_000.0, "buyer_npv_rebuild_cad": 414_000.0,
        "buyer_yield_hold_pct": 4.4, "buyer_yield_enhance_pct": 5.5,
        "buyer_yield_rebuild_pct": 6.1, "residual_price_enhance_cad": 520_000.0,
        "residual_price_rebuild_cad": 864_000.0, "buyer_best_future": "rebuild",
        "enhance_solved": True, "enhance_status": "OPTIMAL", "enhance_added_storeys": 1,
        "enhance_added_floor_area_m2": 200.0, "enhance_added_dwellings": 3,
        "enhance_capital_cost_cad": 450_000.0, "hbu_total_capital_cost_cad": 1_500_000.0,
        "site_costs_cad": 36_000.0, "hbu_num_dwellings": 10, "hbu_floor_area_m2": 1000.0,
    }
    row.update(overrides)
    return row


def test_the_futures_tool_prices_every_future_after_the_purchase(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(queries, "lot_opportunity", lambda uid, **kw: _futures_row())
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "Price to pay: $450,000" in text
    assert "keep: NPV after purchase -$50,000, 4.4% on all-in cost, most you could pay $400,000" in text
    assert "enhance: NPV after purchase +$70,000, 5.5% on all-in cost, most you could pay $520,000" in text
    assert "rebuild: NPV after purchase +$414,000" in text
    assert "costing $1,536,000 to build on top of the price" in text
    assert "<- best" in text.split("tear down and rebuild")[1]
    # The land is inside every line, and the tool says so rather than leaving
    # the reader to assume it the way the owner's version could.
    assert "the land is paid for at the price above in every line" in text


def test_the_futures_tool_states_the_room_over_the_asking_price(monkeypatch):
    """The broker's number: the ceiling the winning future puts over the price."""
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(queries, "lot_opportunity", lambda uid, **kw: _futures_row())
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    # rebuild wins and could bear $864,000 against $450,000 asked.
    assert "Room between the price and what the best future could bear: +$414,000" in text

    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(acquisition_cost_cad=1_000_000.0),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "Room between the price and what the best future could bear: -$136,000" in text


def test_the_futures_tool_names_the_thesis_that_says_who_to_call(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(
            investment_thesis="mixed_use", thesis_rank=3, num_ranked_in_thesis=41
        ),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "Would build: mixed use, ranked 3 of 41 such sites in the borough." in text


def test_the_futures_tool_says_when_a_future_is_not_priced(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(enhance_solved=False, enhance_status="not_underbuilt"),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "enhance: not priced (not_underbuilt)" in text


def test_the_futures_tool_calls_an_enhancement_that_adds_nothing_keeping(monkeypatch):
    """Solved, and the answer is the standing building: no cost, no time, no
    dwellings - the tool says so instead of pricing a second keep."""
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "3 791 023"},
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(
            enhance_added_storeys=0, enhance_added_floor_area_m2=0.0,
            enhance_added_dwellings=0, enhance_capital_cost_cad=0.0,
            enhance_footprint_m2=74.1, existing_footprint_m2=74.1,
            owner_enhance_value_cad=400_000.0, buyer_npv_enhance_cad=-50_000.0,
            residual_price_enhance_cad=400_000.0,
        ),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "3 791 023"})
    enhance_line = next(line for line in text.splitlines() if line.startswith("- enhance"))
    assert "nothing to add" in enhance_line
    assert "keeping" in enhance_line
    for misleading in ("costing", "m² added", "new dwellings", "NPV after purchase", "m² annex"):
        assert misleading not in enhance_line, misleading


def test_the_futures_tool_states_the_annex_only_where_the_plate_grows(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(enhance_footprint_m2=140.0, existing_footprint_m2=100.0),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "(1 storey and a 40 m² annex, 200 m² added, 3 new dwellings)" in text
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(enhance_footprint_m2=100.0, existing_footprint_m2=100.0),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "(1 storey, 200 m² added, 3 new dwellings)" in text


def test_the_thesis_sentence_measures_the_annex_off_the_solve():
    base = {
        "site_thesis": "improvement", "site_yield_on_cost_pct": 6.2,
        "improvement_floor_m2": 200.0, "improvement_added_storeys": 1,
        "improvement_noi_cad": 24_000.0, "improvement_cost_cad": 400_000.0,
        "enhance_solved": True,
    }
    grown = parcel_tools._site_thesis_sentence(
        {**base, "enhance_footprint_m2": 140.0, "existing_footprint_m2": 100.0}
    )
    assert "(1 storey on the standing plate and a 40 m² annex beside it)" in grown
    on_plate = parcel_tools._site_thesis_sentence(
        {**base, "enhance_footprint_m2": 100.0, "existing_footprint_m2": 100.0}
    )
    assert "(1 storey on the standing plate)" in on_plate
    assert "annex beside it" not in on_plate
    # The closed-form estimate has no plate of its own, so the shape stays open.
    estimated = parcel_tools._site_thesis_sentence({**base, "enhance_solved": False})
    assert "(1 storey on the standing plate or an annex beside it)" in estimated


def test_the_futures_tool_refuses_a_lot_the_roll_never_priced(monkeypatch):
    """No price for the ground is no deal - and the tool says where to look."""
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(acquisition_cost_cad=None),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "no price to put on the ground" in text
    assert "ask for the lot's programme instead" in text


# ---------------------------------------------------------------------------
# The returns
# ---------------------------------------------------------------------------


def test_the_layer_and_the_tile_take_the_good_candidate_screen(captured, captured_scalar):
    calls, _ = captured
    queries.opportunities_in_bbox((-73.7, 45.5, -73.6, 45.6), good_only=True)
    sql, params = calls[0]
    assert "o.is_good_candidate" in sql
    assert params["good_only"] is True
    queries.mvt_tile("opportunities", 13, 2400, 2900, good_only=True)
    assert captured_scalar[0][1]["good_only"] is True
    assert tiles._tile_arguments({"good_only": ["1"]}) == {"good_only": True}


def test_lot_opportunity_and_the_rankings_carry_the_returns(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, investment_opportunities=True),
    )
    monkeypatch.setattr(queries, "query_one", lambda sql, params=None: seen.append(sql) or None)
    monkeypatch.setattr(queries, "query", lambda sql, params=None: seen.append(sql) or [])
    queries.lot_opportunity(4211)
    queries.top_site_opportunities(limit=3)
    queries.site_thesis_totals()
    for column in ("o.site_irr_pct", "o.site_all_in_yield_on_cost_pct", "o.is_good_candidate",
                   "o.buyer_irr_rebuild_pct", "o.owner_irr_enhance_pct", "o.rebuild_soft_cost_cad"):
        assert column in seen[0], column
    assert "o.site_irr_pct" in seen[1] and "o.is_good_candidate" in seen[1]
    assert "num_good_candidates" in seen[2] and "median_site_irr_pct" in seen[2]


def test_a_good_candidate_gets_the_green_edge_and_the_legend_says_so():
    good = basemap._opportunity_style(
        {"properties": {"site_thesis": "teardown", "is_good_candidate": True}}
    )
    top = basemap._opportunity_style(
        {"properties": {"site_thesis": "teardown", "is_top_site_opportunity": True}}
    )
    assert good["color"] == basemap.GOOD_CANDIDATE_EDGE
    assert good["weight"] > top["weight"]
    assert good["fillColor"] == basemap.SITE_THESIS_COLORS["teardown"]
    assert any(colour == basemap.GOOD_CANDIDATE_EDGE for colour, _ in basemap.opportunities_legend_rows())
    assert "is_good_candidate" in basemap._style_js("opportunities")


def test_decorate_labels_the_returns():
    feature_set = queries._as_feature_set(
        [opportunity_row(site_irr_pct=14.34, site_all_in_yield_on_cost_pct=6.1,
                         site_yoc_spread_bps=160.0, is_good_candidate=True)],
        layer="opportunities", id_key="lot_number", limit=10,
    )
    basemap.decorate(feature_set, "opportunities")
    props = feature_set.features[0]["properties"]
    assert props["returns_label"] == "IRR 14.3% · yield 6.1% on cost (+160 bps vs cap) · ✔ good candidate"
    feature_set = queries._as_feature_set(
        [opportunity_row()], layer="opportunities", id_key="lot_number", limit=10
    )
    basemap.decorate(feature_set, "opportunities")
    assert feature_set.features[0]["properties"]["returns_label"] == "—"


def test_the_top_sites_tool_reads_the_returns(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        queries, "top_site_opportunities",
        lambda **kwargs: [
            {
                "lot_number": "2 214 147", "lot_area_m2": 2694.0,
                "site_thesis": "teardown", "investment_thesis": "residential",
                "site_thesis_rank": 1, "num_ranked_in_site_thesis": 46,
                "site_yield_on_cost_pct": 5.37, "site_irr_pct": 13.4,
                "site_all_in_yield_on_cost_pct": 6.2, "site_yoc_spread_bps": 170.0,
                "is_good_candidate": True,
                "redevelopment_npv_gain_cad": 3_630_000.0,
                "grid_zone": "C04-083",
            }
        ],
    )
    text = parcel_tools.top_site_opportunities.invoke({"site_thesis": "teardown"})
    assert "IRR 13.4%, 6.2% on all-in cost (+170 bps vs cap)" in text
    assert "GOOD CANDIDATE" in text


def test_lot_efficiency_reports_the_returns_and_the_screen(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(lots=True, redevelopment_gap=True, investment_opportunities=True),
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(
        queries, "lot_capacity",
        lambda uid, **kw: {
            "hbu_status": "solved", "existing_floor_area_m2": 270.0,
            "hbu_floor_area_m2": 2700.0, "used_pct": 10.0, "lot_area_m2": 2694.0,
            "existing_num_assessment_units": 1,
        },
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: {
            "site_thesis": "teardown", "site_thesis_rank": 2, "num_ranked_in_site_thesis": 56,
            "is_top_site_opportunity": True, "site_yield_on_cost_pct": 4.8,
            "demolition_cost_cad": 67_500.0, "site_total_project_cost_cad": 5_000_000.0,
            "site_irr_pct": 9.5, "site_all_in_yield_on_cost_pct": 5.1,
            "market_cap_rate_pct": 4.5, "site_yoc_spread_bps": 60.0,
            "owner_site_irr_pct": 11.2, "clears_cap_rate": False, "clears_hurdle": False,
            "is_good_candidate": False,
        },
    )
    text = parcel_tools.lot_efficiency.invoke({"lot_number": "2 214 147"})
    assert "yield on all-in cost 5.1% against a 4.5% market cap rate (+60 bps)" in text
    assert "buyer's unlevered IRR 9.5%" in text
    assert "owner's IRR on the increment 11.2%" in text
    assert "not a good candidate: misses the cap rate spread, misses the IRR hurdle" in text


def test_a_lot_clearing_one_bar_is_reported_as_a_good_candidate_and_says_which(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: _caps(lots=True, redevelopment_gap=True, investment_opportunities=True),
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4212, "lot_number": "2 784 705"},
    )
    monkeypatch.setattr(
        queries, "lot_capacity",
        lambda uid, **kw: {
            "hbu_status": "solved", "existing_floor_area_m2": 0.0,
            "hbu_floor_area_m2": 900.0, "used_pct": 0.0, "lot_area_m2": 400.0,
            "existing_num_assessment_units": 0,
        },
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: {
            "site_thesis": "infill", "site_thesis_rank": 1, "num_ranked_in_site_thesis": 400,
            "is_top_site_opportunity": True, "site_yield_on_cost_pct": 5.4,
            "site_irr_pct": 5.53, "site_all_in_yield_on_cost_pct": 5.37,
            "market_cap_rate_pct": 4.5, "site_yoc_spread_bps": 87.0,
            "clears_cap_rate": False, "clears_hurdle": True, "is_good_candidate": True,
        },
    )
    text = parcel_tools.lot_efficiency.invoke({"lot_number": "2 784 705"})
    assert "a good candidate: misses the cap rate spread, clears the IRR hurdle and pays against holding" in text


def test_the_futures_tool_states_the_irr(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: _caps(investment_opportunities=True)
    )
    monkeypatch.setattr(
        parcel_tools, "_resolve_lot_uid",
        lambda number: {"lot_uid": 4211, "lot_number": "2 214 147"},
    )
    monkeypatch.setattr(
        queries, "lot_opportunity",
        lambda uid, **kw: _futures_row(
            buyer_irr_rebuild_pct=13.1, buyer_yoc_rebuild_pct=6.0,
            owner_irr_rebuild_pct=15.7, owner_irr_enhance_pct=8.2,
        ),
    )
    text = parcel_tools.lot_futures.invoke({"lot_number": "2 214 147"})
    assert "rebuild: NPV after purchase +$414,000, IRR 13.1%, 6.0% on all-in cost" in text
    # The owner's chair is gone from the tool, but the owner's IRR on the
    # increment is the one owner number a buyer still wants beside their own.
    assert "(to the owner, 15.7% on the increment)" in text
    assert "(to the owner, 8.2% on the increment)" in text
    # A row the proforma never reached keeps the old shape of the line.
    assert "keep: NPV after purchase -$50,000, 4.4% on all-in cost" in text
