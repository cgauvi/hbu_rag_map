"""The utilisation layer, the per-lot comparison, and the borough totals.

The interesting assertions here are about *nulls*. ``gold.lot_redevelopment_gap``
publishes a gap column that is NULL wherever either side is, and the side that
is missing is almost always the existing one — a lot the assessment roll never
reached. Summing that column would silently drop the vacant parcels, which are
the ones with the most headroom, so `queries` computes its own headroom instead.
These tests pin that decision, because it is invisible in the output and wrong
in a way that looks plausible.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.utils import basemap, queries


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
            highest_best_use=True, massing=True,
        ),
    )
    return calls, rows


def capacity_row(**overrides) -> dict:
    row = {
        "lot_uid": 4211,
        "lot_number": "2 170 935",
        "neighborhood": "VSMPE",
        "scrape_date": date(2026, 8, 27),
        "area_m2": 300.0,
        "hbu_status": "solved",
        "has_assessment": True,
        "is_underbuilt": True,
        "existing_floor_area_m2": 220.0,
        "hbu_floor_area_m2": 880.0,
        "floor_area_gap_m2": 660.0,
        "existing_num_dwellings": 2,
        "hbu_num_dwellings": 11,
        "dwelling_gap": 9,
        "used_pct": 25.0,
        "residential_headroom_m2": 600.0,
        "commercial_headroom_m2": 60.0,
        "industrial_headroom_m2": 0.0,
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# The headroom expression
# ---------------------------------------------------------------------------


def test_headroom_reads_a_missing_existing_floor_as_zero():
    """A lot the roll never reached must contribute its whole envelope.

    Using the published ``*_floor_area_gap_m2`` would return NULL for exactly
    these lots and drop them from every sum.
    """
    sql = queries._headroom_m2("residential")
    assert "COALESCE(g.existing_residential_floor_area_m2, 0)" in sql
    assert "hbu_residential_floor_area_m2" in sql


def test_headroom_clamps_so_an_overbuilt_lot_cannot_cancel_a_neighbour():
    assert queries._headroom_m2("commercial").startswith("GREATEST(")
    assert queries._headroom_m2("commercial").rstrip().endswith(", 0)")


def test_used_pct_has_no_zero_denominator():
    """An unsolved lot has no permitted floor, and must not divide by it."""
    assert "NULLIF(g.hbu_floor_area_m2, 0)" in queries._USED_PCT


# ---------------------------------------------------------------------------
# The viewport layer
# ---------------------------------------------------------------------------


def test_capacity_joins_lots_to_the_gap_on_the_whole_partition_key(captured):
    """lot_uid is a bigserial a reload mints again — joining on it alone would
    cross two snapshots and shade this year's parcels with last year's answer."""
    calls, _ = captured
    queries.capacity_in_bbox((-73.7, 45.5, -73.6, 45.6))
    sql, _ = calls[0]
    assert f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap" in sql
    assert "g.lot_uid      = l.lot_uid" in sql
    assert "g.neighborhood = l.neighborhood" in sql
    assert "g.scrape_date  = l.scrape_date" in sql


def test_capacity_asks_for_one_row_past_the_limit(captured):
    calls, _ = captured
    queries.capacity_in_bbox((-73.7, 45.5, -73.6, 45.6), limit=10)
    _sql, params = calls[0]
    assert params["limit"] == 11


def test_capacity_only_underbuilt_narrows_to_the_same_screen_as_massing(captured):
    calls, _ = captured
    queries.capacity_in_bbox((-73.7, 45.5, -73.6, 45.6), only_underbuilt=True)
    sql, params = calls[0]
    assert params["only_underbuilt"] is True
    assert "g.is_underbuilt" in sql


def test_capacity_features_carry_what_the_tooltip_reads(captured):
    _calls, rows = captured
    rows.append(capacity_row())
    found = queries.capacity_in_bbox((-73.7, 45.5, -73.6, 45.6))
    props = found.features[0]["properties"]
    assert found.layer == "capacity"
    assert props["id"] == "2 170 935"
    assert props["used_pct"] == 25.0
    assert props["residential_headroom_m2"] == 600.0


# ---------------------------------------------------------------------------
# Styling — colour carries a finding, so the bands are worth pinning
# ---------------------------------------------------------------------------


def test_an_overbuilt_lot_is_not_drawn_as_the_efficient_end_of_the_ramp():
    """A legal non-conformity is not "maximally efficient" — reading it as the
    top of the ramp would invert the map."""
    over = basemap._capacity_style({"properties": {"used_pct": 140.0}})
    full = basemap._capacity_style({"properties": {"used_pct": 98.0}})
    assert over["fillColor"] == basemap._CAPACITY_OVER_COLOR
    assert over["fillColor"] != full["fillColor"]


def test_an_unsolved_lot_is_grey_rather_than_empty():
    style = basemap._capacity_style({"properties": {"used_pct": None}})
    assert style["fillColor"] == basemap._CAPACITY_NONE_COLOR


def test_emptier_lots_are_darker_so_the_ramp_reads_without_a_legend():
    empty = basemap._capacity_style({"properties": {"used_pct": 5.0}})
    nearly_full = basemap._capacity_style({"properties": {"used_pct": 90.0}})
    assert empty["fillColor"] != nearly_full["fillColor"]
    assert empty["fillColor"] == basemap._CAPACITY_BANDS[0][1]


def test_the_legend_is_generated_from_the_same_bands_the_style_uses():
    rows = basemap.capacity_legend_rows()
    colours = [c for c, _ in rows]
    for _upper, colour, _label in basemap._CAPACITY_BANDS:
        assert colour in colours
    assert basemap._CAPACITY_OVER_COLOR in colours
    assert basemap._CAPACITY_NONE_COLOR in colours


# ---------------------------------------------------------------------------
# Tooltip labels
# ---------------------------------------------------------------------------


def test_an_unsolved_lot_says_why_rather_than_showing_a_blank():
    found = queries.FeatureSet(
        features=[{
            "properties": {
                "used_pct": None,
                "hbu_status": "no_residential_column",
                "area_m2": 300.0,
            }
        }],
        layer="capacity",
    )
    basemap.decorate(found, "capacity")
    assert "résidentiel" in found.features[0]["properties"]["used_label"]


def test_the_renamed_status_has_a_label_of_its_own():
    """no_candidate_column replaced no_residential_column when the solver
    started pricing commerce and industry; both spellings must label."""
    found = queries.FeatureSet(
        features=[{
            "properties": {
                "used_pct": None,
                "hbu_status": "no_candidate_column",
                "area_m2": 300.0,
            }
        }],
        layer="capacity",
    )
    basemap.decorate(found, "capacity")
    assert "valorisable" in found.features[0]["properties"]["used_label"]


def test_headroom_label_names_both_units_and_the_dwellings():
    found = queries.FeatureSet(
        features=[{
            "properties": {
                "used_pct": 25.0,
                "hbu_status": "solved",
                "area_m2": 300.0,
                "existing_floor_area_m2": 220.0,
                "hbu_floor_area_m2": 880.0,
                "residential_headroom_m2": 600.0,
                "commercial_headroom_m2": 60.0,
                "industrial_headroom_m2": 0.0,
                "dwelling_gap": 9,
            }
        }],
        layer="capacity",
    )
    basemap.decorate(found, "capacity")
    label = found.features[0]["properties"]["headroom_label"]
    assert "660 m²" in label
    assert "pi²" in label
    assert "9 logements" in label


# ---------------------------------------------------------------------------
# One lot, and the borough
# ---------------------------------------------------------------------------


def test_lot_capacity_returns_none_when_the_gap_table_is_absent(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: queries.Capabilities(postgis=True)
    )
    assert queries.lot_capacity(4211) is None


def test_lot_capacity_joins_the_massing_only_for_the_fit_caveat(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(
            postgis=True, redevelopment_gap=True, highest_best_use=True, massing=True
        ),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.lot_capacity(4211)
    assert "footprint_fit_pct" in seen[0]
    assert f"{queries.GOLD_SCHEMA}.lot_highest_best_use" in seen[0]


def test_lot_capacity_still_answers_without_the_massing_table(monkeypatch):
    """A missing massing table must cost the caveat, not the answer."""
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(
            postgis=True, redevelopment_gap=True, highest_best_use=True, massing=False
        ),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.lot_capacity(4211)
    assert "footprint_fit_pct" not in seen[0]
    assert f"{queries.GOLD_SCHEMA}.lot_redevelopment_gap" in seen[0]


def test_capacity_totals_reports_the_counts_the_headline_rests_on(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, redevelopment_gap=True),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.capacity_totals(neighborhood="VSMPE")
    sql = seen[0]
    for column in (
        "num_solved", "num_underbuilt", "num_without_assessment", "num_over_built",
    ):
        assert column in sql


def test_capacity_totals_keeps_the_signed_net_beside_the_clamped_headline(monkeypatch):
    """Two different questions, and the row must carry both rather than one.

    The headline sums clamped headroom; the net sums the published signed gap.
    A borough dense enough to be net-negative is a real finding, and reporting
    only the clamped total would hide it.
    """
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, redevelopment_gap=True),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.capacity_totals()
    sql = seen[0]
    assert "sum(g.floor_area_gap_m2)            AS net_floor_area_gap_m2" in sql
    assert "AS total_headroom_m2" in sql


def test_capacity_totals_sums_the_positive_npv_gain_with_its_count(monkeypatch):
    """The developer's verdict travels with the room, and only where it is
    positive — a lot better kept as it stands contributes nothing rather
    than cancelling a neighbour's gain."""
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, redevelopment_gap=True),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.capacity_totals()
    sql = seen[0]
    assert "GREATEST(g.redevelopment_npv_gain_cad, 0)" in sql
    assert "num_npv_gain_positive" in sql


def test_top_npv_gain_lots_needs_the_gap_table(monkeypatch):
    monkeypatch.setattr(
        queries, "capabilities", lambda: queries.Capabilities(postgis=True)
    )
    assert queries.top_npv_gain_lots() == []


def test_top_npv_gain_lots_ranks_on_the_verdict_and_names_the_use(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(postgis=True, redevelopment_gap=True),
    )
    monkeypatch.setattr(
        queries, "query", lambda sql, params=None: seen.append(sql) or []
    )
    queries.top_npv_gain_lots(neighborhood="VSMPE", limit=5)
    sql = seen[0]
    assert "ORDER BY g.redevelopment_npv_gain_cad DESC" in sql
    assert "hbu_dominant_use" in sql
    assert "g.redevelopment_npv_gain_cad > 0" in sql


def test_lot_capacity_carries_the_developer_economics(monkeypatch):
    """The pane cannot say what the choice was worth without the npv trio and
    the one-word use, so the hbu join must select them."""
    seen: list[str] = []
    monkeypatch.setattr(
        queries, "capabilities",
        lambda: queries.Capabilities(
            postgis=True, redevelopment_gap=True, highest_best_use=True
        ),
    )
    monkeypatch.setattr(
        queries, "query_one", lambda sql, params=None: seen.append(sql) or None
    )
    queries.lot_capacity(4211)
    sql = seen[0]
    for column in (
        "h.npv_cad",
        "h.present_value_cad",
        "h.hbu_dominant_use",
        "g.redevelopment_npv_gain_cad",
        "g.existing_present_value_cad",
    ):
        assert column in sql
