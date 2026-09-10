"""The two blocks that compare what stands against what is proposed.

The Lot pane's today-against-proposal table and the Deal pane's *what each
future builds* columns are both pure functions over one gold row, split from
their renderers so they can be read without a browser. What they get wrong is
never a crash: it is a zero printed where the source said nothing, a class of
floor reported as unknown on a lot whose total is stated, or a proposed side
that invents a count no table holds. Every test here pins one of those.

Unit tests: no socket, no `AppTest`. See the ``app_defs`` fixture in
conftest for how the script's definitions are reached.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def gap_row() -> dict:
    """One row shaped like `queries.lot_capacity` returns, on a solved lot.

    A duplex over a shop under a proposal of housing over commerce, so every
    row of the table has something on both sides.
    """
    return {
        "hbu_status": "solved",
        "existing_num_storeys": 2,
        "existing_num_dwellings": 2,
        "existing_num_assessment_units": 3,
        "existing_floor_area_m2": 400.0,
        "existing_residential_floor_area_m2": 300.0,
        "existing_commercial_floor_area_m2": 100.0,
        "existing_industrial_floor_area_m2": 0.0,
        "existing_dominant_income_class": "residential",
        "existing_dominant_use_description": "Logement",
        "floors": 5,
        "hbu_num_dwellings": 14,
        "hbu_floor_area_m2": 1_500.0,
        "hbu_residential_floor_area_m2": 1_200.0,
        "hbu_commercial_floor_area_m2": 200.0,
        "hbu_industrial_floor_area_m2": 100.0,
        "hbu_dominant_use": "residential",
        "footprint_m2": 300.0,
    }


@pytest.fixture
def roll_row() -> dict:
    """One row shaped like `queries.lot_roll_units` returns."""
    return {
        "roll_loaded": True,
        "num_units": 3,
        "num_residential_units": 2,
        "num_nonresidential_units": 1,
        "nonresidential_floor_area_m2": 100.0,
    }


def _by_measure(rows: list[dict]) -> dict[str, dict]:
    return {row["Measure"]: row for row in rows}


class TestUseSides:
    def test_storeys_compare_the_roll_against_the_solver(self, app_defs, gap_row):
        rows = _by_measure(app_defs._use_sides(gap_row, None))
        assert rows["Storeys"]["Today"] == "2"
        assert rows["Storeys"]["Proposed"] == "5"

    def test_an_unsolved_lot_proposes_no_storeys(self, app_defs, gap_row):
        # ``floors`` can survive on a row the solver did not solve - a
        # partially re-materialized partition - and printing it would put a
        # building on a lot the next row says has no programme.
        gap_row["hbu_status"] = "infeasible"
        rows = _by_measure(app_defs._use_sides(gap_row, None))
        assert rows["Storeys"]["Proposed"] == "—"
        assert rows["Use"]["Proposed"] == "no programme"

    def test_the_two_floor_lines_sum_to_the_total(self, app_defs, gap_row):
        rows = _by_measure(app_defs._use_sides(gap_row, None))
        assert rows["Floor area"]["Today"] == "400 m²"
        assert rows["— residential floor"]["Today"] == "300 m²"
        assert rows["— non-residential floor"]["Today"] == "100 m²"
        assert rows["Floor area"]["Proposed"] == "1,500 m²"
        assert rows["— residential floor"]["Proposed"] == "1,200 m²"
        # Commerce and industry together: they are one line to a tenant.
        assert rows["— non-residential floor"]["Proposed"] == "300 m²"

    def test_a_class_the_row_omits_is_nothing_where_the_total_is_stated(
        self, app_defs, gap_row
    ):
        # The gold total is *built* by summing the three classes, so a lot
        # whose residential floor accounts for all of it has no commerce -
        # not an unknown quantity of it. Reported as a dash this read as the
        # roll declining to say, beside a proposed side printing 0 m².
        gap_row["existing_commercial_floor_area_m2"] = None
        gap_row["existing_industrial_floor_area_m2"] = None
        gap_row["existing_residential_floor_area_m2"] = 400.0
        rows = _by_measure(app_defs._use_sides(gap_row, None))
        assert rows["— non-residential floor"]["Today"] == "0 m²"

    def test_a_missing_total_leaves_every_class_unknown(self, app_defs, gap_row):
        # The roll reached no unit here at all, so there is no total to
        # decompose and no class under it. The rule above only turns a missing
        # class into a zero where the *total* was stated.
        gap_row["existing_num_assessment_units"] = 0
        gap_row["existing_floor_area_m2"] = None
        gap_row["existing_residential_floor_area_m2"] = None
        gap_row["existing_commercial_floor_area_m2"] = None
        gap_row["existing_industrial_floor_area_m2"] = None
        rows = _by_measure(app_defs._use_sides(gap_row, None))
        assert rows["Floor area"]["Today"] == "—"
        assert rows["— residential floor"]["Today"] == "—"
        assert rows["— non-residential floor"]["Today"] == "—"

    def test_a_roll_that_states_no_floor_says_so_on_every_floor_line(
        self, app_defs, gap_row
    ):
        # A unit on the lot and no superficie d'etages: unknown, not zero,
        # and the qualification belongs on the split lines as much as on the
        # total, since all three are read off the same silent unit.
        gap_row["existing_floor_area_m2"] = None
        gap_row["existing_num_assessment_units"] = 3
        rows = _by_measure(app_defs._use_sides(gap_row, None))
        assert rows["Floor area"]["Today"] == "not reported"
        assert rows["— residential floor"]["Today"] == "not reported"
        assert rows["— non-residential floor"]["Today"] == "not reported"

    def test_premises_come_from_the_roll_and_the_proposal_has_none(
        self, app_defs, gap_row, roll_row
    ):
        rows = _by_measure(app_defs._use_sides(gap_row, None, roll_row))
        assert rows["Non-residential premises"]["Today"] == "1"
        # The solver sizes commerce in floor area and never as units. A count
        # in this cell would be one the app invented.
        assert rows["Non-residential premises"]["Proposed"] == "floor only"

    def test_a_snapshot_without_the_roll_does_not_report_zero_premises(
        self, app_defs, gap_row, roll_row
    ):
        # silver.assessment_units is partitioned, and a borough can carry the
        # roll on one of its cadastre loads and not the other. Zero there is
        # not "no shop on this lot", which is a finding.
        roll_row["roll_loaded"] = False
        roll_row["num_nonresidential_units"] = 0
        rows = _by_measure(app_defs._use_sides(gap_row, None, roll_row))
        assert rows["Non-residential premises"]["Today"] == "roll not loaded"

    def test_without_the_table_the_row_is_a_dash_and_nothing_else_changes(
        self, app_defs, gap_row
    ):
        rows = _by_measure(app_defs._use_sides(gap_row, None, None))
        assert rows["Non-residential premises"]["Today"] == "—"
        assert rows["Storeys"]["Today"] == "2"

    def test_the_footprint_is_the_measured_one(self, app_defs, gap_row):
        rows = _by_measure(
            app_defs._use_sides(
                gap_row, {"num_footprints": 2, "covered_area_m2": 251.4}
            )
        )
        assert rows["Footprint"]["Today"] == "251 m²"
        assert rows["Footprint"]["Proposed"] == "300 m²"


class TestFootprintLine:
    """The Footprint line on a lot one zone covers, and on each piece of one
    two zones cut. Lot 3 237 014's own numbers: a 14 830 m² building stands
    all but entirely in E04-064, and the E04-065 piece carries 83 m² of it.
    """

    PARCEL = {
        "num_footprints": 4, "covered_area_m2": 15012.4,
        "lot_area_m2": 70094.9, "coverage_pct": 21.4,
    }
    E04_065 = {
        "num_footprints": 1, "covered_area_m2": 82.6,
        "piece_area_m2": 18805.8, "coverage_pct": 0.44,
    }
    E04_064 = {
        "num_footprints": 4, "covered_area_m2": 14929.8,
        "piece_area_m2": 51262.7, "coverage_pct": 29.1,
    }

    def test_a_lot_in_one_zone_reads_as_it_always_did(self, app_defs):
        line, caption = app_defs._footprint_line(self.PARCEL)
        assert line == (
            "**Footprint:** 4 building(s), 15,012 m² of ground — 21% of the lot"
        )
        assert "fall inside this lot" in caption

    def test_a_vacant_lot_has_no_caption(self, app_defs):
        line, caption = app_defs._footprint_line(
            {"num_footprints": 0, "covered_area_m2": 0.0, "coverage_pct": 0.0}
        )
        assert line == "**Footprint:** no building on this lot"
        assert caption is None

    def test_a_piece_reports_its_own_ground_and_where_the_rest_stands(
        self, app_defs
    ):
        # The regression: this piece used to print the parcel's 15,012 m².
        line, caption = app_defs._footprint_line(self.E04_065, self.PARCEL, "E04-065")
        assert line == (
            "**Footprint on the E04-065 piece:** 1 building(s), 83 m² of ground "
            "— under 1% of the E04-065 piece"
        )
        assert "4 building(s) over 15,012 m²" in caption
        assert "the other 14,930 m² stand on its other piece(s)" in caption

    def test_the_piece_with_the_building_says_so(self, app_defs):
        line, caption = app_defs._footprint_line(self.E04_064, self.PARCEL, "E04-064")
        assert line.endswith(
            "4 building(s), 14,930 m² of ground — 29% of the E04-064 piece"
        )
        assert "the other 83 m² stand on its other piece(s)" in caption

    def test_a_vacant_piece_of_a_built_lot_is_not_a_vacant_lot(self, app_defs):
        line, caption = app_defs._footprint_line(
            {"num_footprints": 0, "covered_area_m2": 0.0, "coverage_pct": 0.0},
            self.PARCEL, "E04-065",
        )
        assert line == "**Footprint on the E04-065 piece:** no building on the E04-065 piece"
        assert "all of it on its other piece(s)" in caption

    def test_a_piece_holding_the_whole_building_says_that_too(self, app_defs):
        whole = dict(self.PARCEL, piece_area_m2=60000.0, coverage_pct=25.0)
        _, caption = app_defs._footprint_line(whole, self.PARCEL, "E04-064")
        assert caption.endswith("all of it on this piece.")

    def test_the_table_reads_the_piece_coverage_it_is_handed(self, app_defs, gap_row):
        rows = _by_measure(app_defs._use_sides(gap_row, self.E04_065))
        assert rows["Footprint"]["Today"] == "83 m²"


class TestLotIsSplit:
    def test_two_zones_is_split_and_one_or_nothing_is_not(self, app_defs):
        assert app_defs._lot_is_split({"num_lot_zones": 2})
        assert not app_defs._lot_is_split({"num_lot_zones": 1})
        assert not app_defs._lot_is_split({"num_lot_zones": None})
        assert not app_defs._lot_is_split({})
        assert not app_defs._lot_is_split(None)


@pytest.fixture
def site_row() -> dict:
    """One row shaped like `queries.lot_opportunity` returns, all three futures
    priced and an addition that adds something."""
    return {
        "hbu_status": "solved",
        "buyer_best_future": "rebuild",
        "existing_num_storeys": 2,
        "existing_footprint_m2": 200.0,
        "existing_floor_area_m2": 400.0,
        "existing_num_dwellings": 2,
        "existing_annual_stabilised_noi_cad": 31_000.0,
        "existing_dominant_use_description": "Logement",
        "enhance_solved": True,
        "enhance_status": "OPTIMAL",
        "enhance_floors": 3,
        "enhance_added_storeys": 1,
        "enhance_footprint_m2": 220.0,
        "enhance_gross_floor_area_m2": 620.0,
        "enhance_added_floor_area_m2": 220.0,
        "enhance_num_dwellings": 4,
        "enhance_added_dwellings": 2,
        "enhance_added_commercial_area_m2": 0.0,
        "enhance_added_industrial_area_m2": 0.0,
        "enhance_added_annual_stabilised_noi_cad": 21_000.0,
        "enhance_capital_cost_cad": 620_000.0,
        "hbu_floors": 5,
        "hbu_footprint_m2": 300.0,
        "hbu_floor_area_m2": 1_500.0,
        "hbu_residential_floor_area_m2": 1_200.0,
        "hbu_commercial_floor_area_m2": 300.0,
        "hbu_industrial_floor_area_m2": 0.0,
        "hbu_num_dwellings": 14,
        "hbu_annual_stabilised_noi_cad": 180_000.0,
        "hbu_total_capital_cost_cad": 3_400_000.0,
        "site_costs_cad": 120_000.0,
    }


class TestBuildingLines:
    def test_keep_is_what_the_roll_says_stands(self, app_defs, site_row):
        lines, note = app_defs._building_lines(site_row, "hold")
        assert dict(lines)["Storeys"] == "2"
        assert dict(lines)["Dwellings"] == "2"
        assert dict(lines)["NOI a year"] == "$31,000"
        assert dict(lines)["Works"] == "none"
        assert note == "Logement"

    def test_keep_reads_columns_the_shortlist_query_must_carry(
        self, app_defs, site_row
    ):
        # These three used to be absent from the SELECT behind this row, so
        # the pane reported "0 dwellings" and a dash for the income on every
        # lot in the borough. A row missing them again should read as missing,
        # not as a building nobody would keep.
        for column in (
            "existing_num_dwellings",
            "existing_annual_stabilised_noi_cad",
            "hbu_num_dwellings",
        ):
            assert column in site_row

    def test_enhance_reports_the_total_and_what_the_works_add(
        self, app_defs, site_row
    ):
        lines = dict(app_defs._building_lines(site_row, "enhance")[0])
        assert lines["Storeys"] == "3 (+1)"
        assert lines["Floor area"] == "620 m² (+220 m²)"
        assert lines["Dwellings"] == "4 (+2)"
        # The dataplatform writes the addition's income as what is *added*,
        # so the label and the sign both have to say so.
        assert lines["NOI added a year"] == "+$21,000"
        assert lines["Works"] == "$620,000"

    def test_enhance_names_a_class_only_where_the_addition_has_any(
        self, app_defs, site_row
    ):
        assert "Commerce added" not in dict(app_defs._building_lines(site_row, "enhance")[0])
        site_row["enhance_added_commercial_area_m2"] = 150.0
        lines = dict(app_defs._building_lines(site_row, "enhance")[0])
        assert lines["Commerce added"] == "+150 m²"

    def test_an_unmodelled_enhancement_has_no_lines(self, app_defs, site_row):
        site_row["enhance_solved"] = False
        site_row["enhance_status"] = "not_underbuilt"
        assert app_defs._building_lines(site_row, "enhance")[0] == []

    def test_rebuild_splits_the_plate_and_names_the_site_costs(
        self, app_defs, site_row
    ):
        lines = dict(app_defs._building_lines(site_row, "rebuild")[0])
        assert lines["Storeys"] == "5"
        assert lines["Floor area"] == "1,500 m²"
        assert lines["— housing"] == "1,200 m²"
        assert lines["— commerce"] == "300 m²"
        # Nothing industrial is proposed, so no line claims there is.
        assert "— industry" not in lines
        assert lines["Works"] == "$3,400,000"
        assert lines["Clearing the site"] == "$120,000"

    def test_rebuild_says_what_it_replaces(self, app_defs, site_row):
        _, note = app_defs._building_lines(site_row, "rebuild")
        assert note == "Replaces the 2 storeys and 400 m² standing today."

    def test_rebuild_names_only_the_measures_the_roll_stated(
        self, app_defs, site_row
    ):
        # A storey count with no floor area is 799 lots of this borough, and
        # the sentence used to read "the 2 storeys and — standing today".
        site_row["existing_floor_area_m2"] = None
        assert app_defs._building_lines(site_row, "rebuild")[1] == (
            "Replaces the 2 storeys standing today."
        )
        site_row["existing_num_storeys"] = None
        assert app_defs._building_lines(site_row, "rebuild")[1] is None

    def test_an_unsolved_rebuild_has_no_lines(self, app_defs, site_row):
        site_row["hbu_status"] = "equipment_zone"
        assert app_defs._building_lines(site_row, "rebuild")[0] == []

    def test_a_free_site_does_not_bill_for_clearing_it(self, app_defs, site_row):
        site_row["site_costs_cad"] = 0.0
        assert "Clearing the site" not in dict(
            app_defs._building_lines(site_row, "rebuild")[0]
        )


class TestStatusReasons:
    def test_every_status_the_pipeline_writes_reads_as_a_sentence(self, app_defs):
        # The statuses `urban_rag.hbu` can write. A status with no entry falls
        # through to its own name, which reads as a bug on the pane.
        for status in (
            "no_candidate_column", "no_governing_column", "infeasible",
            "solver_error", "equipment_zone",
        ):
            assert app_defs._hbu_status_reason(status) != status
        for status in (
            "no_building", "not_underbuilt", "no_program", "no_envelope",
            "INFEASIBLE", "ERROR",
        ):
            assert app_defs._enhance_status_reason(status) != status

    def test_a_partition_older_than_the_enhancement_solve_says_so(self, app_defs):
        assert app_defs._enhance_status_reason(None) == (
            "this snapshot predates the enhancement solve"
        )

    def test_both_blocks_of_the_deal_pane_read_one_table(self, app_defs):
        # The reasons were inline in the futures block and are now a constant,
        # because two blocks say them and a renamed status should not be
        # half-updated in one of them.
        assert app_defs._ENHANCE_STATUS_REASONS["no_program"]
