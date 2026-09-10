"""
parcel_tools.py — The geometry half of the agent's toolbox.

These read ``rag.lots``, ``rag.buildings`` and ``rag.features`` — and, where
the pipeline has precomputed them, the ``silver`` joins between those. They are
what the chat panel uses to answer questions the map could also answer by being
clicked. Both paths end in the same functions in ``src.utils.queries`` on
purpose: "what is on lot 2 170 935" typed into the chat and clicked on the map
must not be able to disagree, and neither may the fast path and the fallback
inside one of those functions.

Every tool returns compact text rather than JSON. The map is where shapes go —
handing an LLM a polygon's coordinates costs thousands of tokens and buys
nothing, so a tool that finds a shape sends it to the map through
``src.utils.state`` and tells the model only what it found.
"""

from __future__ import annotations

import logging

from langchain.tools import tool
from langchain_core.tools import ToolException

from src.utils import basemap, queries, state
from src.utils.db import DbError

logger = logging.getLogger(__name__)

#: How many rows a listing tool puts in front of the model. Past this the
#: answer is a count and a suggestion to narrow, not a list.
MAX_ROWS = 25


def _require(capability: str) -> None:
    """Fail with what is missing rather than with a SQL error.

    The three repos land in a database independently, so "relation rag.buildings
    does not exist" is a routine state and deserves a routine explanation.
    """
    caps = queries.capabilities()
    if not getattr(caps, capability, False):
        raise ToolException(
            f"This database has no {capability} yet — it is not loaded. "
            f"Missing: {', '.join(caps.missing(include_advisory=False)) or 'nothing'}. "
            f"Tell the user which part of the pipeline has not run rather than "
            f"retrying."
        )


def _fmt_area(value) -> str:
    return f"{float(value):,.0f} m²" if value else "unknown area"


# ---------------------------------------------------------------------------
# Lots
# ---------------------------------------------------------------------------


@tool
def find_lot(lot_number: str) -> str:
    """Look up a cadastral lot by its number and select it on the map.

    Use this whenever the user names a lot ("lot 2 170 935", "2170935"). The
    map zooms to the lot and the Lot pane fills with its attributes and its
    zoning grid, so you do not need to describe the geometry — say what it is
    and what you found.

    Args:
        lot_number: The Infolot lot number. Spaces and other separators are
            ignored, so "2 170 935" and "2170935" both work.

    Returns:
        A short description of the lot: number, borough, area, snapshot date.
    """
    _require("lots")
    try:
        lot = queries.lot_by_number(lot_number)
    except DbError as exc:
        raise ToolException(str(exc)) from exc

    if not lot:
        raise ToolException(
            f"No lot numbered {lot_number!r} in the loaded snapshots. Check the "
            f"number, or ask the user which borough it is in — only the loaded "
            f"boroughs are searchable."
        )

    state.set_selected_lot(
        lot["lot_number"], lot.get("lon"), lot.get("lat"), lot.get("neighborhood")
    )
    bounds = basemap.bounds_of(lot.get("geometry"))
    state.request_map(
        select_lot=lot["lot_number"],
        fit_bounds=basemap.pad_bounds(bounds) if bounds else None,
        note=f"Lot {lot['lot_number']} selected",
    )
    return (
        f"Lot {lot['lot_number']} — {_fmt_area(lot.get('area_m2'))}, "
        f"{lot.get('neighborhood')}, snapshot {lot.get('scrape_date')}. "
        f"Selected on the map; its zoning grid is in the Lot pane."
    )


@tool
def describe_selected_lot() -> str:
    """Report the lot currently selected on the map.

    Call this FIRST whenever the user says "this lot", "here", "the selected
    one", or asks a question with no lot number in it. It tells you which lot
    the user is looking at, so you do not have to ask them to repeat it.

    Returns:
        The selected lot's number, area and coordinates, or a note that nothing
        is selected.
    """
    selected = state.get_selected_lot()
    if not selected.get("lot_number"):
        return (
            "No lot is selected. Ask the user to click a lot on the map, or to "
            "give a lot number."
        )
    _require("lots")
    lot = queries.lot_by_number(selected["lot_number"])
    if not lot:
        return f"Lot {selected['lot_number']} is selected but no longer in the snapshot."

    buildings = ""
    if queries.capabilities().buildings:
        # `lot_coverage` rather than a sum over `buildings_on_lot`: the ground
        # two overlapping footprints share is covered once, and the lot's own
        # snapshot is the one the rest of this line was read from.
        coverage = queries.lot_coverage(
            lot["lot_number"], scrape_date=lot.get("scrape_date")
        )
        if coverage and coverage["num_footprints"]:
            buildings = (
                f" {coverage['num_footprints']} building footprint(s) on it, "
                f"{float(coverage['covered_area_m2']):,.0f} m² of ground covered."
            )

    return (
        f"Selected: lot {lot['lot_number']} — {_fmt_area(lot.get('area_m2'))}, "
        f"{lot.get('neighborhood')}, snapshot {lot.get('scrape_date')}, "
        f"at {lot.get('lat'):.5f}, {lot.get('lon'):.5f}.{buildings}"
    )


@tool
def list_lots(
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
    neighborhood: str | None = None,
    limit: int = 15,
) -> str:
    """List lots in the area currently shown on the map, optionally by size.

    Use this for questions like "which lots here are over 500 m²" or "show me
    the biggest parcels in view". The search is restricted to the visible
    rectangle — the map is the question's scope, so pan or zoom first if the
    user means somewhere else.

    Args:
        min_area_m2: Only lots at least this large.
        max_area_m2: Only lots at most this large.
        neighborhood: Restrict to one borough code, e.g. "VSMPE".
        limit: How many to list, capped at 25.

    Returns:
        A numbered list of lot numbers with their areas.
    """
    _require("lots")
    bounds = state.get_viewport()
    if bounds is None:
        raise ToolException(
            "The map has not reported a viewport yet. Ask the user to move the "
            "map once, or use find_lot with a specific lot number."
        )

    found = queries.lots_in_bbox(
        bounds,
        zoom=state.get_viewport_zoom(),
        neighborhood=neighborhood,
        min_area_m2=min_area_m2,
        max_area_m2=max_area_m2,
        limit=min(int(limit), MAX_ROWS) if limit else MAX_ROWS,
    )
    if not found.features:
        return (
            "No lots in the current view match those criteria. The view may be "
            "outside the loaded borough, or the size filter may be too narrow."
        )

    rows = sorted(
        found.features,
        key=lambda f: float(f["properties"].get("area_m2") or 0),
        reverse=True,
    )
    listing = "\n".join(
        f"{i}. Lot {f['properties']['lot_number']} — "
        f"{_fmt_area(f['properties'].get('area_m2'))}"
        for i, f in enumerate(rows[: min(int(limit), MAX_ROWS)], 1)
    )
    more = " (more exist in view than are listed)" if found.truncated else ""
    return f"{len(rows)} matching lot(s) in the current view{more}:\n{listing}"


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


@tool
def buildings_on_lot(lot_number: str = "") -> str:
    """Report the building footprints standing on a lot.

    Args:
        lot_number: The lot to inspect. Leave empty to use the lot currently
            selected on the map.

    Returns:
        Footprint count, each footprint's area, and how much ground they cover
        inside the lot — the measured counterpart to the lot coverage (taux
        d'implantation) the zoning grid permits. This is ground covered, not
        floor built: a building of several storeys holds several times this
        much floor area, which is what `lot_efficiency` reports.
    """
    _require("buildings")
    lot_number = lot_number or (state.get_selected_lot().get("lot_number") or "")
    if not lot_number:
        raise ToolException(
            "No lot given and none selected. Ask the user to click a lot or "
            "name one."
        )

    lot = queries.lot_by_number(lot_number)
    if not lot:
        raise ToolException(f"No lot numbered {lot_number!r} in the loaded snapshots.")

    # Both reads are pinned to the lot's own snapshot, so the listing and the
    # total describe one load of the cadastre rather than every load in the
    # database - which is what turned one footprint into two.
    rows = queries.buildings_on_lot(
        lot["lot_number"], scrape_date=lot.get("scrape_date")
    )
    coverage = queries.lot_coverage(
        lot["lot_number"], scrape_date=lot.get("scrape_date")
    )
    if not rows:
        return f"Lot {lot['lot_number']} has no building footprint on it — it reads as vacant."

    # The total is the union of the clipped shapes, not the sum of the rows
    # below: where two footprints overlap, the ground under both is covered
    # once, and adding the rows up can exceed the lot itself.
    covered = float((coverage or {}).get("covered_area_m2") or 0)
    pct = (coverage or {}).get("coverage_pct")
    ratio = f", {pct:.0f}% of the lot" if pct is not None else ""
    each = "; ".join(
        f"{_fmt_area(r.get('area_m2'))} footprint in all "
        f"({_fmt_area(r.get('overlap_m2'))} of it on this lot)"
        for r in rows[:10]
    )
    return (
        f"Lot {lot['lot_number']} ({_fmt_area(lot.get('area_m2'))}): "
        f"{len(rows)} footprint(s) covering {covered:,.0f} m² of its ground"
        f"{ratio}. {each}"
    )


# ---------------------------------------------------------------------------
# Is it used for what it is zoned for
# ---------------------------------------------------------------------------


def _resolve_lot_uid(lot_number: str) -> dict:
    """The lot row a capacity question is about, from a number or the selection."""
    number = (lot_number or "").strip() or state.get_selected_lot().get("lot_number")
    if not number:
        raise ToolException(
            "No lot number given and none selected. Ask the user to click a lot "
            "or give a number."
        )
    lot = queries.lot_by_number(number)
    if not lot:
        raise ToolException(f"No lot {number} in this snapshot.")
    return lot


@tool
def lot_efficiency(lot_number: str = "") -> str:
    """Say whether a lot is used for as much as its zoning permits, and what else fits.

    This is the tool for "is this lot used efficiently", "is it under-built",
    "what else could I build here", "how many more units fit". It compares what
    the assessment roll says stands on the lot today against what the governing
    zoning envelope could hold, and reports the difference per use class.

    Leave ``lot_number`` empty to use the lot selected on the map.

    Args:
        lot_number: The lot to report on. Empty means the map's selection.

    Returns:
        The share of permitted floor area in use, the additional floor area by
        class in m² and sq ft, the dwelling count today against the proposed
        one, and the shape of the proposed building — or the reason no
        programme could be solved for this lot.
    """
    _require("redevelopment_gap")
    lot = _resolve_lot_uid(lot_number)
    row = queries.lot_capacity(
        int(lot["lot_uid"]),
        scrape_date=lot.get("scrape_date"),
        neighborhood=lot.get("neighborhood"),
    )
    if not row:
        return (
            f"Lot {lot['lot_number']} has no highest-and-best-use row in this "
            f"snapshot."
        )

    if row.get("hbu_status") != "solved":
        reason = {
            "no_candidate_column": (
                "every zoning column reaching it authorises none of the uses "
                "the solver prices (housing, commerce, industry) — usually a "
                "community-facilities-only zone"
            ),
            # The former name of no_candidate_column, from when the solver
            # priced dwellings alone; rows written before the rename carry it.
            "no_residential_column": (
                "every zoning column reaching it authorises something other "
                "than housing, and this snapshot predates the solver pricing "
                "commerce and industry — re-run the pipeline to solve it"
            ),
            "no_governing_column": (
                "candidate columns exist but none governs it — usually no "
                "measured frontage under a grid stating a minimum width"
            ),
            "infeasible": (
                "no governing column has a feasible programme — a minimum the "
                "parcel cannot meet; a lot the stalls alone stop is solved "
                "without them and reported solved with the parking waived"
            ),
            "solver_error": "the governing column could not be modelled",
        }.get(row["hbu_status"], row["hbu_status"])
        return (
            f"Lot {lot['lot_number']}: no development programme was solved — "
            f"{reason}. So there is no permitted-floor figure to compare "
            f"against what stands there."
        )

    built = float(row.get("existing_floor_area_m2") or 0)
    permitted = float(row.get("hbu_floor_area_m2") or 0)
    used = row.get("used_pct")
    # The roll assessed a unit here and stated no floor area for it. Answering
    # "0% of the envelope is in use" on that is the one failure mode of this
    # tool that a reader cannot catch: it is a plausible number about a lot
    # with a building standing on it, and nothing else in the answer says the
    # figure was never measured.
    unreported = queries.floor_area_unreported(row)
    parts = [f"Lot {lot['lot_number']} ({_fmt_area(row.get('lot_area_m2'))})."]

    # A parcel a zoning boundary crosses is two development sites, and every
    # figure below is about one of them - the largest, which is what
    # `lot_capacity` returns when no zone is named. Said out loud rather than
    # left implicit, because the failure it prevents is the worst kind this
    # tool has: a confident, plausible answer about 90 % of a parcel, reported
    # as though it were the parcel.
    zones = row.get("num_lot_zones")
    try:
        num_zones = int(zones) if zones is not None else 1
    except (TypeError, ValueError):
        num_zones = 1
    if num_zones > 1:
        parts.append(
            f"This lot is in {num_zones} zones: a zoning boundary crosses it, "
            f"so it is {num_zones} separate development sites with their own "
            f"envelopes, streets and programmes. Everything below is about "
            f"the largest — zone {row.get('feature_id')}, "
            f"{_fmt_area(row.get('piece_area_m2'))} of the "
            f"{_fmt_area(row.get('lot_area_m2'))} parcel. Say so when "
            f"reporting it; the other piece(s) are a different answer, not a "
            f"rounding of this one."
        )
    if row.get("parking_waived"):
        parts.append(
            "The programme these figures come from only exists with its "
            f"parking waived: it is short {int(row.get('waived_stalls') or 0)} "
            "stall(s) of what the assumed ratios ask, because no building "
            "that provides them fits or pays on this parcel. Say so - it "
            "stands on a variance."
        )

    if unreported:
        parts.append(
            f"How much of the permitted floor is in use is NOT KNOWN for this "
            f"lot: the assessment roll has a unit on it but states no floor "
            f"area, so there is nothing to hold against the {permitted:,.0f} "
            f"m² the grid permits. Do not report it as 0%, as under-built, or "
            f"as vacant — say the roll does not give the figure. "
            f"`buildings_on_lot` is what can still be measured here."
        )
    elif used is None:
        parts.append("No utilisation share could be computed.")
    elif float(used) > 100:
        parts.append(
            f"{float(used):,.0f}% of what the grid permits is already standing "
            f"({built:,.0f} m² of floor area against {permitted:,.0f} m² "
            f"permitted) — more floor than today's zoning would allow, i.e. a "
            f"legal non-conformity rather than headroom."
        )
    else:
        verdict = "effectively built out" if float(used) >= 95 else "under-built"
        parts.append(
            f"{float(used):,.0f}% of permitted floor area is in use "
            f"({built:,.0f} m² of floor area today against {permitted:,.0f} m² "
            f"permitted) — {verdict}."
        )
    # Said once, here, because the two numbers above are every storey added up
    # and the coverage figure `buildings_on_lot` reports is the ground under
    # one - and on a multi-storey building the first is several times the
    # second, which reads as a contradiction unless it is named.
    parts.append(
        "Floor area is the sum of the storeys, not the ground the building "
        "covers; that is the footprint, from buildings_on_lot."
    )

    # The unit count, not has_assessment: gold writes that flag true on every
    # row of the table, so this sentence never reached the lots it is about.
    if queries.nothing_assessed(row):
        parts.append(
            "The assessment roll has no unit on this lot, so the floor area "
            "standing today is read as nothing built."
        )

    # Headroom is the same subtraction as the share above, split by class, so
    # an unreported existing floor makes every one of these numbers the whole
    # envelope rather than what is left of it.
    if unreported:
        parts.append(
            "Additional floor area cannot be stated for the same reason: what "
            "already stands is not in the roll, so what is left of the "
            "envelope is unknown."
        )
    else:
        extras = []
        for label, key in (
            ("residential", "residential_headroom_m2"),
            ("commercial", "commercial_headroom_m2"),
            ("industrial", "industrial_headroom_m2"),
        ):
            value = float(row.get(key) or 0)
            if value > 0:
                extras.append(
                    f"{label} {value:,.0f} m² ({value * 10.7639:,.0f} sq ft)"
                )
        parts.append(
            ("Additional floor area that fits: " + "; ".join(extras) + ".")
            if extras else "No additional floor area fits under this grid."
        )

    hbu_d, existing_d = row.get("hbu_num_dwellings"), row.get("existing_num_dwellings")
    if hbu_d is not None:
        parts.append(
            f"Dwellings: {int(existing_d or 0)} today, {int(hbu_d)} proposed."
        )

    # The use on each side, said before the shape: "a residential building"
    # below is the answer, and this is what it is an answer *to*. The roll's
    # own words follow today's class because the class is a filing and the
    # words are the fact - "commercial" is what a church is filed under.
    today_use = row.get("existing_dominant_income_class")
    proposed_use = row.get("hbu_dominant_use")
    if today_use or proposed_use:
        today = str(today_use or "not on the roll").replace("_", " ")
        if row.get("existing_dominant_use_description"):
            today += f" ({row['existing_dominant_use_description']})"
        proposed = str(proposed_use or "no programme").replace("_", " ")
        verdict = ""
        if today_use and proposed_use and today_use not in ("none",) \
                and proposed_use not in ("none",):
            verdict = (
                " The use changes." if today_use != proposed_use
                else " The use stays."
            )
        parts.append(f"Use: {today} today, {proposed} proposed.{verdict}")

    shape = []
    use = row.get("hbu_dominant_use")
    if use and use not in ("none",):
        shape.append(f"a {use.replace('_', ' ')} building")
    if row.get("floors"):
        shape.append(f"{int(row['floors'])} storeys")
    if row.get("height_m"):
        shape.append(f"{float(row['height_m']):.1f} m")
    if row.get("grid_zone"):
        shape.append(f"under zone {row['grid_zone']}")
    if shape:
        parts.append("Proposed building: " + ", ".join(shape) + ".")

    # The developer's arithmetic behind the proposal: what it costs, what the
    # finished building is worth discounted, and whether building it beats
    # keeping what stands. Quoted only when the columns are populated — a
    # snapshot solved before the discounting existed has nothing to say here.
    npv = row.get("hbu_npv_cad") if row.get("hbu_npv_cad") is not None else row.get("npv_cad")
    if npv is not None:
        money = [
            f"discounted net profit ${float(npv):,.0f}",
        ]
        if row.get("total_capital_cost_cad") is not None:
            money.append(
                f"construction ${float(row['total_capital_cost_cad']):,.0f}"
            )
        gain = row.get("redevelopment_npv_gain_cad")
        if gain is not None:
            verdict = (
                f"redeveloping beats holding the current building by "
                f"${float(gain):,.0f}"
                if float(gain) > 0
                else f"holding the current building beats redeveloping by "
                f"${-float(gain):,.0f}"
            )
            money.append(verdict)
        parts.append(
            "Developer economics (land excluded, both futures priced at the "
            "same discount): " + "; ".join(money) + "."
        )

    # The second axis, where the shortlist table is loaded: why the parcel is
    # acquirable, what that costs, and what the heritage rows say. Read by its
    # own query rather than joined into `lot_capacity`, so a database without
    # the table loses this sentence and nothing else.
    if queries.capabilities().investment_opportunities:
        site = queries.lot_opportunity(
            int(lot["lot_uid"]),
            scrape_date=lot.get("scrape_date"),
            neighborhood=lot.get("neighborhood"),
        )
        sentence = _site_thesis_sentence(site)
        if sentence:
            parts.append(sentence)

    fit = row.get("footprint_fit_pct")
    if fit is not None and float(fit) < 99.5:
        parts.append(
            f"CAVEAT: the solved footprint only fits this parcel's shape at "
            f"{float(fit):.0f}%, so the floor areas above are overstated here. "
            f"Say so if you quote them."
        )
    return " ".join(parts)


#: What each site thesis means, in the words the answer uses.
_SITE_THESIS_MEANING = {
    "brownfield": (
        "a contamination-risk use stands on it (the dataplatform's brownfield "
        "thesis), so the ground has to be characterised and cleaned before "
        "the change of use"
    ),
    "teardown": (
        "an obsolete building fills little of an envelope that allows storeys "
        "above it (the teardown thesis), so the play is to demolish and rebuild"
    ),
    "infill": "nothing stands on it (the infill thesis)",
    "improvement": (
        "the building can stay and gain a storey or a rear annex inside its "
        "envelope (the improvement thesis)"
    ),
}


def _enhancement_adds_nothing(site: dict) -> bool:
    """Whether the enhancement solve came back as the standing building.

    The dataplatform normalises a solve where no storey and no annex pays at
    the addition premium (`nothing_pencils`) to the building that stands: 0 m²
    added, no dwellings, no capital. Such a row is *solved*, so `enhance_solved`
    alone does not say it - the added floor does.
    """
    return bool(site.get("enhance_solved")) and not (
        float(site.get("enhance_added_floor_area_m2") or 0) > 0
        or int(site.get("enhance_added_dwellings") or 0) > 0
    )


def _annex_m2(site: dict) -> float | None:
    """The ground the addition takes beside the standing building, in m².

    `enhance_footprint_m2` is the whole plate after the works and the standing
    plate is `existing_footprint_m2` (floor over storeys where the roll gave
    no footprint), so the annex is the difference - 0 where the addition is a
    storey on the plate that is there. None where either is unknown.
    """
    after = site.get("enhance_footprint_m2")
    before = site.get("existing_footprint_m2")
    if before is None:
        floor = site.get("existing_floor_area_m2")
        storeys = site.get("existing_num_storeys")
        if floor is not None and storeys:
            before = float(floor) / float(storeys)
    if after is None or before is None:
        return None
    return max(float(after) - float(before), 0.0)


def _addition_shape(site: dict) -> str:
    """"1 storey on the standing plate and a 40 m² annex beside it", off the
    solve's own geometry; the generic phrasing where the addition is the
    closed-form estimate and has no plate of its own."""
    storeys = int(site.get("improvement_added_storeys") or 0)
    annex = _annex_m2(site) if site.get("enhance_solved") else None
    storey_text = f"{storeys} storey on the standing plate" if storeys else ""
    if annex is None:
        return (
            f"{storey_text} or an annex beside it" if storey_text
            else "a storey on the standing plate or an annex beside it"
        )
    annex_text = f"a {annex:,.0f} m² annex beside it" if annex >= 0.5 else ""
    parts = [part for part in (storey_text, annex_text) if part]
    return " and ".join(parts) if parts else "on the standing plate"


def _site_thesis_sentence(site: dict | None) -> str:
    """One or two sentences on the row's site thesis, or "" where none holds."""
    if not site:
        return ""
    thesis = site.get("site_thesis")
    flags = []
    if site.get("is_heritage_sector"):
        flags.append(
            "the governing zone is a secteur d'intérêt patrimonial, which keeps "
            "the lot out of any thesis that demolishes"
        )
    if site.get("has_piia_review"):
        flags.append(
            f"the zone is in PIIA sector {site.get('piia_sector')}, so a "
            "replacement building faces a discretionary architectural review, "
            "which keeps the lot out of any thesis that demolishes"
        )
    if site.get("demolition_review_required") and not site.get("is_heritage_sector"):
        flags.append(
            "the building predates 1940, so its demolition is subject to the "
            "borough's demolition by-law and heritage review"
        )
    flag_text = ("; ".join(flags) + ".") if flags else ""

    if not thesis or thesis == "none":
        return (
            "Site thesis: none - no site condition (obsolete building, "
            "contamination-risk use, empty lot, or room for an addition) holds "
            "on this lot. " + flag_text
        ).strip()

    meaning = _SITE_THESIS_MEANING.get(thesis, thesis)
    rank = site.get("site_thesis_rank")
    count = site.get("num_ranked_in_site_thesis")
    if rank is not None:
        standing = f"ranked {int(rank)} of {int(count or 0)} {thesis} sites"
        if site.get("is_top_site_opportunity"):
            standing += " and on that thesis's shortlist"
    else:
        standing = (
            "filed but unranked, because at the solve's assumptions the play "
            "does not pay"
        )
    pieces = [f"Site thesis: {thesis} - {meaning}; {standing}."]

    site_yield = site.get("site_yield_on_cost_pct")
    if site_yield is not None:
        if thesis == "improvement":
            pieces.append(
                f"The addition is {float(site.get('improvement_floor_m2') or 0):,.0f} "
                f"m² ({_addition_shape(site)}) earning "
                f"${float(site.get('improvement_noi_cad') or 0):,.0f} a year "
                f"on ${float(site.get('improvement_cost_cad') or 0):,.0f} of "
                f"work, a {float(site_yield):,.1f}% yield on cost."
            )
        else:
            costs = [
                f"demolition ${float(site.get('demolition_cost_cad') or 0):,.0f}"
            ]
            if float(site.get("remediation_cost_cad") or 0):
                costs.append(
                    f"characterisation ${float(site.get('site_assessment_cost_cad') or 0):,.0f}"
                )
                costs.append(
                    f"remediation ${float(site.get('remediation_cost_cad') or 0):,.0f}"
                )
            pieces.append(
                f"Yield on cost with the site's own costs "
                f"({', '.join(costs)}) is {float(site_yield):,.1f}% on "
                f"${float(site.get('site_total_project_cost_cad') or 0):,.0f} "
                "all in, land at its assessed value."
            )
    if site.get("site_irr_pct") is not None or site.get("site_all_in_yield_on_cost_pct") is not None:
        returns_bits = []
        if site.get("site_all_in_yield_on_cost_pct") is not None:
            text = f"yield on all-in cost {float(site['site_all_in_yield_on_cost_pct']):,.1f}%"
            if site.get("market_cap_rate_pct") is not None and site.get("site_yoc_spread_bps") is not None:
                spread = float(site["site_yoc_spread_bps"])
                text += (
                    f" against a {float(site['market_cap_rate_pct']):,.1f}% market cap rate "
                    f"({'+' if spread >= 0 else '-'}{abs(spread):,.0f} bps)"
                )
            returns_bits.append(text)
        if site.get("site_irr_pct") is not None:
            returns_bits.append(f"buyer's unlevered IRR {float(site['site_irr_pct']):,.1f}%")
        if site.get("owner_site_irr_pct") is not None:
            returns_bits.append(f"owner's IRR on the increment {float(site['owner_site_irr_pct']):,.1f}%")
        # Either bar makes a good candidate, so the verdict says which one it
        # was; a lot that clears one and is still not one does not pay.
        screen_bits = ", ".join(
            f"{'clears' if ok else 'misses'} the {name}"
            for name, ok in (
                ("cap rate spread", bool(site.get("clears_cap_rate"))),
                ("IRR hurdle", bool(site.get("clears_hurdle"))),
            )
        )
        verdict_text = (
            f"a good candidate: {screen_bits} and pays against holding"
            if site.get("is_good_candidate")
            else f"not a good candidate: {screen_bits}"
            if not (site.get("clears_cap_rate") or site.get("clears_hurdle"))
            else "not a good candidate: the play does not pay against holding"
        )
        pieces.append("Returns: " + "; ".join(returns_bits) + " - " + verdict_text + ".")
    standing_bits = []
    if site.get("existing_year_built") is not None:
        standing_bits.append(f"built {int(site['existing_year_built'])}")
    if site.get("existing_num_storeys") is not None and site.get("hbu_floors") is not None:
        standing_bits.append(
            f"{int(site['existing_num_storeys'])} storeys where the grid "
            f"takes {int(site['hbu_floors'])}"
        )
    if standing_bits:
        pieces.append("Standing: " + ", ".join(standing_bits) + ".")
    if flag_text:
        pieces.append("Heritage: " + flag_text[0].lower() + flag_text[1:])
    return " ".join(pieces)


_FUTURE_NAMES = {"hold": "keep", "enhance": "enhance", "rebuild": "tear down and rebuild"}


@tool
def lot_futures(lot_number: str = "") -> str:
    """Price a lot's three futures for a buyer - keep, enhance, or rebuild.

    This is the tool for "what is this lot worth to a buyer", "what could I
    pay for it", "does rebuilding beat keeping", "is there a deal here",
    "what would an extra storey earn". The dataplatform priced all three on
    one footing: the standing building's income discounted (keep), the same
    plus a solved addition on the standing building (enhance), and the
    highest-and-best-use rebuild with its income starting after the build and
    the lease-up, less demolition and remediation (rebuild).

    Every figure is a buyer's, and the ground is paid for inside all of them:
    each future is what it is worth to whoever ends up holding the lot, less
    the price of the land. The owner's own arithmetic - each future to
    somebody who already holds the ground, land cancelling - is not reported,
    with one exception: the standing income's worth to its holder sets the
    floor under the asking price, so it is stated as part of the price.

    Args:
        lot_number: The lot to report on. Empty means the map's selection.

    Returns:
        The price the ground would take, then one line per future with its
        NPV after purchase, its unlevered IRR (soft costs, contingency and
        the absorption-driven lease-up in), its yield on everything paid to
        reach it, the
        most a buyer could pay for it, the cost and the timeline, and which
        future wins.
    """
    _require("investment_opportunities")
    lot = _resolve_lot_uid(lot_number)
    site = queries.lot_opportunity(
        int(lot["lot_uid"]),
        scrape_date=lot.get("scrape_date"),
        neighborhood=lot.get("neighborhood"),
    )
    if not site:
        return (
            f"Lot {lot['lot_number']} has no row in the shortlist table for this "
            "snapshot, so its futures are not priced."
        )
    lines = [f"Lot {lot['lot_number']} ({_fmt_area(site.get('lot_area_m2'))}), three futures:"]
    price = site.get("acquisition_cost_cad")
    if price is None:
        return (
            f"Lot {lot['lot_number']}: the assessment roll never reached it, so "
            "there is no price to put on the ground and no deal to price against "
            "it. Its highest and best use is still solved - ask for the lot's "
            "programme instead."
        )
    lines.append(
        f"Price to pay: ${float(price):,.0f} - the larger of the roll's value "
        "times the market factor and what the standing income is worth to "
        "whoever holds it, since a seller keeps the better of the two."
    )
    futures = (
        ("hold", "owner_hold_value_cad", "buyer_npv_hold_cad",
         "buyer_yield_hold_pct", None, None),
        ("enhance", "owner_enhance_value_cad", "buyer_npv_enhance_cad",
         "buyer_yield_enhance_pct", "residual_price_enhance_cad",
         "enhance_capital_cost_cad"),
        ("rebuild", "owner_rebuild_value_cad", "buyer_npv_rebuild_cad",
         "buyer_yield_rebuild_pct", "residual_price_rebuild_cad",
         "hbu_total_capital_cost_cad"),
    )
    best = site.get("buyer_best_future")
    for key, value_key, npv_key, yield_key, residual_key, cost_key in futures:
        name = _FUTURE_NAMES[key]
        value = site.get(value_key)
        if key == "enhance" and not site.get("enhance_solved"):
            lines.append(f"- {name}: not priced ({site.get('enhance_status') or 'no enhancement'}).")
            continue
        if key == "enhance" and _enhancement_adds_nothing(site):
            # Solved, and the answer is the building that stands: no storey
            # and no annex pays at the addition premium. There is nothing to
            # build, so nothing to cost, time or return - it is the keep line.
            lines.append(
                f"- {name}: nothing to add - no storey or annex pays at the "
                "addition premium, so enhancing this building is keeping it; "
                "see the keep line."
            )
            continue
        if key == "rebuild" and site.get("hbu_status") != "solved":
            lines.append(f"- {name}: not priced ({site.get('hbu_status')}).")
            continue
        if value is None:
            lines.append(f"- {name}: not priced.")
            continue
        npv = site.get(npv_key)
        yld = site.get(f"buyer_yoc_{key}_pct")
        if yld is None:
            yld = site.get(yield_key)
        irr = site.get(f"buyer_irr_{key}_pct")
        irr_text = f", IRR {float(irr):,.1f}%" if irr is not None else ""
        residual = site.get(residual_key) if residual_key else value
        part = (
            f"- {name}: NPV after purchase {'+' if float(npv or 0) >= 0 else '-'}"
            f"${abs(float(npv or 0)):,.0f}{irr_text}, {float(yld or 0):,.1f}% on all-in cost, "
            f"most you could pay ${float(residual or 0):,.0f}"
        )
        owner_irr = site.get(f"owner_irr_{key}_pct")
        if owner_irr is not None:
            part += f" (to the owner, {float(owner_irr):,.1f}% on the increment)"
        cost = 0.0 if cost_key is None else float(site.get(cost_key) or 0)
        if key == "rebuild":
            cost += float(site.get("site_costs_cad") or 0)
        if cost:
            part += f", costing ${cost:,.0f} to build on top of the price"
        if key == "enhance":
            annex = _annex_m2(site)
            annex_text = (
                f" and a {annex:,.0f} m² annex" if annex is not None and annex >= 0.5 else ""
            )
            part += (
                f" ({int(site.get('enhance_added_storeys') or 0)} storey{annex_text}, "
                f"{float(site.get('enhance_added_floor_area_m2') or 0):,.0f} m² added, "
                f"{int(site.get('enhance_added_dwellings') or 0)} new dwellings)"
            )
            if site.get("enhance_parking_waived"):
                part += (
                    f"; PARKING WAIVED — short {int(site.get('enhance_waived_stalls') or 0)} "
                    "stall(s) of what is owed, so the addition stands on a variance"
                )
        if key == "rebuild":
            part += (
                f" ({int(site.get('hbu_num_dwellings') or 0)} dwellings, "
                f"{float(site.get('hbu_floor_area_m2') or 0):,.0f} m², income after the build and lease-up)"
            )
            if site.get("hbu_parking_waived"):
                part += (
                    f"; PARKING WAIVED — short {int(site.get('hbu_waived_stalls') or 0)} "
                    "stall(s) of what is owed, so the rebuild stands on a variance"
                )
        if key == best:
            part += " <- best"
        lines.append(part + ".")
    # The room between the asking price and the ceiling the best future puts
    # over it: the whole of the negotiating range, and the first thing anyone
    # brokering the lot wants said.
    ceiling = {
        "hold": site.get("owner_hold_value_cad"),
        "enhance": site.get("residual_price_enhance_cad"),
        "rebuild": site.get("residual_price_rebuild_cad"),
    }.get(best) if best and best != "none" else None
    if best == "none":
        lines.append(
            "No future clears the discount rate at this price - a buyer walks, "
            "or pays no more than the residual prices above."
        )
    elif ceiling is not None:
        room = float(ceiling) - float(price)
        lines.append(
            f"Room between the price and what the best future could bear: "
            f"{'+' if room >= 0 else '-'}${abs(room):,.0f}."
        )
    thesis = str(site.get("investment_thesis") or "none")
    if thesis != "none":
        lines.append(
            f"Would build: {thesis.replace('_', ' ')}"
            + (
                f", ranked {int(site['thesis_rank'])} of "
                f"{int(site.get('num_ranked_in_thesis') or 0)} such sites in the borough"
                if site.get("thesis_rank") is not None else ""
            )
            + "."
        )
    lines.append(
        "Unlevered, at the solve's discount rate, hold and terminal cap; the "
        "land is paid for at the price above in every line, and the IRR and "
        "the yield on all-in cost carry soft costs, contingency, builder's "
        "risk and an absorption-driven lease-up on top of the hard cost. Not "
        "an appraisal."
    )
    return "\n".join(lines)


@tool
def top_site_opportunities(site_thesis: str = "", limit: int = 10) -> str:
    """List the best lots of one site thesis - why a parcel is acquirable.

    This is the tool for "where are the teardowns", "which gas stations could
    become housing", "brownfield sites", "where could an owner add a storey",
    "empty lots worth building on". The dataplatform files every lot under a
    site thesis - brownfield, teardown, infill or improvement - and ranks each
    thesis on its own yield on cost, with demolition, remediation or the
    addition's premium in the denominator.

    Args:
        site_thesis: One of brownfield, teardown, infill, improvement; empty
            lists the top of every thesis together.
        limit: How many lots to list (default 10).

    Returns:
        One line per lot: its rank within the thesis, the yield, the verdict,
        what stands there, and any heritage or PIIA flag.
    """
    _require("investment_opportunities")
    thesis = (site_thesis or "").strip().lower() or None
    if thesis is not None and thesis not in queries.SITE_THESES:
        raise ToolException(
            f"Unknown site thesis {site_thesis!r} - use one of "
            f"{', '.join(queries.SITE_THESES)}, or leave it empty."
        )
    rows = queries.top_site_opportunities(
        site_thesis=thesis, limit=max(1, min(int(limit or 10), 50))
    )
    if not rows:
        return (
            f"No ranked {thesis or 'site'} opportunity in this snapshot - either "
            "the lot_investment_opportunities asset has not been run with the "
            "site theses, or nothing filed under it pays at the solve's "
            "assumptions."
        )
    lines = [
        (
            f"Top {thesis} sites, by that thesis's yield on cost:"
            if thesis
            else "Top sites of each thesis, interleaved by rank:"
        )
    ]
    for row in rows:
        flags = []
        if row.get("is_heritage_sector"):
            flags.append("heritage sector")
        if row.get("has_piia_review"):
            flags.append("PIIA")
        if row.get("demolition_review_required") and not row.get("is_heritage_sector"):
            flags.append("pre-1940")
        if row.get("site_thesis") == "improvement":
            verdict = (
                f"+{float(row.get('improvement_floor_m2') or 0):,.0f} m² earning "
                f"${float(row.get('improvement_noi_cad') or 0):,.0f}/yr"
            )
        else:
            verdict = (
                f"+${float(row.get('redevelopment_npv_gain_cad') or 0):,.0f} vs holding"
            )
        standing = []
        if row.get("existing_year_built") is not None:
            standing.append(f"built {int(row['existing_year_built'])}")
        if row.get("existing_num_storeys") is not None and row.get("hbu_floors") is not None:
            standing.append(
                f"{int(row['existing_num_storeys'])}/{int(row['hbu_floors'])} storeys"
            )
        if row.get("existing_dominant_use_description"):
            standing.append(str(row["existing_dominant_use_description"]))
        returns_text = ""
        if row.get("site_irr_pct") is not None:
            returns_text += f"IRR {float(row['site_irr_pct']):,.1f}%, "
        if row.get("site_all_in_yield_on_cost_pct") is not None:
            returns_text += f"{float(row['site_all_in_yield_on_cost_pct']):,.1f}% on all-in cost"
            if row.get("site_yoc_spread_bps") is not None:
                spread = float(row["site_yoc_spread_bps"])
                returns_text += f" ({'+' if spread >= 0 else '-'}{abs(spread):,.0f} bps vs cap)"
            returns_text += ", "
        elif row.get("site_yield_on_cost_pct") is not None:
            # A row the proforma never reached: the solve's own yield on cost.
            returns_text += f"{float(row['site_yield_on_cost_pct']):,.1f}% on cost, "
        lines.append(
            f"Lot {row.get('lot_number') or '?'} "
            f"({float(row.get('lot_area_m2') or 0):,.0f} m², zone "
            f"{row.get('grid_zone') or '?'}): {row.get('site_thesis')} rank "
            f"{int(row.get('site_thesis_rank') or 0)} of "
            f"{int(row.get('num_ranked_in_site_thesis') or 0)}, "
            + returns_text
            + f"{verdict}"
            + ("; GOOD CANDIDATE" if row.get("is_good_candidate") else "")
            + f"; would build {str(row.get('investment_thesis') or '?').replace('_', ' ')}"
            + (f"; today {', '.join(standing)}" if standing else "")
            + (f"; flags: {', '.join(flags)}" if flags else "")
            + "."
        )
    lines.append(
        "Ranked on the buyer's unlevered IRR of each thesis's own future, "
        "with soft costs, contingency and an absorption-driven lease-up in; "
        "a GOOD CANDIDATE clears the area's cap rate by the spread or the "
        "IRR hurdle and pays against holding. Rates are the row's "
        "screen_assumptions; none is a per-lot survey."
    )
    return "\n".join(lines)


@tool
def development_capacity() -> str:
    """Total how much more could be built across the whole borough under current zoning.

    This is the tool for "how much more could we build", "how many more units
    could fit in the borough", "what is the total unused capacity". It sums the
    per-lot comparison over the loaded partition, not over the viewport.

    Returns:
        Additional residential, commercial and industrial floor area in m² and
        sq ft, the additional dwelling count, and the counts of lots the totals
        rest on.
    """
    _require("redevelopment_gap")
    totals = queries.capacity_totals()
    if not totals or not totals.get("num_lots"):
        return "No redevelopment-gap rows are loaded, so there is nothing to total."

    def area(key: str, modelled_key: str) -> str:
        # A class no solved lot was given any floor of is a finding about the
        # economics, not the by-law: the solver prices all three families and
        # picks the most profitable governing envelope, so zero lots of a
        # class means it never won a single storey at current rents and
        # costs. Distinguished from "0 m² of headroom", which would mean the
        # class was built out.
        if not int(totals.get(modelled_key) or 0):
            return (
                "none proposed (no lot's most profitable programme includes "
                "this class at current rents and construction costs)"
            )
        value = float(totals.get(key) or 0)
        return f"{value:,.0f} m² ({value * 10.7639:,.0f} sq ft)"

    net = totals.get("net_floor_area_gap_m2")
    net_sentence = (
        f" The signed net across assessed lots is {float(net):,.0f} m²."
        if net is not None else ""
    )
    gain = totals.get("redevelopment_npv_gain_cad")
    gain_sentence = ""
    if gain is not None and float(gain) > 0:
        gain_sentence = (
            f" On the developer's arithmetic, redeveloping the "
            f"{int(totals.get('num_npv_gain_positive') or 0):,} lots where it "
            f"beats holding is worth ${float(gain) / 1e6:,.0f}M of discounted "
            f"net gain in total (land excluded)."
        )
    return (
        f"Across {int(totals['num_lots']):,} lots "
        f"({int(totals['num_solved']):,} with a solved programme, "
        f"{int(totals['num_underbuilt']):,} under-built), the additional floor "
        f"area that current zoning would allow is: residential "
        f"{area('residential_headroom_m2', 'num_with_residential')}; commercial "
        f"{area('commercial_headroom_m2', 'num_with_commercial')}; industrial "
        f"{area('industrial_headroom_m2', 'num_with_industrial')}. "
        f"That is {int(totals.get('additional_dwellings') or 0):,} additional "
        f"dwellings. These are positive-headroom totals: "
        f"{int(totals.get('num_over_built') or 0):,} lots already hold more "
        f"floor than today's grid permits and contribute zero rather than a "
        f"negative, and {int(totals.get('num_without_assessment') or 0):,} lots "
        f"have no assessment so their whole envelope counts as headroom."
        + net_sentence
        + gain_sentence
    )


@tool
def top_redevelopment_lots(limit: int = 10) -> str:
    """List the lots where redeveloping beats holding by the most money.

    This is the tool for "where should a developer look", "best redevelopment
    opportunities", "which lots are worth rebuilding". It ranks lots by
    redevelopment_npv_gain_cad — the discounted value of building the lot's
    highest and best use minus the discounted value of keeping the standing
    building, land excluded since the owner holds it either way.

    Args:
        limit: How many lots to list, largest gain first (default 10).

    Returns:
        One line per lot: the gain, what kind of building the programme is,
        its size, and what stands there today.
    """
    _require("redevelopment_gap")
    rows = queries.top_npv_gain_lots(limit=max(1, min(int(limit or 10), 50)))
    if not rows:
        return (
            "No lot shows a positive redevelopment gain in this snapshot — "
            "either the discounted-profit columns have not been materialized "
            "yet, or at current rents and costs holding beats rebuilding "
            "everywhere."
        )
    lines = [
        "Lots where redeveloping beats holding, by discounted net gain "
        "(land excluded):"
    ]
    for row in rows:
        use = (row.get("hbu_dominant_use") or "?").replace("_", " ")
        floors = row.get("floors")
        dwellings = row.get("hbu_num_dwellings")
        shape = ", ".join(
            part
            for part in (
                f"{int(floors)} storeys" if floors else "",
                f"{int(dwellings)} dwellings" if dwellings else "",
            )
            if part
        )
        lines.append(
            f"Lot {row.get('lot_number') or '?'} "
            f"({float(row.get('lot_area_m2') or 0):,.0f} m²): "
            f"+${float(row.get('redevelopment_npv_gain_cad') or 0):,.0f} — "
            f"{use} programme"
            + (f" ({shape})" if shape else "")
            + f"; today {int(row.get('existing_num_dwellings') or 0)} dwelling(s)."
        )
    lines.append(
        "Ranked by money, not by room — development_capacity totals the room. "
        "The biggest gains sit on the biggest parcels, some of them park or "
        "rail-yard scale; check lot area before treating one as a site."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Zoning
# ---------------------------------------------------------------------------


@tool
def zoning_for_lot(lot_number: str = "") -> str:
    """Read the zoning grid that applies to a lot.

    This is the tool for "what can be built here", "how tall", "what uses are
    allowed", "what is the lot coverage". It returns the values off the zoning
    grid (the borough's grille des spécifications) for the zone covering the
    lot, and puts the grid PDF itself in the Lot pane.

    A lot on a zone boundary is covered by more than one zone; they are
    reported in order of how much of the lot each covers, and the first is
    almost always the one meant. One entry per zone, and only zones that
    actually cover the lot - a clip of a square metre or less, or of under one
    per cent of the parcel, is the cadastre and the zoning layer disagreeing,
    and is not reported at all.

    Args:
        lot_number: The lot to look up. Leave empty to use the selected lot.

    Returns:
        The zone code, the grid values, and the URL of the grid PDF.
    """
    _require("features")
    lot_number = lot_number or (state.get_selected_lot().get("lot_number") or "")
    if not lot_number:
        raise ToolException(
            "No lot given and none selected. Ask the user to click a lot or name one."
        )

    lot = queries.lot_by_number(lot_number)
    if not lot:
        raise ToolException(f"No lot numbered {lot_number!r} in the loaded snapshots.")

    # The lot's own snapshot, not "whichever dates are loaded": the lot row
    # above came from one load of the cadastre and the zones that govern it
    # are that load's. Asking across every date returns the same zone once per
    # date, which reads as a lot straddling zones it does not.
    zones = queries.zoning_for_lot(
        lot["lot_number"], scrape_date=lot.get("scrape_date")
    )
    if not zones:
        return (
            f"No zoning polygon covers lot {lot['lot_number']} in the loaded "
            f"snapshot. Either the zoning layer is not loaded for this "
            f"borough, or every zone touching this lot clips it by under "
            f"{queries.MIN_ZONE_OVERLAP_M2:g} m² or under "
            f"{queries.MIN_ZONE_PCT_OF_LOT:g}% of its area, which is a survey "
            f"artefact rather than a zone that governs it."
        )

    state.set_selected_lot(
        lot["lot_number"], lot.get("lon"), lot.get("lat"), lot.get("neighborhood")
    )
    state.request_map(select_lot=lot["lot_number"], note=f"Zoning for lot {lot['lot_number']}")

    parts = [f"Lot {lot['lot_number']} — {_fmt_area(lot.get('area_m2'))}"]
    for zone in zones[:3]:
        attributes = zone.get("attributes") or {}
        share = ""
        if zone.get("overlap_m2") and zone.get("lot_area_m2"):
            share = f" (covers {zone['overlap_m2'] / zone['lot_area_m2'] * 100:.0f}% of the lot)"
        values = [
            f"{label}: {attributes[key]}"
            for key, label in queries.ZONING_FIELDS
            if str(attributes.get(key, "")).strip()
        ]
        url = zone.get("zoning_pdf_url") or ""
        parts.append(
            f"\nZone {zone['zone']}{share}\n  "
            + "\n  ".join(values or ["(the grid carries no values in this snapshot)"])
            + (f"\n  Grid PDF: {url}" if url else "")
        )
    if len(zones) > 3:
        parts.append(f"\n(+{len(zones) - 3} more zones touch this lot)")
    return "\n".join(parts)


@tool
def read_zoning_grid(lot_number: str = "") -> str:
    """Read the full text of the zoning grid PDF for a lot.

    Use this only when zoning_for_lot's structured values do not answer the
    question — a footnote, a conditional use, a note in the margin. It
    downloads the zoning grid PDF and returns its text layer, which is longer
    and noisier than the attributes.

    Args:
        lot_number: The lot whose grid to read. Leave empty for the selected lot.

    Returns:
        The grid's text, truncated if very long.
    """
    from src.utils import documents  # noqa: PLC0415

    _require("features")
    lot_number = lot_number or (state.get_selected_lot().get("lot_number") or "")
    if not lot_number:
        raise ToolException("No lot given and none selected.")

    lot = queries.lot_by_number(lot_number)
    if not lot:
        raise ToolException(f"No lot numbered {lot_number!r}.")

    zones = queries.zoning_for_lot(
        lot["lot_number"], scrape_date=lot.get("scrape_date")
    )
    url = next((z.get("zoning_pdf_url") for z in zones if z.get("zoning_pdf_url")), None)
    if not url and zones:
        url = queries.zoning_pdf_url_fallback(zones[0]["zone"])
    if not url:
        raise ToolException(
            f"No grid PDF is linked from the zoning covering lot {lot['lot_number']}."
        )

    try:
        document = documents.fetch(url)
        text = documents.extract_text(document.content)
    except documents.DocumentError as exc:
        raise ToolException(f"Could not read the grid: {exc}") from exc

    limit = 6000
    body = text[:limit] + ("\n…(truncated)" if len(text) > limit else "")
    return f"Zoning grid, zone {zones[0]['zone']} ({url}):\n\n{body}"


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


@tool
def data_status() -> str:
    """Report what is loaded in the database: boroughs, snapshots, corpus size.

    Call this when a tool says something is missing, or when the user asks what
    data is available, which neighbourhoods are covered, or how current it is.

    Returns:
        A summary of the loaded tables and partitions.
    """
    caps = queries.capabilities()
    present = [
        name
        for name in ("lots", "buildings", "features", "chunks")
        if getattr(caps, name)
    ]
    lines = [f"Present: {', '.join(present) or 'nothing'}"]
    # Required only: the silver joins missing means an answer is computed the
    # slow way, not that it is unavailable, and the model would relay it to the
    # user as a gap in the data either way.
    missing = caps.missing(include_advisory=False)
    if missing:
        lines.append(f"Missing: {', '.join(missing)}")

    if caps.lots:
        for table in ("lots", "buildings", "features"):
            if not getattr(caps, table):
                continue
            hoods = queries.neighborhoods(table)
            dates = queries.scrape_dates(table)
            lines.append(
                f"{table}: {', '.join(hoods) or 'no rows'} · snapshots "
                f"{', '.join(str(d) for d in dates[:3]) or 'none'}"
            )
    if caps.chunks:
        for row in queries.corpus_status()[:5]:
            lines.append(
                f"corpus {row.get('neighborhood')} {row.get('scrape_date')}: "
                f"{row.get('documents')} document(s), {row.get('chunks')} chunk(s)"
            )
    return "\n".join(lines)


PARCEL_TOOLS = [
    find_lot,
    describe_selected_lot,
    list_lots,
    buildings_on_lot,
    lot_efficiency,
    development_capacity,
    top_redevelopment_lots,
    top_site_opportunities,
    lot_futures,
    zoning_for_lot,
    read_zoning_grid,
    data_status,
]
