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
        rows = queries.buildings_on_lot(lot["lot_number"])
        if rows:
            total = sum(float(r.get("overlap_m2") or 0) for r in rows)
            buildings = f" {len(rows)} building footprint(s) on it, {total:,.0f} m² covered."

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
        Footprint count, each footprint's area, and how much of the lot they
        cover — which is the measured counterpart to the zoning grid's
        permitted taux d'implantation.
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

    rows = queries.buildings_on_lot(lot["lot_number"])
    if not rows:
        return f"Lot {lot['lot_number']} has no building footprint on it — it reads as vacant."

    covered = sum(float(r.get("overlap_m2") or 0) for r in rows)
    lot_area = float(lot.get("area_m2") or 0)
    ratio = f", {covered / lot_area * 100:.0f}% of the lot" if lot_area else ""
    each = "; ".join(
        f"{_fmt_area(r.get('area_m2'))} footprint ({_fmt_area(r.get('overlap_m2'))} on this lot)"
        for r in rows[:10]
    )
    return (
        f"Lot {lot['lot_number']}: {len(rows)} footprint(s) covering "
        f"{covered:,.0f} m²{ratio}. {each}"
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
                "pure équipements collectifs zone"
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
            "infeasible": "no governing column has a feasible programme",
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
    parts = [f"Lot {lot['lot_number']} ({_fmt_area(row.get('lot_area_m2'))})."]

    if used is None:
        parts.append("No utilisation share could be computed.")
    elif float(used) > 100:
        parts.append(
            f"{float(used):,.0f}% of what the grid permits is already standing "
            f"({built:,.0f} m² against {permitted:,.0f} m² permitted) — more "
            f"floor than today's zoning would allow, i.e. a legal "
            f"non-conformity rather than headroom."
        )
    else:
        verdict = "effectively built out" if float(used) >= 95 else "under-built"
        parts.append(
            f"{float(used):,.0f}% of permitted floor area is in use "
            f"({built:,.0f} m² standing against {permitted:,.0f} m² permitted) "
            f"— {verdict}."
        )

    if not row.get("has_assessment"):
        parts.append(
            "The assessment roll has no unit on this lot, so the standing "
            "figure is read as nothing built."
        )

    extras = []
    for label, key in (
        ("residential", "residential_headroom_m2"),
        ("commercial", "commercial_headroom_m2"),
        ("industrial", "industrial_headroom_m2"),
    ):
        value = float(row.get(key) or 0)
        if value > 0:
            extras.append(f"{label} {value:,.0f} m² ({value * 10.7639:,.0f} sq ft)")
    parts.append(
        ("Additional floor area that fits: " + "; ".join(extras) + ".")
        if extras else "No additional floor area fits under this grid."
    )

    hbu_d, existing_d = row.get("hbu_num_dwellings"), row.get("existing_num_dwellings")
    if hbu_d is not None:
        parts.append(
            f"Dwellings: {int(existing_d or 0)} today, {int(hbu_d)} proposed."
        )

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

    fit = row.get("footprint_fit_pct")
    if fit is not None and float(fit) < 99.5:
        parts.append(
            f"CAVEAT: the solved footprint only fits this parcel's shape at "
            f"{float(fit):.0f}%, so the floor areas above are overstated here. "
            f"Say so if you quote them."
        )
    return " ".join(parts)


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

    This is the tool for "what can be built here", "how tall", "what usages are
    allowed", "what is the taux d'implantation". It returns the values off the
    grille des spécifications for the zone covering the lot, and puts the grid
    PDF itself in the Lot pane.

    A lot on a zone boundary is covered by more than one zone; they are
    reported in order of how much of the lot each covers, and the first is
    almost always the one meant.

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

    zones = queries.zoning_for_lot(lot["lot_number"])
    if not zones:
        return (
            f"No zoning polygon covers lot {lot['lot_number']} in the loaded "
            f"snapshot. The zoning layer may not be loaded for this borough."
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
    question — a footnote, a conditional usage, a note in the margin. It
    downloads the grille des spécifications and returns its text layer, which
    is longer and noisier than the attributes.

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

    zones = queries.zoning_for_lot(lot["lot_number"])
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
    return f"Grille des spécifications, zone {zones[0]['zone']} ({url}):\n\n{body}"


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
    zoning_for_lot,
    read_zoning_grid,
    data_status,
]
