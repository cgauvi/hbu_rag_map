"""
data_tools.py — the tools that answer a question spanning more than one surface.

The agent's other tools each read one thing and render it as prose, which is
right for "what can I build on this lot" and hopeless for "which lots are zoned
for four storeys, assessed under $500k and not heritage-listed". That second
question needs the cadastre, the parsed grid, the assessment roll and the
heritage layers at once, and it is one read over ``gold.lot_dossier``.

**The model does not write SQL here.** It picks a filter name from this
module's signatures, a column name from ``dossier``'s registry, and supplies
numbers. The statement is composed in ``queries`` from validated identifiers
and bound parameters. That is the whole of the injection story: there is no
string the model controls that becomes SQL, so there is nothing to escape.

What that trades away is open-ended questions. A shape nobody anticipated is
not expressible, and the remedy is to add a filter here - reviewable, testable,
and a permanent capability - rather than to let a model improvise SQL.

The caveats travel with the numbers
-----------------------------------
``lot_efficiency`` refuses to report a share of the envelope when the roll
states no floor area, and warns on a split parcel and on waived parking. A bare
result table would hand the model those same numbers with the reasons to doubt
them removed, so every tool here runs ``queries.dossier_caveats`` over whatever
it returned and prints the sentences underneath. See ``dossier.footnotes``.
"""

from __future__ import annotations

import logging

from langchain.tools import tool
from langchain_core.tools import ToolException

from src.utils import dossier, neighborhoods, queries, state

logger = logging.getLogger(__name__)

#: Leaves room for the header and the footnotes inside the agent's own tool
#: output cap, which truncates from the end - where the footnotes are.
_OUTPUT_BUDGET = 24_000


def _require_dossier() -> None:
    if not queries.capabilities().redevelopment_gap:
        raise ToolException(
            "gold.lot_redevelopment_gap is not loaded in this database, and "
            "it is what the dossier is built on - so there is nothing to "
            "search. Tell the user that rather than retrying."
        )
    if not queries.dossier_loaded():
        raise ToolException(
            "gold.lot_dossier does not exist yet — it is created by "
            "hbu_infra's sql/034_gold_lot_dossier.sql, applied by `make "
            "db-init`. Until then use the per-lot tools (zoning_for_lot, "
            "lot_efficiency) one lot at a time. Do not retry this."
        )


def _scope(neighborhood: str = "") -> tuple[str, object]:
    """Which borough and which snapshot a search runs against.

    In order: what the caller named, the borough of the lot on the map, the
    borough the map is framed on, and - only when there is exactly one - the
    one that is loaded. Guessing between several would answer confidently
    about the wrong city.
    """
    code = (neighborhood or "").strip().upper()
    if not code:
        code = (state.get_selected_lot() or {}).get("neighborhood") or ""
    if not code:
        loaded = queries.neighborhoods()
        if len(loaded) == 1:
            code = loaded[0]
        elif loaded:
            raise ToolException(
                "Several boroughs are loaded and none is selected, so this "
                "would answer about the wrong one. Ask the user which, or "
                "have them click a lot. Loaded: "
                + ", ".join(neighborhoods.label(n) for n in loaded)
                + "."
            )
        else:
            raise ToolException("No borough is loaded in this database.")

    if code not in neighborhoods.NEIGHBORHOODS:
        raise ToolException(
            f"{code!r} is not a place code this app knows. One of: "
            + ", ".join(sorted(neighborhoods.NEIGHBORHOODS))
            + "."
        )

    scrape_date = queries.latest_scrape_date(table="lots", neighborhood=code)
    if scrape_date is None:
        raise ToolException(
            f"{neighborhoods.label(code)} has no loaded snapshot to search."
        )
    return code, scrape_date


def _with_caveats(
    rows: list[dict], columns: list[str], *, header: str, truncated: bool,
    code: str, scrape_date
) -> str:
    """A result table, under its header, over the warnings it owes."""
    table = dossier.render(rows, columns, header=header, truncated=truncated)
    if not rows:
        return table

    numbers = [str(r.get("lot_number") or "") for r in rows]
    notes = []
    try:
        counts = queries.dossier_caveats(
            numbers, neighborhood=code, scrape_date=scrape_date
        )
        notes = dossier.footnotes(counts)
    except Exception as exc:  # noqa: BLE001 - a missing warning must not lose the answer
        logger.warning("Could not read the caveats: %s", exc)
        notes = [
            "⚠ The usual checks on these figures could not be run, so the "
            "caveats below them are missing. Say so before quoting a number."
        ]

    # A lot appearing twice is the grain speaking: two zone pieces, two
    # answers. Said as data rather than left to a prompt rule, which a model
    # follows far less reliably.
    seen = [n for n in numbers if n]
    if len(set(seen)) < len(seen):
        notes.append(
            "⚠ Some lot numbers appear more than once: those parcels are "
            "crossed by a zoning boundary and each row is one PIECE of one. "
            "Add only_primary_zone for one row per lot."
        )
    if not queries.heritage_layers_loaded(code, scrape_date):
        notes.append(
            f"ℹ Heritage layers are not loaded for "
            f"{neighborhoods.name(code)}, so 'no heritage' here means "
            f"'not known', not 'not designated'."
        )
    return dossier.fit(header, table, notes, limit=_OUTPUT_BUDGET)


@tool
def describe_data(topic: str = "overview") -> str:
    """The columns available for searching sites, by topic.

    Call this once before your first find_sites or summarize_sites of a turn,
    with the topic nearest the question. Do not guess a column name.

    Args:
        topic: One of overview, ground, today, money, capacity, zoning, flags.
            Anything else returns the overview.

    Returns:
        The columns of that topic, what each means, and the values the short
        ones take.
    """
    return dossier.describe(topic)


@tool
def find_sites(
    neighborhood: str = "",
    min_permitted_storeys: int = 0,
    max_assessed_value_cad: int = 0,
    min_assessed_value_cad: int = 0,
    min_floor_area_gap_m2: int = 0,
    min_piece_area_m2: int = 0,
    max_year_built: int = 0,
    use_permitted: str = "",
    site_thesis: str = "",
    exclude_heritage: bool = False,
    only_underbuilt: bool = False,
    only_primary_zone: bool = False,
    order_by: str = "floor_area_gap_m2",
    limit: int = 20,
) -> str:
    """Find the sites in a borough matching several conditions at once.

    THE tool for a question that combines surfaces — what the zone permits AND
    what the roll says it is worth AND how under-built it is AND whether it is
    heritage. One call, not one call per condition.

    Every number means "no condition" when left at 0, and every name means "no
    condition" when left empty. The result is one row per lot × zone piece.

    Args:
        neighborhood: Borough code. Empty uses the selected lot's borough.
        min_permitted_storeys: Storeys the solved programme reaches, at least.
        max_assessed_value_cad: The parcel's assessed value, at most.
        min_assessed_value_cad: The parcel's assessed value, at least.
        min_floor_area_gap_m2: Floor area the site is short of its permitted
            envelope, at least. The headroom.
        min_piece_area_m2: Ground area of the zone piece, at least.
        max_year_built: Construction year, at most — for "older than".
        use_permitted: Restrict to zones permitting one of residential,
            commercial, industrial.
        site_thesis: One of brownfield, teardown, infill, improvement.
        exclude_heritage: Drop sites a heritage layer covers.
        only_underbuilt: Only sites holding materially less than permitted.
        only_primary_zone: One row per lot rather than one per zone piece.
        order_by: Any column from describe_data. Largest first.
        limit: How many to return, capped at 50.

    Returns:
        A table of the matching sites, with the caveats that apply to them.
    """
    _require_dossier()
    code, scrape_date = _scope(neighborhood)

    filters = {
        "min_permitted_storeys": min_permitted_storeys or None,
        "max_assessed_value_cad": max_assessed_value_cad or None,
        "min_assessed_value_cad": min_assessed_value_cad or None,
        "min_floor_area_gap_m2": min_floor_area_gap_m2 or None,
        "min_piece_area_m2": min_piece_area_m2 or None,
        "max_year_built": max_year_built or None,
    }
    flags = {
        "exclude_heritage": exclude_heritage,
        "only_underbuilt": only_underbuilt,
        "only_primary_zone": only_primary_zone,
    }
    columns = [
        "lot_number", "grid_zone", "hbu_floors", "parcel_assessed_value_cad",
        "floor_area_gap_m2", "existing_year_built", "site_thesis",
    ]

    try:
        rows, truncated = queries.find_sites(
            neighborhood=code,
            scrape_date=scrape_date,
            columns=columns,
            filters=filters,
            flags=flags,
            use_permitted=use_permitted,
            site_thesis=site_thesis,
            order_by=order_by,
            limit=limit,
        )
    except (ValueError, dossier.UnknownColumn) as exc:
        raise ToolException(str(exc)) from exc

    header = (
        f"Sites in {neighborhoods.label(code)}, snapshot {scrape_date}"
        f" — {len(rows)} shown."
    )
    if not rows:
        return _no_rows(code, scrape_date, filters, flags, header)
    return _with_caveats(
        rows, columns, header=header, truncated=truncated,
        code=code, scrape_date=scrape_date,
    )


def _no_rows(code, scrape_date, filters, flags, header: str) -> str:
    """Why nothing matched, by dropping one condition at a time.

    A model facing an empty result either gives up or invents one, and the
    answer it needs is almost always "this one condition is what emptied it".
    Cheap: one count per supplied condition, each already narrowed to a
    borough.
    """
    lines = [header, ""]
    try:
        total, _ = queries.find_sites(
            neighborhood=code, scrape_date=scrape_date, columns=["lot_number"],
            filters={}, flags={}, limit=1,
        )
        lines.append(
            "No site matched. The borough's dossier is "
            + ("not empty" if total else "empty")
            + "."
        )
        supplied = {k: v for k, v in filters.items() if v} | {
            k: v for k, v in flags.items() if v
        }
        for name in supplied:
            trimmed_f = {k: v for k, v in filters.items() if k != name}
            trimmed_g = {k: v for k, v in flags.items() if k != name}
            rows, more = queries.find_sites(
                neighborhood=code, scrape_date=scrape_date, columns=["lot_number"],
                filters=trimmed_f, flags=trimmed_g, limit=dossier.MAX_ROWS,
            )
            if rows:
                count = f"{len(rows)}+" if more else str(len(rows))
                lines.append(f"  Dropping {name} would return {count}.")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not become the error
        logger.warning("Could not diagnose the empty result: %s", exc)
        lines.append("No site matched.")
    lines.append("")
    lines.append(
        "Say that nothing matched and which condition is the binding one. Do "
        "not widen the question yourself without saying that you did."
    )
    return "\n".join(lines)


@tool
def site_dossier(lot_number: str = "") -> str:
    """Everything the data holds about one lot, every zone piece of it.

    Use this to answer several questions about one parcel at once, or when a
    question mixes what stands there with what the zone permits and what it is
    worth. For one of those alone the single-surface tools are shorter.

    Args:
        lot_number: The lot. Empty uses the one selected on the map.

    Returns:
        One row per zone piece, with every column, and the caveats on them.
    """
    _require_dossier()
    selected = state.get_selected_lot() or {}
    number = lot_number or (selected.get("lot_number") or "")
    if not number:
        raise ToolException(
            "No lot given and none selected. Ask the user to click one or "
            "name it."
        )
    code, scrape_date = _scope(selected.get("neighborhood") or "")

    rows = queries.site_dossier(number, neighborhood=code, scrape_date=scrape_date)
    if not rows:
        raise ToolException(
            f"No lot {number!r} in {neighborhoods.label(code)} at {scrape_date}."
        )

    # Every column is too wide to print; the renderer keeps the first few and
    # names the rest, which is what makes a follow-up question possible.
    columns = [c.name for c in dossier.DOSSIER_COLUMNS]
    header = (
        f"Lot {rows[0].get('lot_number')} in {neighborhoods.label(code)}, "
        f"snapshot {scrape_date} — {len(rows)} zone piece(s)."
    )
    return _with_caveats(
        rows, columns, header=header, truncated=False,
        code=code, scrape_date=scrape_date,
    )


@tool
def compare_sites(lot_numbers: str = "") -> str:
    """Put several lots side by side on the same measures.

    Args:
        lot_numbers: The lots, comma-separated — "2 170 935, 1 740 794".

    Returns:
        One row per lot (per zone piece where a lot has several), on the
        measures a comparison usually turns on.
    """
    _require_dossier()
    numbers = [n.strip() for n in (lot_numbers or "").replace(";", ",").split(",")]
    numbers = [n for n in numbers if n]
    if len(numbers) < 2:
        raise ToolException(
            "Name at least two lots, comma-separated. For one lot use "
            "site_dossier."
        )
    if len(numbers) > 8:
        raise ToolException(
            f"{len(numbers)} lots is more than a comparison can be read from. "
            f"Narrow it to at most 8."
        )

    code, scrape_date = _scope()
    columns = [
        "lot_number", "grid_zone", "piece_area_m2", "existing_floor_area_m2",
        "hbu_floor_area_m2", "floor_area_gap_m2", "parcel_assessed_value_cad",
        "hbu_floors", "best_future",
    ]
    rows = queries.sites_by_number(
        numbers, neighborhood=code, scrape_date=scrape_date, columns=columns
    )
    found = {str(r.get("lot_number") or "") for r in rows}
    header = (
        f"{len(found)} of {len(numbers)} lot(s) in "
        f"{neighborhoods.label(code)}, snapshot {scrape_date}."
    )
    if not rows:
        raise ToolException(
            f"None of those lots is in {neighborhoods.label(code)} at {scrape_date}."
        )
    return _with_caveats(
        rows, columns, header=header, truncated=False,
        code=code, scrape_date=scrape_date,
    )


@tool
def summarize_sites(
    group_by: str = "site_thesis",
    neighborhood: str = "",
    use_permitted: str = "",
    exclude_heritage: bool = False,
    only_underbuilt: bool = False,
    only_primary_zone: bool = False,
) -> str:
    """Count the borough's sites by group — the "how many" question.

    Args:
        group_by: site_thesis, investment_thesis, best_future, hbu_status,
            hbu_dominant_use, existing_dominant_use_description or grid_zone.
        neighborhood: Borough code. Empty uses the selected lot's borough.
        use_permitted: residential, commercial or industrial.
        exclude_heritage: Leave out sites a heritage layer covers.
        only_underbuilt: Count only sites holding less than permitted.
        only_primary_zone: Count lots rather than zone pieces.

    Returns:
        One row per group: how many sites, how many lots, and the totals.
    """
    _require_dossier()
    code, scrape_date = _scope(neighborhood)
    try:
        rows = queries.summarize_sites(
            neighborhood=code,
            scrape_date=scrape_date,
            group_by=group_by,
            filters={},
            flags={
                "exclude_heritage": exclude_heritage,
                "only_underbuilt": only_underbuilt,
                "only_primary_zone": only_primary_zone,
            },
        )
    except (ValueError, dossier.UnknownColumn) as exc:
        raise ToolException(str(exc)) from exc

    if not rows:
        return (
            f"No sites in {neighborhoods.label(code)} at {scrape_date} to "
            f"count."
        )

    lines = [
        f"Sites in {neighborhoods.label(code)}, snapshot {scrape_date}, "
        f"by {group_by}:",
        "",
        "group | sites | lots | floor gap m² | dwelling gap | NPV gain",
        "--- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("group_value") or "—"),
                    f"{int(row.get('num_sites') or 0):,}",
                    f"{int(row.get('num_lots') or 0):,}",
                    f"{float(row.get('total_floor_area_gap_m2') or 0):,.0f}",
                    f"{int(row.get('total_dwelling_gap') or 0):,}",
                    f"${float(row.get('total_npv_gain_cad') or 0):,.0f}",
                ]
            )
        )
    lines.append("")
    lines.append(
        "A site is a lot × zone piece unless only_primary_zone was set, so "
        "sites and lots differ where a zoning boundary crosses a parcel."
    )
    return "\n".join(lines)


DATA_TOOLS = [
    describe_data,
    find_sites,
    site_dossier,
    compare_sites,
    summarize_sites,
]
