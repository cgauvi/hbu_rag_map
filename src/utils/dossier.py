"""
dossier.py — what ``gold.lot_dossier`` holds, and how an answer off it reads.

One registry, four consumers:

* the columns a generated query is allowed to name or order by,
* the text ``describe_data`` hands the model before it asks for any,
* how a value is formatted in a result table,
* the suggestions offered when a name is not recognised.

Keeping them in one list is what stops them drifting. The same arrangement
already exists one layer down in ``queries.ZONING_GRID_COLUMN_FIELDS``, which
pairs each parsed-grid column with the English reading of its French term.

Why a registry and not a schema read
------------------------------------
The model never writes SQL here. It names a column, and the name is checked by
exact membership against this tuple before it is composed into a statement with
``psycopg.sql.Identifier``; everything else it supplies is a bound parameter.
Reading the column list out of ``information_schema`` instead would make the
allowlist whatever the database happens to hold, which is the opposite of an
allowlist - and would silently widen the moment someone adds a column.

The caveats are part of the data
--------------------------------
``lot_efficiency`` refuses to report a share of the envelope when the roll
states no floor area, warns when a zoning boundary splits the parcel, and warns
when a programme only exists with its parking waived. A result table that
dropped all that would be handing the model the same numbers with the reasons
to doubt them stripped off. So the view carries the flags, and ``footnotes``
turns them back into the sentences the prose tools would have written.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

#: The view this describes. One place, because it is also what a caller's
#: FROM clause is built from.
RELATION = "gold.lot_dossier"

#: Topics `describe_data` answers on. Seven, because a model picks reliably
#: from a vocabulary that size and unreliably from twenty.
TOPICS: tuple[str, ...] = (
    "overview",
    "ground",
    "today",
    "money",
    "capacity",
    "zoning",
    "flags",
)


@dataclass(frozen=True)
class Column:
    """One column of the dossier, as the model should understand it."""

    name: str
    topic: str
    #: text | int | float | money | area | pct | bool | date — drives rendering
    #: only; the database's own type is what it is.
    kind: str
    note: str
    #: Enumerated values, where the column has few enough to list. Worth more
    #: than the type is: "site_thesis" is unguessable, its five values are not.
    values: str = ""


DOSSIER_COLUMNS: tuple[Column, ...] = (
    # --- identity and grain ------------------------------------------------
    Column("lot_number", "overview", "text",
           "The cadastral lot, printed with spaces. Always ask for this."),
    Column("lot_number_digits", "overview", "text",
           "The same number with the spaces removed, for matching what a user typed."),
    Column("feature_id", "overview", "text",
           "The zone piece. A lot in two zones has two rows, one per feature_id."),
    Column("grid_zone", "overview", "text",
           "The zone's own code, as the by-law prints it."),
    Column("is_primary_zone", "overview", "bool",
           "The largest piece of this lot. Filter on it for one row per lot."),
    Column("num_lot_zones", "overview", "int",
           "How many zones cross this lot. Above 1 it is two development sites."),
    Column("neighborhood", "overview", "text",
           "Borough code. Already filtered; never needs restating."),
    Column("scrape_date", "overview", "date",
           "The snapshot. Already filtered."),

    # --- the ground ---------------------------------------------------------
    Column("parcel_area_m2", "ground", "area",
           "The WHOLE parcel. Repeated on every piece of it, so never summed."),
    Column("piece_area_m2", "ground", "area",
           "This zone piece's ground. The site's own area; sum this, not the parcel."),
    Column("piece_pct_of_parcel", "ground", "pct",
           "How much of the parcel this piece is."),
    Column("primary_frontage_m", "ground", "float",
           "Longest street frontage, in metres. What a grid's minimum width is read against."),
    Column("buildable_area_m2", "ground", "area",
           "Ground left inside the setbacks - what a building may actually occupy."),
    Column("parcel_num_buildings", "ground", "int", "Footprints standing on the parcel."),
    Column("parcel_built_pct", "ground", "pct", "Share of the parcel already covered."),

    # --- what stands today --------------------------------------------------
    Column("existing_floor_area_m2", "today", "area",
           "Floor area standing, every storey added up. NULL means the roll "
           "does not say - see floor_area_unreported. Not the footprint."),
    Column("existing_num_dwellings", "today", "int", "Dwellings on the roll."),
    Column("existing_num_storeys", "today", "int", "Storeys standing."),
    Column("existing_year_built", "today", "year", "Construction year off the roll."),
    Column("existing_num_assessment_units", "today", "int",
           "Assessment units the roll reached here. Zero means it reached none."),
    Column("existing_dominant_use_code", "today", "text",
           "CUBF use code of the dominant premises."),
    Column("existing_dominant_use_description", "today", "text",
           "That code in words."),

    # --- what it is worth ----------------------------------------------------
    Column("piece_assessed_value_cad", "money", "money",
           "Assessed value attributed to this piece."),
    Column("parcel_assessed_value_cad", "money", "money",
           "Assessed value of the WHOLE parcel. This is what 'assessed at' "
           "means in a question; repeated per piece, so never summed."),
    Column("parcel_roll_year", "money", "year", "Roll year the value is from."),
    Column("parcel_estimated_value_cad", "money", "money",
           "Value estimated from comparable sales, not the roll."),
    Column("parcel_cap_rate_pct", "money", "pct", "Capitalisation rate used for it."),
    Column("parcel_assessed_to_estimated_ratio", "money", "float",
           "Assessed over estimated. Below 1 the roll is behind the market."),

    # --- what the zone permits -----------------------------------------------
    Column("permits_residential", "zoning", "bool", "The zone authorises housing."),
    Column("permits_commercial", "zoning", "bool", "The zone authorises commerce."),
    Column("permits_industrial", "zoning", "bool", "The zone authorises industry."),
    Column("grid_floors_min", "zoning", "int", "Storeys the grid requires at least."),
    Column("grid_floors_max", "zoning", "int",
           "Storeys the grid allows at most. NULL where the grid states none - "
           "which is most of Quebec City - not zero storeys."),
    Column("grid_height_max_m", "zoning", "float", "Height ceiling in metres."),
    Column("grid_site_coverage_max_pct", "zoning", "pct",
           "Taux d'implantation maximal: ground the building may cover."),
    Column("grid_density_max", "zoning", "float", "COS: floor area ratio ceiling."),
    Column("grid_usage_summary", "zoning", "text",
           "The use classes the grid lists, as it writes them."),
    Column("grid_url", "zoning", "text", "The grille des specifications PDF."),

    # --- what could stand ------------------------------------------------------
    Column("hbu_status", "capacity", "text",
           "Whether a programme was solved, and why not when it was not.",
           "solved, infeasible, single_family_zone, no_candidate_column, "
           "no_governing_column, solver_error"),
    Column("hbu_dominant_use", "capacity", "text", "What the solved programme mostly is."),
    Column("hbu_floors", "capacity", "int", "Storeys in the solved programme."),
    Column("hbu_height_m", "capacity", "float", "Its height in metres."),
    Column("hbu_footprint_m2", "capacity", "area", "Ground it would cover."),
    Column("hbu_floor_area_m2", "capacity", "area", "Floor area it would hold."),
    Column("hbu_num_dwellings", "capacity", "int", "Dwellings it would hold."),
    Column("hbu_total_capital_cost_cad", "capacity", "money", "What building it would cost."),
    Column("is_underbuilt", "capacity", "bool",
           "The solved programme holds materially more floor than what stands."),
    Column("floor_area_gap_m2", "capacity", "area",
           "Permitted floor minus standing floor. The headroom."),
    Column("dwelling_gap", "capacity", "int", "Dwellings the gap is worth."),
    Column("storey_headroom", "capacity", "int", "Storeys that could be added."),

    # --- what the difference is worth -------------------------------------------
    Column("hbu_npv_cad", "money", "money", "Net present value of rebuilding."),
    Column("existing_present_value_cad", "money", "money", "Present value of what stands."),
    Column("redevelopment_npv_gain_cad", "money", "money",
           "The difference: what rebuilding is worth over keeping."),
    Column("best_future", "money", "text",
           "Which future pays best for the owner.", "hold, enhance, rebuild"),

    # --- the thesis, and what blocks it -------------------------------------------
    Column("investment_thesis", "capacity", "text",
           "What the solved programme is, as a thesis.",
           "residential, mixed_use, commercial, industrial, none"),
    Column("site_thesis", "capacity", "text",
           "Why the site is acquirable at all.",
           "brownfield, teardown, infill, improvement, none"),
    Column("site_thesis_rank", "capacity", "int", "Rank within that thesis, 1 is best."),
    Column("site_irr_pct", "money", "pct", "Internal rate of return for the site thesis."),
    Column("is_good_candidate", "capacity", "bool",
           "Clears the cap-rate spread or the IRR hurdle, and pays. Rare."),
    Column("is_heritage_sector", "flags", "bool", "Inside a heritage sector."),
    Column("has_piia_review", "flags", "bool",
           "A PIIA applies: discretionary architectural review before a permit."),
    Column("demolition_review_required", "flags", "bool", "Demolition needs a review."),
    Column("parcel_has_heritage", "flags", "bool",
           "A heritage layer covers this parcel. FALSE where the layers are not "
           "loaded for the borough, which is not the same as 'not heritage'."),
    Column("parcel_heritage_layers", "flags", "text", "Which layers, by name."),

    # --- flags that decide how a number may be reported ----------------------------
    Column("floor_area_unreported", "flags", "bool",
           "TRUE where the roll has a unit here and states NO floor area. The "
           "share of the envelope in use is NOT KNOWN. Never report it as 0%, "
           "as under-built, or as vacant."),
    Column("nothing_assessed", "flags", "bool",
           "TRUE where the roll reached no unit at all - vacant ground, a lane, "
           "a park. Here the floor area genuinely is nothing."),
    Column("hbu_parking_waived", "flags", "bool",
           "The programme exists only with its parking requirement waived."),
    Column("grid_parsed", "flags", "bool",
           "FALSE where the parsed grid does not cover this zone, so every "
           "grid_* column is NULL for a missing source rather than for a zone "
           "that permits nothing."),
)

BY_NAME: dict[str, Column] = {c.name: c for c in DOSSIER_COLUMNS}
COLUMN_NAMES: frozenset[str] = frozenset(BY_NAME)

#: `gold.lot_dossier` also selects `cell_partition`, which is deliberately NOT
#: in the registry above. It is the tile the row is partitioned under: useful
#: to this module when it prunes, meaningless to a reader, and not something
#: the model should be able to name or sort by. Absent here means refused
#: there - which is the point, not an omission to fix.

#: What a caller gets when it names no columns: enough to recognise a site.
SPINE: tuple[str, ...] = (
    "lot_number",
    "grid_zone",
    "piece_area_m2",
    "parcel_assessed_value_cad",
    "hbu_floors",
    "floor_area_gap_m2",
)

#: Result-table ceilings. Rows rather than characters, because the agent's
#: output cap truncates from the *end* - which is where the footnotes are, and
#: they are the part that must not be lost.
MAX_ROWS = 50
MAX_RENDER_COLUMNS = 12
_FOOTNOTE_RESERVE = 1_400


class UnknownColumn(ValueError):
    """A column name that is not in the registry, with the nearest matches."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.suggestions = difflib.get_close_matches(name, sorted(COLUMN_NAMES), n=3)
        hint = f" Did you mean {', '.join(self.suggestions)}?" if self.suggestions else ""
        super().__init__(
            f"{name!r} is not a column of {RELATION}.{hint} "
            f"Call describe_data for the list."
        )


def validated(name: str) -> str:
    """The column name, or raise. The only way a name reaches SQL."""
    cleaned = (name or "").strip()
    if cleaned not in COLUMN_NAMES:
        raise UnknownColumn(cleaned)
    return cleaned


def validated_many(names) -> list[str]:
    return [validated(n) for n in names]


# ---------------------------------------------------------------------------
# describe_data
# ---------------------------------------------------------------------------

_OVERVIEW = f"""\
{RELATION} — one row per lot x ZONE PIECE, not per lot.

Already filtered to one borough and one snapshot before you see it, so never
ask about neighborhood or scrape_date.

Three rules that decide whether an answer is right:
  1. A lot crossed by a zoning boundary has one row per piece. Add
     `only_primary_zone` for one row per lot.
  2. Columns named parcel_* are whole-parcel facts repeated on every piece.
     Never add them up.
  3. A NULL is not a zero. `existing_floor_area_m2` NULL means the roll does
     not state it; `grid_floors_max` NULL means the parsed grid does not cover
     the zone. Neither is "none".

Topics, for describe_data:
  ground    the land, its frontage and what is already built on it
  today     what stands there now, off the assessment roll
  money     assessed and estimated value, cap rate, what redevelopment is worth
  capacity  what could be built, and the gap against what is
  zoning    what the zone permits
  flags     heritage, review requirements, and the caveats on the numbers

Columns you will almost always want:
"""


def describe(topic: str = "overview") -> str:
    """The columns of one topic, for the model to read before it asks."""
    wanted = (topic or "overview").strip().lower()
    if wanted not in TOPICS:
        wanted = "overview"

    if wanted == "overview":
        lines = [_OVERVIEW]
        for name in SPINE:
            lines.append(f"  {name:<34} {BY_NAME[name].note}")
        return "\n".join(lines)

    columns = [c for c in DOSSIER_COLUMNS if c.topic == wanted]
    lines = [f"{RELATION} — {wanted}", ""]
    for column in columns:
        lines.append(f"  {column.name:<34} {column.kind:<6} {column.note}")
        if column.values:
            lines.append(f"  {'':<34} {'':<6} one of: {column.values}")
    lines.append("")
    lines.append(
        "Other topics: " + ", ".join(t for t in TOPICS if t != wanted) + "."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format(value: Any, kind: str) -> str:
    if value is None:
        # Distinguishable from a zero at a glance, which for this data is the
        # whole point - see rule 3 in the overview.
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        if kind in ("money",):
            return f"${float(value):,.0f}"
        if kind in ("area", "float"):
            return f"{float(value):,.0f}"
        if kind == "pct":
            return f"{float(value):.1f}%"
        if kind == "int":
            return f"{int(value):,}"
        if kind == "year":
            # No thousands separator: "1,962" reads as a quantity, and a model
            # quoting it writes "built in 1,962".
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return str(value)


def render(rows: list[dict], columns: list[str], *, header: str, truncated: bool) -> str:
    """A result set as a compact table, under the header that frames it."""
    if not rows:
        return f"{header}\n\nNo site matched."

    shown = columns[:MAX_RENDER_COLUMNS]
    dropped = columns[MAX_RENDER_COLUMNS:]

    lines = [header, ""]
    lines.append(" | ".join(shown))
    lines.append(" | ".join("---" for _ in shown))
    for row in rows:
        lines.append(
            " | ".join(
                _format(row.get(name), BY_NAME[name].kind if name in BY_NAME else "text")
                for name in shown
            )
        )
    if dropped:
        lines.append("")
        lines.append(f"({len(dropped)} more column(s) not shown: {', '.join(dropped)})")
    if truncated:
        lines.append("")
        lines.append(
            f"Only the first {len(rows)} are shown. Say so, and narrow the "
            f"question rather than implying this is all of them."
        )
    return "\n".join(lines)


#: Each flag, and the sentence a reader has to be given when any row trips it.
#: Worded as instructions because the reader is a model, and because the two
#: about floor area are the difference between an answer and a confident wrong
#: one. These restate what `lot_efficiency` says in prose; they are the same
#: rules, carried through a table instead of a paragraph.
_FOOTNOTES: tuple[tuple[str, str], ...] = (
    (
        "split_parcel",
        "⚠ A zoning boundary crosses the parcel in {n} of these rows "
        "(num_lot_zones > 1). Every area and money figure above is that "
        "PIECE's, not the parcel's. Say so, and never add two pieces together "
        "and call the total the lot.",
    ),
    (
        "floor_area_unreported",
        "⚠ For {n} of these the roll has an assessment unit and states NO "
        "floor area. How much of the envelope is in use is NOT KNOWN for them. "
        "Never report them as 0%, as under-built, or as vacant, and do not "
        "fill the figure in from the footprint.",
    ),
    (
        "nothing_assessed",
        "ℹ The roll reached no assessment unit at all in {n} of these — vacant "
        "ground, a lane, a park. There the existing floor area genuinely is "
        "nothing.",
    ),
    (
        "hbu_parking_waived",
        "⚠ The parking requirement was waived to make the programme fit in {n} "
        "of these. They stand on a variance; say so before quoting a storey or "
        "dwelling count.",
    ),
    (
        "unsolved",
        "⚠ No programme was solved for {n} of these, so their hbu_* and gap "
        "columns are empty. There is no permitted-floor figure to compare for "
        "them.",
    ),
    (
        "grid_unparsed",
        "ℹ The parsed grid does not cover {n} of these zones, so grid_* is "
        "NULL for a missing source rather than for a zone that permits "
        "nothing. zoning_for_lot has the polygon's own values.",
    ),
)


def footnotes(counts: dict[str, int]) -> list[str]:
    """The sentences owed to a result set, from one pass of counts over it."""
    return [
        text.format(n=counts[key])
        for key, text in _FOOTNOTES
        if counts.get(key)
    ]


def fit(header: str, table: str, notes: list[str], *, limit: int) -> str:
    """Header, table and footnotes within the agent's tool-output budget.

    Rows are shed, never the footnotes. The agent truncates a long tool result
    from the end, which is where the footnotes sit - so a table allowed to fill
    the budget would arrive with exactly the caveats that make it safe cut off.
    """
    tail = "\n\n".join(notes)
    room = max(limit - len(tail) - _FOOTNOTE_RESERVE, 1_000)
    lines = table.split("\n")
    while len("\n".join(lines)) > room and len(lines) > 4:
        lines.pop()
        lines[-1] = "… (rows dropped to leave room for the notes below)"
    body = "\n".join(lines)
    return f"{body}\n\n{tail}" if tail else body
