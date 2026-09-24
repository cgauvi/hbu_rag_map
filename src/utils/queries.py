"""
queries.py — Every statement this app sends, in one file.

Two reasons it is not spread across the tools that call it. The first is that
the schema belongs to two other repos — ``rag.features``, ``rag.lots``, the
``silver``/``gold`` tables and the search functions to `hbu_infra`,
``rag.chunks`` to `hbu_dataplatform` — so the set of assumptions this app makes
about their shape is worth being able to read in one sitting. The second is
that the map and the agent ask the *same* questions: clicking a lot and asking
"what applies here" both end at ``rag.search_at_lot``, and duplicating that in
two layers is how they drift.

**Two schemas, and the difference matters when reading a query below.**
``rag`` holds what the scrape loaded — the lots, the buildings, the map
features, the corpus — and is queried live. ``silver`` holds the joins the
pipeline has *already computed* between them, one table per asset, partitioned
by ``(neighborhood, scrape_date)``. Four of the reads here have a fast path off
a silver table and a fallback that computes the same thing with
``ST_Intersection``: `buildings_on_lot`, `zoning_for_lot`, `buildings_in_bbox`
and the ``buildings`` viewport read. The fallback is not dead code — a borough
loaded this morning has its ``rag`` rows before the silver assets have run over
them — so both paths have to keep returning the same column names, and
`capabilities()` is what chooses between them, except on the viewport read
where `_building_lots_available` memoises the one bit that matters.

**The map's geometry does not leave this file at all any more.** The nine
layers are PMTiles archives the dataplatform renders per partition - with the
``ST_AsMVT`` SQL that used to be here - and the browser reads off S3; see
`src/utils/tiles.py`. The ``*_in_bbox`` reads return the same nine layers as
GeoJSON for a whole viewport, capped at ``DEFAULT_FEATURE_LIMIT`` — still what
the agent's `find_lots_in_view` tool wants, and still what
``HBU_MAP_RENDERER=geojson`` selects, but not how the map is drawn: a tile is
bounded by its own area, and a cap was never the fix for a borough.

Either way the wire carries the vertices that get *drawn* rather than the
vertices Infolot recorded — a borough's worth of lots is several megabytes of
coordinates at full precision and a few hundred kilobytes at screen precision.
The viewport reads get there with ``ST_SimplifyPreserveTopology`` and a
tolerance in degrees; a tile gets there by quantising onto its own grid.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from src.utils.db import (
    GOLD_SCHEMA,
    SCHEMA,
    SILVER_SCHEMA,
    query,
    query_one,
    scalar,
)

logger = logging.getLogger(__name__)

#: The scraped layer whose rows carry a link to a zoning grid PDF, and the
#: attribute holding it. Mirrors ``DOCUMENT_SOURCES`` in the dataplatform's
#: rag/documents.py — the same registry, read from the other end. Values in
#: ``rag.features.source_table`` are the file *slug* the scrape writes, not the
#: Spectrum path, because that is what ``rag.chunks.source_table`` holds and
#: the two are joined on it.
ZONING_SOURCE_TABLE = os.environ.get(
    "HBU_ZONING_SOURCE_TABLE", "Reglement_urbanisme__VSP_REG_ZONE"
)
ZONING_URL_ATTRIBUTE = os.environ.get("HBU_ZONING_URL_ATTRIBUTE", "LIEN_GRILLE")

#: Every scraped layer that *is* zoning, one per city the database holds:
#: Montreal's zone table above, and Quebec City's 'Zonage en vigueur' layer,
#: which carries no PDF link on the polygon (see `ZONING_PDF_URL_TEMPLATES`).
#: Mirrors ``ZONING_SOURCES`` in the dataplatform's rag/documents.py. The zone
#: reads below match any of these, so a viewport over either city draws its
#: zones; `ZONING_SOURCE_TABLE` stays the one the corpus is scoped to.
ZONING_SOURCE_TABLES: tuple[str, ...] = tuple(
    name.strip()
    for name in os.environ.get(
        "HBU_ZONING_SOURCE_TABLES",
        f"{ZONING_SOURCE_TABLE},Zonage__ZONAGE_EN_VIGUEUR",
    ).split(",")
    if name.strip()
)

#: Where a city serves a zone's grid when its layer carries no link to one,
#: keyed by ``source_table`` and formatted with ``{zone}``.
#:
#: Montreal has no entry and needs none: its polygons carry a ``LIEN_GRILLE``
#: to a static PDF per zone, and a link the scrape *recorded* is better than
#: one this app constructs — it survives the city reorganising its URLs.
#:
#: Quebec City carries no such attribute, which read as "this city publishes
#: no grid" and is not true: it serves one per zone from a handler keyed on the
#: zone code, which is the id `rag.features.feature_id` already holds. So the
#: link is derivable where it is not recorded, and a derived link beats no
#: link at all. It is tried *after* the corpus, so a URL that was actually
#: indexed still wins.
#:
#: Overridable as ``slug=template`` pairs separated by commas — the slugs
#: carry no ``=`` and the templates no ``,``, and the split is on the first
#: ``=`` so a template with one of its own survives.
ZONING_PDF_URL_TEMPLATES: dict[str, str] = {
    slug.strip(): template.strip()
    for slug, _, template in (
        pair.partition("=")
        for pair in os.environ.get(
            "HBU_ZONING_PDF_URL_TEMPLATES",
            "Zonage__ZONAGE_EN_VIGUEUR="
            "https://carte.ville.quebec.qc.ca/GrillesZonage/HandlerZonage.ashx?{zone}",
        ).split(",")
    )
    if slug.strip() and template.strip()
}

#: The zoning-grid attributes worth showing next to a lot, in reading order,
#: with an English label for the French field the grid itself carries.
#: Anything not listed still reaches the agent through the raw attributes;
#: this is what the Lot pane renders.
#:
#: The labels are a translation and the *values* are not: a grid says
#: "H.1-3" or "isolée" whatever this column is called, and the PDF rendered
#: below the table is the French document. Two of these are terms of art the
#: by-law names rather than describes - taux d'implantation is the share of
#: the lot a building may cover, COS the ratio of floor area to lot area - so
#: they are labelled with the English planning term a reader can act on and
#: the French one they will find on the page.
ZONING_FIELDS: tuple[tuple[str, str], ...] = (
    ("NUMERO_COMPLET", "Zone"),
    ("USAGE", "Permitted uses"),
    ("USAGE_AUT", "Permitted uses (cont.)"),
    ("USAGE_EXC", "Excluded uses"),
    ("ETAGE_MIN", "Storeys min"),
    ("ETAGE_MAX", "Storeys max"),
    ("METRE_MIN", "Height min (m)"),
    ("METRE_MAX", "Height max (m)"),
    ("TAUX_IMP_MIN", "Lot coverage min (%) — taux d'implantation"),
    ("TAUX_IMP_MAX", "Lot coverage max (%) — taux d'implantation"),
    ("COS_MIN", "Floor area ratio min — COS"),
    ("COS_MAX", "Floor area ratio max — COS"),
    ("IMPLANTATION", "Siting"),
    ("RDC_COMMERCIAL", "Ground-floor commercial"),
    ("SECTEUR_PAT", "Heritage sector"),
    ("SECTEUR_PIIA", "Site-plan review (PIIA) sector"),
    ("CAT_AFFICHAGE", "Signage category"),
)

#: The attribute the zoning layer's own number lives in - the label the map
#: puts on a zone, and the id every grid join is keyed on. Read off
#: ``ZONING_FIELDS`` rather than written down again, so the Lot pane and the
#: map cannot disagree about which attribute that is.
ZONE_LABEL_ATTRIBUTE = ZONING_FIELDS[0][0]

#: The attributes carrying the *uses* a grid permits - the "H.1-3", "C.4",
#: "E.2" codes the by-law names a zone's programme with.
#:
#: Two of them, because the grid splits one list across a second column when it
#: runs long: a zone whose uses spill into ``USAGE_AUT`` permits more than its
#: first column says, rather than being a different kind of zone. Derived from
#: the same table the values are rendered from, for the same reason
#: `ZONE_LABEL_ATTRIBUTE` is - so a column renamed in one place cannot go on
#: being read in the other.
#:
#: ``USAGE_EXC`` is deliberately absent. That column is the grid's
#: *exclusions*, and folding it into the same list would report a use as
#: permitted that this zone specifically forbids.
ZONING_USE_ATTRIBUTES: tuple[str, ...] = tuple(
    key for key, label in ZONING_FIELDS if label.startswith("Permitted uses")
)

#: The parsed grid, column by column, as the Regulations pane renders it —
#: ``silver.zoning_grid_columns``' planning fields in the order a *grille des
#: spécifications* prints them, with an English label for each.
#:
#: This is the second reading of the same by-law and not a duplicate of
#: `ZONING_FIELDS`. That tuple is what one city's *map layer* happens to carry
#: on its polygons; this one is what the dataplatform parsed off the sheet
#: itself. They come apart in both directions and that is the point of having
#: both: Montreal publishes ``SECTEUR_PAT``/``SECTEUR_PIIA`` on the polygon and
#: nowhere else, while Quebec City publishes *nothing* on the polygon — its
#: layer carries ``NATURE``, ``STATUT`` and the shape's own measurements — and
#: states every norm in a workbook, which is what this table holds.
#:
#: One *column* is one programme the zone permits, not one zone: a mixed zone
#: prints a residential column beside a commercial one, each with its own uses
#: and its own storey range, and collapsing them would report a height against
#: a use that may not reach it.
#:
#: The margins are last for the same reason the grid prints them last: they
#: decide where a building sits once its use, height and coverage are settled.
ZONING_GRID_COLUMN_FIELDS: tuple[tuple[str, str], ...] = (
    ("usage_habitation", "Housing uses — habitation"),
    ("usage_commerce", "Commercial uses — commerce"),
    ("usage_industrie", "Industrial uses — industrie"),
    ("usage_equipements", "Institutional uses — équipements"),
    ("only_permitted_usages", "Only these uses — usages permis"),
    ("excluded_usages", "Excluded uses — usages exclus"),
    ("floors_min", "Storeys min"),
    ("floors_max", "Storeys max"),
    ("height_min_m", "Height min (m)"),
    ("height_max_m", "Height max (m)"),
    ("site_coverage_min_pct", "Lot coverage min (%) — taux d'implantation"),
    ("site_coverage_max_pct", "Lot coverage max (%) — taux d'implantation"),
    ("density_min", "Floor area ratio min — COS"),
    ("density_max", "Floor area ratio max — COS"),
    # Quebec City's *Normes de densité*, and a different quantity from the two
    # above it: a dwelling count per hectare of lot rather than a floor-area
    # ratio. Labelled in the by-law's own units so the two cannot be read as
    # alternatives — a zone states 65 log/ha and no COS at all.
    ("dwelling_density_min_per_ha", "Dwellings per hectare min — log/ha"),
    ("dwelling_density_max_per_ha", "Dwellings per hectare max — log/ha"),
    ("max_dwellings", "Dwellings max"),
    ("specific_use_area_max_m2", "Floor area max for the use (m²)"),
    # The tighter of the grid's retail and administration ceilings, per
    # building — see `quebec._commercial_floor_cap` in the dataplatform.
    ("commercial_floor_max_m2", "Commercial floor area max per building (m²)"),
    ("implantation_mode", "Siting — mode d'implantation"),
    ("min_lot_width_m", "Lot width min (m)"),
    ("front_margin_min_m", "Front margin min (m)"),
    ("front_margin_max_m", "Front margin max (m)"),
    ("secondary_front_margin_min_m", "Secondary front margin min (m)"),
    ("secondary_front_margin_max_m", "Secondary front margin max (m)"),
    ("side_margin_min_m", "Side margin min (m)"),
    ("rear_margin_min_m", "Rear margin min (m)"),
)

#: The four ``usage_*`` columns above, which together are what a grid column
#: *permits*. Derived from the rendered tuple for the same reason
#: `ZONING_USE_ATTRIBUTES` is: a column renamed in one place cannot go on being
#: read in the other. ``only_permitted_usages`` and ``excluded_usages`` are
#: deliberately absent — the first is a reference into the borough's own usage
#: numbering rather than a list of codes, and the second is the grid's
#: *exclusions*, which folding in would report as permitted.
ZONING_GRID_USE_COLUMNS: tuple[str, ...] = tuple(
    key for key, _ in ZONING_GRID_COLUMN_FIELDS if key.startswith("usage_")
)

#: What separates one use code from the next inside a use column. A semicolon
#: is what the Montreal scrape stores — "C.4;H" is one zone permitting two
#: things — and the comma is both what a handful of those rows use instead and
#: what the Quebec City workbook uses throughout ("C1, C2, C3").
#:
#: Neither is a delimiter this app chose; both are the by-law's punctuation as
#: the scrape found it. Here rather than beside the one renderer that used to
#: split on it, because the same codes now arrive from two tables — the
#: polygon's attributes and the parsed grid — and one list of separators is
#: what keeps a zone's uses reading the same whichever of them answered.
USE_CODE_SEPARATORS = re.compile(r"[;,]")

#: How much of a lot a zone has to actually cover, in square metres, before
#: that zone is reported as covering it.
#:
#: A cadastral boundary and a zoning boundary are drawn by two publishers from
#: two surveys, so they miss each other by centimetres all along a street and
#: every lot clips a corner of its neighbour's zone. Those clips are real
#: polygons with real area - `silver.lot_features` records them, deliberately
#: and without a threshold, because the cutoff belongs to the question being
#: asked rather than to the geometry (see 005_silver_lot_features.sql). This is
#: that cutoff, for the question the Lot pane asks: *which zones govern this
#: lot*. A square metre of a zone does not, and offering it beside the real one
#: as though the reader had a choice to make is what this number prevents.
#:
#: One square metre, because half the artefact has an absolute size - a survey
#: disagreement measured in centimetres, times the length of a lot line, is the
#: same few square metres on a 200 m2 duplex parcel as at the corner of Parc
#: Jarry, where a percentage of 1.59 km2 would be far too small a number to
#: threshold on. `MIN_ZONE_PCT_OF_LOT` is the other half. A lot smaller than
#: this threshold itself would lose every zone to it; there is no such
#: development site, and the pane says plainly that nothing covers it.
#:
#: The dataplatform applies the same cutoff one layer over, in
#: `ZonePieceConfig.min_overlap_m2`, which is what keeps the zone this pane
#: shows and the zone the solver priced from disagreeing.
MIN_ZONE_OVERLAP_M2 = float(os.environ.get("HBU_MIN_ZONE_OVERLAP_M2", 1.0))

#: How much of a lot a zone has to cover, as a percentage of it, before that
#: zone is reported as covering it.
#:
#: The other half of the same artefact cutoff, and the half that catches the
#: sliver a square metre lets through - because a square metre is not a large
#: number on a lot line. Lot 6 291 714 is the case the Lot pane was getting
#: wrong: 438 m2, of which the two publishers put 437.23 in H03-126 and 1.19 -
#: 0.27 per cent - in C03-130. That 1.19 m2 clears `MIN_ZONE_OVERLAP_M2`, so
#: the pane offered a commercial zone governing a quarter of a per cent of the
#: parcel beside the residential one governing the rest, and rounded its share
#: to "0 per cent of the lot" while doing it. Two zones on a lot are a mapping
#: disagreement rather than a menu, and this is what says so.
#:
#: One per cent is a judgement about the borough rather than a property of the
#: data. Over Villeray-Saint-Michel-Parc-Extension it drops 930 of 28 850
#: lot x zone pairs and takes the lots reported as split between two zones from
#: 2 529 to 1 861. `silver.lot_features` is thresholded at neither cutoff, on
#: purpose, so the rows are still there to be read back at another value.
#:
#: The dataplatform's `ZonePieceConfig.min_pct_of_lot` is the same one per cent,
#: for the same reason and applied one layer over.
MIN_ZONE_PCT_OF_LOT = float(os.environ.get("HBU_MIN_ZONE_PCT_OF_LOT", 1.0))

#: How much of a building has to land inside a lot, in square metres, before
#: that footprint is reported as standing on it.
#:
#: The building-side twin of the two zone cutoffs above, and it exists for a
#: different artefact. BDOI draws a terrace or a shopping strip as one
#: contiguous outline running through the party walls, so clipping it to the
#: cadastre is what gives each lot its own house. The clip also hands each lot
#: the few square metres of its neighbour's house that fall on this side of a
#: lot line the two surveys draw differently, and those have area: they survive
#: the dimension screen, and they are counted. Lot 3 791 059 is the case the
#: Lot pane was getting wrong - a 155 m2 house standing on it, plus 4.15 m2 of
#: the house next door and 0.93 m2 of a shed touching the corner, reported as
#: three buildings.
#:
#: The dataplatform applies this same five square metres in
#: `postgis.MIN_BUILDING_OVERLAP_M2`, which is what keeps
#: `silver.building_lot_intersections` and the fallbacks below the same set -
#: and unlike the zone cutoffs, this one *is* applied when the table is built,
#: because no reader of it wants the neighbour's wall.
MIN_BUILDING_OVERLAP_M2 = float(os.environ.get("HBU_MIN_BUILDING_OVERLAP_M2", 5.0))

#: How much of a building has to land inside a lot, as a percentage of the
#: building, before that footprint is reported as standing on it.
#:
#: **An or, where the two zone cutoffs are an and**, and the difference is the
#: contiguous outline above. A zone is large and a lot is small, so a genuine
#: zone covers most of a lot and a small share of one is a survey artefact - a
#: row failing either cutoff is dropped. A footprint is the other way round: a
#: townhouse standing wholly on its own parcel is a small *percentage* of the
#: block-long shape it was digitised into, and over VSMPE the median clip
#: between 3 and 10 per cent of its building is around 100 m2 - a whole house.
#: A percentage floor applied the way the zone floor is would drop 9 224 of
#: 31 815 rows, most of them real buildings.
#:
#: So this is an escape hatch rather than a requirement: a row under
#: `MIN_BUILDING_OVERLAP_M2` survives if it is at least this much of its
#: footprint, which is what keeps a 4 m2 garage sitting entirely on its lot.
#: Over VSMPE it rescues 2 235 rows from the absolute cutoff, 896 of them
#: structures standing at least half on the lot they are reported against.
#: Together the pair drops 7.1 per cent of the rows and 0.09 per cent of the
#: clipped area - numerous, and occupying almost nothing.
#:
#: `postgis.MIN_BUILDING_PCT_OF_BUILDING` is the same ten per cent one repo
#: over.
MIN_BUILDING_PCT_OF_BUILDING = float(
    os.environ.get("HBU_MIN_BUILDING_PCT_OF_BUILDING", 10.0)
)

#: The three tests that separate a footprint standing on a lot from a
#: neighbour's wall crossing the line, for a query computing its own clip.
#:
#: Written once because four fallbacks need it and they are the paths nobody
#: watches: they answer for a borough whose silver join has not been built yet,
#: so a screen added to `compute_intersections` and not to them makes the map
#: mean one thing before the pipeline runs and another after. The dimension
#: test was already copied into three of them and the area test into none, and
#: that is exactly the drift this constant exists to stop.
#:
#: Expects the clipped geometry as ``clip.geom`` and the whole footprint as
#: ``b.geom``, and the caller to supply ``min_building_overlap_m2`` and
#: ``min_building_pct_of_building`` — `_building_screen_params` does that.
_BUILDING_CLIP_SCREEN = """(
               -- A party wall on the lot line intersects and clips to a line
               -- or a point. That is two buildings meeting at a boundary, not
               -- one standing on the parcel.
               NOT ST_IsEmpty(clip.geom)
               AND ST_Dimension(clip.geom) = 2
               -- And an area clip is not enough either: the neighbour's wall
               -- drawn a hand's breadth over the line clips to a thin polygon
               -- that has area and is still the house next door. Kept if the
               -- slice is large enough to be a building, *or* is enough of its
               -- footprint to be one.
               AND (
                     ST_Area(clip.geom::geography) >= %(min_building_overlap_m2)s
                  OR (ST_Area(b.geom::geography) > 0
                      AND 100.0 * ST_Area(clip.geom::geography)
                                / ST_Area(b.geom::geography)
                          >= %(min_building_pct_of_building)s)
               )
           )"""


def _building_screen_params() -> dict[str, float]:
    """The two cutoffs `_BUILDING_CLIP_SCREEN` reads, as query parameters."""
    return {
        "min_building_overlap_m2": MIN_BUILDING_OVERLAP_M2,
        "min_building_pct_of_building": MIN_BUILDING_PCT_OF_BUILDING,
    }

#: A viewport query returns at most this many shapes. Past it the map is a
#: solid block of outlines and the browser is the bottleneck, not the database.
#:
#: This is the *GeoJSON* renderer's cap, and it is why that renderer is no
#: longer the default: a cap is not a fix. Two thousand lots is still two
#: thousand full coordinate lists embedded in the page on every rerun, and the
#: browser gives out well below a borough. The tile renderer needs no such
#: number, because a tile is bounded by its own area - see `src/utils/tiles.py`.
DEFAULT_FEATURE_LIMIT = int(os.environ.get("HBU_MAP_FEATURE_LIMIT", 2000))


# ---------------------------------------------------------------------------
# What the database actually has
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    """Which halves of the schema are present.

    Checked once per session rather than assumed, because the three repos land
    in a database independently: `hbu_infra` creates the tables, the
    dataplatform's ``document_index`` asset creates ``rag.chunks`` on its first
    load, and ``rag.buildings`` arrives with whoever loads BDOI. A map that
    silently draws nothing is worse than one that says which table is missing.

    ``building_lots`` and ``lot_features`` are the two silver joins, and they
    are a different kind of absent from the rest. Everything else here missing
    means a question cannot be answered; these two missing only mean it is
    answered the slow way, by computing the intersection per click. So neither
    `can_map` nor `can_retrieve` turns on them, and `missing()` takes an
    argument for which of the two readings the caller wants.
    """

    postgis: bool = False
    pgvector: bool = False
    lots: bool = False
    buildings: bool = False
    building_lots: bool = False
    lot_features: bool = False
    #: ``silver.assessment_units`` - the roll, one row per premises, geocoded.
    #: Advisory, and the narrowest dependency on this list: the Lot pane's
    #: today-against-proposal table loses the count of non-residential
    #: premises standing on the lot and keeps every other row, because every
    #: other row is read off gold. Nothing else in the app asks for it - the
    #: dataplatform is what sums this table onto a parcel, and the map reads
    #: those sums rather than repeating them.
    assessment_units: bool = False
    #: ``silver.lot_addresses`` - Adresses Quebec's civic address points, put
    #: on the parcel and, within it, on the zone piece this platform answers
    #: at. Advisory, and the thinnest dependency here: without it the Lot pane
    #: loses its address line and nothing else changes. Off the dataplatform's
    #: daily schedules - it has a job and a `make addresses` of its own - so a
    #: borough materialized before that was run has every other table and no
    #: addresses, which is a greyed line rather than a fault.
    lot_addresses: bool = False
    features: bool = False
    #: ``silver.zoning_grid_columns`` - the *grille des spécifications* parsed
    #: into one row per column. Advisory, and the reading depends on the city:
    #: without it a Montreal zone still states its norms on the polygon and
    #: only loses the cross-check, while a Quebec City zone loses them
    #: entirely, because that layer publishes none. Which is why the Regulations
    #: pane says which of the two it is showing rather than drawing an empty
    #: table either way.
    zoning_grid_columns: bool = False
    #: ``silver.neighborhood_streets`` - the RQTT road network, cut to a borough.
    #: Advisory like the two silver joins above: without it the Streets layer
    #: is greyed out and every other layer draws exactly as before.
    streets: bool = False
    massing: bool = False
    #: ``gold.lot_surface_parking`` - the other polygon the massing asset
    #: draws. Advisory and separately probed rather than folded into
    #: `massing`, because the two tables genuinely come apart: a borough
    #: materialized before this table existed has every building and no
    #: asphalt, and greying out the one toggle is a truer thing to show than
    #: an empty layer that looks like a borough which parks nowhere.
    surface_parking: bool = False
    highest_best_use: bool = False
    redevelopment_gap: bool = False
    #: ``gold.lot_investment_opportunities`` - the two shortlists over the gap
    #: table: what to build (`investment_thesis`) and why the parcel is
    #: acquirable (`site_thesis`). Advisory: without it the Opportunities
    #: layer and the Deal pane's price and site-thesis blocks are absent, and the
    #: subtraction the gap table holds is unaffected.
    investment_opportunities: bool = False
    chunks: bool = False
    #: ``rag.lot_documents`` - the lot x document join, from hbu_infra's
    #: 006_lot_documents.sql. Advisory, and for a reason worth stating: without
    #: it the Regulations pane still finds a lot's grid, by way of the
    #: ``LIEN_GRILLE`` on the zoning row. What it loses is every document that
    #: is *not* reached that way - a layer the dataplatform starts indexing
    #: whose attributes carry no link, or a zone whose link an older scrape
    #: dropped. So a fallback, not a fault.
    lot_documents: bool = False
    search_at_lot: bool = False
    search_near: bool = False

    @property
    def can_map(self) -> bool:
        return self.postgis and (self.lots or self.buildings or self.features)

    @property
    def can_retrieve(self) -> bool:
        return self.pgvector and self.chunks

    def missing(self, *, include_advisory: bool = True) -> list[str]:
        """Human-readable names of what is absent.

        ``include_advisory=True`` is the operator's reading — the sidebar pane
        and `scripts/doctor.py` — where "the pipeline has not joined this
        borough yet" is worth seeing.

        ``include_advisory=False`` is what a *user-facing* message wants, and
        what the agent's tools pass. Telling someone that
        `silver.lot_features` is missing when the answer arrived anyway, just
        by the slower route, is reporting a fault that did not happen.
        """
        checks = {
            "postgis extension": (self.postgis, True),
            "vector extension": (self.pgvector, True),
            f"{SCHEMA}.lots": (self.lots, True),
            f"{SCHEMA}.buildings": (self.buildings, True),
            f"{SILVER_SCHEMA}.building_lot_intersections": (self.building_lots, False),
            f"{SILVER_SCHEMA}.lot_features": (self.lot_features, False),
            f"{SILVER_SCHEMA}.lot_addresses": (self.lot_addresses, False),
            f"{SILVER_SCHEMA}.assessment_units": (self.assessment_units, False),
            f"{SCHEMA}.features": (self.features, True),
            f"{SILVER_SCHEMA}.zoning_grid_columns": (self.zoning_grid_columns, False),
            f"{SILVER_SCHEMA}.neighborhood_streets": (self.streets, False),
            f"{GOLD_SCHEMA}.lot_building_massing": (self.massing, False),
            f"{GOLD_SCHEMA}.lot_surface_parking": (self.surface_parking, False),
            f"{GOLD_SCHEMA}.lot_highest_best_use": (self.highest_best_use, False),
            f"{GOLD_SCHEMA}.lot_redevelopment_gap": (self.redevelopment_gap, False),
            f"{GOLD_SCHEMA}.lot_investment_opportunities": (
                self.investment_opportunities, False,
            ),
            f"{SCHEMA}.chunks": (self.chunks, True),
            f"{SCHEMA}.lot_documents": (self.lot_documents, False),
            f"{SCHEMA}.search_at_lot()": (self.search_at_lot, True),
            f"{SCHEMA}.search_near()": (self.search_near, True),
        }
        return [
            name
            for name, (present, required) in checks.items()
            if not present and (required or include_advisory)
        ]


def capabilities() -> Capabilities:
    """One round trip that answers every "does this exist" question."""
    row = query_one(
        """
        SELECT
          (SELECT count(*) FROM pg_extension WHERE extname = 'postgis')  > 0 AS postgis,
          (SELECT count(*) FROM pg_extension WHERE extname = 'vector')   > 0 AS pgvector,
          to_regclass(%(schema)s || '.lots')     IS NOT NULL AS lots,
          to_regclass(%(schema)s || '.buildings') IS NOT NULL AS buildings,
          to_regclass(%(silver)s || '.building_lot_intersections')
            IS NOT NULL AS building_lots,
          to_regclass(%(silver)s || '.lot_features') IS NOT NULL AS lot_features,
          to_regclass(%(silver)s || '.assessment_units')
            IS NOT NULL AS assessment_units,
          to_regclass(%(silver)s || '.lot_addresses')
            IS NOT NULL AS lot_addresses,
          to_regclass(%(schema)s || '.features') IS NOT NULL AS features,
          to_regclass(%(silver)s || '.zoning_grid_columns')
            IS NOT NULL AS zoning_grid_columns,
          to_regclass(%(silver)s || '.neighborhood_streets')
            IS NOT NULL AS streets,
          to_regclass(%(gold)s || '.lot_building_massing')
            IS NOT NULL AS massing,
          to_regclass(%(gold)s || '.lot_surface_parking')
            IS NOT NULL AS surface_parking,
          to_regclass(%(gold)s || '.lot_highest_best_use')
            IS NOT NULL AS highest_best_use,
          to_regclass(%(gold)s || '.lot_redevelopment_gap')
            IS NOT NULL AS redevelopment_gap,
          to_regclass(%(gold)s || '.lot_investment_opportunities')
            IS NOT NULL AS investment_opportunities,
          to_regclass(%(schema)s || '.chunks')   IS NOT NULL AS chunks,
          -- A view, and to_regclass answers for one the same as for a table.
          to_regclass(%(schema)s || '.lot_documents')
            IS NOT NULL AS lot_documents,
          (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %(schema)s AND p.proname = 'search_at_lot') > 0 AS search_at_lot,
          (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = %(schema)s AND p.proname = 'search_near') > 0 AS search_near
        """,
        {"schema": SCHEMA, "silver": SILVER_SCHEMA, "gold": GOLD_SCHEMA},
    )
    return Capabilities(**row) if row else Capabilities()


def corpus_dimension() -> int | None:
    """The vector width the corpus was built at, from ``rag.chunks_meta``.

    Querying a 1024-wide corpus with a 384-wide encoder does not error, it
    returns confident nonsense — so the width is checked before the first
    search rather than after a user disbelieves an answer.
    """
    if scalar(f"SELECT to_regclass('{SCHEMA}.chunks_meta') IS NOT NULL") is not True:
        return None
    value = scalar(
        f"SELECT value FROM {SCHEMA}.chunks_meta WHERE key = 'dimension'"
    )
    return int(value) if value is not None else None


def corpus_model() -> str | None:
    """The encoder the corpus was built with, so a query can match it."""
    if scalar(f"SELECT to_regclass('{SCHEMA}.chunks_meta') IS NOT NULL") is not True:
        return None
    return scalar(f"SELECT value FROM {SCHEMA}.chunks_meta WHERE key = 'embedding_model'")


def corpus_status() -> list[dict]:
    """Per neighborhood and scrape date: documents, chunks, features loaded."""
    if scalar(f"SELECT to_regclass('{SCHEMA}.corpus_status') IS NOT NULL") is True:
        return query(f"SELECT * FROM {SCHEMA}.corpus_status")
    # The view lives in 003_spatial_search.sql, which is skipped until
    # rag.chunks exists. Fall back to counting the table directly.
    return query(
        f"""
        SELECT neighborhood, scrape_date,
               count(DISTINCT doc_id) AS documents,
               count(*)               AS chunks
          FROM {SCHEMA}.chunks
         GROUP BY neighborhood, scrape_date
         ORDER BY scrape_date DESC, neighborhood
        """
    )


# ---------------------------------------------------------------------------
# Partitions — which snapshot the map is looking at
# ---------------------------------------------------------------------------


def neighborhoods(table: str = "lots") -> list[str]:
    return [
        r["neighborhood"]
        for r in query(
            f"SELECT DISTINCT neighborhood FROM {SCHEMA}.{_safe(table)} ORDER BY 1"
        )
    ]


def neighborhood_bounds(
    neighborhood: str, scrape_date: date | None = None
) -> list | None:
    """``[[south, west], [north, east]]`` of one borough's loaded lots, or None.

    What the sidebar frames when a borough is picked: the map opens on
    Montreal (`basemap.DEFAULT_CENTER`) and a Quebec City arrondissement is
    250 km off that frame, so the extent has to come from the data. Read off
    `rag.lots` because that is the table the borough list itself comes from.
    """
    row = query_one(
        f"""
        SELECT ST_XMin(e) AS west, ST_YMin(e) AS south,
               ST_XMax(e) AS east, ST_YMax(e) AS north
          FROM (
            SELECT ST_Extent(geom) AS e
              FROM {SCHEMA}.lots
             WHERE neighborhood = %(neighborhood)s
               AND (%(scrape_date)s::date IS NULL OR scrape_date = %(scrape_date)s)
          ) extent
        """,
        {"neighborhood": neighborhood, "scrape_date": scrape_date},
    )
    if not row or row.get("west") is None:
        return None
    return [[row["south"], row["west"]], [row["north"], row["east"]]]


def scrape_dates(table: str = "lots", neighborhood: str | None = None) -> list[date]:
    """Snapshot dates present, newest first."""
    return [
        r["scrape_date"]
        for r in query(
            f"""
            SELECT DISTINCT scrape_date
              FROM {SCHEMA}.{_safe(table)}
             WHERE (%s::text IS NULL OR neighborhood = %s)
             ORDER BY scrape_date DESC
            """,
            (neighborhood, neighborhood),
        )
    ]


def latest_scrape_date(table: str = "lots", neighborhood: str | None = None) -> date | None:
    dates = scrape_dates(table, neighborhood)
    return dates[0] if dates else None


#: The tables the map can draw but the pipeline may not have written yet, as
#: ``layer -> (schema, table, the asset an operator has to run)``. All three
#: are *advisory* — the map works without any of them — so the pane says which
#: asset is missing rather than leaving the layer silently blank, which reads
#: as "nothing can be built here", as "every lot is fully used" and as "this
#: borough has no streets" respectively.
#:
#: The schema is part of the value rather than assumed: two of these are gold
#: answers and the third is a silver scrape, and the only thing they have in
#: common is that they are partitioned on the same ``(scrape_date,
#: neighborhood)`` pair this asks about.
MAP_PARTITION_TABLES = {
    "massing": (GOLD_SCHEMA, "lot_building_massing", "lot_building_massing"),
    # The same asset writes both, so the asset name repeats: a borough with
    # buildings and no asphalt has not failed to park, it has parked in
    # structure - which is why the empty-layer note names the asset and the
    # reason together.
    "surface_parking": (
        GOLD_SCHEMA,
        "lot_surface_parking",
        "lot_building_massing",
    ),
    "capacity": (GOLD_SCHEMA, "lot_redevelopment_gap", "lot_redevelopment_gap"),
    # Today's side comes off the gap table and the proposed side off the HBU
    # one, and the gap table is the one probed: it is the join's driving side,
    # and a borough with a gap row and no solve still has a use to draw.
    "land_use": (GOLD_SCHEMA, "lot_redevelopment_gap", "lot_redevelopment_gap"),
    "opportunities": (
        GOLD_SCHEMA,
        "lot_investment_opportunities",
        "lot_investment_opportunities",
    ),
    "streets": (SILVER_SCHEMA, "neighborhood_streets", "neighborhood_streets"),
}


#: The site theses `gold.lot_investment_opportunities.site_thesis` takes, in
#: the order the dataplatform resolves them - `urban_rag.opportunities.
#: SITE_THESES` over there. Mirrored here because a tile URL names one and
#: a value the table cannot hold is a filter that draws nothing; `tiles.py`
#: refuses anything else before it reaches a query. 'none' is deliberately
#: not in it: the layer is the lots that carry a thesis.
SITE_THESES: tuple[str, ...] = ("brownfield", "teardown", "infill", "improvement")

#: The use classes the Land use layer colours, on either side of the proposal.
#:
#: Two columns feed it and they do not take the same values. Today's side is
#: ``lot_redevelopment_gap.existing_dominant_income_class`` - the class the
#: dataplatform files the roll's dominant CUBF under, and it is never
#: ``mixed`` because one assessment unit has one code. The proposed side is
#: ``lot_highest_best_use.hbu_dominant_use``, which *is* ``mixed`` where the
#: solver stacked commercial floor under residential. One palette over the
#: union, so a lot that changes use is the same two colours whichever side is
#: showing. ``none`` is on both: vacant ground on the roll, no programme on
#: the solve. A NULL is neither - it is a lot the roll never reached, or one
#: the solver has no row for - and draws grey rather than as a class.
LAND_USE_CLASSES: tuple[str, ...] = (
    "residential", "commercial", "industrial", "mixed", "none",
)

#: Which side of the proposal the Land use layer colours by. ``existing`` is
#: what the roll says stands there; ``hbu`` is what the solver would build.
#: A tile URL names one, and the tile carries *both* classes whichever it is
#: coloured by, so the hover can say the change without a second request.
LAND_USE_SIDES: tuple[str, ...] = ("existing", "hbu")


def partition_has_rows(
    layer: str,
    *,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
) -> bool:
    """Does this borough-snapshot hold any rows of ``layer``'s own table?

    A partition-level fact rather than a viewport one, which is what makes it
    worth a query of its own: under the tile renderer nothing on the Python
    side ever sees a feature, so "the massing asset has not run for this
    borough" has to be asked directly instead of inferred from an empty layer.
    ``EXISTS`` stops at the first row, so it costs an index probe rather than
    a count over the partition.
    """
    schema, table, _asset = MAP_PARTITION_TABLES[layer]
    return bool(
        scalar(
            f"""
            SELECT EXISTS (
                SELECT 1
                  FROM {schema}.{table}
                 WHERE (%(scrape_date)s::date IS NULL
                        OR scrape_date = %(scrape_date)s)
                   AND (%(neighborhood)s::text IS NULL
                        OR neighborhood = %(neighborhood)s)
            )
            """,
            {"scrape_date": scrape_date, "neighborhood": neighborhood},
        )
    )


# ---------------------------------------------------------------------------
# Viewport queries
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """A GeoJSON FeatureCollection plus what the limit did to it."""

    features: list[dict] = field(default_factory=list)
    truncated: bool = False
    layer: str = ""

    @property
    def count(self) -> int:
        return len(self.features)

    def collection(self) -> dict:
        return {"type": "FeatureCollection", "features": self.features}


def _bbox_params(bounds: tuple[float, float, float, float]) -> dict:
    west, south, east, north = bounds
    return {"west": west, "south": south, "east": east, "north": north}


def simplify_tolerance(zoom: int) -> float:
    """Degrees of simplification appropriate to a zoom level.

    A screen pixel is roughly ``360 / (256 * 2**zoom)`` degrees of longitude, so
    collapsing vertices below that is invisible by construction. Doubling it
    trades a pixel of fidelity for roughly half the coordinates.
    """
    return 360.0 / (256.0 * (2 ** max(zoom, 1))) * 2.0


def lots_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
    lot_numbers: list[str] | None = None,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Cadastral lots intersecting the visible rectangle.

    ``geom && ST_MakeEnvelope`` is the part the GiST index answers; the
    ``ST_Intersects`` after it is what turns the index's bounding-box answer
    into a real one. Both are needed — the operator alone returns lots whose
    envelope overlaps the viewport while their shape does not.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "min_area": min_area_m2,
            "max_area": max_area_m2,
            "lot_numbers": lot_numbers,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.neighborhood,
               l.scrape_date,
               COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2,
               l.attributes,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(l.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SCHEMA}.lots l
         WHERE l.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(l.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR l.neighborhood = %(neighborhood)s)
           AND (%(min_area)s::float8 IS NULL
                OR COALESCE(l.area_m2, ST_Area(l.geom::geography)) >= %(min_area)s)
           AND (%(max_area)s::float8 IS NULL
                OR COALESCE(l.area_m2, ST_Area(l.geom::geography)) <= %(max_area)s)
           AND (%(lot_numbers)s::text[] IS NULL OR l.lot_number = ANY(%(lot_numbers)s))
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="lots", id_key="lot_number", limit=limit)


def buildings_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Building footprints in the visible rectangle, **clipped to their lots**.

    The ``buildings`` tile layer written for the GeoJSON renderer, and it
    means the same thing: one feature per (building, lot), carrying the part of
    the footprint that stands on that parcel and that part's area. The layer is
    the intersection rather than the footprint because BDOI draws a terrace as
    one outline across every party wall, and the unclipped shape overstates
    every parcel it crosses.

    Same fast-path-and-fallback pair as `buildings_on_lot`, chosen the same way
    and returning the same columns either way, so nothing above here can tell
    which answered.

    Same shape as :func:`lots_in_bbox` otherwise, with one difference worth
    knowing: ``rag.buildings`` has no natural key. BDOI carries no id that
    survives an extract, so a load replaces a whole ``(neighborhood,
    scrape_date)`` partition and the surrogate ``building_uid`` is the only
    handle a footprint has — which means it is stable within a snapshot and
    meaningless across two. ``building_lot_key`` pairs it with the lot's own
    surrogate and is stable on exactly the same terms.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "limit": limit + 1,
            **_building_screen_params(),
        }
    )
    if _building_lots_available():
        rows = query(
            f"""
            SELECT bl.building_uid,
                   bl.building_uid::text || ':' || bl.lot_uid::text
                       AS building_lot_key,
                   bl.lot_number,
                   bl.neighborhood,
                   bl.scrape_date,
                   bl.intersection_area_m2 AS area_m2,
                   b.attributes,
                   ST_AsGeoJSON(
                       ST_SimplifyPreserveTopology(bl.geom, %(tolerance)s)
                   )::json AS geometry
              FROM {SILVER_SCHEMA}.building_lot_intersections bl
              JOIN {SCHEMA}.buildings b ON b.building_uid = bl.building_uid
             WHERE bl.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
               AND ST_Intersects(bl.geom,
                       ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
               AND (%(scrape_date)s::date IS NULL OR bl.scrape_date = %(scrape_date)s)
               AND (%(neighborhood)s::text IS NULL
                    OR bl.neighborhood = %(neighborhood)s)
             LIMIT %(limit)s
            """,
            params,
        )
        return _as_feature_set(
            rows, layer="buildings", id_key="building_lot_key", limit=limit
        )

    # The clip computed rather than read, joined within a snapshot for the
    # reason the tile fallback gives. The envelope is still tested against
    # ``b.geom`` — the indexed column — and the intersection taken only over
    # what survives it.
    rows = query(
        f"""
        SELECT b.building_uid,
               b.building_uid::text || ':' || l.lot_uid::text AS building_lot_key,
               l.lot_number,
               b.neighborhood,
               b.scrape_date,
               ST_Area(clip.geom::geography) AS area_m2,
               b.attributes,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(clip.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SCHEMA}.buildings b
          JOIN {SCHEMA}.lots l
            ON l.geom && b.geom
           AND l.scrape_date  = b.scrape_date
           AND l.neighborhood = b.neighborhood
           AND ST_Intersects(l.geom, b.geom)
          CROSS JOIN LATERAL (
              SELECT ST_Intersection(b.geom, l.geom) AS geom
          ) clip
         WHERE b.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(b.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND {_BUILDING_CLIP_SCREEN}
           AND (%(scrape_date)s::date IS NULL OR b.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR b.neighborhood = %(neighborhood)s)
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(
        rows, layer="buildings", id_key="building_lot_key", limit=limit
    )


def zones_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 14,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Zoning polygons — the layer that carries the link to the grid PDF.

    Far coarser than lots (633 against 24,953 for one borough), so this is the
    layer worth drawing when zoomed out past the point where a lot is a pixel.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "source_tables": list(ZONING_SOURCE_TABLES),
            "url_attribute": ZONING_URL_ATTRIBUTE,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT f.feature_id,
               f.neighborhood,
               f.scrape_date,
               f.attributes,
               f.attributes ->> %(url_attribute)s AS zoning_pdf_url,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(f.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SCHEMA}.features f
         WHERE f.source_table = ANY(%(source_tables)s)
           AND f.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(f.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR f.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR f.neighborhood = %(neighborhood)s)
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="zones", id_key="feature_id", limit=limit)


def streets_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 15,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Street segments intersecting the visible rectangle.

    Centre lines: the MRNF's RQTT draws one line down the axis of each segment
    of roadway. Montreal's géobase double used to draw two, one per curb, and
    the column is still called ``cote_rue_id`` from that time — the name is a
    *côté de rue* and nothing here is one any more, but it is in two primary
    keys and re-keying them to win a noun is the more expensive mistake.

    The pipeline has already clipped each segment to the borough it is
    partitioned under, so a segment crossing a borough line is short here and
    ``length_m`` is the length of the surviving piece rather than the published
    one.

    ``street_name`` is nullable and that is not a defect: an unnamed service
    road is a real road, and the tooltip says so rather than hiding it.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT s.cote_rue_id,
               s.neighborhood,
               s.scrape_date,
               s.street_name,
               s.length_m,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(s.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SILVER_SCHEMA}.neighborhood_streets s
         WHERE s.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(s.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR s.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR s.neighborhood = %(neighborhood)s)
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="streets", id_key="cote_rue_id", limit=limit)


def massing_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    only_underbuilt: bool = False,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """The proposed building of each lot — the one layer that is not a scrape.

    Every other layer on this map is something a publisher drew: a cadastral
    lot, a BDOI footprint, a zoning polygon. This one is an *answer* — the
    highest-and-best-use program the dataplatform solved for the lot, drawn as
    a rectangle inside that lot's setback envelope so the zone's four margins
    are respected by the shape itself. Which is why it is the only read here
    that reaches into `gold`.

    ``ST_SimplifyPreserveTopology`` is applied for symmetry with the layers
    above and does nothing: a massing is a rectangle and has four corners at
    any tolerance. It stays so the shape of this function does not have to be
    remembered as the exceptional one.

    ``only_underbuilt`` narrows to the proposals that are bigger than what
    stands on the lot today — the screen `gold.lot_redevelopment_gap` exists
    for, applied here as a semi-join rather than a second layer, so the map can
    show "where redevelopment is worth drawing" without the caller assembling
    two feature sets and hiding one.

    Rows with no rectangle are not in the table at all: a lot whose footprint
    could not be drawn has no geometry, and `urban_rag.warehouse` skips a
    geometry-less row on the way into a spatial table. So this returns the
    drawn massings and nothing else, and a lot missing from the layer is not a
    lot missing from the answer — see that asset's `massing_status`.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "only_underbuilt": only_underbuilt,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT m.lot_uid,
               m.lot_number,
               -- The zone whose piece this building stands on. In the key
               -- since the pieces: a lot two zones cut in two gets two
               -- rectangles, and a lot number alone would collide them.
               m.feature_id,
               m.neighborhood,
               m.scrape_date,
               m.massing_status,
               m.footprint_m2,
               m.placed_footprint_m2,
               m.footprint_fit_pct,
               m.aspect_ratio,
               m.width_m,
               m.depth_m,
               m.floors,
               m.height_m,
               m.num_dwellings,
               m.commercial_floors,
               m.placed_gross_floor_area_m2,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(m.geom, %(tolerance)s)
               )::json AS geometry
          FROM {GOLD_SCHEMA}.lot_building_massing m
         WHERE m.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(m.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR m.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR m.neighborhood = %(neighborhood)s)
           AND (NOT %(only_underbuilt)s::boolean OR EXISTS (
                   SELECT 1
                     FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
                    WHERE g.scrape_date = m.scrape_date
                      AND g.neighborhood = m.neighborhood
                      AND g.lot_uid = m.lot_uid
                      -- And this piece of it: a lot two zones cut in two
                      -- has two answers, and the building drawn on the
                      -- half with no headroom must not be shown because
                      -- the other half has some.
                      AND g.feature_id = m.feature_id
                      AND g.is_underbuilt
               ))
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="massing", id_key="piece_key", limit=limit)


def surface_parking_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    only_underbuilt: bool = False,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """The surface parking of each proposal — the massing's other polygon.

    Drawn apart from the building because it *is* apart: a surface stall has no
    floor area, no storey and no height, so it is not part of the massing and a
    map that extruded it would raise a solid where there is asphalt. It is
    fitted into the parcel less the building — the lot, not the setback
    envelope, because a margin is what a *building* keeps — and it can be a
    MultiPolygon, since a building across the middle of its parcel leaves a
    front yard and a rear one and stalls go in both.

    ``only_underbuilt`` takes the same screen `massing_in_bbox` takes, so the
    two layers on together with the filter set cannot show parking for a
    building the filter has hidden.

    Rows with no asphalt are not in the table at all: a program that parks
    underground, on a deck or in a ground-floor bay has no polygon, and
    `urban_rag.warehouse` skips a geometry-less row on the way into a spatial
    table. So a lot missing from this layer is usually a lot that parks
    somewhere else rather than one that failed to park — `parking_status` on
    `gold.lot_building_massing` is what tells the two apart.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "only_underbuilt": only_underbuilt,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT p.lot_uid,
               p.lot_number,
               p.feature_id,
               p.neighborhood,
               p.scrape_date,
               p.parking_status,
               p.surface_stalls,
               p.placed_surface_stalls,
               p.surface_parking_area_m2,
               p.placed_surface_parking_m2,
               p.surface_parking_fit_pct,
               p.parking_width_m,
               p.parking_depth_m,
               p.num_parking_bays,
               p.yard_area_m2,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(p.geom, %(tolerance)s)
               )::json AS geometry
          FROM {GOLD_SCHEMA}.lot_surface_parking p
         WHERE p.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(p.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR p.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR p.neighborhood = %(neighborhood)s)
           AND (NOT %(only_underbuilt)s::boolean OR EXISTS (
                   SELECT 1
                     FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
                    WHERE g.scrape_date = p.scrape_date
                      AND g.neighborhood = p.neighborhood
                      AND g.lot_uid = p.lot_uid
                      AND g.feature_id = p.feature_id
                      AND g.is_underbuilt
               ))
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(
        rows, layer="surface_parking", id_key="piece_key", limit=limit
    )


#: "What could still be added here", per class, as a SQL expression over
#: ``gold.lot_redevelopment_gap`` aliased ``g``.
#:
#: That table already carries ``{class}_floor_area_gap_m2``, and this does not
#: use it, for one reason worth stating: the gap column is NULL wherever
#: *either* side is, and the side that is missing is almost always the existing
#: one — a lot the assessment roll never reached is usually a lot with nothing
#: standing on it. Summing the published gap would therefore drop exactly the
#: vacant parcels, which are the ones carrying the most headroom, and would do
#: it silently.
#:
#: So a missing existing floor is read as zero here. That is not a new
#: assumption: it is the rule ``is_underbuilt`` is documented to follow, and
#: this keeps the two consistent. ``GREATEST`` also clamps at zero, so a lot
#: built past what today's zoning would allow contributes nothing rather than
#: cancelling a neighbour's headroom — those lots are counted separately, and
#: the signed total is reported beside this one by `capacity_totals`.
#:
#: On an unsolved lot every ``hbu_*`` column is NULL, and PostgreSQL's
#: ``GREATEST`` ignores NULL arguments rather than propagating them, so the
#: expression yields 0 and the lot adds nothing to a sum. That is the intended
#: reading — "no answer" is not "no room", and `num_solved` is what says how
#: much of the borough the totals actually speak for.
def _headroom_m2(cls: str) -> str:
    return (
        f"GREATEST(g.hbu_{cls}_floor_area_m2"
        f" - COALESCE(g.existing_{cls}_floor_area_m2, 0), 0)"
    )


#: How much of what the zoning would permit is standing today, as a percentage.
#: NULL on a lot with no solved programme — there is no denominator — which is
#: what the map draws in its "no programme" colour rather than as 0%.
#:
#: NULL again, and for the same reason from the other side, where the roll has
#: a unit on the lot but states no floor area for it: the numerator is unknown,
#: not zero. Coalescing it drew a standing building as the emptiest parcel in
#: the borough — lot 3 237 014 carries an assessed office, CUBF 6599, and no
#: superficie d'étages at all, and read at 0% it took the darkest band of the
#: ramp and the top of every under-built list.
#:
#: `existing_num_assessment_units` is what separates the two missing existing
#: floors, and it is the count rather than `has_assessment` for a reason: that
#: flag is true on every row of the table. gold writes it as "any column of the
#: roll join is non-null", and a lot the roll never reached still joins to a
#: unit count of 0 — a non-null — so the flag says "assessed" for the parcels
#: it exists to exclude. The count is the roll's own answer to the same
#: question and does discriminate: 0 units against 2,481 lots of VSMPE, one or
#: more against the rest.
#:
#: Zero units is nothing assessed, which is genuinely nothing standing — a
#: vacant parcel, the case `is_underbuilt` exists to find — so it keeps its 0
#: and the panes name it. A unit with no area is the roll declining to say,
#: which is not a finding about the lot: 799 lots of VSMPE, an assessed value
#: and a CUBF code on every one of them.
_USED_PCT = (
    "CASE WHEN g.existing_num_assessment_units > 0"
    "      AND g.existing_floor_area_m2 IS NULL THEN NULL"
    " ELSE 100.0 * COALESCE(g.existing_floor_area_m2, 0)"
    " / NULLIF(g.hbu_floor_area_m2, 0) END"
)


#: The class the Land use layer colours by, chosen server-side from the URL's
#: ``use_side`` so the browser's style function reads one property and never
#: has to know which side the map is showing. Anything but ``hbu`` - including
#: the NULL an omitted parameter binds - is today's side, so a cached URL from
#: before the parameter existed draws the roll rather than nothing.
_USE_CLASS = (
    "CASE WHEN %(use_side)s::text = 'hbu' THEN h.hbu_dominant_use"
    "      ELSE g.existing_dominant_income_class END"
)


#: The three answer layers, drawn on the ground each answer is *about*.
#:
#: **One feature per (lot, zone), not per lot.** A zoning boundary does not
#: have to follow a lot line, and on a large parcel it usually does not — so
#: since ``silver.lot_zone_pieces`` (hbu_infra sql/025) the whole chain behind
#: these layers is one row per piece of ground: two zones cutting a parcel in
#: two are two sites, each solved over its own area, against its own street and
#: under its own margins, and each with its own capacity, its own gap and its
#: own yield. Drawing them on the parcel would put two answers on one polygon
#: and force the map to pick; drawing the clip puts each answer exactly where
#: it applies. ``lot_number`` is on every feature, so the pieces of one parcel
#: are still one thing to a reader, to a click and to a `GROUP BY`.
#:
#: **The geometry comes from the piece, so no join to ``rag.lots`` is needed
#: at all** — which incidentally removes the reload hazard the joins here have
#: always had to work around. ``lot_uid`` is a bigserial ``load_lots`` mints
#: again on every load, and the gold tables are keyed on it; the piece carries
#: both that surrogate *and* its own polygon, so this joins the answer on the
#: cadastral number and the zone, which survive a reload together.
_PIECE_SOURCE = """{silver}.lot_zone_pieces l"""

#: What every piece-drawn feature says about the ground it covers and the
#: parcel it belongs to. Kept in one string so the three layers cannot drift
#: about what a split lot looks like on a map.
#:
#: ``num_lot_zones`` is the one a renderer must read: it is 1 on the great
#: majority of features, and where it is not the feature is a *part* of the
#: parcel its ``lot_number`` names. A legend that sums ``piece_area_m2`` over a
#: viewport is summing ground and is right; one that sums ``lot_area_m2`` is
#: counting split parcels once per piece and is not.
_PIECE_COLUMNS = """
               l.lot_uid,
               l.lot_number,
               l.feature_id,
               l.piece_area_m2,
               l.lot_area_m2,
               l.num_lot_zones,
               l.is_primary_zone,
               l.primary_street_name"""


def floor_area_unreported(row: Mapping | None) -> bool:
    """True where the roll has a unit on this lot but states no floor for it.

    The Python side of `_USED_PCT`'s CASE, and how a caller tells "nothing is
    built here" from "the roll does not say". Everything that renders a floor
    area today — the Lot pane, the map tooltips, the agent's `lot_efficiency`
    — asks this rather than testing the two columns itself, so the three
    cannot end up disagreeing about which lots are unknown.
    """
    if not row:
        return False
    return (
        float(row.get("existing_num_assessment_units") or 0) > 0
        and row.get("existing_floor_area_m2") is None
    )


def nothing_assessed(row: Mapping | None) -> bool:
    """True where the roll reached no unit on this lot at all.

    The other half of `floor_area_unreported`, and the case whose floor area
    genuinely is zero. Not `has_assessment`: see `_USED_PCT` above on why that
    column is true everywhere and cannot be asked this.
    """
    if not row:
        return False
    return float(row.get("existing_num_assessment_units") or 0) <= 0


def capacity_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 15,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    only_underbuilt: bool = False,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Each lot shaded by how much of its permitted floor is actually built.

    The question "is this lot used efficiently" asked of every parcel in view
    at once. ``gold.lot_redevelopment_gap`` holds the subtraction but carries
    no geometry, so the shape comes from ``rag.lots`` and the finding from the
    join.

    **Joined on ``lot_number`` within the partition, never on ``lot_uid``.**
    ``lot_uid`` is a bigserial minted fresh on every load of a partition, so
    reloading one borough-day renumbers every lot in it and orphans every gold
    table already materialized against the old numbers. On the surrogate that
    fails silently and totally — the join matches nothing, every lot loses its
    finding at once, and the layer goes blank rather than wrong, which reads as
    a solver that never ran. ``lot_number`` is the cadastral number the load is
    keyed *to* rather than one it happens to mint, so a gold partition
    materialized hours before a reload still joins. The borough and the date
    travel with it because the number is unique within a partition and not
    across two.

    ``only_underbuilt`` narrows to the lots the gap table flags — the same
    screen the massing layer takes, applied to the same rows, so turning both
    layers on with the filter set cannot show a proposal on a lot this layer
    has hidden.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "only_underbuilt": only_underbuilt,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.feature_id,
               l.neighborhood,
               l.scrape_date,
               -- The ground this answer is about. `area_m2` keeps its name
               -- because every renderer and tooltip reads it; what changed is
               -- that on a split parcel it is now the piece rather than the
               -- whole lot, with `lot_area_m2` beside it saying what the piece
               -- is a piece of.
               l.piece_area_m2 AS area_m2,
               l.lot_area_m2,
               l.num_lot_zones,
               l.is_primary_zone,
               l.primary_street_name,
               g.hbu_status,
               g.has_assessment,
               g.existing_num_assessment_units,
               g.is_underbuilt,
               g.existing_floor_area_m2,
               g.hbu_floor_area_m2,
               g.floor_area_gap_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               g.dwelling_gap,
               {_USED_PCT} AS used_pct,
               {_headroom_m2("residential")} AS residential_headroom_m2,
               {_headroom_m2("commercial")}  AS commercial_headroom_m2,
               {_headroom_m2("industrial")}  AS industrial_headroom_m2,
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(l.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SILVER_SCHEMA}.lot_zone_pieces l
          JOIN {GOLD_SCHEMA}.lot_redevelopment_gap g
            ON g.lot_number   = l.lot_number
           AND g.feature_id   = l.feature_id
           AND g.neighborhood = l.neighborhood
           AND g.scrape_date  = l.scrape_date
         WHERE l.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(l.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR l.neighborhood = %(neighborhood)s)
           AND (NOT %(only_underbuilt)s::boolean OR g.is_underbuilt)
         LIMIT %(limit)s
        """,
        params,
    )
    # Keyed on the piece rather than on the lot: a split parcel has two
    # features here and a lot number would collide them, so the second would
    # silently replace the first in whatever the caller keys by.
    return _as_feature_set(rows, layer="capacity", id_key="piece_key", limit=limit)


#: What the Land use layer carries per piece, in both renderers: the class on
#: each side and the one the map is coloured by, the roll's own words for
#: today's, and the three measures the two sides are compared on - floor,
#: footprint, dwellings. `hbu_status` travels so a blank proposed side can say
#: why. The description is the one text column on the tile and it earns the
#: bytes: "residential" is the class, "Logement" is what the roll actually
#: says, and a reader hovering a church wants the second.
#:
#: `existing_footprint_m2` is the piece's own, off `silver.lot_zone_pieces`.
#: It used to be a lateral sum over `silver.building_lot_intersections` at the
#: *lot* grain - which was the same number while a lot was one feature, and on
#: a split parcel would now paint the whole parcel's footprint onto each of its
#: pieces. The pieces table measures the same clip one cut further and carries
#: it on the row, so the join is gone rather than repaired, and with it the
#: fallback spec for a database that had not built the clip: the layer draws
#: that table's geometry, so without it there is nothing to draw either way.
_LAND_USE_COLUMNS = f"""
               g.hbu_status,
               g.existing_num_assessment_units,
               g.existing_dominant_income_class AS existing_use,
               g.existing_dominant_use_description,
               h.hbu_dominant_use               AS hbu_use,
               {_USE_CLASS}                     AS use_class,
               g.existing_floor_area_m2,
               g.hbu_floor_area_m2,
               l.existing_footprint_m2,
               h.footprint_m2                   AS hbu_footprint_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings"""


def land_use_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 15,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    use_side: str = "existing",
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """Each lot coloured by what it is used for - today, or as proposed.

    The Utilisation layer asks *how much* of the envelope stands; this one
    asks *what*, on both sides of the solve. ``gold.lot_redevelopment_gap``
    carries today's class and the floor and dwellings on both sides;
    ``gold.lot_highest_best_use`` carries the solver's class and footprint;
    the silver clip carries the footprint that was measured. Joined to the
    cadastre on ``lot_number`` within the partition, for the reason
    `capacity_in_bbox` gives at length.

    ``use_side`` picks which class ``use_class`` is - the one the style reads
    - and both classes travel regardless, so the hover can name the change.
    A lot the solver has no row for draws its today's side either way, and a
    grey on the proposed side is "no programme", which the hover says.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "use_side": use_side,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.feature_id,
               l.neighborhood,
               l.scrape_date,
               l.piece_area_m2 AS area_m2,
               l.lot_area_m2,
               l.num_lot_zones,
               l.is_primary_zone,
               l.primary_street_name,{_LAND_USE_COLUMNS},
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(l.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SILVER_SCHEMA}.lot_zone_pieces l
          JOIN {GOLD_SCHEMA}.lot_redevelopment_gap g
            ON g.lot_number   = l.lot_number
           AND g.feature_id   = l.feature_id
           AND g.neighborhood = l.neighborhood
           AND g.scrape_date  = l.scrape_date
          LEFT JOIN {GOLD_SCHEMA}.lot_highest_best_use h
            ON h.lot_uid      = g.lot_uid
           AND h.feature_id   = g.feature_id
           AND h.neighborhood = g.neighborhood
           AND h.scrape_date  = g.scrape_date
         WHERE l.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(l.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR l.neighborhood = %(neighborhood)s)
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(rows, layer="land_use", id_key="piece_key", limit=limit)


#: What the Opportunities layer carries per lot, in both renderers: the two
#: theses, the rank and the yield the site thesis is ordered on, the verdict,
#: and what the screen read - year, storeys, headroom, the use in words, the
#: heritage flags, and the addition on an improvement. Enough for a tooltip
#: and a colour; the pane reads the whole row again by lot.
_OPPORTUNITY_COLUMNS = """
               o.site_thesis,
               o.investment_thesis,
               o.site_thesis_rank,
               o.is_top_site_opportunity,
               o.site_yield_on_cost_pct,
               o.redevelopment_npv_gain_cad,
               o.existing_year_built,
               o.existing_num_storeys,
               o.hbu_floors,
               o.storey_headroom,
               o.existing_dominant_use_description,
               o.is_heritage_sector,
               o.has_piia_review,
               o.demolition_review_required,
               o.improvement_added_storeys,
               o.improvement_floor_m2,
               o.owner_best_future,
               o.buyer_best_future,
               o.site_irr_pct,
               o.site_all_in_yield_on_cost_pct,
               o.site_yoc_spread_bps,
               o.market_cap_rate_pct,
               o.is_good_candidate,
               o.clears_cap_rate,
               o.clears_hurdle"""

#: The screen the layer applies, in the parameters both renderers bind: only
#: the lots that carry a site thesis, narrowed to one thesis and to the top of
#: each when asked. `%(site_thesis)s` is NULL for "every thesis".
_OPPORTUNITY_SCREEN = """
           o.site_thesis IS NOT NULL AND o.site_thesis <> 'none'
           AND (%(site_thesis)s::text IS NULL OR o.site_thesis = %(site_thesis)s)
           AND (NOT %(top_only)s::boolean OR o.is_top_site_opportunity)
           AND (NOT %(good_only)s::boolean OR o.is_good_candidate)"""


def opportunities_in_bbox(
    bounds: tuple[float, float, float, float],
    *,
    zoom: int = 15,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    site_thesis: str | None = None,
    top_only: bool = False,
    good_only: bool = False,
    limit: int = DEFAULT_FEATURE_LIMIT,
) -> FeatureSet:
    """The pieces of ground that carry a site thesis, shaped by the zoning clip.

    The few hundred sites in a borough the screens in
    ``gold.lot_investment_opportunities`` filed under `brownfield`,
    `teardown`, `infill` or `improvement` - why the ground is acquirable -
    each with the rank and the yield its thesis is ordered on. The table
    carries no geometry, so the shape comes from ``silver.lot_zone_pieces``
    and the finding from the join, **on ``lot_number`` and the zone within the
    partition** for the reason `capacity_in_bbox` gives at length.

    One feature per (lot, zone), not per lot - see `_PIECE_SOURCE`. A parcel a
    zoning boundary crosses can carry two theses and does: a commercial strip
    worth redeveloping in front of a yard that is not is exactly the shape this
    layer is for, and it was invisible while one thesis had to answer for the
    whole parcel.

    ``site_thesis`` narrows to one thesis; ``top_only`` to the first
    ``site_top_n`` of each, which is the shortlist the dataplatform marked.
    """
    params = _bbox_params(bounds)
    params.update(
        {
            "tolerance": simplify_tolerance(zoom),
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "site_thesis": site_thesis,
            "top_only": top_only,
            "good_only": good_only,
            "limit": limit + 1,
        }
    )
    rows = query(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.feature_id,
               l.neighborhood,
               l.scrape_date,
               l.piece_area_m2 AS area_m2,
               l.lot_area_m2,
               l.num_lot_zones,
               l.is_primary_zone,
               l.primary_street_name,{_OPPORTUNITY_COLUMNS},
               ST_AsGeoJSON(
                   ST_SimplifyPreserveTopology(l.geom, %(tolerance)s)
               )::json AS geometry
          FROM {SILVER_SCHEMA}.lot_zone_pieces l
          JOIN {GOLD_SCHEMA}.lot_investment_opportunities o
            ON o.lot_number   = l.lot_number
           AND o.feature_id   = l.feature_id
           AND o.neighborhood = l.neighborhood
           AND o.scrape_date  = l.scrape_date
         WHERE l.geom && ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326)
           AND ST_Intersects(l.geom,
                   ST_MakeEnvelope(%(west)s, %(south)s, %(east)s, %(north)s, 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR l.neighborhood = %(neighborhood)s)
           AND {_OPPORTUNITY_SCREEN}
         LIMIT %(limit)s
        """,
        params,
    )
    return _as_feature_set(
        rows, layer="opportunities", id_key="piece_key", limit=limit
    )


# ---------------------------------------------------------------------------
# The map's tiles are not queried here any more
#
# The nine layers the map draws are **PMTiles archives** - one file per layer
# per (scrape_date, neighborhood) partition - rendered by the dataplatform's
# `map_tiles` asset with the ``ST_AsMVT`` SQL that used to live in this
# module, and fetched by the browser straight off S3 with range requests. See
# `src/utils/tiles.py` for where they are and `urban_rag.map_tiles` over there
# for what each tile carries.
#
# What stays here is what the *page* has to know about those archives to
# draw a legend, write a note and gate a layer: which zoom each layer holds
# its own features from, which layers have dissolved cells below that, and
# which cell level a zoom is filled from. They are facts about the tiles'
# content, mirrored from the dataplatform's `urban_rag.map_tiles` and
# `tile_grid`, and they have to agree with what was built or the notes under
# the map name the wrong zoom.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Below a layer's detail zoom, a tile comes from the aggregates instead
#
# Every layer here is too dense to draw whole at a borough-wide zoom - a lot is
# sub-pixel below 15 and twenty-five thousand of them is a grey rectangle - so
# each one has a zoom it starts drawing its own features at. Below that zoom
# the archive holds the cells of `gold.map_cell_aggregates` instead: the same
# layer, dissolved onto the Web Mercator tile grid by the dataplatform's
# `map_cell_aggregates` asset, one feature per cell.
#
# **It is the same MVT layer name either way**, which is the point. Leaflet
# holds one VectorGrid per layer, keyed on that name, so crossing the threshold
# changes what a tile contains and nothing else - no second layer in the
# control, no visibility to keep in step, no remount. The style and tooltip in
# `basemap` branch on `agg_level`, which only an aggregate tile carries.
# ---------------------------------------------------------------------------

#: The zoom each layer starts drawing its own features at. Below it the tile is
#: an aggregate; at or above it, the layer itself.
#:
#: These live here rather than in `basemap` because they are a fact about the
#: data's density rather than about the drawing - "a lot is sub-pixel at 13" is
#: a statement about the cadastre. **They mirror `DETAIL_ZOOM` in the
#: dataplatform's `urban_rag.map_tiles`**, which is what actually decides what
#: an archive holds at each zoom; these are what the legend and the notes read.
MVT_DETAIL_ZOOM: dict[str, int] = {
    # Zoning is block-sized already and draws all the way down as itself,
    # which is why it has no aggregate and no threshold worth naming. 0 rather
    # than absent, so a caller may look every layer up.
    "zones": 0,
    "capacity": 15,
    # A class per lot over the whole cadastre, so it takes the lots' gate for
    # the lots' reason: at 14 a tile holds a quarter of a borough. No
    # aggregate behind it - `map_cell_aggregates` carries no use kind - so
    # this is a floor, like `opportunities`, rather than a handover.
    "land_use": 15,
    "streets": 14,
    "lots": 15,
    "buildings": 16,
    "massing": 16,
    # A parking bay is smaller than the building beside it, so it earns at
    # least the same gate. It has no aggregate to fall back to below this -
    # `map_cell_aggregates` builds five layers and this is not one of them -
    # so this is a floor rather than a handover, and `TILE_LAYER_MIN_ZOOM`
    # stops Leaflet asking below it.
    "surface_parking": 16,
    # A few hundred lots in a borough rather than twenty-five thousand, so it
    # can draw itself from further out than the cadastre can - and it has to,
    # because "where are the opportunities" is a question asked of a borough.
    # No aggregate behind it, so like `surface_parking` this is a floor.
    "opportunities": 12,
}

#: How many zooms finer than the display zoom an aggregate cell is. **This has
#: to match `urban_rag.tile_grid.ZOOM_OFFSET` in the dataplatform**, because it
#: is what turns a requested tile into the level of cells to read: a tile at
#: zoom Z is filled from cells at Z + 4. Getting it wrong does not error - it
#: reads a level that exists and draws cells four times too coarse or too fine.
#:
#: It is also the bound this whole path rests on: a tile at Z contains exactly
#: 4**4 = 256 cells at Z + 4, so an aggregate tile carries at most 256 features
#: however dense the borough is, with no fuse needed on it.
AGGREGATE_ZOOM_OFFSET = 4

#: The cell levels the dataplatform builds, and therefore the display zooms
#: an archive can hold cells for: 1..19 covers -3..15, which is every zoom Leaflet can be at
#: and then some. **This mirrors `urban_rag.tile_grid.CELL_ZOOMS`** over there,
#: and the two disagreeing does not error - a level named here that was never
#: built comes back as a borough with no data.
#:
#: 10..19 is the part the map reads today, because `MAP_MIN_ZOOM` is 6. The
#: rest is built, so lowering that floor is a change to the archives' zoom
#: range over there rather than a re-dissolve of every borough.
AGGREGATE_CELL_ZOOMS = tuple(range(1, 20))

#: The layers that have an aggregate to fall back to.
AGGREGATE_LAYERS: tuple[str, ...] = (
    "capacity",
    "streets",
    "lots",
    "buildings",
    "massing",
)


def aggregate_cell_zoom(zoom: int) -> int:
    """The cell level a tile at display ``zoom`` is filled from.

    Clamped to the levels that exist. Below the range the coarsest level is
    the honest answer - it is still a true summary, just of more ground than a
    cell should cover; above it the layer is drawing itself and this is not
    called.
    """
    cell = zoom + AGGREGATE_ZOOM_OFFSET
    return max(AGGREGATE_CELL_ZOOMS[0], min(AGGREGATE_CELL_ZOOMS[-1], cell))


def serves_aggregate(layer: str, zoom: int) -> bool:
    """Whether a tile of ``layer`` at ``zoom`` holds dissolved cells."""
    return layer in AGGREGATE_LAYERS and zoom < MVT_DETAIL_ZOOM[layer]


#: The display zoom below which an aggregate tile stops being a summary and
#: becomes an **outline**: the same dissolved cells, shaded by the same
#: `value`, simplified, and carrying nothing a tooltip could read.
#:
#: There are two different things a reader wants from a zoomed-out map and they
#: part company here. From 12 up a cell is a *finding* - 44 lots, 61% of the
#: permitted floor - and it is worth hovering. From 11 down a cell is sixteen
#: screen pixels of a borough that is itself a few dozen across, so the only
#: thing it can honestly convey is shape and shading; everything else on the
#: row is weight carried to no end, and `attributes` is a jsonb blob per cell
#: for a tooltip nobody can aim at.
#:
#: It is also where the *geometry* stops being free. Below 12 the cell level is
#: 15 and coarser, and from about level 11 down a borough fits inside a single
#: cell whose geometry is its entire dissolved union - see the
#: `map_cell_aggregates` header in the dataplatform's `urban_rag.postgis` on
#: why those copies are stored intact rather than simplified. This is the zoom
#: below which that union stops being transformed vertex by vertex for a shape
#: a dozen pixels wide; the dataplatform thins it before projecting.
AGGREGATE_OUTLINE_ZOOM = 12


def serves_outline(zoom: int) -> bool:
    """Whether an aggregate tile at ``zoom`` is an outline rather than a summary.

    A property of the zoom alone rather than of the layer: `serves_aggregate`
    has already decided that *some* aggregate answers, and this decides which
    of its two shapes. Every layer crosses this boundary at the same zoom,
    because what changes at it is what a reader can see rather than how dense
    any one layer happens to be.
    """
    return zoom < AGGREGATE_OUTLINE_ZOOM


#: The layers the map draws as tiles, in draw order - one archive per name per
#: partition. `basemap` builds one Leaflet layer per entry and `tiles` resolves
#: one archive per entry, so a layer added to the dataplatform's
#: `urban_rag.map_tiles.LAYERS` reaches both by being added here.
MVT_LAYER_NAMES: tuple[str, ...] = (
    "zones",
    "land_use",
    "capacity",
    "opportunities",
    "streets",
    "lots",
    "buildings",
    "surface_parking",
    "massing",
)

#: How long a viewport read trusts its answer to "has the building x lot join
#: been built on this database". `app._capabilities` is memoised for the same
#: span; this is the one bit the GeoJSON path's buildings read needs without
#: the rest of that probe.
#:
#: Five minutes is what being wrong costs: a borough whose silver join lands
#: mid-session draws its buildings the slow way for up to that much longer, and
#: the shapes are the same either way.
TILE_CAPABILITY_TTL_S = 300.0

#: ``(expires_at, present)`` for the probe below, or None before the first one.
_building_lots_probe: tuple[float, bool] | None = None


def _building_lots_available() -> bool:
    """Whether ``silver.building_lot_intersections`` exists, cached per process.

    `capabilities()` answers this too, and answers eight other questions with
    it in one round trip - which is what the panes want and what a viewport
    read repeated per pan does not. Memoised for `TILE_CAPABILITY_TTL_S`.
    """
    global _building_lots_probe
    now = time.monotonic()
    if _building_lots_probe is None or _building_lots_probe[0] <= now:
        present = bool(
            scalar(
                "SELECT to_regclass(%(silver)s "
                "|| '.building_lot_intersections') IS NOT NULL",
                {"silver": SILVER_SCHEMA},
            )
        )
        _building_lots_probe = (now + TILE_CAPABILITY_TTL_S, present)
    return _building_lots_probe[1]


def _as_feature_set(rows: list[dict], *, layer: str, id_key: str, limit: int) -> FeatureSet:
    """Turn query rows into GeoJSON features, noting whether the limit bit.

    Each query asks for ``limit + 1`` rows; getting them back is how the caller
    learns the viewport holds more than it is being shown, without a second
    ``count(*)`` over the same predicate.
    """
    truncated = len(rows) > limit
    features = []
    for row in rows[:limit]:
        geometry = row.pop("geometry", None)
        if geometry is None:
            continue
        attributes = row.pop("attributes", None) or {}
        properties = {k: _jsonable(v) for k, v in row.items()}
        properties["layer"] = layer
        # The three answer layers are one feature per *piece* of a lot, so the
        # lot number is not unique among them - `piece_key` is the composite
        # the caller keys by, built here rather than in five SQL statements so
        # the shape of it is decided once. A layer whose grain is the parcel
        # names its own column and never reaches this branch.
        if id_key == "piece_key":
            properties["piece_key"] = _piece_key(properties)
        properties["id"] = properties.get(id_key)
        properties["attributes"] = attributes
        features.append(
            {"type": "Feature", "geometry": geometry, "properties": properties}
        )
    if truncated:
        logger.info("%s: %d shapes in view, showing %d", layer, len(rows), limit)
    return FeatureSet(features=features, truncated=truncated, layer=layer)


def _piece_key(properties: Mapping) -> str:
    """The identity of one lot x zone piece, as a string a renderer can key on.

    ``<lot number>@<zone>``, and the separator is chosen to read: a person
    seeing ``1 740 794@C04-083`` in a tooltip or a URL should be able to say
    which parcel and which grid without being told the format. A feature with
    no zone - a partition from before the pieces, or a lot no grid reached -
    falls back to the lot number alone, which is what it was before and still
    unique among such rows.
    """
    lot = properties.get("lot_number") or properties.get("lot_uid")
    zone = properties.get("feature_id")
    return f"{lot}@{zone}" if zone else str(lot)


def _jsonable(value):
    """Dates and Decimals through folium's JSON serialiser unharmed."""
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "__float__") and not isinstance(value, (int, float, bool)):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# One lot
# ---------------------------------------------------------------------------


def lot_at_point(lon: float, lat: float, *, scrape_date: date | None = None) -> dict | None:
    """The lot a map click fell in.

    Resolved server-side from the click's coordinates rather than from whatever
    shape the browser thinks was hit: a click near a boundary, on a lot the
    viewport limit left undrawn, or on a simplified edge all still land on the
    right parcel this way. Newest snapshot wins when several are loaded — a
    cadastre changes rarely, but the current shape is the one that matters.
    """
    return query_one(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.neighborhood,
               l.scrape_date,
               COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2,
               l.attributes,
               ST_AsGeoJSON(l.geom)::json AS geometry,
               ST_X(ST_PointOnSurface(l.geom)) AS lon,
               ST_Y(ST_PointOnSurface(l.geom)) AS lat,
               ARRAY[ST_XMin(l.geom), ST_YMin(l.geom),
                     ST_XMax(l.geom), ST_YMax(l.geom)]::float8[] AS bbox
          FROM {SCHEMA}.lots l
         WHERE ST_Intersects(l.geom, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326))
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
         ORDER BY l.scrape_date DESC
         LIMIT 1
        """,
        {"lon": lon, "lat": lat, "scrape_date": scrape_date},
    )


def lot_by_number(lot_number: str, *, scrape_date: date | None = None) -> dict | None:
    """A lot by its cadastral number.

    Infolot spells it with thin spaces — ``"2 170 935"`` — and nobody types it
    that way, so digits are compared with the separators stripped from both
    sides.
    """
    return query_one(
        f"""
        SELECT l.lot_uid,
               l.lot_number,
               l.neighborhood,
               l.scrape_date,
               COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2,
               l.attributes,
               ST_AsGeoJSON(l.geom)::json AS geometry,
               ST_X(ST_PointOnSurface(l.geom)) AS lon,
               ST_Y(ST_PointOnSurface(l.geom)) AS lat,
               ARRAY[ST_XMin(l.geom), ST_YMin(l.geom),
                     ST_XMax(l.geom), ST_YMax(l.geom)]::float8[] AS bbox
          FROM {SCHEMA}.lots l
         WHERE regexp_replace(l.lot_number, '\\D', '', 'g')
             = regexp_replace(%(lot_number)s, '\\D', '', 'g')
           AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
         ORDER BY l.scrape_date DESC
         LIMIT 1
        """,
        {"lot_number": lot_number, "scrape_date": scrape_date},
    )


# ---------------------------------------------------------------------------
# Addresses — a lot from a street and a number
# ---------------------------------------------------------------------------
#
# ``silver.lot_addresses`` is the dataplatform's spatial join of Adresses
# Québec's points onto the cadastre - hbu_infra's sql/026_silver_lot_addresses
# says why a join is the only way to get there: the publisher records no lot
# number on any point. One row per address *point*, so a walk-up is its door
# row plus one row per unit, all carrying the same civic address and the same
# lot. The reads below collapse that to one row per lot, which is what a
# question asked in an address is really asking for.
#
# Matching is on a folded street name rather than on the column: a person
# types "Lajeunesse", "st-michel" or "14th avenue" and the layer prints "Rue
# Lajeunesse", "Boulevard Saint-Michel" and "14e Avenue". Both sides are
# lower-cased, stripped of accents and punctuation, and the stored name loses
# its leading type, so "boul. St-Michel" and "Boulevard Saint-Michel" fold to
# the same key. The database has no `unaccent` extension, so the folding is a
# `translate` over the letters French actually uses. It runs over the whole
# partition on every call and ignores the `(lower(street_name), civic_number)`
# index - a borough is under a hundred thousand points, and the scan costs
# milliseconds.

#: The generic word Adresses Québec prints at the head of a street name and a
#: person leaves out - "Lajeunesse" for "Rue Lajeunesse" - with the
#: abbreviations it gets typed as. Stripped only as the *leading* word, so
#: "Place" goes from "Place Ville-Marie" and stays in "Rue Placide". Numbered
#: streets ("14e Avenue") carry their type last and keep it, on both sides.
STREET_TYPES: tuple[str, ...] = (
    "rue", "avenue", "ave", "av", "boulevard", "boul", "blvd", "bd", "chemin", "ch",
    "place", "pl", "terrasse", "allee", "montee", "cote", "croissant", "carre",
    "cours", "impasse", "promenade", "rang", "route", "rte", "ruelle", "square",
    "voie", "autoroute",
)

#: The same rule for Python and for Postgres: a leading type word and the
#: whitespace after it - or nothing after it, so a bare "rue" folds to an
#: empty key and is refused as no street at all. Postgres' ARE regexes read
#: `\s` and a plain group.
STREET_TYPE_PREFIX_RE = r"^(" + "|".join(STREET_TYPES) + r")(\s+|$)"

_FOLD_FROM = "àâäéèêëîïôöùûüçñ"
_FOLD_TO = "aaaeeeeiioouuucn"

#: ``street_name`` folded in SQL exactly as `street_key` folds it in Python,
#: minus the type strip, which `_STREET_CORE_SQL` adds. `lower` first so the
#: translate only has to know the lower-case letters.
_STREET_KEY_SQL = (
    "regexp_replace(regexp_replace(regexp_replace(regexp_replace("
    "translate(replace(replace(lower(a.street_name), "
    f"'œ', 'oe'), 'æ', 'ae'), '{_FOLD_FROM}', '{_FOLD_TO}'), "
    "'[-''’.]', ' ', 'g'), '\\s+', ' ', 'g'), "
    # The same two expansions `street_key` makes, so a layer that printed
    # "St-" would still fold onto what a person typed as "Saint-".
    "'\\mst\\M', 'saint', 'g'), '\\mste\\M', 'sainte', 'g')"
)
_STREET_CORE_SQL = f"regexp_replace({_STREET_KEY_SQL}, %(type_prefix)s, '')"

_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th|e|er|re|ere|ème|eme)\b")
_SAINT_RE = re.compile(r"\bst\b")
_SAINTE_RE = re.compile(r"\bste\b")
_STREET_TYPE_PREFIX = re.compile(STREET_TYPE_PREFIX_RE)


def street_key(name: str) -> str:
    """A street name as it is compared: folded, and without its leading type.

    ``"boul. St-Michel"`` and ``"Boulevard Saint-Michel"`` both become
    ``"saint michel"``; ``"14th Avenue"`` becomes ``"14e avenue"``, which is
    how the city numbers it, and ``"1st"`` becomes ``"1re"`` for the same
    reason. Everything else is lower-cased, unaccented and stripped of
    hyphens, apostrophes and dots, so the key is a substring of the stored
    name's key whenever the person meant that street.
    """
    text = unicodedata.normalize("NFKD", (name or "").replace("œ", "oe").replace("æ", "ae"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[-'’.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _ORDINAL_RE.sub(lambda m: "1re" if m.group(1) == "1" else f"{m.group(1)}e", text)
    text = _SAINT_RE.sub("saint", text)
    text = _SAINTE_RE.sub("sainte", text)
    return _STREET_TYPE_PREFIX.sub("", text).strip()


_CIVIC_RE = re.compile(
    r"""^\s*
    (?:[^\s,]*?-)?                       # a unit prefix: 204- , PH4-  (dropped)
    (?P<civic>\d+)
    (?:\s*(?P<suffix>[A-Za-z]|\d+/\d+))?  # A , B , 1/2
    \s+(?P<street>\S.*?)\s*
    (?:,.*)?$                            # ", Montréal H2R2H8" (dropped)
    """,
    re.VERBOSE,
)


def split_civic(text: str) -> tuple[int | None, str | None, str]:
    """``(civic number, suffix, street)`` from an address typed as one string.

    The tool takes the number and the street as separate arguments, but a
    model handed "7430 rue Lajeunesse" will put the whole thing in one of
    them often enough that refusing costs a turn. Same shape as the
    dataplatform's ``ADDRESS_RE``, less strict: a unit prefix and anything
    after a comma are dropped, and a string with no leading number is a
    street name.
    """
    match = _CIVIC_RE.match(text or "")
    if match is None:
        return None, None, (text or "").split(",")[0].strip()
    suffix = match.group("suffix")
    return int(match.group("civic")), (suffix.upper() if suffix else None), match.group("street")


def _like_pattern(key: str) -> str:
    """``key`` as a LIKE substring, with its own wildcards made literal."""
    escaped = key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


#: The newest load of each borough's addresses. Several snapshots of one
#: borough would otherwise answer the same address once per snapshot, the
#: way `zoning_for_lot` explains for zones.
_NEWEST_ADDRESSES = (
    "a.scrape_date = (SELECT max(x.scrape_date) FROM {silver}.lot_addresses x "
    "WHERE x.neighborhood = a.neighborhood)"
)


def lots_by_address(
    street: str,
    civic_number: int | None = None,
    *,
    civic_suffix: str | None = None,
    neighborhood: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    limit: int = 10,
) -> list[dict]:
    """The lots a civic address stands on, one row per lot, best match first.

    Best is: a street whose folded name *equals* what was typed over one that
    merely contains it ("Jarry" names Jarry Est and Jarry Ouest, and the
    number is the same on both), then a point inside ``bounds`` - the map's
    viewport, so the borough the reader is looking at wins a tie between two
    boroughs printing the same street - then the lot with the most address
    rows. ``limit`` caps the lots, not the points.

    Each row carries the lot and borough, the zone piece the door row stands
    in, the address as the layer prints it, and the counts the table states:
    ``num_addresses`` is address rows on that lot matching the query
    (addressable units, the door among them), ``num_civic_addresses`` the
    distinct doors, ``num_lot_addresses`` everything on the parcel.
    ``match_basis`` says whether the point fell inside the parcel or was
    snapped to it - see 026's header.
    """
    key = street_key(street)
    if not key:
        return []
    west, south, east, north = bounds if bounds else (None, None, None, None)
    return query(
        f"""
        WITH matched AS (
            SELECT a.lot_number, a.neighborhood, a.scrape_date, a.lot_uid,
                   a.feature_id, a.source_table, a.piece_basis,
                   a.match_basis, a.snap_distance_m,
                   a.civic_number, a.civic_address, a.street_name,
                   a.municipality, a.unit, a.address_rank, a.is_primary_address,
                   a.num_lot_addresses,
                   {_STREET_CORE_SQL} = %(core)s AS exact_name,
                   (%(west)s::float8 IS NOT NULL
                    AND a.geom && ST_MakeEnvelope(%(west)s, %(south)s,
                                                  %(east)s, %(north)s, 4326))
                       AS in_view
              FROM {SILVER_SCHEMA}.lot_addresses a
             WHERE {_STREET_KEY_SQL} LIKE %(pattern)s
               AND (%(civic)s::int IS NULL OR a.civic_number = %(civic)s)
               AND (%(suffix)s::text IS NULL
                    OR lower(coalesce(a.civic_suffix, '')) = lower(%(suffix)s))
               AND (%(neighborhood)s::text IS NULL OR a.neighborhood = %(neighborhood)s)
               AND {_NEWEST_ADDRESSES.format(silver=SILVER_SCHEMA)}
        )
        SELECT lot_number, neighborhood, scrape_date, lot_uid,
               bool_or(exact_name)                         AS exact_name,
               bool_or(in_view)                            AS in_view,
               count(*)::int                               AS num_addresses,
               count(DISTINCT civic_address)::int          AS num_civic_addresses,
               max(num_lot_addresses)::int                 AS num_lot_addresses,
               count(DISTINCT feature_id)::int             AS num_pieces,
               min(civic_number)                           AS civic_min,
               max(civic_number)                           AS civic_max,
               (array_agg(civic_address ORDER BY address_rank))[1]  AS civic_address,
               (array_agg(street_name   ORDER BY address_rank))[1]  AS street_name,
               (array_agg(municipality  ORDER BY address_rank))[1]  AS municipality,
               (array_agg(feature_id    ORDER BY address_rank))[1]  AS feature_id,
               (array_agg(source_table  ORDER BY address_rank))[1]  AS source_table,
               (array_agg(piece_basis   ORDER BY address_rank))[1]  AS piece_basis,
               (array_agg(match_basis   ORDER BY address_rank))[1]  AS match_basis,
               (array_agg(snap_distance_m ORDER BY address_rank))[1] AS snap_distance_m
          FROM matched
         GROUP BY lot_number, neighborhood, scrape_date, lot_uid
         ORDER BY bool_or(exact_name) DESC, bool_or(in_view) DESC,
                  count(*) DESC, lot_number
         LIMIT %(limit)s
        """,
        {
            "core": key,
            "pattern": _like_pattern(key),
            "type_prefix": STREET_TYPE_PREFIX_RE,
            "civic": civic_number,
            "suffix": civic_suffix,
            "neighborhood": neighborhood,
            "west": west, "south": south, "east": east, "north": north,
            "limit": max(1, int(limit)),
        },
    )


def street_summary(street: str, *, neighborhood: str | None = None) -> list[dict]:
    """What the loaded addresses hold for a street: one row per spelling and borough.

    "Jarry" comes back as Rue Jarry Est and Rue Jarry Ouest, each with its
    lots, its doors and the span of its civic numbers. What a tool answers
    with when a number is missing or matched nothing - "the street is loaded
    and runs 7000 to 8999" tells a person more than "no match".
    """
    key = street_key(street)
    if not key:
        return []
    return query(
        f"""
        SELECT a.neighborhood, a.scrape_date, a.street_name,
               {_STREET_CORE_SQL} = %(core)s          AS exact_name,
               count(DISTINCT a.lot_number)::int       AS num_lots,
               count(DISTINCT a.civic_address)::int    AS num_civic_addresses,
               min(a.civic_number)                     AS civic_min,
               max(a.civic_number)                     AS civic_max
          FROM {SILVER_SCHEMA}.lot_addresses a
         WHERE {_STREET_KEY_SQL} LIKE %(pattern)s
           AND (%(neighborhood)s::text IS NULL OR a.neighborhood = %(neighborhood)s)
           AND {_NEWEST_ADDRESSES.format(silver=SILVER_SCHEMA)}
         GROUP BY a.neighborhood, a.scrape_date, a.street_name
         ORDER BY exact_name DESC, num_lots DESC, a.street_name, a.neighborhood
        """,
        {
            "core": key,
            "pattern": _like_pattern(key),
            "type_prefix": STREET_TYPE_PREFIX_RE,
            "neighborhood": neighborhood,
        },
    )


def address_coverage() -> list[dict]:
    """Which boroughs have addresses, at which snapshot, and how many.

    Not the same set as `neighborhoods("lots")`: the address join is its own
    asset and runs after a borough's lots and zone pieces have landed, so a
    borough can be on the map and not yet be searchable by address. The
    tool says which it is.
    """
    return query(
        f"""
        SELECT neighborhood, max(scrape_date) AS scrape_date,
               count(*)::int                  AS num_addresses,
               count(DISTINCT lot_number)::int AS num_lots
          FROM {SILVER_SCHEMA}.lot_addresses a
         WHERE {_NEWEST_ADDRESSES.format(silver=SILVER_SCHEMA)}
         GROUP BY neighborhood
         ORDER BY neighborhood
        """
    )


def buildings_on_lot(lot_number: str, *, scrape_date: date | None = None) -> list[dict]:
    """Footprints standing on a lot, largest overlap first. One row per building.

    Reads ``silver.building_lot_intersections`` when it is there. That table
    already holds each footprint clipped to each lot it falls in, with the
    clipped area and the share of the building it represents, so this is an
    index lookup rather than an ``ST_Intersection`` per query — and it gets the
    hard case right for free: a school or a tower spanning several parcels has
    one row per lot, each carrying only the portion inside it, with no notion
    of a "primary" lot to guess at.

    Looked up by ``lot_number`` directly rather than through a join to
    ``rag.lots``: the table carries the number itself, because ``lot_uid`` is a
    bigserial a reload mints again and the number is what survives one.

    **One footprint is one row here, and that is not what the table's grain
    gives you.** Its grain is the *intersection* — one row per (building, lot)
    — so a caller that counts rows is counting intersections, while the Lot
    pane and `buildings_on_lot` both report footprints. Two things put the same
    footprint on this lot more than once, and they need different answers:

    * the same building against two rows carrying this lot number, which the
      ``DISTINCT ON`` collapses to the largest overlap. The pipeline enforces
      one row per lot number within a partition, so this is the cross-borough
      case and the belt to the braces below;
    * the same building in every snapshot the database holds, which no distinct
      over the id can collapse: ``building_uid`` is a bigserial reminted on
      each load, so one footprint carries a different id per date. A call with
      no ``scrape_date`` is therefore answered from the newest date this lot
      has rows for — the snapshot the rest of the pane is reading, and the same
      choice `zoning_for_lot` makes for the same reason.

    Both duplicated the count *and* the covered-area sum the callers add up
    from these rows, which is the kind of error that reads as a plausible
    number rather than as a crash.

    Falls back to computing the intersection when the join table has not been
    built for this partition yet, so a freshly loaded borough still answers.
    """
    if capabilities().building_lots:
        rows = query(
            f"""
            WITH matched AS (
                SELECT bl.building_uid,
                       bl.scrape_date,
                       bl.building_area_m2       AS area_m2,
                       bl.intersection_area_m2   AS overlap_m2,
                       bl.pct_of_building,
                       b.attributes
                  FROM {SILVER_SCHEMA}.building_lot_intersections bl
                  JOIN {SCHEMA}.buildings b ON b.building_uid = bl.building_uid
                 WHERE bl.lot_number = %(lot_number)s
                   AND (%(scrape_date)s::date IS NULL
                        OR bl.scrape_date = %(scrape_date)s)
            ),
            footprints AS (
                SELECT DISTINCT ON (building_uid)
                       building_uid, area_m2, overlap_m2, pct_of_building,
                       attributes
                  FROM matched
                 WHERE scrape_date = (SELECT max(scrape_date) FROM matched)
                 ORDER BY building_uid, overlap_m2 DESC
            )
            SELECT * FROM footprints
             ORDER BY overlap_m2 DESC
             LIMIT 50
            """,
            {"lot_number": lot_number, "scrape_date": scrape_date},
        )
        if rows:
            return rows

    # The same screens as above, spelled for the shapes rather than for the
    # join table. The `lot` CTE already picks one lot row, so a building can
    # only repeat here by being present in more than one snapshot — which the
    # `max(scrape_date)` filter is for, and why there is no second distinct.
    #
    # The clip is lateral so it is computed once and read three times, by the
    # area, by the share and by `_BUILDING_CLIP_SCREEN` — which this path had
    # none of until now. It intersected and returned, so a lot answered through
    # the fallback listed the neighbours whose walls cross its line, while the
    # same lot answered from silver did not: two different building counts for
    # one parcel, decided by whether the pipeline had run.
    #
    # ``pct_of_building`` is computed rather than left NULL for the same
    # reason. It is half of that screen, so a fallback that did not have it
    # could not apply it — and the column the pane reads is now populated on
    # both paths instead of only on one.
    return query(
        f"""
        WITH lot AS (
            SELECT geom FROM {SCHEMA}.lots
             WHERE lot_number = %(lot_number)s
             ORDER BY scrape_date DESC
             LIMIT 1
        ),
        matched AS (
            SELECT b.building_uid,
                   b.scrape_date,
                   COALESCE(b.area_m2, ST_Area(b.geom::geography)) AS area_m2,
                   ST_Area(clip.geom::geography) AS overlap_m2,
                   CASE WHEN ST_Area(b.geom::geography) > 0
                        THEN 100.0 * ST_Area(clip.geom::geography)
                                   / ST_Area(b.geom::geography)
                        ELSE 0.0
                   END AS pct_of_building,
                   b.attributes
              FROM {SCHEMA}.buildings b, lot
              CROSS JOIN LATERAL (
                  SELECT ST_Intersection(b.geom, lot.geom) AS geom
              ) clip
             WHERE b.geom && lot.geom
               AND ST_Intersects(b.geom, lot.geom)
               AND {_BUILDING_CLIP_SCREEN}
               AND (%(scrape_date)s::date IS NULL OR b.scrape_date = %(scrape_date)s)
        )
        SELECT building_uid, area_m2, overlap_m2, pct_of_building, attributes
          FROM matched
         WHERE scrape_date = (SELECT max(scrape_date) FROM matched)
         ORDER BY overlap_m2 DESC
         LIMIT 50
        """,
        {
            "lot_number": lot_number,
            "scrape_date": scrape_date,
            **_building_screen_params(),
        },
    )


def coverage_pct(covered_area_m2: float | None, lot_area_m2: float | None) -> float | None:
    """The share of a lot its footprints cover, or None when there is no share.

    Split out of the SQL and out of the pane because it is the one number here
    a reader acts on — it is compared against the *taux d'implantation* the
    grid permits — and because both of the ways it goes wrong are arithmetic
    rather than geometry. A lot with no recorded area gives no denominator, and
    ``0 / 0`` is not "0% built"; a covered area larger than the lot is not a
    percentage at all, it is a signal that the two numbers came from different
    snapshots or that the same footprint was counted twice.

    Returns None in the first case. In the second it returns the ratio it was
    given, over 100 and visibly wrong, rather than clamping: `lot_coverage`
    makes the case unreachable by unioning the clipped shapes, and a number
    quietly pinned at 100% would hide the day that stops being true.
    """
    if not lot_area_m2 or float(lot_area_m2) <= 0:
        return None
    return float(covered_area_m2 or 0) / float(lot_area_m2) * 100.0


def lot_coverage(lot_number: str, *, scrape_date: date | None = None) -> dict | None:
    """How much of a lot is built on: the measured *taux d'implantation*.

    One row — ``lot_area_m2``, ``num_footprints``, ``covered_area_m2`` and
    ``coverage_pct`` — or None when no lot carries that number.

    **This exists because the sum a caller can do for itself is the wrong
    sum.** `buildings_on_lot` returns one row per footprint with the area of
    the part inside this lot, and adding those up double-counts every square
    metre two footprints share. Overlapping footprints are rare and real: the
    building layer records an extension or a re-digitised outline as its own
    shape, and the ground under both is covered once. The union of the clipped
    shapes is what "how much of this lot is built on" means, so it is computed
    where the geometry is, and the answer cannot exceed the lot.

    **The snapshot is the lot's own, not the newest one anywhere.** Everything
    else the Lot pane shows — the area, the zoning, the gap row — is paired
    with the lot row on screen, and a coverage read across every date in the
    database is how one footprint became two and 53% of a parcel became 106%
    of it. `buildings_on_lot` screens for this after the fact with a
    ``max(scrape_date)``; here the lot's date *is* the join key, so there is
    nothing to screen.

    Reads ``silver.building_lot_intersections`` when it is there, which already
    holds each footprint clipped to each lot. The fallback clips against
    ``rag.buildings`` for a borough whose silver join has not been built yet,
    and applies the same two-dimensional test the pipeline does: a footprint
    sharing an edge or a corner with the lot line intersects it and covers none
    of it.

    A lot the fast path reports as empty is re-asked the slow way, the same
    trade `buildings_on_lot` makes. A genuinely vacant lot pays one cheap
    index-bounded query for it, and a borough loaded this morning gets a real
    answer instead of "nothing is built here".
    """
    lot_cte = f"""
        WITH lot AS (
            SELECT l.lot_uid,
                   l.lot_number,
                   l.neighborhood,
                   l.scrape_date,
                   l.geom,
                   COALESCE(l.area_m2, ST_Area(l.geom::geography)) AS area_m2
              FROM {SCHEMA}.lots l
             WHERE regexp_replace(l.lot_number, '\\D', '', 'g')
                 = regexp_replace(%(lot_number)s, '\\D', '', 'g')
               AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
             ORDER BY l.scrape_date DESC
             LIMIT 1
        )"""
    # ``count(DISTINCT building_uid)`` and not ``count(*)``: the silver table's
    # grain is the (building, lot) intersection, and the pipeline's uniqueness
    # check is on the lot number within a partition rather than across
    # boroughs. Counting rows counts intersections.
    tail = """
        SELECT lot.lot_number,
               lot.lot_uid,
               lot.neighborhood,
               lot.scrape_date,
               lot.area_m2                                     AS lot_area_m2,
               COALESCE(c.num_footprints, 0)                   AS num_footprints,
               COALESCE(c.covered_area_m2, 0.0)                AS covered_area_m2
          FROM lot
          LEFT JOIN LATERAL (
              SELECT count(DISTINCT clipped.building_uid)              AS num_footprints,
                     ST_Area(ST_Union(clipped.geom)::geography)        AS covered_area_m2
                FROM clipped
          ) c ON TRUE
    """

    row = None
    if capabilities().building_lots:
        row = query_one(
            lot_cte
            + f""",
        clipped AS (
            SELECT bl.building_uid, bl.geom
              FROM {SILVER_SCHEMA}.building_lot_intersections bl, lot
             WHERE bl.lot_number  = lot.lot_number
               AND bl.neighborhood = lot.neighborhood
               AND bl.scrape_date  = lot.scrape_date
        )"""
            + tail,
            {"lot_number": lot_number, "scrape_date": scrape_date},
        )

    if row is None or not row["num_footprints"]:
        fallback = query_one(
            lot_cte
            + f""",
        clipped AS (
            SELECT b.building_uid, clip.geom
              FROM {SCHEMA}.buildings b, lot
              CROSS JOIN LATERAL (
                  SELECT ST_Intersection(b.geom, lot.geom) AS geom
              ) clip
             WHERE b.neighborhood = lot.neighborhood
               AND b.scrape_date  = lot.scrape_date
               AND b.geom && lot.geom
               AND ST_Intersects(b.geom, lot.geom)
               AND {_BUILDING_CLIP_SCREEN}
        )"""
            + tail,
            {
                "lot_number": lot_number,
                "scrape_date": scrape_date,
                **_building_screen_params(),
            },
        )
        if fallback is not None and (row is None or fallback["num_footprints"]):
            row = fallback

    if row is None:
        return None
    return dict(
        row,
        coverage_pct=coverage_pct(row["covered_area_m2"], row["lot_area_m2"]),
    )


def piece_coverage(
    lot_number: str,
    feature_id: str,
    *,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
) -> dict | None:
    """How much of one zone piece of a lot is built on.

    `lot_coverage` for a piece: the same union of the same lot-clipped
    footprints, cut once more to the (lot × zone) polygon
    ``silver.lot_zone_pieces`` holds for ``feature_id``. One row —
    ``piece_area_m2``, ``lot_area_m2``, ``num_footprints``, ``covered_area_m2``
    and ``coverage_pct`` *of the piece* — or None where no piece of that lot
    carries that zone, or there is no silver clip to cut.

    **This exists because the Lot pane compares today against a proposal that
    is solved per piece.** Since the piece became the unit of work, every
    figure on the proposed side — the plate, the storeys, the floor — belongs
    to one zone's part of the parcel, and the gap table divides the roll's
    floor between the pieces by where the building stands. The footprint was
    the one measure on today's side still read for the whole parcel. Lot
    3 237 014 is 70 095 m², 51 263 of it in E04-064 and 18 806 in E04-065,
    with a 14 830 m² building standing all but entirely in the first; the
    E04-065 piece reported 15 012 m² of ground under four buildings against
    a 6 587 m² plate proposed for a piece that carries 83 m² of building.

    The shapes are ``silver.building_lot_intersections``' — each footprint
    already clipped to *this* lot — so the second clip is against a polygon
    that is only ever inside the parcel, and the pieces' figures sum to the
    parcel's within measurement tolerance. That is also why there is no
    ``rag.buildings`` fallback the way `lot_coverage` has one: the pieces
    table is computed *from* the silver clip, so a database with pieces has
    the clip, and one without pieces has nothing to cut to.

    Unioned rather than summed, for the reason `lot_coverage` gives: ground
    under two overlapping footprints is covered once. The pieces table's own
    ``existing_footprint_m2`` is a planar *sum* of the same clips in the
    metric CRS the pipeline measured its shares in — the right number to
    divide the roll by, and a few square metres off the geodesic union this
    reports, which is the one that adds up to the parcel line above it. A
    clip that is a line or a point is a footprint touching the zone boundary,
    not standing on the piece, and neither counts nor covers; only the
    polygonal part of a clip is kept, so a footprint that both crosses and
    grazes the boundary is unioned as the ground it covers.

    Matched on the lot number and the piece's own snapshot, like everything
    else on the pane; ``lot_uid`` is minted again on every load.
    """
    caps = capabilities()
    if not (caps.building_lots and caps.lot_features):
        return None
    row = query_one(
        f"""
        WITH piece AS (
            SELECT p.lot_uid,
                   p.lot_number,
                   p.feature_id,
                   p.neighborhood,
                   p.scrape_date,
                   p.geom,
                   p.lot_area_m2,
                   p.piece_area_m2,
                   p.num_lot_zones,
                   p.is_primary_zone
              FROM {SILVER_SCHEMA}.lot_zone_pieces p
             WHERE regexp_replace(p.lot_number, '\\D', '', 'g')
                 = regexp_replace(%(lot_number)s, '\\D', '', 'g')
               AND p.feature_id = %(feature_id)s
               AND (%(scrape_date)s::date IS NULL OR p.scrape_date = %(scrape_date)s)
               AND (%(neighborhood)s::text IS NULL OR p.neighborhood = %(neighborhood)s)
             ORDER BY p.scrape_date DESC
             LIMIT 1
        ),
        clipped AS (
            SELECT bl.building_uid, clip.geom
              FROM {SILVER_SCHEMA}.building_lot_intersections bl, piece
              CROSS JOIN LATERAL (
                  SELECT ST_CollectionExtract(
                             ST_Intersection(bl.geom, piece.geom), 3
                         ) AS geom
              ) clip
             WHERE bl.lot_number  = piece.lot_number
               AND bl.neighborhood = piece.neighborhood
               AND bl.scrape_date  = piece.scrape_date
               AND bl.geom && piece.geom
               AND ST_Intersects(bl.geom, piece.geom)
               AND NOT ST_IsEmpty(clip.geom)
        )
        SELECT piece.lot_number,
               piece.lot_uid,
               piece.feature_id,
               piece.neighborhood,
               piece.scrape_date,
               piece.lot_area_m2,
               piece.piece_area_m2,
               piece.num_lot_zones,
               piece.is_primary_zone,
               COALESCE(c.num_footprints, 0)                   AS num_footprints,
               COALESCE(c.covered_area_m2, 0.0)                AS covered_area_m2
          FROM piece
          LEFT JOIN LATERAL (
              SELECT count(DISTINCT clipped.building_uid)              AS num_footprints,
                     ST_Area(ST_Union(clipped.geom)::geography)        AS covered_area_m2
                FROM clipped
          ) c ON TRUE
        """,
        {
            "lot_number": lot_number,
            "feature_id": feature_id,
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
        },
    )
    if row is None:
        return None
    return dict(
        row,
        coverage_pct=coverage_pct(row["covered_area_m2"], row["piece_area_m2"]),
    )


#: A CUBF whose leading digit is one of these is *not* housing: manufacturing
#: (2 and 3, one category over two digits), transport and public services (4),
#: commerce (5), services (6), culture and recreation (7), extraction (8).
#: 1 is *habitation* and 9 is vacant land, water and buildings under
#: construction — nothing standing to be a premises of.
#:
#: The same reading `urban_rag.comparables.CUBF_CLASSES` makes, restated in
#: SQL because this count is taken where the geometry is. It is deliberately
#: the leading digit and nothing else: `rl0105a` is a classification whose
#: first character is the category, so 1000 and 4000 are housing and commerce
#: rather than three thousand apart.
_NONRESIDENTIAL_CUBF_DIGITS = "'2','3','4','5','6','7','8'"

#: Every code from 4510 to 4599 is a piece of the public way — a street, a
#: lane, a right of way, the land under a rail line. The roll files them as
#: assessment units so the ground has a row, then carries them at a nominal
#: $100 with no floor and no storeys. They sit inside the commercial category
#: and are not premises anybody leases, so they are excluded here for the same
#: reason `urban_rag.comparables.ROAD_USE_PREFIX` exists.
_ROAD_CUBF_PREFIX = "45"

#: The non-residential test, once, so the count and the floor area beside it
#: cannot end up asking two different questions.
_NONRESIDENTIAL_UNIT = (
    "u.use_code ~ '^[0-9]{4}$'"
    f" AND left(u.use_code, 1) IN ({_NONRESIDENTIAL_CUBF_DIGITS})"
    f" AND left(u.use_code, 2) <> '{_ROAD_CUBF_PREFIX}'"
)


def lot_roll_units(
    lot_number: str,
    *,
    scrape_date: date | None = None,
    feature_id: str | None = None,
    neighborhood: str | None = None,
) -> dict | None:
    """How many of the roll's records on a lot are non-residential premises.

    The Lot pane's today-against-proposal table compares floor, and floor
    alone cannot say whether 310 m² of commerce is one restaurant or four
    bays. The roll can: it files one assessment unit per premises, and the
    CUBF on each says what that premises is.

    **Not ``rl0312a``.** The roll publishes a *nombre de locaux non
    résidentiels* field and it is very nearly empty — filled on 17 of the
    26,318 assessment units of this borough, so a column reading straight off
    it would print 0 on every lot a reader is likely to click. What is dense
    is the units themselves: 3,364 lots here carry a non-residential dominant
    class. So this counts the records rather than trusting the count they
    carry, which is the same substitution `num_units_by_point` makes upstream.

    **Counted by point in polygon**, the way the dataplatform places a unit
    on a lot at all — the roll gives an address, not a lot number, so a
    spatial test is the only join there is. The polygon is the parcel's, or,
    when ``feature_id`` names a zone, the (lot × zone) piece
    ``silver.lot_zone_pieces`` holds for it: the same test one cut further,
    which is what `piece_coverage` does to the footprints and for the same
    reason. A pane whose floor areas, dwellings and footprint are one piece's
    cannot report the parcel's premises beside them — on lot 3 237 014 that
    put the one unit on the roll on a piece carrying 83 m² of building, when
    its address stands on the other one. A unit does not divide, so this is
    not a share: each record is on exactly one piece, the one its address
    point falls in, and the pieces' counts sum to the parcel's.

    ``roll_loaded`` is what separates "no shop on this lot" from "the roll for
    this snapshot is not in silver". Both are a count of zero and only the
    first is a fact about the lot: ``silver.assessment_units`` here holds the
    2026-09 partition and not the 2026-08 one, so without this flag every lot
    on the older snapshot would report itself as having no premises at all.

    None where the table is absent, which is the Lot pane losing one row of
    one table and nothing else.
    """
    caps = capabilities()
    if not caps.assessment_units:
        return None
    if feature_id is not None and caps.lot_features:
        # The piece, in the piece's own snapshot. Aliased `lot` so the count
        # below is one statement whichever ground it is taken over.
        ground = f"""
            SELECT p.lot_number,
                   p.neighborhood,
                   p.scrape_date,
                   p.geom
              FROM {SILVER_SCHEMA}.lot_zone_pieces p
             WHERE regexp_replace(p.lot_number, '\\D', '', 'g')
                 = regexp_replace(%(lot_number)s, '\\D', '', 'g')
               AND p.feature_id = %(feature_id)s
               AND (%(scrape_date)s::date IS NULL OR p.scrape_date = %(scrape_date)s)
               AND (%(neighborhood)s::text IS NULL OR p.neighborhood = %(neighborhood)s)
             ORDER BY p.scrape_date DESC
             LIMIT 1"""
    else:
        ground = f"""
            SELECT l.lot_number,
                   l.neighborhood,
                   l.scrape_date,
                   l.geom
              FROM {SCHEMA}.lots l
             WHERE regexp_replace(l.lot_number, '\\D', '', 'g')
                 = regexp_replace(%(lot_number)s, '\\D', '', 'g')
               AND (%(scrape_date)s::date IS NULL OR l.scrape_date = %(scrape_date)s)
               AND (%(neighborhood)s::text IS NULL OR l.neighborhood = %(neighborhood)s)
             ORDER BY l.scrape_date DESC
             LIMIT 1"""
    return query_one(
        f"""
        WITH lot AS ({ground}
        )
        SELECT lot.lot_number,
               lot.neighborhood,
               lot.scrape_date,
               EXISTS (
                   SELECT 1
                     FROM {SILVER_SCHEMA}.assessment_units u
                    WHERE u.neighborhood = lot.neighborhood
                      AND u.scrape_date  = lot.scrape_date
                    LIMIT 1
               ) AS roll_loaded,
               COALESCE(p.num_units, 0)                 AS num_units,
               COALESCE(p.num_residential_units, 0)     AS num_residential_units,
               COALESCE(p.num_nonresidential_units, 0)  AS num_nonresidential_units,
               p.nonresidential_floor_area_m2
          FROM lot
          LEFT JOIN LATERAL (
              SELECT count(*)                                  AS num_units,
                     count(*) FILTER (
                         WHERE left(u.use_code, 1) = '1'
                     )                                          AS num_residential_units,
                     count(*) FILTER (WHERE {_NONRESIDENTIAL_UNIT})
                                                                AS num_nonresidential_units,
                     sum(u.floor_area_m2) FILTER (WHERE {_NONRESIDENTIAL_UNIT})
                                                                AS nonresidential_floor_area_m2
                FROM {SILVER_SCHEMA}.assessment_units u
               WHERE u.neighborhood = lot.neighborhood
                 AND u.scrape_date  = lot.scrape_date
                 AND u.geom && lot.geom
                 AND ST_Intersects(lot.geom, u.geom)
          ) p ON TRUE
        """,
        {
            "lot_number": lot_number,
            "scrape_date": scrape_date,
            "feature_id": feature_id,
            "neighborhood": neighborhood,
        },
    )


#: How many doors one site's address list is read down to. A parcel with more
#: than this is a tower's worth of civic numbers and the pane says how many it
#: did not draw rather than paging through them - the counts on every row are
#: taken before the cut, so "showing 250 of 412" is still a true sentence.
MAX_SITE_ADDRESSES = 250


def lot_addresses(
    lot_number: str,
    *,
    scrape_date: date | None = None,
    feature_id: str | None = None,
    neighborhood: str | None = None,
    limit: int = MAX_SITE_ADDRESSES,
) -> list[dict]:
    """The civic addresses standing on a lot, or on one zone piece of it.

    Adresses Quebec publishes ten fields per address point and not one of them
    is cadastral - no lot number, no matricule - so "the address of lot
    2 784 705" is not a question that publisher can be asked. It is answered
    upstream, by putting the point on the polygon:
    ``silver.lot_addresses`` is that join, one row per address point, carrying
    the parcel it fell in and the zone piece within it. See hbu_infra
    sql/026_silver_lot_addresses.sql.

    **A row of that table is an addressable unit, not a door.** Roughly half
    the points in a dense borough carry a unit prefix - ``204-7430 Rue
    Lajeunesse`` and ``7430 Rue Lajeunesse`` are two rows - so counting rows
    would make one walk-up look like eight. This groups them back to the
    *civic* address, which is the grain a door is counted at, and states the
    units under each one in ``num_addressable_units``.

    ``feature_id`` is the piece picker's choice, and narrows the list to the
    addresses standing on that piece - the same cut `piece_coverage` makes to
    the footprints and `lot_roll_units` to the roll, and made here for the
    identical reason: on a lot a zoning boundary crosses, the pane's floor,
    dwellings and footprint are one piece's, and the parcel's addresses beside
    them would name a door on the other site. None is the parcel, which is
    what a lot one zone covers whole means. The app reads both - the piece to
    label it, the parcel to say where the rest of the doors are - the pairing
    `lot_coverage` and `piece_coverage` already make.

    **The counts are computed here rather than read off the table's own
    ``num_piece_civic_addresses``.** That column is the piece's, and this
    function answers at two scopes; one arithmetic that holds for both beats
    a column that is right at one of them and silently wrong at the other.

    Ordered by street and then by civic number, so a corner lot's two streets
    come out grouped rather than interleaved. ``snapped`` is true where any
    unit behind the door was matched by proximity rather than containment -
    the point fell just outside every parcel, within the join's 2 m
    allowance - which is worth a word on the pane, since that address is the
    cadastre's nearest answer rather than its own.

    An empty list where the table is absent, where the borough has not had
    ``make addresses`` run over it, or where the site genuinely holds no
    address - a vacant lot, a lane, a strip of right of way. The pane draws
    nothing in all four cases, which is the one fact they share.
    """
    caps = capabilities()
    if not caps.lot_addresses:
        return []
    # The lot's own snapshot, or the newest this table holds for it. A lot
    # number namespaces on nothing across loads, so without this a borough
    # carrying two partitions reports every door twice - the same trap
    # `lot_coverage` pairs its date against.
    return query(
        f"""
        WITH scoped AS (
            SELECT a.*
              FROM {SILVER_SCHEMA}.lot_addresses a
             WHERE regexp_replace(a.lot_number, '\\D', '', 'g')
                 = regexp_replace(%(lot_number)s, '\\D', '', 'g')
               AND (%(neighborhood)s::text IS NULL
                    OR a.neighborhood = %(neighborhood)s)
               AND (%(feature_id)s::text IS NULL
                    OR a.feature_id = %(feature_id)s)
               AND a.scrape_date = COALESCE(
                       %(scrape_date)s::date,
                       (SELECT max(b.scrape_date)
                          FROM {SILVER_SCHEMA}.lot_addresses b
                         WHERE regexp_replace(b.lot_number, '\\D', '', 'g')
                             = regexp_replace(%(lot_number)s, '\\D', '', 'g')
                           AND (%(neighborhood)s::text IS NULL
                                OR b.neighborhood = %(neighborhood)s))
                   )
        ),
        totals AS (
            SELECT count(*) AS num_addresses,
                   count(DISTINCT
                         COALESCE(civic_address, formatted_address))
                       AS num_civic_addresses
              FROM scoped
        )
        SELECT COALESCE(s.civic_address, s.formatted_address) AS address,
               max(s.street_name)    AS street_name,
               max(s.civic_number)   AS civic_number,
               max(s.civic_suffix)   AS civic_suffix,
               max(s.municipality)   AS municipality,
               max(s.postal_code)    AS postal_code,
               max(s.scrape_date)    AS scrape_date,
               max(s.source_version) AS source_version,
               count(*)              AS num_addressable_units,
               bool_or(s.match_basis = 'snapped') AS snapped,
               t.num_addresses,
               t.num_civic_addresses
          FROM scoped s
          CROSS JOIN totals t
         GROUP BY COALESCE(s.civic_address, s.formatted_address),
                  t.num_addresses, t.num_civic_addresses
         -- The suffix sorts after the bare number it hangs off rather
         -- than with it: ordering on the address string alone put `8558 A
         -- Boulevard Pie-IX` ahead of `8558 Boulevard Pie-IX`, because a
         -- space and an `A` sort under a `B`. NULLS FIRST is what keeps the
         -- plain door first on lot 2 214 032, which has both.
         ORDER BY min(s.street_name) NULLS LAST,
                  min(s.civic_number) NULLS LAST,
                  max(s.civic_suffix) NULLS FIRST,
                  1
         LIMIT %(limit)s
        """,
        {
            "lot_number": lot_number,
            "scrape_date": scrape_date,
            "feature_id": feature_id,
            "neighborhood": neighborhood,
            "limit": int(limit),
        },
    )


def zoning_for_lot(lot_number: str, *, scrape_date: date | None = None) -> list[dict]:
    """The zoning polygons covering a lot, and the grid PDF each links to.

    One row per *distinct zone*, ordered by how much of the lot each covers,
    because a lot on a zone boundary intersects both and only one of them is
    the answer. The overlap is in square metres, not square degrees, either
    way.

    **Distinct is doing work here, and it is not a tidying.** A zone is
    identified by ``(neighborhood, feature_id)`` and nothing else: the
    ``source_table`` slug carries no borough namespace, so C01-001 exists in
    every borough that publishes a VSP_REG_ZONE, while the *same* zone recurs
    once per snapshot the database holds. Without the ``DISTINCT ON`` a caller
    that passes no ``scrape_date`` gets one row per zone per date, and the Lot
    pane offers the reader a choice between a zone and itself. The row kept is
    the newest snapshot's, which is the one the rest of the pane is reading.

    **A sliver is not coverage.** Rows under `MIN_ZONE_OVERLAP_M2` *or* under
    `MIN_ZONE_PCT_OF_LOT` are dropped rather than ranked last: the cadastre and
    the zoning layer are drawn by two publishers who disagree by centimetres,
    so a lot clipping a square metre of the block next door is a survey
    artefact and not a second set of rules anybody could build under. Both
    cutoffs, because a sliver can be large in one measure and not the other -
    1.19 m2 of a commercial zone is over the absolute cutoff and is 0.27 per
    cent of lot 6 291 714. `silver.lot_features` keeps those rows on purpose -
    see the constants - and this is where the question being asked supplies the
    cutoff.

    Reads ``silver.lot_features`` when it is there — the same trade
    `buildings_on_lot` makes, and the same table the pipeline computes once per
    partition instead of once per click. The join to ``rag.features`` is still
    needed for ``attributes``, which the Lot pane renders and the silver table
    does not carry; it is an index lookup on the identity the two share.

    The fast path is also the more correct of the two when no ``scrape_date`` is
    given. The fallback intersects the *newest* lot geometry against features of
    every date; the precomputed rows pair each date's lot with that date's
    features, which is what they actually mean.

    A lot whose every zone is a sliver runs both queries and gets nothing from
    either, since "no rows" is what sends the fast path to the fallback and the
    two apply the same cutoffs. That is a wasted query on the hundred or so
    parcels in a borough that have no zoning but a clipped corner, and not a
    disagreement: the answer is empty because it should be.
    """
    if capabilities().lot_features:
        rows = query(
            f"""
            WITH covering AS (
                SELECT DISTINCT ON (lf.neighborhood, lf.feature_id)
                       lf.feature_id                       AS zone,
                       lf.source_table,
                       lf.neighborhood,
                       lf.scrape_date,
                       f.attributes,
                       f.attributes ->> %(url_attribute)s  AS zoning_pdf_url,
                       lf.overlap_area_m2                  AS overlap_m2,
                       lf.lot_area_m2
                  FROM {SILVER_SCHEMA}.lot_features lf
                  JOIN {SCHEMA}.features f
                    ON f.source_table = lf.source_table
                   AND f.feature_id   = lf.feature_id
                   AND f.neighborhood = lf.neighborhood
                   AND f.scrape_date  = lf.scrape_date
                 WHERE lf.lot_number = %(lot_number)s
                   AND lf.source_table = ANY(%(source_tables)s)
                   AND (%(scrape_date)s::date IS NULL OR lf.scrape_date = %(scrape_date)s)
                   AND lf.overlap_area_m2 >= %(min_overlap_m2)s
                   AND lf.pct_of_lot >= %(min_pct_of_lot)s
                 ORDER BY lf.neighborhood, lf.feature_id,
                          lf.scrape_date DESC, lf.overlap_area_m2 DESC
            )
            SELECT * FROM covering
             ORDER BY overlap_m2 DESC NULLS LAST, zone
            """,
            {
                "lot_number": lot_number,
                "scrape_date": scrape_date,
                "source_tables": list(ZONING_SOURCE_TABLES),
                "url_attribute": ZONING_URL_ATTRIBUTE,
                "min_overlap_m2": MIN_ZONE_OVERLAP_M2,
                "min_pct_of_lot": MIN_ZONE_PCT_OF_LOT,
            },
        )
        if rows:
            return rows

    return query(
        f"""
        WITH lot AS (
            SELECT geom FROM {SCHEMA}.lots
             WHERE lot_number = %(lot_number)s
             ORDER BY scrape_date DESC
             LIMIT 1
        ),
        -- The clip once, not twice: the threshold below filters on the same
        -- area the row reports, and ST_Intersection is the expensive half of
        -- this query while ST_Area is not. Named `clipped` and not `overlaps`
        -- because OVERLAPS is a reserved word - the SQL standard's interval
        -- operator - and a CTE cannot take it.
        clipped AS (
            SELECT f.feature_id                          AS zone,
                   f.source_table,
                   f.neighborhood,
                   f.scrape_date,
                   f.attributes,
                   f.attributes ->> %(url_attribute)s    AS zoning_pdf_url,
                   ST_Area(ST_Intersection(f.geom, lot.geom)::geography) AS overlap_m2,
                   ST_Area(lot.geom::geography)                          AS lot_area_m2
              FROM {SCHEMA}.features f, lot
             WHERE f.source_table = ANY(%(source_tables)s)
               AND f.geom && lot.geom
               AND ST_Intersects(f.geom, lot.geom)
               AND (%(scrape_date)s::date IS NULL OR f.scrape_date = %(scrape_date)s)
        ),
        covering AS (
            -- Both cutoffs, matching the fast path above. The percentage is
            -- computed here rather than read off a column because this branch
            -- has no `silver.lot_features` to read it from. A lot of no area
            -- is not divided by: it keeps whatever cleared the absolute
            -- cutoff, rather than being emptied by a division with no answer.
            SELECT DISTINCT ON (neighborhood, zone) *
              FROM clipped
             WHERE overlap_m2 >= %(min_overlap_m2)s
               AND (lot_area_m2 IS NULL OR lot_area_m2 <= 0
                    OR 100.0 * overlap_m2 / lot_area_m2 >= %(min_pct_of_lot)s)
             ORDER BY neighborhood, zone, scrape_date DESC, overlap_m2 DESC
        )
        SELECT * FROM covering
         ORDER BY overlap_m2 DESC NULLS LAST, zone
        """,
        {
            "lot_number": lot_number,
            "scrape_date": scrape_date,
            "source_tables": list(ZONING_SOURCE_TABLES),
            "url_attribute": ZONING_URL_ATTRIBUTE,
            "min_overlap_m2": MIN_ZONE_OVERLAP_M2,
            "min_pct_of_lot": MIN_ZONE_PCT_OF_LOT,
        },
    )


def zoning_at_point(lon: float, lat: float, *, scrape_date: date | None = None) -> list[dict]:
    """The same, for a click that did not land on any lot.

    No area threshold: a point is inside a zone or it is not, and there is no
    lot for a sliver to be a sliver *of*. The ``DISTINCT ON`` is the same one
    `zoning_for_lot` needs and for the same reason - without a ``scrape_date``
    the newest snapshot's zone would otherwise arrive once per snapshot loaded.
    """
    return query(
        f"""
        WITH at_point AS (
            SELECT DISTINCT ON (f.neighborhood, f.feature_id)
                   f.feature_id                       AS zone,
                   f.source_table,
                   f.neighborhood,
                   f.scrape_date,
                   f.attributes,
                   f.attributes ->> %(url_attribute)s AS zoning_pdf_url,
                   NULL::float8                       AS overlap_m2,
                   NULL::float8                       AS lot_area_m2
              FROM {SCHEMA}.features f
             WHERE f.source_table = ANY(%(source_tables)s)
               AND ST_Intersects(f.geom, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326))
               AND (%(scrape_date)s::date IS NULL OR f.scrape_date = %(scrape_date)s)
             ORDER BY f.neighborhood, f.feature_id, f.scrape_date DESC
        )
        SELECT * FROM at_point
         ORDER BY scrape_date DESC, zone
        """,
        {
            "lon": lon,
            "lat": lat,
            "scrape_date": scrape_date,
            "source_tables": list(ZONING_SOURCE_TABLES),
            "url_attribute": ZONING_URL_ATTRIBUTE,
        },
    )


def zoning_grid_columns(
    zone: str,
    *,
    neighborhood: str | None = None,
    source_table: str | None = None,
    scrape_date: date | None = None,
) -> list[dict]:
    """The parsed *grille des spécifications* for one zone, column by column.

    The second reading of the by-law the Regulations pane draws, and on one of
    the two cities it is the *only* one. Montreal's zoning layer carries its
    norms on the polygon, so `zoning_for_lot` already returns them in
    ``attributes``; Quebec City's carries ``NATURE``, ``STATUT`` and the
    shape's own measurements and states every norm in a city-wide workbook,
    which the dataplatform parses into this table. A pane reading only
    ``attributes`` therefore renders nothing at all over La Cité-Limoilou
    while the values sit one table away.

    One row per *column* of the grid, in the order the sheet prints them. A
    column is one programme the zone permits and not one zone: a mixed zone
    prints a residential column beside a commercial one, each with its own
    uses and its own storey range.

    ``neighborhood`` narrows on purpose and the caller should pass it. The
    zone number carries no city namespace — Quebec's ``11004Mc`` and a
    Montreal ``C04-083`` do not collide today, but nothing in the id prevents
    it — and `zoning_for_lot` has the borough on every row it returns.

    Empty is a real answer twice over: a zone the workbook does not cover, and
    a database whose silver zoning has not been built. The caller separates
    them with `Capabilities.zoning_grid_columns`, which is why this returns
    ``[]`` rather than raising when the table is absent.
    """
    if not capabilities().zoning_grid_columns:
        return []
    return query(
        f"""
        WITH columns AS (
            -- One row per column of one grid, from the newest snapshot that
            -- has it. Without this a caller passing no `scrape_date` gets the
            -- zone's columns once per snapshot loaded and the pane offers the
            -- reader the same grid twice - the trap `zoning_for_lot`'s own
            -- DISTINCT ON exists for.
            SELECT DISTINCT ON (neighborhood, source_table, column_index)
                   feature_id AS zone,
                   grid_zone,
                   column_index,
                   neighborhood,
                   source_table,
                   scrape_date,
                   doc_id,
                   url,
                   usages,
                   levels,
                   permits_residential,
                   residential_floors,
                   solver_ready,
                   solver_error,
                   parse_notes,
                   {", ".join(key for key, _ in ZONING_GRID_COLUMN_FIELDS)}
              FROM {SILVER_SCHEMA}.zoning_grid_columns
             WHERE feature_id = %(zone)s
               AND (%(neighborhood)s::text IS NULL
                    OR neighborhood = %(neighborhood)s)
               AND (%(source_table)s::text IS NULL
                    OR source_table = %(source_table)s)
               AND (%(scrape_date)s::date IS NULL
                    OR scrape_date = %(scrape_date)s)
             ORDER BY neighborhood, source_table, column_index, scrape_date DESC
        )
        SELECT * FROM columns
         ORDER BY neighborhood, source_table, column_index
        """,
        {
            "zone": zone,
            "neighborhood": neighborhood,
            "source_table": source_table,
            "scrape_date": scrape_date,
        },
    )


#: How a per-piece read picks its row when the caller names no zone.
#:
#: The primary piece - the largest - which is the answer these panes gave
#: before a parcel could have more than one, and the row `is_primary_zone`
#: marks on every table downstream of `silver.lot_zone_pieces`. `feature_id`
#: is the tiebreak so the same click always opens the same piece rather than
#: whichever the planner returned first.
#:
#: A caller that wants a *particular* piece passes ``feature_id``; the Lot pane
#: does exactly that once the reader picks one out of `lot_zone_pieces_of`.
_PIECE_ORDER = "ORDER BY g.is_primary_zone DESC NULLS LAST, g.feature_id"


def lot_zone_pieces_of(
    lot_uid: int, *, scrape_date: date | None = None, neighborhood: str | None = None
) -> list[dict]:
    """Every piece of one lot, largest first.

    A zoning boundary does not have to follow a lot line, so a parcel can be
    two sites: lot 1 740 794 is 27 044 m² with 24 596 in H04-072 and 2 440 in
    C04-083, and each has its own street, its own envelope and its own answer.
    This is what lets the Lot pane say so, and offer the reader the choice of
    which one to open - see `lot_capacity`, which takes the zone it returns.

    One row on the great majority of parcels, which is why every caller can
    treat "the first" as "the answer" and only has to do more where the list
    is longer than one.
    """
    if not capabilities().lot_features:
        return []
    return query(
        f"""
        SELECT p.lot_uid,
               p.feature_id,
               p.lot_number,
               p.lot_area_m2,
               p.piece_area_m2,
               p.pct_of_lot,
               p.num_lot_zones,
               p.zone_rank,
               p.is_primary_zone,
               p.primary_frontage_m,
               p.primary_street_name,
               p.secondary_frontage_m,
               p.secondary_street_name,
               p.existing_footprint_m2,
               p.footprint_share,
               p.footprint_share_basis
          FROM {SILVER_SCHEMA}.lot_zone_pieces p
         WHERE p.lot_uid = %(lot_uid)s
           AND (%(scrape_date)s::date IS NULL OR p.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR p.neighborhood = %(neighborhood)s)
         ORDER BY p.zone_rank, p.feature_id
        """,
        {"lot_uid": lot_uid, "scrape_date": scrape_date, "neighborhood": neighborhood},
    )


def lot_capacity(
    lot_uid: int,
    *,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    feature_id: str | None = None,
) -> dict | None:
    """Whether one piece of a lot is used for what it is zoned for.

    ``feature_id`` names which piece. Omitted, this returns the parcel's
    primary one - the largest, and the answer this function gave before a
    parcel could have more than one. `lot_zone_pieces_of` is what lists them.

    The two gold tables read together, because they answer two halves of one
    question and a pane showing either alone would mislead.
    ``gold.lot_redevelopment_gap`` is the subtraction — what stands against
    what the governing envelope would hold. ``gold.lot_highest_best_use`` is
    the programme behind the proposed side: the storeys, the height, the unit
    mix, and the zone whose grid authorised them.

    ``gold.lot_building_massing`` is joined only when it is there, and only for
    ``footprint_fit_pct``. That column is the one caveat this pane cannot
    honestly omit: the solver caps a footprint on the lesser of two *areas*,
    and an area is not a shape, so a fit below 100 means the proposed floor
    area is overstated for this parcel. Reporting a dwelling count without it
    would state a number the lot's own geometry refuses.

    **The caller's ``lot_uid`` is resolved through ``rag.lots``, and the gold
    row is matched on ``lot_number``.** The uid is what the map has — the tile
    carries it and a click hands it back — but it is a bigserial minted fresh
    on every load, so reading gold by it returns nothing at all once the
    cadastre has been reloaded behind a materialized partition, and this pane
    then reports "no highest-and-best-use row for this lot in this snapshot"
    for every parcel in the borough. That reads as a solver that never reached
    the lot rather than as two tables sitting on different generations of the
    same surrogate. The cadastral number survives a reload, so the join is made
    on it and the uid only ever says *which* lot was clicked.
    """
    caps = capabilities()
    if not caps.redevelopment_gap:
        return None

    massing_select, massing_join = "", ""
    if caps.massing:
        massing_select = """,
               m.massing_status,
               m.footprint_fit_pct,
               m.placed_gross_floor_area_m2"""
        massing_join = f"""
          LEFT JOIN {GOLD_SCHEMA}.lot_building_massing m
                 ON m.lot_uid      = g.lot_uid
                AND m.feature_id   = g.feature_id
                AND m.neighborhood = g.neighborhood
                AND m.scrape_date  = g.scrape_date"""

    hbu_select, hbu_join = "", ""
    if caps.highest_best_use:
        hbu_select = """,
               h.grid_zone,
               h.usages,
               h.permits_residential,
               h.permits_commercial,
               h.permits_industrial,
               h.hbu_dominant_use,
               h.buildable_area_m2,
               h.num_candidates,
               h.num_zones,
               h.units,
               h.floors,
               h.height_m,
               h.footprint_m2,
               h.residential_floors,
               h.commercial_floors,
               h.industrial_floors,
               h.total_stalls,
               h.parking_waived,
               h.waived_stalls,
               h.total_capital_cost_cad,
               h.npv_cad,
               h.present_value_cad,
               h.annual_stabilised_noi_cad,
               h.binding"""
        hbu_join = f"""
          LEFT JOIN {GOLD_SCHEMA}.lot_highest_best_use h
                 ON h.lot_uid      = g.lot_uid
                AND h.feature_id   = g.feature_id
                AND h.neighborhood = g.neighborhood
                AND h.scrape_date  = g.scrape_date"""

    return query_one(
        f"""
        SELECT l.lot_uid,
               g.lot_number,
               g.feature_id,
               g.neighborhood,
               g.scrape_date,
               -- The parcel, and the ground this answer is about. They are the
               -- same number wherever one zone covers a lot whole; where they
               -- differ the pane has to show both, or a reader sees a building
               -- priced on a tenth of the area beside it.
               g.lot_area_m2,
               g.piece_area_m2,
               g.num_lot_zones,
               g.is_primary_zone,
               g.primary_frontage_m,
               g.hbu_status,
               g.has_assessment,
               g.is_underbuilt,
               g.existing_floor_area_m2,
               g.hbu_floor_area_m2,
               g.floor_area_gap_m2,
               g.floor_area_gap_sqft,
               g.existing_residential_floor_area_m2,
               g.hbu_residential_floor_area_m2,
               g.existing_commercial_floor_area_m2,
               g.hbu_commercial_floor_area_m2,
               g.existing_industrial_floor_area_m2,
               g.hbu_industrial_floor_area_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               g.dwelling_gap,
               -- The roll's storey count, against ``h.floors`` below. The
               -- one measure of the standing building the Lot pane compared
               -- nothing on, and the one a reader asks for first: 910 m² of
               -- proposed floor over 104 m² standing is a number, and "three
               -- storeys where one stands" is the same fact as a building.
               --
               -- Carried onto every piece of a split parcel unchanged rather
               -- than divided - see `urban_rag.hbu._UNALLOCATED`. Half a
               -- triplex is still three storeys, so this is the parcel's
               -- storey count on a row whose floor areas are the piece's.
               g.existing_num_storeys,
               g.existing_num_assessment_units,
               g.existing_dominant_use_code,
               g.existing_dominant_use_description,
               g.existing_dominant_income_class,
               g.existing_total_assessed_value,
               g.hbu_total_capital_cost_cad,
               g.annual_stabilised_noi_gap_cad,
               g.hbu_npv_cad,
               g.existing_present_value_cad,
               g.redevelopment_npv_gain_cad,
               {_USED_PCT} AS used_pct,
               {_headroom_m2("residential")} AS residential_headroom_m2,
               {_headroom_m2("commercial")}  AS commercial_headroom_m2,
               {_headroom_m2("industrial")}  AS industrial_headroom_m2{hbu_select}{massing_select}
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
          JOIN {SCHEMA}.lots l
            ON l.lot_number   = g.lot_number
           AND l.neighborhood = g.neighborhood
           AND l.scrape_date  = g.scrape_date{hbu_join}{massing_join}
         WHERE l.lot_uid = %(lot_uid)s
           AND (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
           AND (%(feature_id)s::text IS NULL OR g.feature_id = %(feature_id)s)
         ORDER BY g.scrape_date DESC, g.is_primary_zone DESC NULLS LAST,
                  g.feature_id
         LIMIT 1
        """,
        {
            "lot_uid": lot_uid,
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "feature_id": feature_id,
        },
    )


def lot_program(
    lot_uid: int,
    *,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    feature_id: str | None = None,
) -> dict | None:
    """The whole programme proposed for one piece of a lot, as the solver stated it.

    ``feature_id`` names which piece; omitted, this returns the parcel primary
    one - see `lot_capacity`, which resolves the same way and off the same
    `is_primary_zone` flag.

    `lot_capacity` answers *how much more* — a subtraction, three headroom
    figures and a dwelling count. This answers *what, exactly*: the storeys by
    use and the order they stack in, the unit mix by CMHC bedroom class, the
    stalls and where they go, the cost of each part, and the printed caps the
    answer is pressed against. They are different reads because they are
    different questions, and a pane detailing a proposal wants every column of
    the chosen row rather than the four the comparison needed.

    Read off ``gold.lot_highest_best_use`` alone, which is where the *chosen*
    envelope's programme is restated whole — so this needs neither the gap
    table nor a join back to ``silver.lot_development_programs``. The massing
    is joined when it is there, and for more than the fit caveat: the rectangle
    it drew is the only place the proposal has a width, a depth and a bearing,
    and a pane describing a building that names no dimension is describing an
    area.

    All three stall columns are taken rather than ``total_stalls`` alone. The
    three places cost an order of magnitude apart and answer to different
    norms — a dug level is outside the *superficie de plancher* and the site
    coverage both, on a plate of its own under the parcel; a garage bay is
    floor area without being a storey; and a stall on the yard is not in a
    building at all — so the split is the finding and the total on its own
    would hide it.

    Resolved through ``rag.lots`` on ``lot_number``, for the reason
    `lot_capacity` gives at length: the uid the map hands back is a bigserial
    that the next load of the cadastre mints again, and reading gold by it is
    what turns a reload into a borough of lots that all report an unsolved
    programme.

    Returns ``None`` when the table is absent or the lot has no row. A row with
    ``hbu_status`` other than ``solved`` comes back in full: the status, the
    candidate counts and the zone are the answer then, and they are on it.
    """
    caps = capabilities()
    if not caps.highest_best_use:
        return None

    massing_select, massing_join = "", ""
    if caps.massing:
        # Aliased away from `h.footprint_m2` and `h.gross_floor_area_m2`, which
        # are the same measures on the *solved* building rather than on the
        # drawn one. Two columns of one name in a dict row is one column.
        massing_select = """,
               m.massing_status,
               m.placed_footprint_m2,
               m.footprint_shortfall_m2,
               m.footprint_fit_pct,
               m.placed_gross_floor_area_m2,
               m.aspect_ratio,
               m.width_m       AS massing_width_m,
               m.depth_m       AS massing_depth_m,
               m.rotation_deg,
               -- The parking summary rides on the massing table rather than
               -- on the parking one, and that is what makes it readable here:
               -- a lot whose yard took nothing has no row in
               -- gold.lot_surface_parking at all, so joining that table would
               -- give a NULL indistinguishable from a lot that parks
               -- underground. These columns say which.
               m.parking_status,
               m.surface_parking_area_m2,
               m.placed_surface_parking_m2,
               m.placed_surface_stalls,
               m.surface_parking_fit_pct"""
        massing_join = f"""
          LEFT JOIN {GOLD_SCHEMA}.lot_building_massing m
                 ON m.lot_uid      = h.lot_uid
                AND m.feature_id   = h.feature_id
                AND m.neighborhood = h.neighborhood
                AND m.scrape_date  = h.scrape_date"""

    return query_one(
        f"""
        SELECT l.lot_uid,
               h.lot_number,
               h.feature_id,
               h.neighborhood,
               h.scrape_date,
               h.lot_area_m2,
               h.piece_area_m2,
               h.num_lot_zones,
               h.is_primary_zone,
               h.primary_frontage_m,
               h.primary_street_name,
               h.hbu_status,
               h.status,
               h.solved,
               h.solve_error,

               h.num_candidates,
               h.num_governing_candidates,
               h.num_zones,
               h.grid_zone,
               h.source_table,
               h.column_index,
               h.pct_of_lot,
               h.usages,
               h.permits_residential,
               h.permits_commercial,
               h.permits_industrial,
               h.buildable_area_m2,

               h.hbu_dominant_use,
               h.units,
               h.num_dwellings,
               h.floors,
               h.height_m,
               h.footprint_m2,
               h.gross_floor_area_m2,
               h.residential_area_m2,
               h.unit_area_m2,
               h.commercial_area_m2,
               h.industrial_area_m2,
               h.underground_area_m2,
               h.underground_plate_m2,
               h.garage_area_m2,
               h.residential_floors,
               h.commercial_floors,
               h.industrial_floors,
               h.underground_levels,
               h.floor_stack,

               h.underground_stalls,
               h.surface_stalls,
               h.garage_stalls,
               h.total_stalls,
               -- Whether those three are zero because the stalls were waived:
               -- the model was infeasible with the parking and solved without
               -- it. The count beside it is what the programme owes at the
               -- assumed ratios, and the pane says so before any figure.
               h.parking_waived,
               h.waived_stalls,
               -- What the stalls earn and what they buy: the ones rented and
               -- their rent a year, the coverage, and the lease-up months
               -- that coverage saves with the present value of the saving.
               h.rented_stalls,
               h.annual_parking_gross_revenue_cad,
               h.parking_coverage,
               h.lease_up_months_saved,
               h.absorption_value_cad,

               h.construction_cost_cad,
               h.commercial_cost_cad,
               h.industrial_cost_cad,
               h.parking_cost_cad,
               h.total_capital_cost_cad,
               h.annual_gross_revenue_cad,
               h.annual_net_operating_income_cad,
               h.annual_stabilised_noi_cad,
               h.present_value_cad,
               h.npv_cad,

               h.binding,
               h.unpriced_types,
               h.program_assumptions{massing_select}
          FROM {GOLD_SCHEMA}.lot_highest_best_use h
          JOIN {SCHEMA}.lots l
            ON l.lot_number   = h.lot_number
           AND l.neighborhood = h.neighborhood
           AND l.scrape_date  = h.scrape_date{massing_join}
         WHERE l.lot_uid = %(lot_uid)s
           AND (%(scrape_date)s::date IS NULL OR h.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR h.neighborhood = %(neighborhood)s)
           AND (%(feature_id)s::text IS NULL OR h.feature_id = %(feature_id)s)
         ORDER BY h.scrape_date DESC, h.is_primary_zone DESC NULLS LAST,
                  h.feature_id
         LIMIT 1
        """,
        {
            "lot_uid": lot_uid,
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "feature_id": feature_id,
        },
    )


def capacity_totals(
    *, neighborhood: str | None = None, scrape_date: date | None = None
) -> dict | None:
    """How much more the borough could hold, added up over every lot.

    One row. The headline figures are the *positive* headroom — what could be
    added on the lots that have room, with over-built lots contributing zero
    rather than a negative — because "how much more could we build" is a
    question about the parcels where building is possible, and letting a
    century-old six-storey walk-up on a now-three-storey zone cancel a vacant
    lot next door would answer a different question quietly.

    ``net_floor_area_gap_m2`` is that different question, kept beside it: the
    signed sum over the lots the roll actually reached, which is what the
    borough's floor area would become if every parcel were rebuilt exactly to
    today's zoning. In an old and dense borough it can be negative, and that is
    a real finding about the by-law rather than an error.

    The counts are not decoration. ``num_solved`` is how much of the borough
    these totals speak for at all; ``num_without_assessment`` is how many lots
    contributed their whole envelope because nothing is recorded standing on
    them; ``num_over_built`` is how many were clamped. A total read without
    them is a number with no error bar.
    """
    if not capabilities().redevelopment_gap:
        return None
    return query_one(
        f"""
        -- Rows are *pieces* of lots since `silver.lot_zone_pieces`, so the two
        -- counts are two different questions and both are answered: how many
        -- development sites the borough has, and how many parcels they sit on.
        -- A `count(*)` labelled "lots" would have quietly grown by the 3 138
        -- extra pieces VSMPE's split parcels contribute.
        SELECT count(*)                                        AS num_sites,
               count(DISTINCT g.lot_number)                     AS num_lots,
               count(*) FILTER (WHERE g.hbu_status = 'solved')  AS num_solved,
               count(*) FILTER (WHERE g.is_underbuilt)          AS num_underbuilt,
               count(DISTINCT g.lot_number) FILTER (WHERE g.num_lot_zones > 1)
                                                               AS num_split_lots,
               count(*) FILTER (WHERE NOT g.has_assessment)     AS num_without_assessment,
               count(*) FILTER (
                   WHERE g.existing_floor_area_m2 > g.hbu_floor_area_m2
               )                                               AS num_over_built,

               sum({_headroom_m2("residential")}) AS residential_headroom_m2,
               sum({_headroom_m2("commercial")})  AS commercial_headroom_m2,
               sum({_headroom_m2("industrial")})  AS industrial_headroom_m2,
               sum({_headroom_m2("residential")}
                   + {_headroom_m2("commercial")}
                   + {_headroom_m2("industrial")}) AS total_headroom_m2,

               sum(GREATEST(g.hbu_num_dwellings
                            - COALESCE(g.existing_num_dwellings, 0), 0))
                                                   AS additional_dwellings,

               -- How many solved lots were given *any* floor of each class.
               -- Zero headroom and zero proposals are different findings: the
               -- first says the envelopes are full, the second says nothing
               -- of that class was ever modelled, and a bare 0 m² reads as
               -- the first while usually meaning the second. The governing
               -- envelope is by construction a residential column, so its
               -- `permits_commercial` is false and `solve_program` caps
               -- commercial and industrial floors at zero — the pane needs to
               -- be able to say that rather than report "no room".
               count(*) FILTER (WHERE g.hbu_residential_floor_area_m2 > 0)
                                                   AS num_with_residential,
               count(*) FILTER (WHERE g.hbu_commercial_floor_area_m2 > 0)
                                                   AS num_with_commercial,
               count(*) FILTER (WHERE g.hbu_industrial_floor_area_m2 > 0)
                                                   AS num_with_industrial,

               sum(g.existing_floor_area_m2)       AS existing_floor_area_m2,
               sum(g.hbu_floor_area_m2)            AS hbu_floor_area_m2,
               sum(g.floor_area_gap_m2)            AS net_floor_area_gap_m2,
               sum(g.existing_num_dwellings)       AS existing_num_dwellings,
               sum(g.hbu_num_dwellings)            AS hbu_num_dwellings,

               -- The developer's verdict, summed where it is positive: what
               -- redeveloping every lot it pays to redevelop would be worth
               -- over keeping what stands, at the assumptions the solve ran
               -- with. The count says how many lots that is.
               sum(GREATEST(g.redevelopment_npv_gain_cad, 0))
                                                   AS redevelopment_npv_gain_cad,
               count(*) FILTER (WHERE g.redevelopment_npv_gain_cad > 0)
                                                   AS num_npv_gain_positive
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
         WHERE (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood},
    )


def top_capacity_lots(
    *,
    neighborhood: str | None = None,
    scrape_date: date | None = None,
    limit: int = 10,
) -> list[dict]:
    """The lots contributing most of the borough's headroom, largest first.

    A borough total is an average of 25,000 parcels and a handful of enormous
    ones can carry a third of it. In Villeray the two largest contributors are
    a 159 ha and a 94 ha parcel — the scale of a park or a rail yard rather
    than a development site — and a reader given only the total has no way to
    see that, or to judge whether the sites driving it are ones anybody would
    build on.

    So this is not a "top opportunities" list. It is the total showing its own
    working, which is why it returns lot *area* beside the proposed dwellings:
    the parcels worth disbelieving are the ones whose area is implausible for a
    development site, and that is visible at a glance in the two columns
    together.
    """
    if not capabilities().redevelopment_gap:
        return []
    return query(
        f"""
        SELECT g.lot_number,
               g.feature_id,
               g.num_lot_zones,
               g.is_primary_zone,
               g.piece_area_m2,
               g.lot_area_m2,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               g.hbu_floor_area_m2,
               GREATEST(g.hbu_num_dwellings
                        - COALESCE(g.existing_num_dwellings, 0), 0)
                   AS additional_dwellings
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
         WHERE g.hbu_status = 'solved'
           AND (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
         ORDER BY additional_dwellings DESC NULLS LAST
         LIMIT %(limit)s
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood, "limit": limit},
    )


def top_npv_gain_lots(
    *,
    neighborhood: str | None = None,
    scrape_date: date | None = None,
    limit: int = 10,
) -> list[dict]:
    """The lots where redevelopment beats holding by the most, largest first.

    The developer's shortlist, on the developer's number:
    ``redevelopment_npv_gain_cad`` is the discounted value of building the
    lot's highest and best use less the discounted value of keeping what
    stands, at the assumptions the solve ran with. `top_capacity_lots` beside
    this ranks by *room*; this ranks by *money*, and the two lists disagree
    exactly where a big envelope does not pencil.

    The join to ``lot_highest_best_use`` brings the one-word answer to "a
    building of what" (``hbu_dominant_use``) and the program's own economics,
    so the list reads as a shortlist rather than a column of dollars.
    """
    if not capabilities().redevelopment_gap:
        return []
    return query(
        f"""
        SELECT g.lot_number,
               g.feature_id,
               g.num_lot_zones,
               g.is_primary_zone,
               g.piece_area_m2,
               g.lot_area_m2,
               g.redevelopment_npv_gain_cad,
               g.hbu_npv_cad,
               g.existing_present_value_cad,
               g.existing_num_dwellings,
               g.hbu_num_dwellings,
               h.hbu_dominant_use,
               h.floors,
               h.num_dwellings,
               h.commercial_area_m2,
               h.industrial_area_m2,
               h.total_capital_cost_cad
          FROM {GOLD_SCHEMA}.lot_redevelopment_gap g
          LEFT JOIN {GOLD_SCHEMA}.lot_highest_best_use h
                 ON h.lot_uid      = g.lot_uid
                AND h.feature_id   = g.feature_id
                AND h.neighborhood = g.neighborhood
                AND h.scrape_date  = g.scrape_date
         WHERE g.hbu_status = 'solved'
           AND g.redevelopment_npv_gain_cad > 0
           AND (%(scrape_date)s::date IS NULL OR g.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR g.neighborhood = %(neighborhood)s)
         ORDER BY g.redevelopment_npv_gain_cad DESC NULLS LAST
         LIMIT %(limit)s
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood, "limit": limit},
    )


def lot_opportunity(
    lot_uid: int,
    *,
    scrape_date: date | None = None,
    neighborhood: str | None = None,
    feature_id: str | None = None,
) -> dict | None:
    """One piece of a lot, whole, from ``gold.lot_investment_opportunities``.

    ``feature_id`` names which piece; omitted, this returns the parcel's
    primary one - see `lot_capacity`, which resolves the same way. A parcel a
    zoning boundary crosses can carry two theses, and the two can disagree: a
    commercial strip worth redeveloping in front of a yard that is not.

    Both axes and everything behind them: the investment thesis and its rank,
    the site thesis and its rank, the yield each is ordered on, the costs the
    site thesis carries into its denominator, the addition an improvement
    proposes, the heritage flags, and the thresholds the run screened with
    (``screen_assumptions``). The pane that explains *why this lot* reads all
    of it, which is why nothing is left out.

    The caller's ``lot_uid`` is resolved through ``rag.lots`` and the gold row
    matched on ``lot_number``, for the reason `lot_capacity` gives: the uid is
    a bigserial the cadastre load mints again, and this table has already been
    found stranded on an old generation of it once.
    """
    if not capabilities().investment_opportunities:
        return None
    return query_one(
        f"""
        SELECT l.lot_uid,
               o.lot_number,
               o.neighborhood,
               o.scrape_date,
               o.lot_area_m2,
               o.hbu_status,
               o.is_underbuilt,
               o.investment_thesis,
               o.thesis_rank,
               o.is_top_opportunity,
               o.num_ranked_in_thesis,
               o.yield_on_cost_pct,
               o.total_project_cost_cad,
               o.is_land_assessed,
               o.existing_dominant_income_class,
               o.existing_dominant_use_code,
               o.existing_dominant_use_description,
               o.existing_year_built,
               o.existing_num_storeys,
               o.existing_footprint_m2,
               o.existing_floor_area_m2,
               -- What stands there, as a building rather than as a price.
               -- The *Keep* column of both blocks on the Deal pane is these
               -- two and the storey count above them, and both used to read
               -- them off a row that did not carry them: the pane reported
               -- "0 dwellings" and a dash for the income on every lot in the
               -- borough, which is a proposal nobody would keep.
               o.existing_num_dwellings,
               o.existing_annual_stabilised_noi_cad,
               o.existing_total_assessed_value,
               o.hbu_floors,
               o.hbu_footprint_m2,
               o.hbu_floor_area_m2,
               o.hbu_num_dwellings,
               -- The proposed plate split by the family that would occupy it.
               -- The total above says how much would stand; these three say
               -- what it is, which is the whole of the difference between a
               -- rebuild that houses people and one that leases shops.
               o.hbu_residential_floor_area_m2,
               o.hbu_commercial_floor_area_m2,
               o.hbu_industrial_floor_area_m2,
               o.hbu_annual_stabilised_noi_cad,
               o.hbu_total_capital_cost_cad,
               o.hbu_parking_waived,
               o.hbu_waived_stalls,
               o.grid_zone,
               o.heritage_sector,
               o.piia_sector,
               o.storey_headroom,
               o.built_share,
               o.redevelopment_npv_gain_cad,
               o.is_brownfield_use,
               o.is_heritage_sector,
               o.has_piia_review,
               o.demolition_review_required,
               o.is_demolition_restricted,
               o.is_brownfield_site,
               o.is_teardown_site,
               o.is_infill_site,
               o.is_improvement_site,
               o.site_thesis,
               o.improvement_added_storeys,
               o.improvement_floor_m2,
               o.improvement_cost_cad,
               o.improvement_noi_cad,
               o.improvement_yield_pct,
               o.demolition_cost_cad,
               o.site_assessment_cost_cad,
               o.remediation_cost_cad,
               o.site_total_project_cost_cad,
               o.site_yield_on_cost_pct,
               o.site_thesis_rank,
               o.is_top_site_opportunity,
               o.num_ranked_in_site_thesis,
               o.site_verdict_cad,
               o.improvement_source,
               o.screen_assumptions,
               -- the three futures, as the gap solved them
               o.existing_present_value_cad,
               o.hbu_npv_cad,
               o.enhance_status,
               o.enhance_solved,
               o.enhance_floors,
               o.enhance_added_storeys,
               o.enhance_footprint_m2,
               o.enhance_gross_floor_area_m2,
               o.enhance_added_floor_area_m2,
               o.enhance_added_dwellings,
               o.enhance_num_dwellings,
               o.enhance_units,
               o.enhance_added_commercial_area_m2,
               o.enhance_added_industrial_area_m2,
               o.enhance_surface_stalls,
               o.enhance_parking_waived,
               o.enhance_waived_stalls,
               o.enhance_capital_cost_cad,
               o.enhance_added_annual_gross_income_cad,
               o.enhance_added_annual_stabilised_noi_cad,
               o.enhance_present_value_cad,
               o.enhance_npv_cad,
               o.enhance_disruption_cad,
               o.enhance_gain_cad,
               o.enhance_assumptions,
               o.hold_value_cad,
               o.enhance_value_cad,
               o.rebuild_value_cad,
               o.best_future,
               -- and priced for the owner and for a buyer
               o.site_costs_cad,
               o.owner_hold_value_cad,
               o.owner_enhance_value_cad,
               o.owner_rebuild_value_cad,
               o.owner_gain_enhance_cad,
               o.owner_gain_rebuild_cad,
               o.owner_best_future,
               o.acquisition_cost_cad,
               o.buyer_npv_hold_cad,
               o.buyer_npv_enhance_cad,
               o.buyer_npv_rebuild_cad,
               o.buyer_yield_hold_pct,
               o.buyer_yield_enhance_pct,
               o.buyer_yield_rebuild_pct,
               o.residual_price_enhance_cad,
               o.residual_price_rebuild_cad,
               o.buyer_best_future,
               -- and the returns: the all-in budget, the IRRs, the screens
               o.comparable_cap_rate_pct,
               o.market_cap_rate_pct,
               o.rebuild_budget_cad,
               o.rebuild_soft_cost_cad,
               o.rebuild_contingency_cad,
               o.rebuild_builders_risk_cad,
               o.rebuild_lease_up_months,
               o.rebuild_total_development_cost_cad,
               o.enhance_budget_cad,
               o.enhance_lease_up_months,
               o.enhance_total_development_cost_cad,
               o.buyer_yoc_hold_pct,
               o.buyer_yoc_enhance_pct,
               o.buyer_yoc_rebuild_pct,
               o.buyer_irr_hold_pct,
               o.buyer_irr_enhance_pct,
               o.buyer_irr_rebuild_pct,
               o.buyer_multiple_rebuild,
               o.buyer_multiple_enhance,
               o.owner_yoc_rebuild_pct,
               o.owner_yoc_enhance_pct,
               o.owner_irr_rebuild_pct,
               o.owner_irr_enhance_pct,
               o.yoc_spread_rebuild_bps,
               o.yoc_spread_enhance_bps,
               o.site_irr_pct,
               o.owner_site_irr_pct,
               o.site_all_in_yield_on_cost_pct,
               o.site_yoc_spread_bps,
               o.clears_cap_rate,
               o.clears_hurdle,
               o.is_good_candidate
          FROM {GOLD_SCHEMA}.lot_investment_opportunities o
          JOIN {SCHEMA}.lots l
            ON l.lot_number   = o.lot_number
           AND l.neighborhood = o.neighborhood
           AND l.scrape_date  = o.scrape_date
         WHERE l.lot_uid = %(lot_uid)s
           AND (%(scrape_date)s::date IS NULL OR o.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR o.neighborhood = %(neighborhood)s)
           AND (%(feature_id)s::text IS NULL OR o.feature_id = %(feature_id)s)
         ORDER BY o.scrape_date DESC, o.is_primary_zone DESC NULLS LAST,
                  o.feature_id
         LIMIT 1
        """,
        {
            "lot_uid": lot_uid,
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "feature_id": feature_id,
        },
    )


#: The three futures, in the order the dataplatform resolves a tie.
FUTURES: tuple[str, ...] = ("hold", "enhance", "rebuild")


def futures_totals(
    *, neighborhood: str | None = None, scrape_date: date | None = None
) -> list[dict]:
    """One row per future: how many lots it wins for the owner and for a
    buyer, and what those wins are worth. The borough-level read of the two
    panes, for the Overview."""
    if not capabilities().investment_opportunities:
        return []
    return query(
        f"""
        SELECT f.future,
               count(*) FILTER (WHERE o.owner_best_future = f.future) AS owner_wins,
               count(*) FILTER (WHERE o.buyer_best_future = f.future) AS buyer_wins,
               sum(CASE f.future
                     WHEN 'enhance' THEN o.owner_gain_enhance_cad
                     WHEN 'rebuild' THEN o.owner_gain_rebuild_cad
                     ELSE 0 END)
                 FILTER (WHERE o.owner_best_future = f.future) AS owner_gain_cad,
               sum(CASE f.future
                     WHEN 'hold' THEN o.buyer_npv_hold_cad
                     WHEN 'enhance' THEN o.buyer_npv_enhance_cad
                     ELSE o.buyer_npv_rebuild_cad END)
                 FILTER (WHERE o.buyer_best_future = f.future) AS buyer_npv_cad
          FROM (VALUES ('hold'), ('enhance'), ('rebuild')) AS f(future)
          CROSS JOIN {GOLD_SCHEMA}.lot_investment_opportunities o
         WHERE o.owner_best_future IS NOT NULL
           AND (%(scrape_date)s::date IS NULL OR o.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR o.neighborhood = %(neighborhood)s)
         GROUP BY f.future
         ORDER BY array_position(ARRAY['hold','enhance','rebuild'], f.future)
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood},
    )


def top_site_opportunities(
    *,
    site_thesis: str | None = None,
    neighborhood: str | None = None,
    scrape_date: date | None = None,
    limit: int = 10,
) -> list[dict]:
    """The best-ranked lots of one site thesis, or of every thesis at once.

    Ordered the way the dataplatform ordered them - by ``site_thesis_rank``
    within the thesis, which is that thesis's own yield on cost with the
    verdict as tiebreak - so this list and the table's `is_top_site_
    opportunity` flag agree. Asked for every thesis, it interleaves them
    rank by rank rather than letting one thesis's yields bury the others,
    which is what the faceting exists for.
    """
    if not capabilities().investment_opportunities:
        return []
    return query(
        f"""
        SELECT o.lot_number,
               o.feature_id,
               o.num_lot_zones,
               o.is_primary_zone,
               o.piece_area_m2,
               o.lot_area_m2,
               o.site_thesis,
               o.investment_thesis,
               o.site_thesis_rank,
               o.num_ranked_in_site_thesis,
               o.site_yield_on_cost_pct,
               o.site_total_project_cost_cad,
               o.redevelopment_npv_gain_cad,
               o.improvement_noi_cad,
               o.improvement_floor_m2,
               o.existing_year_built,
               o.existing_num_storeys,
               o.hbu_floors,
               o.existing_dominant_use_description,
               o.grid_zone,
               o.is_heritage_sector,
               o.has_piia_review,
               o.demolition_review_required,
               o.site_irr_pct,
               o.site_all_in_yield_on_cost_pct,
               o.site_yoc_spread_bps,
               o.market_cap_rate_pct,
               o.is_good_candidate
          FROM {GOLD_SCHEMA}.lot_investment_opportunities o
         WHERE o.site_thesis_rank IS NOT NULL
           AND (%(site_thesis)s::text IS NULL OR o.site_thesis = %(site_thesis)s)
           AND (%(scrape_date)s::date IS NULL OR o.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR o.neighborhood = %(neighborhood)s)
         ORDER BY o.site_thesis_rank ASC, o.site_thesis ASC
         LIMIT %(limit)s
        """,
        {
            "site_thesis": site_thesis,
            "scrape_date": scrape_date,
            "neighborhood": neighborhood,
            "limit": limit,
        },
    )


def site_thesis_totals(
    *, neighborhood: str | None = None, scrape_date: date | None = None
) -> list[dict]:
    """One row per site thesis: how many lots it filed, how many pay, and
    what the ranked ones yield - the borough-level read of the second axis,
    for the Overview pane. A thesis nothing fell in is absent rather than
    zero; the pane fills the gaps from `SITE_THESES` so an empty facet is
    still an answer.
    """
    if not capabilities().investment_opportunities:
        return []
    return query(
        f"""
        SELECT o.site_thesis,
               -- Sites, and the parcels they sit on. A row is a piece of a
               -- lot, and a parcel a zoning boundary crosses can file two
               -- pieces under the same thesis.
               count(*)                                   AS num_lots,
               count(DISTINCT o.lot_number)               AS num_parcels,
               count(o.site_thesis_rank)                  AS num_ranked,
               count(*) FILTER (WHERE o.is_top_site_opportunity) AS num_top,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY o.site_yield_on_cost_pct)
                 FILTER (WHERE o.site_thesis_rank IS NOT NULL) AS median_site_yield_on_cost_pct,
               max(o.site_yield_on_cost_pct)
                 FILTER (WHERE o.site_thesis_rank IS NOT NULL) AS best_site_yield_on_cost_pct,
               sum(o.redevelopment_npv_gain_cad)
                 FILTER (WHERE o.site_thesis_rank IS NOT NULL
                           AND o.site_thesis <> 'improvement') AS ranked_npv_gain_cad,
               sum(o.improvement_noi_cad)
                 FILTER (WHERE o.site_thesis_rank IS NOT NULL
                           AND o.site_thesis = 'improvement') AS ranked_improvement_noi_cad,
               -- The **piece** area, not the parcel's: summing `lot_area_m2`
               -- over rows that are pieces counts a split parcel once per
               -- piece, so a 27 044 m² lot filed under one thesis twice would
               -- contribute 54 088 m² of ground the borough does not have.
               -- COALESCE for a partition written before the pieces, where
               -- the row *is* the lot and the two are the same number.
               sum(COALESCE(o.piece_area_m2, o.lot_area_m2))
                 FILTER (WHERE o.site_thesis_rank IS NOT NULL) AS ranked_lot_area_m2,
               count(*) FILTER (WHERE o.is_heritage_sector) AS num_heritage_sector,
               count(*) FILTER (WHERE o.has_piia_review)    AS num_piia_review,
               count(*) FILTER (WHERE o.is_good_candidate)  AS num_good_candidates,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY o.site_irr_pct)
                 FILTER (WHERE o.site_thesis_rank IS NOT NULL) AS median_site_irr_pct,
               max(o.site_irr_pct)
                 FILTER (WHERE o.site_thesis_rank IS NOT NULL) AS best_site_irr_pct
          FROM {GOLD_SCHEMA}.lot_investment_opportunities o
         WHERE o.site_thesis IS NOT NULL AND o.site_thesis <> 'none'
           AND (%(scrape_date)s::date IS NULL OR o.scrape_date = %(scrape_date)s)
           AND (%(neighborhood)s::text IS NULL OR o.neighborhood = %(neighborhood)s)
         GROUP BY o.site_thesis
        """,
        {"scrape_date": scrape_date, "neighborhood": neighborhood},
    )


def lot_documents(
    lot_uid: int,
    *,
    source_table: str | None = None,
    min_overlap_m2: float = MIN_ZONE_OVERLAP_M2,
    min_pct_of_lot: float = MIN_ZONE_PCT_OF_LOT,
) -> list[dict]:
    """Every by-law document that applies to one lot, most of the lot first.

    ``rag.lot_documents`` - hbu_infra's 006_lot_documents.sql - is the join
    this reads, and it is the last hop of a chain the other two repos have
    already walked: ``silver.lot_features`` says which map features cover the
    lot, ``rag.chunks.feature_ids`` says which features cite each document, and
    the view puts the two together so "lot 2 170 935" becomes "these sheets".

    Next to `zoning_for_lot`, which answers a neighbouring question and is not
    a substitute. That one reads the *grid values* off the zoning row and finds
    the PDF in an attribute, so it sees exactly the one layer whose scrape
    carries a ``LIEN_GRILLE``. This one comes at it from the corpus side: a
    document reaches a lot because it was embedded citing a feature that covers
    it, which holds for a zone whose attribute an older scrape dropped and for
    any layer the dataplatform starts indexing later, without this file
    changing.

    One row per *document*, not per (document, feature). A grid cited by both
    zones of a split lot is one sheet with two zones on it, and the view - one
    row per feature - would otherwise offer the reader the same PDF twice.
    ``zones`` collects those feature ids and ``pct_of_lot`` keeps the largest
    share any one of them covers, which is what ranks the sheets against each
    other; ``overlap_m2`` is their total, the share of the lot the document
    governs at all.

    ``min_overlap_m2`` and ``min_pct_of_lot`` are the survey-artefact cutoffs
    `MIN_ZONE_OVERLAP_M2` and `MIN_ZONE_PCT_OF_LOT` describe, applied here for
    the same reason the Lot pane applies them: under about a square metre, or
    under one per cent of the parcel, the cadastre and the zoning layer have
    simply missed each other along a lot line, and the sheet that comes back is
    the block next door's. Both, because a clip can be over one cutoff and
    under the other - 1.19 m2 on a 438 m2 lot is 0.27 per cent of it, and used
    to bring back a second grid. The view carries both columns and thresholds
    neither on purpose, because the cutoff belongs to the question - this is a
    question.

    Keyed on ``lot_uid``, like `lot_capacity` and like the view itself: one
    ``lot_uid`` is one lot in one snapshot, so there is no ``scrape_date`` to
    pass and no way for a lot's 2025 zones to arrive beside its 2026 ones.
    """
    return query(
        f"""
        WITH applies AS (
            -- One row per (document, feature) before anything is grouped. The
            -- view's `documents` CTE is DISTINCT over feature_ids among other
            -- columns, so a document whose chunks disagree about which
            -- features cite it arrives more than once per feature - and would
            -- be counted that many times in the sum below.
            SELECT DISTINCT ON (d.doc_id, d.feature_id)
                   d.doc_id, d.url, d.title, d.source_table,
                   d.neighborhood, d.scrape_date, d.feature_id,
                   d.pct_of_lot, d.overlap_area_m2, d.coverage_rank
              FROM {SCHEMA}.lot_documents d
             WHERE d.lot_uid = %(lot_uid)s
               AND d.overlap_area_m2 >= %(min_overlap_m2)s
               AND d.pct_of_lot >= %(min_pct_of_lot)s
               AND (%(source_table)s::text IS NULL
                    OR d.source_table = %(source_table)s)
             ORDER BY d.doc_id, d.feature_id, d.pct_of_lot DESC
        )
        SELECT doc_id, url, title, source_table, neighborhood, scrape_date,
               array_agg(feature_id ORDER BY pct_of_lot DESC, feature_id)
                                    AS zones,
               max(pct_of_lot)      AS pct_of_lot,
               sum(overlap_area_m2) AS overlap_m2,
               -- Rank is per (lot, layer), so the minimum is "this document
               -- is the dominant one for its own layer" and not a comparison
               -- with a document from another.
               min(coverage_rank)   AS coverage_rank
          FROM applies
         GROUP BY doc_id, url, title, source_table, neighborhood, scrape_date
         ORDER BY min(coverage_rank), max(pct_of_lot) DESC NULLS LAST, doc_id
        """,
        {
            "lot_uid": lot_uid,
            "min_overlap_m2": min_overlap_m2,
            "min_pct_of_lot": min_pct_of_lot,
            "source_table": source_table,
        },
    )


def zoning_pdf_url_fallback(zone: str, *, source_table: str | None = None) -> str | None:
    """The grid PDF for a zone, when its own row does not carry the link.

    Two routes, in this order.

    ``rag.chunks`` records the URL it embedded and the feature ids that cited
    it, so a zone whose ``attributes`` lost its link — an older scrape, a layer
    the city reshaped — can still be resolved from the document side. Every
    zoning layer and not just the corpus's own, matching the reads above:
    scoping this to `ZONING_SOURCE_TABLE` alone meant a Quebec City zone was
    looked up under Montreal's slug and could only ever miss.

    Failing that, `ZONING_PDF_URL_TEMPLATES` builds the link from the zone
    code, which is what reaches a city that serves its grids from a handler
    rather than linking them from the layer. Second and not first on purpose:
    a URL that was actually indexed is a fact, and a constructed one is this
    app's guess about how a city arranges its site.

    The corpus is skipped rather than assumed when `rag.chunks` is absent, so a
    database with no corpus at all still reaches the derived link — a borough
    loaded this morning has its zones long before ``document_index`` runs.
    """
    if capabilities().chunks:
        url = scalar(
            f"""
            SELECT c.url
              FROM {SCHEMA}.chunks c
             WHERE c.source_table = ANY(%(source_tables)s)
               AND c.feature_ids ? %(zone)s
             ORDER BY c.scrape_date DESC
             LIMIT 1
            """,
            {"source_tables": list(ZONING_SOURCE_TABLES), "zone": zone},
        )
        if url:
            return url
    return zoning_pdf_url_from_template(zone, source_table=source_table)


def zoning_pdf_url_from_template(
    zone: str, *, source_table: str | None = None
) -> str | None:
    """The grid's URL built from the zone code, for a city that serves them.

    Only where `ZONING_PDF_URL_TEMPLATES` has an entry for this layer, and
    nothing is guessed when the caller does not say which layer the zone came
    off: the zone code carries no city namespace, so picking a template by
    trying each in turn would hand a Montreal zone to Quebec City's handler and
    get back a page that is not a grid.
    """
    if not zone or not source_table:
        return None
    template = ZONING_PDF_URL_TEMPLATES.get(source_table)
    return template.format(zone=zone) if template else None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _vector_literal(embedding: list[float]) -> str:
    """pgvector's text input format.

    Passed as a string with an explicit ``::vector`` cast rather than through
    pgvector's psycopg adapter, so retrieval needs no type registration on a
    connection borrowed from the pool.
    """
    return "[" + ",".join(f"{float(v):.7g}" for v in embedding) + "]"


def search_at_lot(
    embedding: list[float], lon: float, lat: float, *, match_count: int = 5
) -> list[dict]:
    """Everything the corpus says about the lot a point falls in.

    ``rag.search_at_lot`` — containment rather than proximity, so there is no
    radius to pick and no neighbouring zone bleeding into the answer.
    """
    return query(
        f"SELECT * FROM {SCHEMA}.search_at_lot(%s::vector, %s, %s, %s)",
        (_vector_literal(embedding), lon, lat, match_count),
    )


def search_near(
    embedding: list[float],
    lon: float,
    lat: float,
    *,
    radius_m: float = 500,
    match_count: int = 5,
    on_scrape_date: date | None = None,
) -> list[dict]:
    """Vector search narrowed to a radius around a point (``rag.search_near``).

    The spatial filter runs first and is selective, so the planner scans the
    surviving chunks rather than using the HNSW index — which makes recall here
    exact rather than approximate.
    """
    return query(
        f"SELECT * FROM {SCHEMA}.search_near(%s::vector, %s, %s, %s, %s, %s)",
        (_vector_literal(embedding), lon, lat, radius_m, match_count, on_scrape_date),
    )


def search_corpus(
    embedding: list[float],
    *,
    match_count: int = 5,
    neighborhood: str | None = None,
    scrape_date: date | None = None,
) -> list[dict]:
    """Unfiltered vector search — the question with no place attached.

    ``hnsw.ef_search`` is widened because the index returns its candidates and
    the ``WHERE`` clause is applied to them afterwards: filtering by
    neighborhood is a reason to ask for more candidates, not fewer.
    """
    from psycopg.rows import dict_row  # noqa: PLC0415

    from src.utils.db import connection  # noqa: PLC0415

    # SET takes no bound parameters, so the value is inlined — as an int, which
    # is where the coercion belongs rather than in a format string.
    ef_search = int(max(100, 4 * match_count))
    # SET LOCAL is scoped to a transaction, and the pool's connections are
    # autocommit, so without an explicit one this would be a no-op that
    # Postgres only warns about. The explicit block also returns the setting to
    # its default when the connection goes back to the pool.
    with connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
        cur.execute(
            f"""
            SELECT c.chunk_id, c.doc_id, c.url, c.title, c.source_table,
                   c.neighborhood, c.scrape_date, c.text AS chunk_text,
                   1 - (c.embedding <=> %(embedding)s::vector) AS similarity
              FROM {SCHEMA}.chunks c
             WHERE (%(neighborhood)s::text IS NULL OR c.neighborhood = %(neighborhood)s)
               AND (%(scrape_date)s::date IS NULL OR c.scrape_date = %(scrape_date)s)
             ORDER BY c.embedding <=> %(embedding)s::vector
             LIMIT %(match_count)s
            """,
            {
                "embedding": _vector_literal(embedding),
                "neighborhood": neighborhood,
                "scrape_date": scrape_date,
                "match_count": match_count,
            },
        )
        return list(cur.fetchall())


def _safe(identifier: str) -> str:
    """Guard the two table names that reach SQL as text rather than as a param."""
    allowed = {"lots", "buildings", "features", "chunks"}
    if identifier not in allowed:
        raise ValueError(f"{identifier!r} is not one of {sorted(allowed)}")
    return identifier
