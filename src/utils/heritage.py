"""What a lot's heritage rows mean, and how much each one weighs.

`queries.heritage_for_lot` hands back raw feature rows - a layer slug and the
attributes its publisher wrote. This module turns each into a `Protection`: a
designation a reader recognises, the name of the thing protected, the detail
that grades it, a link, and a *tier* saying how hard it binds.

Pure on purpose: no database and no Streamlit, so the tiers are tested here
rather than through a rendered pane.

**The tiers are this app's reading, not a statutory scale.** Nothing in the
Loi sur le patrimoine culturel ranks a classement against a citation. What
they say is what a developer meets:

* ``High`` - an authorisation is required before demolition or exterior
  work (a provincial classement, a municipal citation, a heritage site), or
  the city's own inventory grades the building *exceptionnel* or
  *supérieur*.
* ``Moderate`` - the work is reviewed rather than authorised: a protection
  area around someone else's monument, a Montreal heritage sector or listed
  building, a building the inventory grades *bon*.
* ``Low`` - recognition with little or no permit consequence of its own: a
  federal designation on a private lot, a building graded *faible*, a
  Montreal building of local interest, a studied building since demolished.
* ``Ungraded`` - Quebec City's inventory codes 5 *présumé* and 6
  *confirmé*: an interest was presumed or confirmed and never graded. They
  are *not* below *faible* (see `hbu_dataplatform`'s
  `quebec.UNGRADED_CODES`); they are unknown, and sort last only because a
  reader wants the known constraints first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

HIGH = "High"
MODERATE = "Moderate"
LOW = "Low"
UNGRADED = "Ungraded"

#: Display order, most binding first.
TIERS: tuple[str, ...] = (HIGH, MODERATE, LOW, UNGRADED)

#: A marker per tier, for a table cell where colour is not available.
TIER_MARKERS: dict[str, str] = {
    HIGH: "🔴",
    MODERATE: "🟠",
    LOW: "🟡",
    UNGRADED: "⚪",
}

#: Quebec City's studied-building grade, by the code
#: ``EVALUATION_VALEUR_PATRIMO_NO`` carries. 5 and 6 are not grades.
QUEBEC_GRADES: dict[int, tuple[str, str]] = {
    1: ("exceptionnel", HIGH),
    2: ("supérieur", HIGH),
    3: ("bon", MODERATE),
    4: ("faible", LOW),
    5: ("présumé", UNGRADED),
    6: ("confirmé", UNGRADED),
}


@dataclass(frozen=True)
class Layer:
    """One heritage layer: what it is called and what it means on a lot."""

    designation: str
    tier: str
    jurisdiction: str
    #: Whether the layer's polygons are drawn around one immovable rather than
    #: an area, which is what decides the cutoff `queries.heritage_for_lot`
    #: applies to a neighbour's strip of it.
    footprint: bool
    meaning: str


#: Quebec City's layers, keyed on the slug `_quebec_heritage` files them
#: under. The tier of ``BATIMENT_ETUDIE`` is its grade's, set per row.
QUEBEC_LAYERS: dict[str, Layer] = {
    "Patrimoine__IMMEUBLE_CLASSE": Layer(
        "Immeuble patrimonial classé", HIGH, "Provincial", True,
        "Classified under the *Loi sur le patrimoine culturel*: altering, "
        "restoring, repairing or demolishing it, in whole or in part, needs "
        "the Minister of Culture's authorisation.",
    ),
    "Patrimoine__IMMEUBLE_CITE": Layer(
        "Immeuble patrimonial cité", HIGH, "Municipal", True,
        "Cited by the city under the *Loi sur le patrimoine culturel*: "
        "demolishing or moving it needs council's authorisation, and "
        "alterations follow the conditions council sets.",
    ),
    "Patrimoine__SITE_DECLARE_CLASSE": Layer(
        "Site patrimonial déclaré ou classé", HIGH, "Provincial", False,
        "Inside a heritage site designated by the province: subdividing, "
        "building, altering a building's exterior or demolishing needs the "
        "Minister of Culture's authorisation.",
    ),
    "Patrimoine__SITE_CITE": Layer(
        "Site patrimonial cité", HIGH, "Municipal", False,
        "Inside a heritage site the city has cited: demolition needs "
        "council's authorisation, and new construction and exterior "
        "alterations follow the conditions it sets.",
    ),
    "Patrimoine__AIRE_PROTECTION": Layer(
        "Aire de protection", MODERATE, "Provincial", False,
        "Inside the protection area drawn around a classified immovable "
        "nearby: building, demolishing and subdividing here need the "
        "Minister's authorisation, which is judged on the setting of that "
        "monument rather than on this lot's own building.",
    ),
    "Patrimoine__DESIGNE_FEDERAL": Layer(
        "Désignation fédérale", LOW, "Federal", True,
        "Designated by the federal government - a national historic site or "
        "a federal heritage building. It binds federal owners; on its own it "
        "puts no permit condition on a private one. A building the province "
        "has also classified is listed again above, at that tier.",
    ),
    "Patrimoine__BATIMENT_ETUDIE": Layer(
        "Bâtiment d'intérêt patrimonial", UNGRADED, "Municipal", True,
        "In the city's inventory of *patrimoine bâti*, with the heritage "
        "value it graded the building at. The grade is what the city weighs "
        "when demolition or exterior work comes before it; the fiche has the "
        "year, style and the reasons.",
    ),
}

#: Montreal's, by the table's suffix: a Spectrum slug carries the borough's
#: own prefix (``Reglement_urbanisme__VSP_REG_...`` in VSMPE), so the same
#: layer from another borough differs only before it.
MONTREAL_LAYERS: dict[str, Layer] = {
    "_REG_BATIMENT_PATRIMONIAL": Layer(
        "Bâtiment de valeur patrimoniale", MODERATE, "Borough", False,
        "Listed by the borough as a building of heritage value: demolition "
        "goes to the demolition committee, and exterior work is reviewed "
        "against what the listing protects.",
    ),
    "_REG_SECTEUR_PATRIMONAL": Layer(
        "Secteur d'intérêt patrimonial", MODERATE, "Borough", False,
        "Inside a heritage sector of the plan d'urbanisme: demolition goes "
        "to the borough's demolition committee with a heritage study, and "
        "new work is reviewed for how it fits the sector.",
    ),
    "_REG_BATIMENT_INTERET_LOCAL": Layer(
        "Bâtiment d'intérêt local", LOW, "Borough", False,
        "Noted by the borough as a building of local interest - a "
        "recognition rather than a permit condition of its own.",
    ),
}
# The city spells its own layer both ways across boroughs.
MONTREAL_LAYERS["_REG_SECTEUR_PATRIMONIAL"] = MONTREAL_LAYERS["_REG_SECTEUR_PATRIMONAL"]

#: Every heritage slug, as one POSIX regex for the SQL to filter on.
SOURCE_TABLE_PATTERN = (
    "^(" + "|".join(re.escape(slug) for slug in QUEBEC_LAYERS) + ")$"
    "|(" + "|".join(re.escape(suffix) for suffix in MONTREAL_LAYERS) + ")$"
)

#: The slugs whose polygons are buildings, for the footprint cutoff.
FOOTPRINT_TABLES: tuple[str, ...] = tuple(
    slug for slug, layer in QUEBEC_LAYERS.items() if layer.footprint
)


#: Within a tier, which designation leads: an authorisation regime before the
#: inventory that informs one, so a classified house graded *exceptionnel*
#: headlines as classified.
LAYER_ORDER: dict[str, int] = {
    layer.designation: i
    for i, layer in enumerate([*QUEBEC_LAYERS.values(), *MONTREAL_LAYERS.values()])
}

#: The status layers' jurisdiction labels, in the language the rest of the
#: pane is in.
JURISDICTIONS: dict[str, str] = {
    "fédérale / provinciale": "Federal / Provincial",
    "fédérale": "Federal",
    "provinciale": "Provincial",
    "municipale": "Municipal",
}


def layer_for(source_table: str) -> Layer | None:
    """The layer a slug names, or None for one that is not a heritage layer."""
    if source_table in QUEBEC_LAYERS:
        return QUEBEC_LAYERS[source_table]
    for suffix, layer in MONTREAL_LAYERS.items():
        if source_table.endswith(suffix):
            return layer
    return None


@dataclass(frozen=True)
class Protection:
    """One heritage row, read."""

    tier: str
    designation: str
    jurisdiction: str
    name: str | None
    detail: str | None
    url: str | None
    meaning: str
    coverage: str
    source_table: str
    feature_id: str

    @property
    def rank(self) -> tuple[int, int]:
        """Tier first; within a tier, the statutory designation before the inventory."""
        return TIERS.index(self.tier), LAYER_ORDER.get(self.designation, len(LAYER_ORDER))


def _text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _grade(code: object) -> int | None:
    try:
        return int(float(code))  # published as 2.0 as often as 2
    except (TypeError, ValueError):
        return None


def _coverage(row: Mapping, source_table: str) -> str:
    """Where on the lot the feature is, in words.

    Only the inventory's outlines are buildings. A classified or federally
    designated immovable is drawn with its grounds - Maison Hamel-Bruneau's
    is 7 398 m2 - so it is reported as ground covered, like a site.
    """
    overlap = float(row.get("overlap_m2") or 0)
    if overlap <= 0:
        return "Marked on the lot"
    if source_table == "Patrimoine__BATIMENT_ETUDIE":
        return f"Building on the lot ({overlap:,.0f} m²)"
    pct = row.get("pct_of_lot")
    if pct is not None and float(pct) >= 99.5:
        return "Covers the whole lot"
    if pct is not None:
        return f"Covers {float(pct):.0f}% of the lot"
    return f"Covers {overlap:,.0f} m² of the lot"


def classify(row: Mapping) -> Protection | None:
    """One `queries.heritage_for_lot` row as a `Protection`, or None."""
    source_table = str(row.get("source_table") or "")
    layer = layer_for(source_table)
    if layer is None:
        return None
    attributes = row.get("attributes") or {}

    tier = layer.tier
    jurisdiction = layer.jurisdiction
    detail: str | None = None

    if source_table.startswith("Patrimoine__"):
        name = _text(attributes.get("DENOMINATION_PRINCIPALE"))
        url = _text(attributes.get("LIEN_FICHE")) or _text(
            attributes.get("LIEN_URL_GOUVERNEMENTALE")
        )
        # The status layers say whose designation it is, and one building can
        # be both federal and provincial - which is worth reading as such.
        stated = _text(attributes.get("JURIDICTION_NO_LIBELLE"))
        if stated:
            jurisdiction = JURISDICTIONS.get(stated.lower(), stated)

        if source_table == "Patrimoine__BATIMENT_ETUDIE":
            code = _grade(attributes.get("EVALUATION_VALEUR_PATRIMO_NO"))
            label, tier = QUEBEC_GRADES.get(code, (None, UNGRADED))
            label = _text(attributes.get("EVALUATION_VALEUR_PATRIMO_NO_LIBELLE")) or label
            if code in (1, 2, 3, 4):
                detail = f"Heritage value: {label} ({code} of 4, 1 highest)"
            elif label:
                detail = f"Interest {label}, not graded"
            else:
                detail = "Not graded"
            if str(attributes.get("DEMOLI") or "").upper() == "O":
                tier = LOW
                detail += " · recorded as demolished"
        elif source_table == "Patrimoine__AIRE_PROTECTION":
            radius = attributes.get("RAYON_PROTECTION")
            if radius:
                detail = f"{float(radius):,.0f} m around the protected immovable"

        if str(attributes.get("STATUT_ACTIF") or "A").upper() != "A":
            status = _text(attributes.get("STATUT_ACTIF_LIBELLE")) or "not in force"
            tier = LOW
            detail = f"{detail} · {status}" if detail else status
    else:
        # Montreal's listing puts the building's name in DESCRIPTION, and on
        # a house with no name, its address instead.
        name = _text(attributes.get("DESCRIPTION"))
        address = _text(attributes.get("ADRESSE"))
        if name and address and address != name:
            name = f"{name} — {address}"
        elif address and not name:
            name = address
        category = _text(attributes.get("NOM_CAT"))
        detail = category if category and category != layer.designation else None
        if name == layer.designation:
            # A sector's DESCRIPTION is its category again, not a name.
            name = None
        url = _text(attributes.get("EN_SAVOIR_PLUS"))

    return Protection(
        tier=tier,
        designation=layer.designation,
        jurisdiction=jurisdiction,
        name=name,
        detail=detail,
        url=url,
        meaning=layer.meaning,
        coverage=_coverage(row, source_table),
        source_table=source_table,
        feature_id=str(row.get("feature_id") or ""),
    )


def protections(rows: Iterable[Mapping]) -> list[Protection]:
    """Every heritage row on a lot, read and ordered most binding first."""
    read = [p for p in (classify(row) for row in rows) if p is not None]
    return sorted(read, key=lambda p: (p.rank, p.name or ""))
