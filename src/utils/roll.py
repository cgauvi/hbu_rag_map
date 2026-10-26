"""
roll.py — The assessment roll's identifiers, spelled the way each city's own
roll lookup wants them typed.

The open roll (the MAMH GeoPackage the dataplatform snapshots) publishes no
owner: the whole RL02 section - name, status, mailing address - is withheld,
and the licence forbids re-identifying it. The cities' *own* online rolls do
show the owner, lawfully, one unit at a time, behind a search box that takes
the unit's matricule, its lot number or its address. So the one thing this
app can do about "who owns it" is hand the reader the exact string to paste
into that box, and the box's address. This module is that string and that
address.

Three cities, three lookups, kept by hand like `neighborhoods.NEIGHBORHOODS`:

* Québec (code_mun 23027) - the city's own page, searchable by address, lot
  number, matricule or postal code. Behind a reCAPTCHA, which is the city's
  way of saying one at a time, by a person. The matricule is typed with its
  dashes, `4885-62-2726-1-000-0000`, and the lot number bare, `5342219`.
* Montréal (66023) - montreal.ca's lookup, by address, renewed lot number or
  matricule; shows the owner's name and mailing address.
* Saguenay (94068) - PG Solutions' ImmoNet portal the city links to, by
  address (typed without accents, the page warns) or matricule.

A city not listed here still gets the matricule, since the string is the
MAMH's and not the city's; it just gets no link.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: How a matricule is written on every city's lookup page and on a tax bill:
#: `4885-62-2726-1-000-0000`. The 18 characters the roll stores it as
#: (`mat18`) are these seven groups run together.
_MATRICULE_GROUPS = (4, 2, 4, 1, 3, 4)


@dataclass(frozen=True)
class RollLookup:
    """One city's online roll: where it is and what it accepts."""

    code_mun: str
    city: str
    url: str
    #: The identifiers the search box takes, in the words the page uses.
    accepts: str
    #: What the page says about using it, if it says anything.
    terms: str = ""


ROLL_LOOKUPS: dict[str, RollLookup] = {
    lookup.code_mun: lookup
    for lookup in (
        RollLookup(
            "23027",
            "Québec",
            "https://www.ville.quebec.qc.ca/citoyens/taxes_evaluation/"
            "evaluation_fonciere/role/index.aspx",
            "address, lot number, matricule or postal code",
            "the city's page is for personal consultation, one unit at a "
            "time, behind a reCAPTCHA",
        ),
        RollLookup(
            "66023",
            "Montréal",
            "https://montreal.ca/role-evaluation-fonciere",
            "address, renewed lot number or matricule",
            "shows the owner's name and mailing address",
        ),
        RollLookup(
            "94068",
            "Saguenay",
            "https://pdi.pgmunicipal.com/immosoft/controller/ImmoNetPub/U4051/"
            "trouverParAdresse?language=fr&fourn_seq=1312",
            "address (typed without accents) or matricule",
        ),
    )
}


def format_matricule(mat18: str | None) -> str | None:
    """`488562272610000000` as `4885-62-2726-1-000-0000`; None when it is not 18 digits.

    The roll stores the matricule as one 18-character string, and every
    city's lookup, every tax bill and every assessor's letter writes it in
    seven dashed groups. A string that is not 18 digits is handed back as
    typed rather than cut into groups that would mean nothing - a test
    fixture's `"1" * 18` formats; a blank does not.
    """
    if mat18 is None:
        return None
    digits = str(mat18).strip()
    if len(digits) != sum(_MATRICULE_GROUPS) or not digits.isdigit():
        return digits or None
    parts, start = [], 0
    for width in _MATRICULE_GROUPS:
        parts.append(digits[start:start + width])
        start += width
    return "-".join(parts)


def lookup_for(code_mun: str | None) -> RollLookup | None:
    """The city's online roll for a municipality code, or None if none is registered."""
    return ROLL_LOOKUPS.get(str(code_mun or "").strip())


#: Québec City's roll page, and what `scripts/roll_lookup.py` types into it.
#: The field ids are the page's as of 2026-10-05 (a Telerik RadTabStrip over
#: an ASP.NET form); this is the one place they live.
QUEBEC_CODE_MUN = "23027"
_FORM = "ctl00_ctl00_contenu_texte_page_fichePropriete_RechercheAdresse1_"
TAB_LOT = "a.rtsLink.onglet-lot"
TAB_MATRICULE = "a.rtsLink.onglet-matricule"
#: The lot number goes in one masked box, seven digits.
LOT_FIELD = _FORM + "RadMaskedTextLot"
#: The matricule goes in six masked boxes, one per dashed group.
MATRICULE_FIELDS = tuple(_FORM + f"RmTextMatricule{i}" for i in range(1, 7))

_LOT_DIGITS = re.compile(r"^\d{7}$")


@dataclass(frozen=True)
class Step:
    """One field to fill on the city's page: its element id and the text to type."""

    field_id: str
    text: str


@dataclass(frozen=True)
class LookupPlan:
    """Which tab of the city's page to open and what to type where."""

    tab: str
    steps: tuple[Step, ...]
    label: str


def lookup_plan(lot: str | None = None, matricule: str | None = None) -> LookupPlan:
    """What `scripts/roll_lookup.py` types where, for a lot number or a matricule.

    The lot number is typed with its spaces removed (`5 342 219` as
    `5342219`), the way the city's masked box wants it. The matricule is
    taken apart on its dashes, or cut into 4-2-4-1-3-4 when given as the 18
    digits the roll stores, one group per box. One of the two, not both: the
    page has one tab open at a time. A `ValueError` names what is wrong, so
    nothing opens a browser for a number that cannot be typed.
    """
    if bool(lot) == bool(matricule):
        raise ValueError("give exactly one of a lot number or a matricule")
    if lot:
        digits = lot_key(lot) or ""
        if not _LOT_DIGITS.match(digits):
            raise ValueError(
                f"{lot!r} is not a seven-digit lot number (spaces are fine)"
            )
        return LookupPlan(TAB_LOT, (Step(LOT_FIELD, digits),), f"lot {digits}")

    text = str(matricule).strip()
    if "-" in text:
        groups = text.split("-")
    else:
        groups, start = [], 0
        for width in _MATRICULE_GROUPS:
            groups.append(text[start:start + width])
            start += width
    if len(groups) != len(_MATRICULE_GROUPS) or any(
        not g.isdigit() or len(g) != w
        for g, w in zip(groups, _MATRICULE_GROUPS, strict=True)
    ):
        raise ValueError(
            f"{matricule!r} is not a matricule: 4885-62-3131-1-000-0000, or "
            "its 18 digits run together"
        )
    return LookupPlan(
        TAB_MATRICULE,
        tuple(Step(f, g) for f, g in zip(MATRICULE_FIELDS, groups, strict=True)),
        f"matricule {'-'.join(groups)}",
    )


def lot_key(lot_number: str | None) -> str | None:
    """A lot number the way the roll's lookups and `lot_numbers` spell it: no spaces.

    Infolot and this app write `5 342 219`; the roll, the cities' search
    boxes and the dataplatform's `lot_key` all write `5342219`. Every kind
    of space comes out and nothing else does - a `PC-` prefix stays, because
    a common-parts lot is not on any roll and should miss rather than match.
    """
    if lot_number is None:
        return None
    text = "".join(str(lot_number).split())
    return text or None
