"""Open Québec City's roll lookup in your own Chrome, with the lot or matricule typed in.

    python scripts/roll_lookup.py --lot 5342219
    python scripts/roll_lookup.py --matricule 4885-62-3131-1-000-0000
    make roll-lookup LOT=5342219

What it does, and deliberately no more: it starts Chrome through Selenium,
loads the city's page, switches to the *Lot* or *Matricule* tab and types the
number into the field. Then it stops and leaves the window open. **You** click
*Rechercher*, pass whatever reCAPTCHA challenge the page puts up, and read the
result. The script never clicks the search button, never reads the result
page, and never writes anything down.

That line is where it is on purpose. The page is for personal consultation,
one unit at a time, behind a reCAPTCHA; the owner's name it shows is personal
information the open roll withholds and whose re-identification the open
data licence forbids. Typing the number for you is a convenience. Fetching
the answer for you would be a scraper, and caching it would be a database of
owners this project is not allowed to build. See the README's *On the roll*.

Chrome runs on its own profile under your local app data (so the city's
cookies and any challenge it remembers persist between runs) and is left
running when the script exits (`detach`). `--close` is for a smoke test: it
reads the field back, prints it, and quits the browser.

The field ids are the page's as of 2026-10-05 (Telerik RadTabStrip over an
ASP.NET form). `plan()` is the one place they live, and the only part of
this file a test can read without a browser.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROLL_URL = (
    "https://www.ville.quebec.qc.ca/citoyens/taxes_evaluation/"
    "evaluation_fonciere/role/index.aspx"
)

#: The ASP.NET control every field on the page hangs off.
_PREFIX = "ctl00_ctl00_contenu_texte_page_fichePropriete_RechercheAdresse1_"

#: The tab strip's anchors, by the class the page gives each tab.
TAB_LOT = "a.rtsLink.onglet-lot"
TAB_MATRICULE = "a.rtsLink.onglet-matricule"

#: The lot number goes in one masked box, seven digits.
LOT_FIELD = _PREFIX + "RadMaskedTextLot"

#: The matricule goes in six masked boxes, one per dashed group: 4-2-4-1-3-4.
MATRICULE_FIELDS = tuple(_PREFIX + f"RmTextMatricule{i}" for i in range(1, 7))
MATRICULE_WIDTHS = (4, 2, 4, 1, 3, 4)

_LOT = re.compile(r"^\d{7}$")


@dataclass(frozen=True)
class Step:
    """One field to fill: its element id and the text to type."""

    field_id: str
    text: str


@dataclass(frozen=True)
class Plan:
    """Which tab to open and what to type where."""

    tab: str
    steps: tuple[Step, ...]
    label: str


def plan(lot: str | None = None, matricule: str | None = None) -> Plan:
    """What to type where, for a lot number or a matricule.

    The lot number is typed with its spaces removed (`5 342 219` as
    `5342219`), the way the city's masked box wants it. The matricule is
    taken apart on its dashes, or cut into 4-2-4-1-3-4 when given as the
    18 digits the roll stores, one group per box. One of the two, not both:
    the page has one tab open at a time.
    """
    if bool(lot) == bool(matricule):
        raise ValueError("give exactly one of --lot or --matricule")
    if lot:
        digits = "".join(str(lot).split())
        if not _LOT.match(digits):
            raise ValueError(
                f"{lot!r} is not a seven-digit lot number (spaces are fine)"
            )
        return Plan(TAB_LOT, (Step(LOT_FIELD, digits),), f"lot {digits}")

    text = str(matricule).strip()
    groups = text.split("-") if "-" in text else [
        text[sum(MATRICULE_WIDTHS[:i]):sum(MATRICULE_WIDTHS[:i + 1])]
        for i in range(len(MATRICULE_WIDTHS))
    ]
    if len(groups) != 6 or any(
        not g.isdigit() or len(g) != w for g, w in zip(groups, MATRICULE_WIDTHS, strict=True)
    ):
        raise ValueError(
            f"{matricule!r} is not a matricule: 4885-62-3131-1-000-0000, or "
            "its 18 digits run together"
        )
    return Plan(
        TAB_MATRICULE,
        tuple(Step(f, g) for f, g in zip(MATRICULE_FIELDS, groups, strict=True)),
        f"matricule {'-'.join(groups)}",
    )


def profile_dir() -> Path:
    """A Chrome profile of this script's own, so your running Chrome is not locked."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or "~"
    return Path(base).expanduser() / "hbu_rag_map" / "chrome-profile"


def open_lookup(which: Plan, *, close: bool = False, timeout: float = 30.0) -> list[str]:
    """Start Chrome, load the page, open the tab, type the number. Return what the fields hold."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:  # pragma: no cover - the extra is not installed
        sys.exit(
            "selenium is not installed in this venv: "
            "uv pip install --python .venv/Scripts/python.exe selenium"
        )

    options = webdriver.ChromeOptions()
    profile = profile_dir()
    profile.mkdir(parents=True, exist_ok=True)
    options.add_argument(f"--user-data-dir={profile}")
    options.add_argument("--window-size=1200,1000")
    if not close:
        # The window outlives the script: the rest of the consultation is yours.
        options.add_experimental_option("detach", True)

    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, timeout)
    try:
        driver.get(ROLL_URL)
        tab = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, which.tab)))
        tab.click()
        typed = []
        for step in which.steps:
            box = wait.until(EC.visibility_of_element_located((By.ID, step.field_id)))
            box.click()
            # A Telerik masked box: Home then the digits overwrite the mask
            # in place, which is what a person does with the mouse and keys.
            box.send_keys(Keys.HOME)
            box.send_keys(step.text)
            typed.append(box.get_attribute("value") or "")
        print(
            f"Typed {which.label} into the city's roll lookup. Click "
            "Rechercher in the Chrome window to search; the owner's name is on "
            "the result, for your eyes."
        )
        return typed
    finally:
        if close:
            driver.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lot", help="seven-digit lot number, spaces allowed")
    parser.add_argument("--matricule", help="4885-62-3131-1-000-0000, or its 18 digits")
    parser.add_argument(
        "--close", action="store_true",
        help="smoke test: read the fields back, print them and quit Chrome",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        which = plan(lot=args.lot, matricule=args.matricule)
    except ValueError as exc:
        parser.error(str(exc))
    typed = open_lookup(which, close=args.close, timeout=args.timeout)
    if args.close:
        print("fields:", typed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
