"""The vector tile path: the SQL, the URL, the key, and the server.

The map was crashing because every shape in the viewport was fetched, embedded
in the page and re-sent on every rerun. These tests hold the three things that
replaced it and are easy to get quietly wrong:

* the tile SQL asks the *4326* index for its candidates while measuring
  against the *3857* envelope — swap them and the query still returns the right
  tile, just by scanning the borough for it;
* the key on a tile URL is derived from the app password, so a public listener
  does not put the cadastre on the internet;
* a tile that fails comes back empty rather than 500, because Leaflet retries a
  500 and does not retry an empty tile.

Nothing here opens a database. `queries.scalar` is stubbed the way
`test_queries` stubs `queries.query`.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import date

import pytest

from src.utils import basemap, documents, queries, tiles


@pytest.fixture
def captured_scalar(monkeypatch):
    """Capture the (sql, params) of the tile query, returning canned bytes."""
    calls: list[tuple[str, object]] = []
    answer: list[object] = [b"\x1a\x0f"]

    def fake_scalar(sql, params=None):
        calls.append((sql, params))
        return answer[0]

    monkeypatch.setattr(queries, "scalar", fake_scalar)
    return calls, answer


# ---------------------------------------------------------------------------
# The SQL
# ---------------------------------------------------------------------------


def test_every_layer_builds_a_tile(captured_scalar):
    calls, _ = captured_scalar
    for layer in queries.MVT_LAYER_NAMES:
        queries.mvt_tile(layer, 15, 9646, 11732)
    assert len(calls) == len(queries.MVT_LAYER_NAMES) == 6


def test_an_unknown_layer_never_reaches_the_database(captured_scalar):
    """The layer comes off a URL, so it is checked before it is interpolated."""
    calls, _ = captured_scalar
    with pytest.raises(ValueError):
        queries.mvt_tile("lots; DROP TABLE rag.lots", 15, 1, 1)
    assert calls == []


def test_the_bbox_filter_is_in_4326_and_the_clip_in_3857(captured_scalar):
    """The whole performance of a tile is in this one line.

    The GiST indexes are on the 4326 geometry column. `ST_AsMVTGeom` needs a
    3857 envelope. Comparing the row against the projected one is correct and
    unindexable, which turns every tile into a scan of the borough — and it
    looks fine until the borough is loaded.
    """
    calls, _ = captured_scalar
    queries.mvt_tile("lots", 15, 9646, 11732)
    sql, _params = calls[0]

    assert "l.geom && envelope.lonlat" in sql
    assert "ST_Transform(\n                       ST_TileEnvelope" in sql
    assert "ST_AsMVTGeom(\n                       ST_Transform(l.geom, 3857)" in sql
    assert "envelope.mercator" in sql


def test_the_tile_carries_the_slack_that_hides_the_seams(captured_scalar):
    """Selected wider than the tile, kept wider than the tile, by the same slack."""
    calls, _ = captured_scalar
    queries.mvt_tile("lots", 15, 9646, 11732)
    _sql, params = calls[0]

    assert params["buffer"] == queries.MVT_BUFFER
    assert params["margin"] == pytest.approx(
        queries.MVT_BUFFER / queries.MVT_EXTENT
    )


def test_filters_reach_the_tile_query(captured_scalar):
    calls, _ = captured_scalar
    queries.mvt_tile(
        "lots", 16, 1, 1,
        scrape_date=date(2026, 8, 27),
        neighborhood="VSMPE",
        min_area_m2=500.0,
        max_area_m2=900.0,
    )
    _sql, params = calls[0]

    assert params["scrape_date"] == date(2026, 8, 27)
    assert params["neighborhood"] == "VSMPE"
    assert params["min_area"] == 500.0
    assert params["max_area"] == 900.0


def test_the_underbuilt_screen_reaches_both_layers_that_take_it(captured_scalar):
    """Turning it on must narrow the shading and the proposal to the same lots."""
    calls, _ = captured_scalar
    queries.mvt_tile("capacity", 16, 1, 1, only_underbuilt=True)
    queries.mvt_tile("massing", 16, 1, 1, only_underbuilt=True)

    capacity_sql, capacity_params = calls[0]
    massing_sql, massing_params = calls[1]
    assert capacity_params["only_underbuilt"] is True
    assert massing_params["only_underbuilt"] is True
    assert "g.is_underbuilt" in capacity_sql
    assert "g.is_underbuilt" in massing_sql


def test_the_capacity_tile_joins_on_the_whole_partition_triple(captured_scalar):
    """lot_uid is a bigserial a reload mints again — joining on it alone would
    shade this year's parcels with last year's answer."""
    calls, _ = captured_scalar
    queries.mvt_tile("capacity", 16, 1, 1)
    sql, _params = calls[0]

    assert "g.lot_uid      = l.lot_uid" in sql
    assert "g.neighborhood = l.neighborhood" in sql
    assert "g.scrape_date  = l.scrape_date" in sql


def test_the_streets_tile_reads_silver_through_the_4326_index(captured_scalar):
    """The one line layer, and the one tile out of `silver` rather than `rag`.

    Same trap as every other layer: the GiST index on
    `silver.neighborhood_streets` is on the 4326 column, so a tile that
    compares against the projected envelope scans the borough for every square
    it draws — and looks fine until the borough is loaded.
    """
    calls, _ = captured_scalar
    queries.mvt_tile("streets", 15, 9646, 11732)
    sql, params = calls[0]

    assert f"{queries.SILVER_SCHEMA}.neighborhood_streets s" in sql
    assert "s.geom && envelope.lonlat" in sql
    assert "ST_AsMVTGeom(\n                       ST_Transform(s.geom, 3857)" in sql
    assert params["layer"] == "streets"


def test_the_streets_tile_carries_the_name_and_the_length(captured_scalar):
    """What the tooltip reads, and nothing else — a tile pays per feature."""
    calls, _ = captured_scalar
    queries.mvt_tile("streets", 15, 9646, 11732)
    sql, _params = calls[0]

    assert "s.street_name" in sql
    assert "s.length_m" in sql
    assert "s.attributes" not in sql


def test_the_streets_tile_takes_the_partition_filters(captured_scalar):
    """A borough's street sides are partitioned the same way its lots are, so
    the snapshot the sidebar picked has to reach this layer too."""
    calls, _ = captured_scalar
    queries.mvt_tile(
        "streets", 15, 9646, 11732,
        scrape_date=date(2026, 8, 27), neighborhood="VSMPE",
    )
    sql, params = calls[0]

    assert params["scrape_date"] == date(2026, 8, 27)
    assert params["neighborhood"] == "VSMPE"
    assert "s.scrape_date = %(scrape_date)s" in sql
    assert "s.neighborhood = %(neighborhood)s" in sql


def test_the_tile_carries_no_attribute_bag(captured_scalar):
    """Two dozen Infolot columns per lot, on every tile, on every pan."""
    calls, _ = captured_scalar
    queries.mvt_tile("lots", 16, 1, 1)
    sql, _params = calls[0]
    assert "l.attributes" not in sql


def test_an_empty_tile_is_empty_bytes_not_none(captured_scalar):
    _calls, answer = captured_scalar
    answer[0] = None
    assert queries.mvt_tile("lots", 16, 1, 1) == b""


def test_a_memoryview_comes_back_as_bytes(captured_scalar):
    """psycopg returns bytea as a memoryview; an HTTP body wants bytes."""
    _calls, answer = captured_scalar
    answer[0] = memoryview(b"\x1a\x0f")
    assert queries.mvt_tile("lots", 16, 1, 1) == b"\x1a\x0f"


# ---------------------------------------------------------------------------
# Paths and URLs
# ---------------------------------------------------------------------------


def test_a_tile_path_parses():
    assert tiles.parse_path("/tiles/lots/15/9646/11732.mvt") == ("lots", 15, 9646, 11732)
    assert tiles.parse_path("/tiles/zones/15/9646/11732.pbf") == ("zones", 15, 9646, 11732)


@pytest.mark.parametrize(
    "path",
    [
        "/tiles/nope/15/1/1.mvt",          # not a layer this app serves
        "/tiles/lots/15/1/1.png",          # not a tile format
        "/tiles/lots/15/1.mvt",            # missing a coordinate
        "/tiles/lots/x/1/1.mvt",           # not a number
        "/tiles/lots/99/1/1.mvt",          # outside the grid
        "/tiles/lots/2/9/1.mvt",           # x past 2**z
        "/lots/15/1/1.mvt",                # outside the prefix
    ],
)
def test_a_malformed_path_is_refused(path):
    assert tiles.parse_path(path) is None


def test_a_grid_path_parses():
    assert tiles.parse_grid_path("/tiles/grid/784a0b4f710d1785.pdf") == (
        "784a0b4f710d1785"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/tiles/grid/784a0b4f710d1785.png",       # not a PDF
        "/tiles/grid/784A0B4F710D1785.pdf",       # a document_id is lower case
        "/tiles/grid/784a0b4f710d178.pdf",        # fifteen characters
        "/tiles/grid/784a0b4f710d17855.pdf",      # seventeen
        "/tiles/grid/../../etc/passwd.pdf",       # nothing to traverse with
        "/tiles/grid/.pdf",                       # no id at all
        "/grid/784a0b4f710d1785.pdf",             # outside the prefix
    ],
)
def test_a_malformed_grid_path_is_refused(path):
    assert tiles.parse_grid_path(path) is None


def test_a_grid_url_is_same_origin_behind_the_balancer(monkeypatch):
    """The whole point of the route: relative, so it inherits the page's
    scheme, so an https page can frame it."""
    monkeypatch.delenv("HBU_APP_PASSWORD", raising=False)
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    assert tiles.grid_url("784a0b4f710d1785") == "/tiles/grid/784a0b4f710d1785.pdf"


def test_a_grid_url_carries_the_key_when_there_is_one(monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    assert tiles.grid_url("784a0b4f710d1785") == (
        f"/tiles/grid/784a0b4f710d1785.pdf?k={tiles.tile_key()}"
    )


def test_no_url_is_built_for_something_that_is_not_a_document_id():
    assert tiles.grid_url("../../etc/passwd") is None
    assert tiles.grid_url("") is None


def test_the_url_leaves_the_tile_coordinates_for_leaflet(monkeypatch):
    monkeypatch.delenv("HBU_APP_PASSWORD", raising=False)
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    url = tiles.layer_url("lots")
    assert url == "/tiles/lots/{z}/{x}/{y}.mvt"


def test_filters_travel_in_the_query_string_not_the_path(monkeypatch):
    """So the browser's own cache keys on them and a toggled-back layer is free."""
    monkeypatch.delenv("HBU_APP_PASSWORD", raising=False)
    monkeypatch.setenv("HBU_TILE_BASE_URL", "same-origin")
    url = tiles.layer_url(
        "capacity", {"scrape_date": date(2026, 8, 27), "neighborhood": "VSMPE"}
    )
    assert url.startswith("/tiles/capacity/{z}/{x}/{y}.mvt?")
    assert "scrape_date=2026-08-27" in url
    assert "neighborhood=VSMPE" in url


def test_empty_filters_are_left_out(monkeypatch):
    monkeypatch.delenv("HBU_APP_PASSWORD", raising=False)
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    url = tiles.layer_url("lots", {"neighborhood": None, "min_area": None})
    assert "?" not in url


def test_a_laptop_gets_an_absolute_url(monkeypatch):
    monkeypatch.delenv("HBU_APP_PASSWORD", raising=False)
    monkeypatch.delenv("HBU_TILE_BASE_URL", raising=False)
    assert tiles.layer_url("lots").startswith(f"http://localhost:{tiles.DEFAULT_PORT}/")


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_no_password_means_no_key(monkeypatch):
    """The gate is off exactly when `auth`'s is — which is every local run."""
    monkeypatch.delenv("HBU_APP_PASSWORD", raising=False)
    assert tiles.tile_key() is None
    assert "k=" not in tiles.layer_url("lots")


def test_the_placeholder_password_grants_nothing(monkeypatch):
    """Terraform writes it before anyone sets a value, and `auth` refuses it."""
    monkeypatch.setenv("HBU_APP_PASSWORD", "PLACEHOLDER")
    assert tiles.tile_key() is None


def test_the_key_is_derived_from_the_password_and_is_not_the_password(monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    key = tiles.tile_key()
    assert key and "hunter2" not in key
    assert len(key) == 32


def test_every_task_derives_the_same_key(monkeypatch):
    """Which is why the tile target group needs no stickiness."""
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    first = tiles.tile_key()
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    assert tiles.tile_key() == first


def test_the_key_is_on_the_url_when_there_is_one(monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    assert f"k={tiles.tile_key()}" in tiles.layer_url("lots")


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


@pytest.fixture
def running(monkeypatch):
    """A tile server on an ephemeral port, with the database stubbed out."""
    served: list[tuple] = []

    def fake_tile(layer, z, x, y, **kwargs):
        served.append((layer, z, x, y, kwargs))
        return b"tile:" + layer.encode()

    monkeypatch.setattr(queries, "mvt_tile", fake_tile)
    monkeypatch.setenv("HBU_TILE_BIND", "127.0.0.1")
    tiles.stop()
    port = tiles.start(0)  # 0 lets the OS pick, so a busy 8502 is not a flake
    assert port
    yield f"http://127.0.0.1:{port}", served
    tiles.stop()


def _get(url: str) -> tuple[int, bytes, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def test_the_server_answers_a_tile(running):
    base, served = running
    status, body, headers = _get(f"{base}/tiles/lots/15/9646/11732.mvt")

    assert status == 200
    assert body == b"tile:lots"
    assert headers["Content-Type"] == "application/vnd.mapbox-vector-tile"
    assert served[0][:4] == ("lots", 15, 9646, 11732)


def test_a_tile_is_cacheable_by_the_browser(running):
    base, _served = running
    _status, _body, headers = _get(f"{base}/tiles/lots/15/9646/11732.mvt")
    assert "max-age" in headers["Cache-Control"]


def test_the_second_request_for_a_tile_does_not_hit_the_database(running):
    base, served = running
    _get(f"{base}/tiles/lots/15/9646/11732.mvt")
    _get(f"{base}/tiles/lots/15/9646/11732.mvt")
    assert len(served) == 1


def test_a_different_filter_is_a_different_tile(running):
    base, served = running
    _get(f"{base}/tiles/capacity/15/9646/11732.mvt")
    _get(f"{base}/tiles/capacity/15/9646/11732.mvt?underbuilt=1")
    assert len(served) == 2
    assert served[1][4] == {"only_underbuilt": True}


def test_the_filters_on_the_url_reach_the_query(running):
    base, served = running
    _get(
        f"{base}/tiles/lots/15/9646/11732.mvt"
        "?scrape_date=2026-08-27&neighborhood=VSMPE&min_area=500&max_area=900"
    )
    assert served[0][4] == {
        "scrape_date": date(2026, 8, 27),
        "neighborhood": "VSMPE",
        "min_area_m2": 500.0,
        "max_area_m2": 900.0,
    }


def test_an_unparseable_filter_is_ignored_rather_than_fatal(running):
    """A browser can replay a cached URL from before a parameter changed."""
    base, served = running
    status, _body, _headers = _get(
        f"{base}/tiles/lots/15/9646/11732.mvt?scrape_date=yesterday&min_area=lots"
    )
    assert status == 200
    assert served[0][4] == {}


def test_a_failing_tile_is_empty_rather_than_a_500(running, monkeypatch):
    """Leaflet retries a 500 forever and fills the console; it accepts empty."""
    base, _served = running

    def explode(*_a, **_k):
        raise RuntimeError("gold.lot_building_massing does not exist")

    monkeypatch.setattr(queries, "mvt_tile", explode)
    status, body, _headers = _get(f"{base}/tiles/massing/16/1/1.mvt")
    assert status == 200
    assert body == b""


@pytest.fixture
def published_grid():
    """One PDF in the registry, and nothing on disk under it."""
    documents.forget_published()
    doc_id = documents.document_id("http://example.test/zone/C01-001.pdf")
    documents.publish(doc_id, b"%PDF-1.4 grille")
    yield doc_id
    documents.forget_published()


def test_the_server_answers_a_published_grid(running, published_grid, tmp_path):
    base, _served = running
    status, body, headers = _get(f"{base}/tiles/grid/{published_grid}.pdf")

    assert status == 200
    assert body == b"%PDF-1.4 grille"
    assert headers["Content-Type"] == "application/pdf"
    # `inline`, so a link opens the browser's viewer rather than saving a file.
    assert headers["Content-Disposition"].startswith("inline")
    # These bytes came off a municipal web server; a document that lied about
    # its type is refused rather than sniffed into whatever it actually is.
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_a_grid_is_served_from_the_disk_cache_too(running, tmp_path, monkeypatch):
    """A process restart empties the registry; the cache directory survives it,
    and so does the link the page handed out."""
    documents.forget_published()
    monkeypatch.setattr(documents, "DEFAULT_CACHE_DIR", tmp_path)
    doc_id = documents.document_id("http://example.test/zone/C02-002.pdf")
    (tmp_path / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4 from disk")

    base, _served = running
    status, body, _headers = _get(f"{base}/tiles/grid/{doc_id}.pdf")

    assert (status, body) == (200, b"%PDF-1.4 from disk")


def test_an_unpublished_grid_is_a_404_rather_than_a_fetch(running, tmp_path, monkeypatch):
    """The security property of the route, as a test: an id nobody in this
    process has fetched a URL for resolves to nothing at all. There is no
    address in the request for the server to go and get."""
    documents.forget_published()
    monkeypatch.setattr(documents, "DEFAULT_CACHE_DIR", tmp_path)
    fetched: list[str] = []
    monkeypatch.setattr(
        documents, "fetch", lambda *a, **k: fetched.append(a) or (_ for _ in ()).throw(
            AssertionError("the route must never fetch")
        )
    )

    base, _served = running
    status, _body, _headers = _get(f"{base}/tiles/grid/{'0' * 16}.pdf")

    assert status == 404
    assert fetched == []


def test_a_grid_needs_the_key(running, published_grid, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    base, _served = running

    assert _get(f"{base}/tiles/grid/{published_grid}.pdf")[0] == 403
    assert _get(
        f"{base}/tiles/grid/{published_grid}.pdf?k={tiles.tile_key()}"
    )[0] == 200


def test_a_grid_is_not_cached_by_a_shared_cache(running, published_grid):
    """`private`: the sheet is a public document, but which sheets this
    deployment holds is a trace of what has been looked at."""
    base, _served = running
    _status, _body, headers = _get(f"{base}/tiles/grid/{published_grid}.pdf")
    assert headers["Cache-Control"].startswith("private")


def test_an_unknown_path_is_a_404(running):
    base, _served = running
    status, _body, _headers = _get(f"{base}/tiles/nope/15/1/1.mvt")
    assert status == 404


def test_the_health_path_needs_no_key(running, monkeypatch):
    """The ALB's health check carries no credentials, and a target group that
    cannot reach its own health path drains the service on the first deploy."""
    base, _served = running
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    status, body, _headers = _get(f"{base}{tiles.HEALTH_PATH}")
    assert status == 200
    assert body == b"ok"


def test_a_tile_without_the_key_is_refused_when_a_password_is_set(running, monkeypatch):
    base, served = running
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")

    status, _body, _headers = _get(f"{base}/tiles/lots/15/9646/11732.mvt")
    assert status == 403
    assert served == []

    status, _body, _headers = _get(
        f"{base}/tiles/lots/15/9646/11732.mvt?k={tiles.tile_key()}"
    )
    assert status == 200


def test_tiles_are_readable_cross_origin(running):
    """Streamlit is on one port and this is on another, on a laptop."""
    base, _served = running
    _status, _body, headers = _get(f"{base}/tiles/lots/15/9646/11732.mvt")
    assert headers["Access-Control-Allow-Origin"] == "*"


# ---------------------------------------------------------------------------
# The vendored library
#
# `streamlit_folium` awaits every `default_js` URL before it draws anything and
# catches nothing if one rejects — and it populates the map's own div inside
# that promise. A library the browser cannot fetch therefore does not cost the
# vector layers, it costs the whole map, silently. These keep the fetch inside
# this deployment.
# ---------------------------------------------------------------------------


def test_the_server_hands_out_the_vectorgrid_library(running):
    base, _served = running
    status, body, headers = _get(f"{base}{tiles.VENDOR_PREFIX}/{tiles.VECTORGRID_FILE}")

    assert status == 200
    assert headers["Content-Type"].startswith("application/javascript")
    # Byte-for-byte what is committed, so what a browser runs is what a digest
    # can be taken over.
    assert body == (tiles.VENDOR_DIR / tiles.VECTORGRID_FILE).read_bytes()
    # And it is the library rather than an error page that happens to be 200.
    assert b"vectorGrid" in body


def test_the_library_needs_no_key(running, monkeypatch):
    """Keyless like the health path: public code, carrying no cadastre. It is
    also fetched by the same promise the map's existence hangs on, so a stale
    key here would blank the pane rather than empty a layer."""
    base, _served = running
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")

    status, _body, _headers = _get(f"{base}{tiles.VENDOR_PREFIX}/{tiles.VECTORGRID_FILE}")
    assert status == 200


def test_the_library_is_readable_cross_origin(running):
    """The laptop shape puts Streamlit on 8501 and this on 8502."""
    base, _served = running
    _status, _body, headers = _get(f"{base}{tiles.VENDOR_PREFIX}/{tiles.VECTORGRID_FILE}")
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_the_library_is_cached_hard_because_its_url_carries_its_version(running):
    base, _served = running
    _status, _body, headers = _get(f"{base}{tiles.VENDOR_PREFIX}/{tiles.VECTORGRID_FILE}")
    assert "immutable" in headers["Cache-Control"]
    assert f"max-age={tiles.VENDOR_CACHE_SECONDS}" in headers["Cache-Control"]


@pytest.mark.parametrize(
    "path",
    [
        "/tiles/vendor/",
        "/tiles/vendor/other.js",
        "/tiles/vendor/leaflet-vectorgrid-1.2.0.js",
        "/tiles/vendor/../tiles.py",
        "/tiles/vendor/%2e%2e/tiles.py",
    ],
)
def test_only_the_one_vendored_file_is_published(running, path):
    """Nothing from the URL is joined onto a directory, so there is no path for
    a request to traverse - the name either is the published one or is a 404."""
    base, _served = running
    status, _body, _headers = _get(f"{base}{path}")
    assert status == 404


# ---------------------------------------------------------------------------
# The map the URLs end up on
# ---------------------------------------------------------------------------


def _rendered(**kwargs) -> str:
    urls = {
        layer: f"/tiles/{layer}/{{z}}/{{x}}/{{y}}.mvt"
        for layer in basemap.TILE_LAYER_ORDER
    }
    return basemap.build_map(tile_layers=urls, **kwargs).get_root().render()


# ---------------------------------------------------------------------------
# What the component keys itself on
#
# `streamlit_folium` hashes the map's JavaScript into the component's key, and
# a key that changes remounts the iframe: Leaflet is thrown away and every
# basemap and vector tile is fetched again. So the question "does this change
# the map's script" is really "does this reload the map", and the answers
# below are what `app.py`'s anchor exists to arrange.
# ---------------------------------------------------------------------------


def _component_key(**kwargs) -> str:
    """The key `st_folium` would give a map built with these arguments."""
    from streamlit_folium import _get_map_string, generate_js_hash

    urls = {
        layer: f"/tiles/{layer}/{{z}}/{{x}}/{{y}}.mvt"
        for layer in basemap.TILE_LAYER_ORDER
    }
    fmap = basemap.build_map(tile_layers=urls, **kwargs)
    fmap.get_root().render()
    return generate_js_hash(_get_map_string(fmap), "zoning_map", False)


_SELECTED = {
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[-73.62, 45.55], [-73.619, 45.55], [-73.619, 45.551],
                         [-73.62, 45.551], [-73.62, 45.55]]],
    },
    "lot_number": "2 170 935",
    "lat": 45.5505,
    "lon": -73.6195,
}


def test_two_maps_built_the_same_way_share_a_key():
    """The premise everything below rests on: the random ids folium stamps
    into a map are stripped before hashing, so an identical rebuild is the
    same component and the pane is not remounted."""
    assert _component_key() == _component_key()


def test_moving_the_map_would_remount_it_which_is_why_a_pan_does_not():
    """A centre baked into the script is a new key, and a new key is a
    reload. Leaflet reports a new centre at the end of every drag, so `app.py`
    keeps an anchor a pan does not touch - see "Where the map is" there."""
    here = _component_key(center=(45.5535, -73.6200), zoom=15)
    assert here != _component_key(center=(45.5551, -73.6188), zoom=15)
    assert here != _component_key(center=(45.5535, -73.6200), zoom=16)


def test_the_selection_is_kept_out_of_the_map_so_a_click_costs_no_reload():
    """The selected lot changes on a click, which is the most frequent thing a
    reader does. Drawn into the map object it would remount the pane every
    time; handed to `st_folium` as a feature group it is evaluated into the
    map already on screen."""
    assert _component_key() == _component_key()
    group = basemap.selection_layer(_SELECTED)
    assert group is not None
    assert basemap.selection_layer(None) is None
    # ... and nothing about the map itself changed to accommodate it.
    assert _component_key() == _component_key()


def test_a_layer_toggle_does_change_the_key():
    """The remounts that are left are the ones that have to be: the map really
    is a different map. `app.py` moves its anchor onto the browser's position
    at exactly these moments, so the reload lands where the user was."""
    visible = {layer: True for layer in basemap.TILE_LAYER_ORDER}
    assert _component_key(tile_visibility=visible) != _component_key(
        tile_visibility={**visible, "massing": False}
    )


def test_the_library_is_fetched_from_this_app_and_not_from_a_cdn():
    """The bug this prevents: `streamlit_folium` awaits every `default_js` URL
    before it renders and catches no failure, and the map's div is filled
    inside that promise. Pointed at a third-party CDN, one blocked or flaky
    host does not degrade the map - it deletes it, with nothing on the page to
    say why. Served from `tiles`, the library is reachable on exactly the
    condition the tiles are, which is the condition this renderer already
    requires."""
    grid = basemap._vector_grid_class()
    urls = [src for _name, src in grid.default_js]

    assert urls == [tiles.vectorgrid_url()]
    assert not any(url.startswith("http") and "//" in url.split("/tiles")[0]
                   and "localhost" not in url for url in urls), urls
    for host in ("unpkg.com", "cdn.jsdelivr.net", "cdnjs.cloudflare.com"):
        assert host not in urls[0]


def test_the_library_url_follows_the_tiles_behind_the_load_balancer(monkeypatch):
    """Same origin as the tiles under every deployment shape, so it rides the
    ALB rule that is already routing /tiles/* rather than needing its own."""
    monkeypatch.setenv("HBU_TILE_BASE_URL", "same-origin")
    assert tiles.vectorgrid_url().startswith(f"{tiles.PATH_PREFIX}/vendor/")

    monkeypatch.delenv("HBU_TILE_BASE_URL", raising=False)
    monkeypatch.setenv("HBU_TILE_PUBLIC_HOST", "localhost")
    assert tiles.vectorgrid_url().startswith("http://localhost:")


def test_the_vendored_library_is_actually_in_the_checkout():
    """A packaging fault here is a blank map, so it is worth an assertion of
    its own rather than only showing up through the route."""
    assert (tiles.VENDOR_DIR / tiles.VECTORGRID_FILE).is_file()


def _st_folium_script(fmap) -> str:
    """The javascript `st_folium` actually ships, not folium's own page.

    It is not the same string: st_folium regenerates the script from the
    element tree and rewrites every `_id` to a stable `div_N` on the way
    through. That rewrite is what these tests are about.
    """
    import streamlit_folium  # noqa: PLC0415

    fmap.render()
    return streamlit_folium._get_map_string(fmap)


def _dangling(js: str) -> set[str]:
    """Vector-grid variables the script uses without declaring."""
    declared = set(re.findall(r"var (vector_grid_protobuf_\w+) = L\.vectorGrid", js))
    used = set(re.findall(r"(vector_grid_protobuf_\w+)", js))
    return used - declared


def test_the_map_survives_being_rendered_more_than_once():
    """The bug: a cached map object re-rendered on the next rerun.

    `st_folium` rewrites each element's `_id` to a stable `div_N`, and holds a
    mapping from the old id to the new one so stale references can be
    repaired. On a second render the ids are *already* `div_N`, so the
    original ids are no longer in that mapping — and anything that captured a
    *name* instead of an element still holds one. What reaches the browser is
    `vector_grid_protobuf_<32 hex>.addTo(map_div)` for a variable that was
    never declared: an uncaught ReferenceError, thrown before
    `initComponent`, which leaves the pane blank rather than the layer empty.
    """
    fmap = basemap.build_map(
        tile_layers={
            layer: f"/tiles/{layer}/{{z}}/{{x}}/{{y}}.mvt"
            for layer in basemap.TILE_LAYER_ORDER
        },
        tile_visibility={
            layer: (i % 2 == 0) for i, layer in enumerate(basemap.TILE_LAYER_ORDER)
        },
    )
    for render in range(1, 4):
        js = _st_folium_script(fmap)
        assert _dangling(js) == set(), f"render #{render} left a dangling reference"
        assert not re.search(r"vector_grid_protobuf_[0-9a-f]{32}", js), (
            f"render #{render} shipped a raw folium id"
        )


def test_the_tooltip_binding_names_the_layer_at_render_time():
    """`add_tile_layers` hands the interaction element the grid *objects*.

    Capturing `get_name()` at build time freezes an id st_folium is about to
    rewrite — the same failure as above, from our side of the line rather than
    folium's.
    """
    fmap = basemap.build_map(
        tile_layers={"lots": "/tiles/lots/{z}/{x}/{y}.mvt"},
        tile_visibility={"lots": True},
    )
    js = _st_folium_script(fmap)
    bound = re.findall(r"hbuBindVectorLayer\(\s*(vector_grid_protobuf_\w+)", js)
    declared = re.findall(r"var (vector_grid_protobuf_\w+) = L\.vectorGrid", js)

    assert bound == declared == ["vector_grid_protobuf_div_1"]


def _declarations(js: str) -> list[tuple[str, str]]:
    """Every vector grid the script declares, as (variable, layer)."""
    return re.findall(
        r"var (vector_grid_protobuf_\w+) = L\.vectorGrid\.protobuf\(\s*'/tiles/(\w+)/",
        js,
    )


def _bindings(js: str) -> list[tuple[str, str, str, dict]]:
    """Every `hbuBindVectorLayer` call, as (variable, layer, map, highlight).

    The javascript *declaration* of that function takes its arguments
    unquoted, so only the calls match.
    """
    return [
        (var, layer, target, json.loads(highlight))
        for var, layer, target, highlight in re.findall(
            r"hbuBindVectorLayer\(\s*(\w+),\s*\"(\w+)\",\s*(\w+),\s*(\{[^}]*\})",
            js,
        )
    ]


def test_every_layer_is_bound_to_its_own_grid():
    """The pairing, not the count.

    `add_tile_layers` fills `bindings` in draw order and the template consumes
    it positionally, so the grid and the layer name it is bound with are held
    together by nothing but that order. Drift between the two - a filter
    applied on one side, a sort, a zip against the wrong sequence - is silent
    in a way the other tests here cannot see: the page still loads, all six
    grids still draw, the layer control is still right, and only the tooltip
    is wrong, reading `capacity`'s fields off a lot and highlighting it in
    another layer's colour. One layer cannot show that, which is why this
    renders all six and checks each variable against the URL it was declared
    with.
    """
    fmap = basemap.build_map(
        tile_layers={
            layer: f"/tiles/{layer}/{{z}}/{{x}}/{{y}}.mvt"
            for layer in basemap.TILE_LAYER_ORDER
        }
    )
    js = _st_folium_script(fmap)
    declared = _declarations(js)
    bound = _bindings(js)

    assert [layer for _var, layer in declared] == list(basemap.TILE_LAYER_ORDER)
    assert [layer for _var, layer, _map, _hl in bound] == list(basemap.TILE_LAYER_ORDER)

    grid_of = dict(declared)
    for var, layer, target, highlight in bound:
        assert var in grid_of, f"{layer} is bound to an undeclared {var}"
        assert grid_of[var] == layer, f"{layer} is bound to the {grid_of[var]} grid"
        # The style the hover paints, from the one place it is written.
        assert highlight == basemap._TILE_HIGHLIGHT[layer]
        # The map, so the re-fired click reaches what streamlit_folium reads.
        assert target == "map_div"


def test_a_map_short_of_a_layer_still_pairs_the_rest():
    """`add_tile_layers` skips a layer with no URL, and a skip shifts every
    binding after it by one. A short `tile_layers` is a normal state rather
    than a broken one - the app rebuilds the dict every rerun - so the gap has
    to close on both sides at once."""
    subset = ["zones", "lots", "massing"]
    fmap = basemap.build_map(
        tile_layers={
            layer: f"/tiles/{layer}/{{z}}/{{x}}/{{y}}.mvt" for layer in subset
        }
    )
    js = _st_folium_script(fmap)
    grid_of = dict(_declarations(js))
    bound = _bindings(js)

    assert [layer for _var, layer, _map, _hl in bound] == subset
    for var, layer, _target, _highlight in bound:
        assert grid_of[var] == layer, f"{layer} is bound to the {grid_of[var]} grid"


def test_two_maps_built_from_the_same_inputs_ship_the_same_script():
    """Which is why the map can be rebuilt every rerun without the pane
    blinking: st_folium keys its component on a hash that strips the variable
    suffixes, so an independently built map is the same component."""
    import streamlit_folium  # noqa: PLC0415

    def build():
        return basemap.build_map(
            tile_layers={
                layer: f"/tiles/{layer}/{{z}}/{{x}}/{{y}}.mvt"
                for layer in basemap.TILE_LAYER_ORDER
            },
            tile_visibility=dict.fromkeys(basemap.TILE_LAYER_ORDER, True),
        )

    first, second = _st_folium_script(build()), _st_folium_script(build())
    assert first == second
    assert streamlit_folium.generate_js_hash(
        first, "zoning_map", False
    ) == streamlit_folium.generate_js_hash(second, "zoning_map", False)


def test_the_map_draws_one_vector_grid_per_layer():
    html = _rendered()
    assert html.count("L.vectorGrid.protobuf(") == len(basemap.TILE_LAYER_ORDER)


def test_a_tile_map_embeds_no_geometry():
    """The whole point: the page carries URLs, not coordinates."""
    html = _rendered()
    assert "FeatureCollection" not in html


def test_the_zoom_gates_are_handed_to_leaflet():
    """So crossing one costs no rerun and no query."""
    html = _rendered()
    assert f"minZoom: {basemap.MIN_LOT_ZOOM}" in html
    assert f"minZoom: {basemap.MIN_BUILDING_ZOOM}" in html


def test_the_click_is_forwarded_to_the_map():
    """VectorGrid's `interactive` stops the map's own click, which is what
    streamlit_folium reports as last_clicked and what selects a lot."""
    html = _rendered()
    assert "map.fire('click'" in html


def test_the_capacity_bands_reach_the_browser_from_the_one_place_they_live():
    html = _rendered()
    for _upper, colour, _label in basemap._CAPACITY_BANDS:
        assert colour in html
    assert basemap._CAPACITY_OVER_COLOR in html
    assert basemap._CAPACITY_NONE_COLOR in html


def test_the_legend_and_the_tile_style_cannot_disagree():
    """Both read `_CAPACITY_BANDS`; this is the assertion that keeps it so."""
    html = _rendered()
    for colour, _label in basemap.capacity_legend_rows():
        assert colour in html


def test_the_streets_are_drawn_as_lines_rather_than_filled():
    """Leaflet fills a path by closing it across its two ends, so a filled
    street side paints a wedge across the block instead of a line along the
    curb. The style has to say so, and this is the assertion that keeps it."""
    html = _rendered()
    assert basemap._STREET_STYLE["color"] in html
    assert basemap._STREET_STYLE["fill"] is False
    assert '"fill": false' in html


def test_the_streets_sit_under_the_cadastre():
    """A click on this map means "select the lot under the cursor". An
    interactive line layer above the lots would swallow that click along every
    frontage, which is where a reader is most likely to aim."""
    order = list(basemap.TILE_LAYER_ORDER)
    assert order.index("streets") > order.index("capacity")
    assert order.index("streets") < order.index("lots")


def test_the_street_tooltip_labels_an_unnamed_lane_rather_than_blanking_it():
    """An unnamed service lane is a real street side. Both twins say so —
    `decorate` in Python and `hbuStreetLabel` in the browser."""
    html = _rendered()
    assert "hbuStreetLabel" in html
    assert "unnamed lane" in html

    features = queries.FeatureSet(
        features=[
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                "properties": {"street_name": None, "length_m": 82.4},
            }
        ],
        layer="streets",
    )
    basemap.decorate(features, "streets")
    props = features.features[0]["properties"]
    assert props["street_label"] == "unnamed lane"
    assert props["length_label"] == "82 m"


def test_the_massing_colours_are_the_python_ones():
    html = _rendered()
    assert basemap._MASSING_FITTED_STYLE["fillColor"] in html
    assert basemap._MASSING_SHRUNK_STYLE["fillColor"] in html


def test_the_tile_script_is_pinned_rather_than_latest():
    """A map whose rendering changes overnight is a map nobody can bisect.

    The pin now lives in the *filename* of the vendored copy rather than in a
    CDN's version selector, so the same guarantee is read from there — and
    folium's own ``@latest`` must not survive the subclass either way.
    """
    html = _rendered()
    assert "1.3.0" in tiles.VECTORGRID_FILE
    assert tiles.VECTORGRID_FILE in html
    assert "vectorgrid@latest" not in html


def test_the_browser_side_labels_are_pure_ascii():
    """They travel inside an srcdoc iframe; \\uXXXX cannot be mangled by a
    charset guess anywhere on that path, and 'm²' can."""
    assert all(ord(c) < 128 for c in basemap._TOOLTIP_JS)


def test_a_layer_the_sidebar_has_unticked_is_still_offered():
    """It is added to the layer control but not to the map, so turning it on
    costs no rerun."""
    html = _rendered(tile_visibility={"massing": False})
    assert "/tiles/massing/" in html
    assert "Proposed massing" in html


# ---------------------------------------------------------------------------
# The entrypoint
# ---------------------------------------------------------------------------


def test_the_tile_server_starts_before_streamlit(monkeypatch):
    """The whole reason `serve.py` exists.

    Streamlit runs the app script per *session*, so the tile server that
    `app.py` starts comes up on the first page load. Behind the load balancer
    that is a deployment loop: a task nobody has visited fails the tile target
    group's health check, ECS replaces it, and the replacement is never
    visited either.
    """
    import streamlit.web.cli as streamlit_cli

    import serve

    order: list[str] = []
    monkeypatch.setattr(tiles, "start", lambda *_a, **_k: order.append("tiles") or 8502)
    monkeypatch.setattr(streamlit_cli, "main", lambda *_a, **_k: order.append("streamlit"))
    monkeypatch.setattr("sys.argv", ["serve", "--server.port=8501"])

    serve.main()

    assert order == ["tiles", "streamlit"]


def test_the_entrypoint_passes_its_arguments_through(monkeypatch):
    import streamlit.web.cli as streamlit_cli

    import serve

    seen: list[list[str]] = []
    monkeypatch.setattr(tiles, "start", lambda *_a, **_k: 8502)
    monkeypatch.setattr(
        streamlit_cli, "main", lambda *_a, **_k: seen.append(list(__import__("sys").argv))
    )
    monkeypatch.setattr("sys.argv", ["serve", "--server.port=9000", "--server.headless=true"])

    serve.main()

    assert seen[0][:2] == ["streamlit", "run"]
    assert seen[0][2].endswith("app.py")
    assert seen[0][3:] == ["--server.port=9000", "--server.headless=true"]


def test_a_tile_server_that_cannot_bind_does_not_stop_the_app(monkeypatch, capsys):
    """It falls back to the GeoJSON renderer and says so, which is more use
    than a container that will not start."""
    import streamlit.web.cli as streamlit_cli

    import serve

    started: list[bool] = []
    monkeypatch.setattr(tiles, "start", lambda *_a, **_k: None)
    monkeypatch.setattr(streamlit_cli, "main", lambda *_a, **_k: started.append(True))
    monkeypatch.setattr("sys.argv", ["serve"])

    serve.main()

    assert started == [True]
    assert "GeoJSON" in capsys.readouterr().err
