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

import urllib.error
import urllib.request
from datetime import date

import pytest

from src.utils import basemap, queries, tiles


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
    assert len(calls) == len(queries.MVT_LAYER_NAMES) == 5


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
# The map the URLs end up on
# ---------------------------------------------------------------------------


def _rendered(**kwargs) -> str:
    urls = {
        layer: f"/tiles/{layer}/{{z}}/{{x}}/{{y}}.mvt"
        for layer in basemap.TILE_LAYER_ORDER
    }
    return basemap.build_map(tile_layers=urls, **kwargs).get_root().render()


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


def test_the_massing_colours_are_the_python_ones():
    html = _rendered()
    assert basemap._MASSING_FITTED_STYLE["fillColor"] in html
    assert basemap._MASSING_SHRUNK_STYLE["fillColor"] in html


def test_the_tile_script_is_pinned_rather_than_latest():
    """A map whose rendering changes overnight is a map nobody can bisect."""
    html = _rendered()
    assert "leaflet.vectorgrid@1.3.0" in html
    assert "vectorgrid@latest" not in html


def test_the_browser_side_labels_are_pure_ascii():
    """They travel inside an srcdoc iframe; \\uXXXX cannot be mangled by a
    charset guess anywhere on that path, and 'étages' can."""
    assert all(ord(c) < 128 for c in basemap._TOOLTIP_JS)


def test_a_layer_the_sidebar_has_unticked_is_still_offered():
    """It is added to the layer control but not to the map, so turning it on
    costs no rerun."""
    html = _rendered(tile_visibility={"massing": False})
    assert "/tiles/massing/" in html
    assert "Massing propos" in html


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
