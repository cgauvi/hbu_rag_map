"""The tile archives: where they are, the URLs, the key, and the server.

The map used to render every vector tile in this process, one query per tile
against the same Postgres the panes read. It now reads PMTiles archives the
dataplatform wrote, straight off S3, and this process only says where they
are. These tests hold the things that are easy to get quietly wrong in that:

* the archive root's three shapes resolve to the right kind of URL - a
  presigned S3 GET, a plain join, or a keyed route on this server - and a
  layer the dataplatform did not build gets no URL at all;
* the key on a locally served archive or a grid is derived from the app
  password, so a public listener does not put the cadastre on the internet;
* the local route answers byte ranges the way the PMTiles reader needs them,
  because a whole-file answer to a range request is a reader that gives up.

Nothing here opens a database or an AWS connection: the S3 client is stubbed
where a test needs one.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pytest

from src.utils import basemap, documents, queries, tiles

DATE = date(2026, 9, 1)
MANIFEST = {
    "scrape_date": "2026-09-01",
    "neighborhood": "VSMPE",
    "layers": {
        "lots": {"file": "lots.pmtiles", "min_zoom": 6, "max_zoom": 19},
        "zones": {"file": "zones.pmtiles", "min_zoom": 6, "max_zoom": 19},
    },
    "empty_layers": ["massing"],
}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(tiles.TILES_URL_ENV, raising=False)
    tiles.forget_manifests()
    yield
    tiles.forget_manifests()


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    """A directory shaped like the dataplatform's gold/map_tiles tree."""
    partition = tmp_path / "2026-09-01" / "VSMPE"
    partition.mkdir(parents=True)
    (partition / "map_tiles.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (partition / "lots.pmtiles").write_bytes(bytes(range(256)) * 4)
    monkeypatch.setenv(tiles.TILES_URL_ENV, str(tmp_path))
    return tmp_path


class _FakeS3:
    """Enough of a boto3 S3 client for the manifest read and the presign."""

    class exceptions:  # noqa: N801 - boto's own spelling
        class NoSuchKey(Exception):
            pass

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.calls: list[tuple[str, str]] = []

    def get_object(self, Bucket, Key):  # noqa: N803 - boto's own spelling
        self.calls.append(("get", Key))
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()
        import io

        return {"Body": io.BytesIO(self.objects[Key])}

    def generate_presigned_url(self, operation, Params, ExpiresIn):  # noqa: N803
        self.calls.append(("presign", Params["Key"]))
        return (
            f"https://{Params['Bucket']}.s3.amazonaws.com/{Params['Key']}"
            f"?X-Amz-Expires={ExpiresIn}&X-Amz-Signature=deadbeef"
        )


@pytest.fixture
def s3_root(monkeypatch):
    client = _FakeS3(
        {"dev/gold/map_tiles/2026-09-01/VSMPE/map_tiles.json": json.dumps(MANIFEST).encode()}
    )
    monkeypatch.setenv(tiles.TILES_URL_ENV, "s3://urban-rag-dataplatform/dev/gold/map_tiles")
    monkeypatch.setattr(tiles, "_s3_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# Where the archives are
# ---------------------------------------------------------------------------


def test_unset_means_no_tiles():
    assert tiles.source() is None
    assert not tiles.configured()
    assert tiles.describe() == "not configured"
    assert tiles.archive_url(DATE, "VSMPE", "lots") is None


def test_an_s3_root_is_split_into_bucket_and_prefix(monkeypatch):
    monkeypatch.setenv(tiles.TILES_URL_ENV, "s3://bucket/dev/gold/map_tiles/")
    configured = tiles.source()
    assert configured.kind == "s3"
    assert configured.bucket == "bucket"
    assert configured.prefix == "dev/gold/map_tiles"


def test_an_s3_root_with_no_bucket_is_refused(monkeypatch):
    monkeypatch.setenv(tiles.TILES_URL_ENV, "s3://")
    assert tiles.source() is None


def test_an_http_root_is_used_as_given(monkeypatch):
    monkeypatch.setenv(tiles.TILES_URL_ENV, "https://tiles.example.org/map_tiles/")
    assert tiles.source().kind == "http"
    assert tiles.source().root == "https://tiles.example.org/map_tiles"


def test_anything_else_is_a_directory(local_root):
    assert tiles.source().kind == "local"
    assert Path(tiles.source().root) == local_root


def test_the_archive_key_is_the_dataplatforms_layout():
    assert tiles.archive_key(DATE, "VSMPE", "lots") == "2026-09-01/VSMPE/lots.pmtiles"
    assert tiles.archive_key("2026-09-01", "CIL", "zones") == "2026-09-01/CIL/zones.pmtiles"
    assert tiles.manifest_key(DATE, "VSMPE") == "2026-09-01/VSMPE/map_tiles.json"


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_the_manifest_says_which_layers_were_built(local_root):
    listed = tiles.manifest(DATE, "VSMPE")
    assert set(listed["layers"]) == {"lots", "zones"}
    assert tiles.built_layers(DATE, ["VSMPE"]) == {"lots", "zones"}


def test_a_partition_with_no_manifest_has_no_tiles(local_root):
    assert tiles.manifest(DATE, "CIL") is None
    assert tiles.built_layers(DATE, ["CIL"]) == set()
    assert tiles.layer_archives("lots", DATE, ["CIL"]) == []


def test_the_manifest_is_read_once_per_partition_not_once_per_layer(s3_root):
    for layer in queries.MVT_LAYER_NAMES:
        tiles.archive_url(DATE, "VSMPE", layer)
    reads = [key for kind, key in s3_root.calls if kind == "get"]
    assert reads == ["dev/gold/map_tiles/2026-09-01/VSMPE/map_tiles.json"]


def test_a_missing_object_on_s3_is_no_tiles_not_an_error(s3_root):
    assert tiles.manifest(DATE, "SAG") is None


def test_an_unreadable_manifest_is_no_tiles_not_an_error(local_root):
    (local_root / "2026-09-01" / "VSMPE" / "map_tiles.json").write_text("{not json", encoding="utf-8")
    assert tiles.manifest(DATE, "VSMPE") is None


def test_no_snapshot_means_no_archives(local_root):
    assert tiles.layer_archives("lots", None, ["VSMPE"]) == []
    assert tiles.built_layers(None, ["VSMPE"]) == set()


# ---------------------------------------------------------------------------
# The URLs
# ---------------------------------------------------------------------------


def test_an_s3_archive_is_presigned_under_the_prefix(s3_root):
    url = tiles.archive_url(DATE, "VSMPE", "lots")
    assert url.startswith(
        "https://urban-rag-dataplatform.s3.amazonaws.com/dev/gold/map_tiles/2026-09-01/VSMPE/lots.pmtiles?"
    )
    assert "X-Amz-Signature" in url
    assert f"X-Amz-Expires={tiles.PRESIGN_SECONDS}" in url


def test_a_presigned_url_is_reused_so_a_rerun_does_not_remount_the_map(s3_root):
    first = tiles.archive_url(DATE, "VSMPE", "lots")
    second = tiles.archive_url(DATE, "VSMPE", "lots")
    assert first == second
    assert sum(kind == "presign" for kind, _ in s3_root.calls) == 1


def test_a_layer_the_dataplatform_did_not_build_gets_no_url(s3_root):
    assert tiles.archive_url(DATE, "VSMPE", "massing") is None
    assert not any(kind == "presign" for kind, _ in s3_root.calls)


def test_an_http_archive_is_a_plain_join(monkeypatch):
    monkeypatch.setenv(tiles.TILES_URL_ENV, "https://tiles.example.org/map_tiles")
    monkeypatch.setattr(tiles, "_read_text", lambda source, key: json.dumps(MANIFEST))
    assert (
        tiles.archive_url(DATE, "VSMPE", "lots")
        == "https://tiles.example.org/map_tiles/2026-09-01/VSMPE/lots.pmtiles"
    )


def test_a_local_archive_is_served_from_this_process(local_root, monkeypatch):
    monkeypatch.delenv("HBU_TILE_BASE_URL", raising=False)
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    url = tiles.archive_url(DATE, "VSMPE", "lots")
    assert url == f"http://localhost:{tiles.DEFAULT_PORT}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles"


def test_a_local_archive_url_carries_the_key_when_there_is_one(local_root, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    url = tiles.archive_url(DATE, "VSMPE", "lots")
    assert url == f"/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles?k={tiles.tile_key()}"


def test_all_loaded_is_one_archive_per_borough(local_root):
    partition = local_root / "2026-09-01" / "CIL"
    partition.mkdir()
    (partition / "map_tiles.json").write_text(json.dumps({"layers": {"lots": {}}}), encoding="utf-8")
    urls = tiles.layer_archives("lots", DATE, ["VSMPE", "CIL", "SAG"])
    assert [url.split("/")[-2] for url in urls] == ["VSMPE", "CIL"]
    # A layer only one borough built is still drawn there.
    assert len(tiles.layer_archives("zones", DATE, ["VSMPE", "CIL"])) == 1


# ---------------------------------------------------------------------------
# Paths and ranges
# ---------------------------------------------------------------------------


def test_an_archive_path_parses():
    assert tiles.parse_archive_path("/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles") == (
        "2026-09-01", "VSMPE", "lots"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/tiles/pmtiles/2026-09-01/VSMPE/parks.pmtiles",  # not a layer
        "/tiles/pmtiles/2026-09-01/VSMPE/lots.parquet",  # not an archive
        "/tiles/pmtiles/../VSMPE/lots.pmtiles",  # not a date
        "/tiles/pmtiles/2026-09-01/../lots.pmtiles",  # not a borough
        "/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles/extra",
        "/tiles/pmtiles/lots.pmtiles",
        "/tiles/lots/15/9646/11732.mvt",  # the old route
    ],
)
def test_a_malformed_archive_path_is_refused(path):
    assert tiles.parse_archive_path(path) is None


def test_a_grid_path_parses():
    assert tiles.parse_grid_path("/tiles/grid/784a0b4f710d1785.pdf") == "784a0b4f710d1785"


@pytest.mark.parametrize(
    "path",
    ["/tiles/grid/not-hex.pdf", "/tiles/grid/784a0b4f710d1785.txt", "/tiles/grid/../x.pdf"],
)
def test_a_malformed_grid_path_is_refused(path):
    assert tiles.parse_grid_path(path) is None


@pytest.mark.parametrize(
    "header,size,expected",
    [
        (None, 100, None),
        ("bytes=0-15", 100, (0, 15)),
        ("bytes=90-", 100, (90, 99)),
        ("bytes=-10", 100, (90, 99)),
        ("bytes=0-1000", 100, (0, 99)),
        ("bytes=200-300", 100, (100, 99)),  # unsatisfiable: told, not truncated
        ("items=0-1", 100, None),
        ("bytes=x-y", 100, None),
        ("bytes=5-2", 100, None),
    ],
)
def test_a_range_header_is_read_the_way_the_reader_writes_it(header, size, expected):
    assert tiles.parse_range(header, size) == expected


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_no_password_means_no_key(monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    assert tiles.tile_key() is None


def test_the_placeholder_password_grants_nothing(monkeypatch):
    from src.utils import auth

    monkeypatch.setenv("HBU_APP_PASSWORD", auth._PLACEHOLDER)
    assert tiles.tile_key() is None


def test_the_key_is_derived_from_the_password_and_is_not_the_password(monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    key = tiles.tile_key()
    assert key and key != "hunter2"
    assert re.fullmatch(r"[0-9a-f]{32}", key)


def test_every_task_derives_the_same_key(monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    assert tiles.tile_key() == tiles.tile_key()


def test_a_grid_url_is_same_origin_behind_the_balancer(monkeypatch):
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    assert tiles.grid_url("784a0b4f710d1785") == "/tiles/grid/784a0b4f710d1785.pdf"


def test_a_grid_url_carries_the_key_when_there_is_one(monkeypatch):
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    assert tiles.grid_url("784a0b4f710d1785") == (
        f"/tiles/grid/784a0b4f710d1785.pdf?k={tiles.tile_key()}"
    )


def test_no_url_is_built_for_something_that_is_not_a_document_id():
    assert tiles.grid_url("../etc/passwd") is None


def test_a_laptop_gets_absolute_library_urls(monkeypatch):
    monkeypatch.delenv("HBU_TILE_BASE_URL", raising=False)
    assert tiles.vectorgrid_url() == (
        f"http://localhost:{tiles.DEFAULT_PORT}/tiles/vendor/{tiles.VECTORGRID_FILE}"
    )
    assert tiles.pmtiles_url() == (
        f"http://localhost:{tiles.DEFAULT_PORT}/tiles/vendor/{tiles.PMTILES_FILE}"
    )


def test_the_libraries_follow_the_app_behind_the_load_balancer(monkeypatch):
    monkeypatch.setenv("HBU_TILE_BASE_URL", "same-origin")
    assert tiles.vectorgrid_url() == f"/tiles/vendor/{tiles.VECTORGRID_FILE}"
    assert tiles.pmtiles_url() == f"/tiles/vendor/{tiles.PMTILES_FILE}"


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


@pytest.fixture
def running(monkeypatch):
    """The server on an ephemeral port."""
    monkeypatch.setenv("HBU_TILE_BIND", "127.0.0.1")
    tiles.stop()
    port = tiles.start(0)  # 0 lets the OS pick, so a busy 8502 is not a flake
    assert port
    yield f"http://127.0.0.1:{port}"
    tiles.stop()


def _get(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    request = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def test_the_health_path_needs_no_key(running, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    status, body, _headers = _get(f"{running}{tiles.HEALTH_PATH}")
    assert (status, body) == (200, b"ok")


@pytest.mark.parametrize("name", tiles.VENDOR_FILES)
def test_the_server_hands_out_both_vendored_libraries(running, name):
    status, body, headers = _get(f"{running}/tiles/vendor/{name}")
    assert status == 200
    assert body == (tiles.VENDOR_DIR / name).read_bytes()
    assert headers["Content-Type"].startswith("application/javascript")
    assert "immutable" in headers["Cache-Control"]
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_the_libraries_need_no_key(running, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    for name in tiles.VENDOR_FILES:
        status, _body, _headers = _get(f"{running}/tiles/vendor/{name}")
        assert status == 200


@pytest.mark.parametrize(
    "path",
    ["/tiles/vendor/../tiles.py", "/tiles/vendor/README.md", "/tiles/vendor/leaflet.js"],
)
def test_only_the_vendored_files_are_published(running, path):
    status, _body, _headers = _get(f"{running}{path}")
    assert status == 404


def test_the_old_tile_route_is_gone(running):
    status, _body, _headers = _get(f"{running}/tiles/lots/15/9646/11732.mvt")
    assert status == 404


def test_a_local_archive_is_served_whole(running, local_root, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    status, body, headers = _get(f"{running}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles")
    assert status == 200
    assert body == bytes(range(256)) * 4
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["ETag"]
    assert "ETag" in headers["Access-Control-Expose-Headers"]
    assert "Content-Range" in headers["Access-Control-Expose-Headers"]


def test_a_local_archive_answers_a_byte_range(running, local_root, monkeypatch):
    """The whole point of the route: the reader asks for the header first,
    then the directory, then one tile, and each is a range."""
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    status, body, headers = _get(
        f"{running}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles",
        headers={"Range": "bytes=256-259"},
    )
    assert status == 206
    assert body == bytes([0, 1, 2, 3])
    assert headers["Content-Range"] == "bytes 256-259/1024"
    assert headers["Content-Length"] == "4"


def test_an_unsatisfiable_range_is_a_416_with_the_size(running, local_root, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    status, _body, headers = _get(
        f"{running}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles",
        headers={"Range": "bytes=5000-6000"},
    )
    assert status == 416
    assert headers["Content-Range"] == "bytes */1024"


def test_a_range_request_is_preflighted_and_allowed(running, local_root):
    status, _body, headers = _get(
        f"{running}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Headers": "range"},
        method="OPTIONS",
    )
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert headers["Access-Control-Allow-Headers"] == "*"


def test_an_archive_needs_the_key_when_a_password_is_set(running, local_root, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    status, _body, _headers = _get(f"{running}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles")
    assert status == 403
    status, _body, _headers = _get(
        f"{running}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles?k={tiles.tile_key()}"
    )
    assert status == 200


def test_an_archive_that_is_not_there_is_a_404(running, local_root, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    status, _body, _headers = _get(f"{running}/tiles/pmtiles/2026-09-01/VSMPE/zones.pmtiles")
    assert status == 404


def test_the_local_route_never_proxies_a_bucket(running, monkeypatch):
    """An S3 root is fetched by the browser; a URL shaped like the local
    route must not make this process go and get it."""
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    monkeypatch.setenv(tiles.TILES_URL_ENV, "s3://bucket/prefix")
    status, _body, _headers = _get(f"{running}/tiles/pmtiles/2026-09-01/VSMPE/lots.pmtiles")
    assert status == 404


@pytest.fixture
def published_grid():
    """One PDF in the registry, and nothing on disk under it."""
    documents.forget_published()
    body = b"%PDF-1.4 grille"
    doc_id = documents.document_id("http://example.test/zone/C01-001.pdf")
    documents.publish(doc_id, body)
    yield doc_id, body
    documents.forget_published()


def test_the_server_answers_a_published_grid(running, published_grid, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "")
    doc_id, body = published_grid
    status, served, headers = _get(f"{running}/tiles/grid/{doc_id}.pdf")
    assert status == 200
    assert served == body
    assert headers["Content-Type"] == "application/pdf"
    assert headers["Content-Disposition"].startswith("inline")


def test_a_grid_needs_the_key(running, published_grid, monkeypatch):
    monkeypatch.setenv("HBU_APP_PASSWORD", "hunter2")
    doc_id, _body = published_grid
    status, _served, _headers = _get(f"{running}/tiles/grid/{doc_id}.pdf")
    assert status == 403


def test_an_unknown_path_is_a_404(running):
    status, _body, _headers = _get(f"{running}/tiles/nothing/here")
    assert status == 404


# ---------------------------------------------------------------------------
# The map the archives end up on
# ---------------------------------------------------------------------------


def _rendered(**kwargs) -> str:
    archives = {
        layer: [f"https://tiles.example.org/2026-09-01/VSMPE/{layer}.pmtiles"]
        for layer in basemap.TILE_LAYER_ORDER
    }
    return basemap.build_map(tile_layers=archives, **kwargs).get_root().render()


def test_the_map_draws_one_pmtiles_grid_per_layer():
    html = _rendered()
    assert html.count("L.vectorGrid.pmtiles(") == len(basemap.TILE_LAYER_ORDER)
    # The class is defined once, ahead of the first grid.
    assert html.count("L.VectorGrid.PMTiles = L.VectorGrid.Protobuf.extend(") == 1
    assert html.index("L.VectorGrid.PMTiles = ") < html.index("L.vectorGrid.pmtiles(")


def test_a_tile_map_embeds_no_geometry():
    """The whole point: the page carries archive URLs, not coordinates."""
    html = _rendered()
    assert "FeatureCollection" not in html
    assert "2026-09-01/VSMPE/lots.pmtiles" in html


def test_both_libraries_are_fetched_from_this_app_and_not_from_a_cdn(monkeypatch):
    monkeypatch.setenv("HBU_TILE_BASE_URL", "")
    html = _rendered()
    assert f"/tiles/vendor/{tiles.VECTORGRID_FILE}" in html
    assert f"/tiles/vendor/{tiles.PMTILES_FILE}" in html
    assert "unpkg.com" not in html


def test_the_vendored_libraries_are_actually_in_the_checkout():
    for name in tiles.VENDOR_FILES:
        assert (tiles.VENDOR_DIR / name).is_file(), name
    # The PMTiles build exposes the global the glue constructs from.
    assert b"var pmtiles=" in (tiles.VENDOR_DIR / tiles.PMTILES_FILE).read_bytes()[:64]


def test_a_layer_with_no_archive_is_left_off_the_map():
    archives = {"lots": ["https://tiles.example.org/lots.pmtiles"], "zones": []}
    html = basemap.build_map(tile_layers=archives).get_root().render()
    assert html.count("L.vectorGrid.pmtiles(") == 1
    assert '"Zoning" :' not in html


def test_all_loaded_hands_every_borough_archive_to_one_grid():
    archives = {
        "lots": [
            "https://tiles.example.org/2026-09-01/VSMPE/lots.pmtiles",
            "https://tiles.example.org/2026-09-01/CIL/lots.pmtiles",
        ]
    }
    html = basemap.build_map(tile_layers=archives).get_root().render()
    assert html.count("L.vectorGrid.pmtiles(") == 1
    assert "VSMPE/lots.pmtiles" in html and "CIL/lots.pmtiles" in html


def test_every_layer_is_requested_all_the_way_down():
    """Leaflet's floor is the map's, not each layer's detail zoom: below its
    detail zoom a layer's archive holds dissolved cells, and a `minZoom` of 15
    would mean the browser never asked."""
    html = _rendered()
    for layer in queries.AGGREGATE_LAYERS:
        assert basemap.TILE_LAYER_MIN_ZOOM[layer] == basemap.MAP_MIN_ZOOM
    assert f"minZoom: {basemap.MAP_MIN_ZOOM}" in html
    assert f"maxNativeZoom: {basemap.TILE_MAX_NATIVE_ZOOM}" in html
    assert basemap.TILE_LAYER_MIN_ZOOM["zones"] == 0
    assert "surface_parking" not in queries.AGGREGATE_LAYERS
    assert (
        basemap.TILE_LAYER_MIN_ZOOM["surface_parking"]
        == queries.MVT_DETAIL_ZOOM["surface_parking"]
    )


def test_the_detail_zoom_decides_what_the_archive_holds():
    for layer, detail in queries.MVT_DETAIL_ZOOM.items():
        if layer not in queries.AGGREGATE_LAYERS:
            assert not queries.serves_aggregate(layer, 0)
            continue
        assert queries.serves_aggregate(layer, detail - 1)
        assert not queries.serves_aggregate(layer, detail)


def test_a_tile_is_filled_from_cells_four_zooms_finer():
    for zoom in range(basemap.MAP_MIN_ZOOM, max(queries.MVT_DETAIL_ZOOM.values())):
        cell = queries.aggregate_cell_zoom(zoom)
        assert cell in queries.AGGREGATE_CELL_ZOOMS
        if zoom + queries.AGGREGATE_ZOOM_OFFSET <= queries.AGGREGATE_CELL_ZOOMS[-1]:
            assert cell == zoom + queries.AGGREGATE_ZOOM_OFFSET


def test_the_cell_branch_is_a_property_and_not_a_zoom():
    html = _rendered()
    assert "properties.agg_level" in html
    # And the browser-side screens never touch a cell.
    assert "props.agg_level !== null" in html


# ---------------------------------------------------------------------------
# The screens, applied in the browser
# ---------------------------------------------------------------------------


def test_no_screen_means_no_filter_function():
    assert basemap._filter_js("lots", {}) is None
    assert basemap._filter_js("lots", None) is None
    assert basemap._filter_js("zones", {"underbuilt": 1}) is None


def test_the_lot_area_range_reads_area_m2():
    predicate = basemap._filter_js("lots", {"min_area": 200, "max_area": 900.5})
    assert "p.area_m2 >= 200.0" in predicate
    assert "p.area_m2 <= 900.5" in predicate
    assert predicate.startswith("function (p) { return ")


def test_the_underbuilt_screen_reaches_the_three_layers_that_take_it():
    for layer in ("capacity", "massing", "surface_parking"):
        assert basemap._filter_js(layer, {"underbuilt": 1}) == (
            "function (p) { return !!p.is_underbuilt; }"
        )
        assert basemap._filter_js(layer, {"underbuilt": 0}) is None
    assert basemap._filter_js("lots", {"underbuilt": 1}) is None


def test_the_opportunity_screens_read_the_thesis_and_the_two_flags():
    predicate = basemap._filter_js(
        "opportunities", {"site_thesis": "TearDown", "top_only": 1, "good_only": True}
    )
    assert 'p.site_thesis === "teardown"' in predicate
    assert "!!p.is_top_site_opportunity" in predicate
    assert "!!p.is_good_candidate" in predicate
    # A thesis the table cannot hold draws every thesis, not none.
    assert basemap._filter_js("opportunities", {"site_thesis": "renamed"}) is None


def test_the_land_use_side_is_written_onto_the_feature():
    assert basemap._decorate_js("land_use", {"use_side": "hbu"}) == (
        "function (p) { p.use_class = p.hbu_use; }"
    )
    assert basemap._decorate_js("land_use", {}) == (
        "function (p) { p.use_class = p.existing_use; }"
    )
    # A side the layer cannot draw is today's side, not a blank map.
    assert "existing_use" in basemap._decorate_js("land_use", {"use_side": "proposed"})
    assert basemap._decorate_js("lots", {"use_side": "hbu"}) is None


def test_the_screens_reach_the_layer_options():
    html = basemap.build_map(
        tile_layers={
            "lots": ["https://tiles.example.org/lots.pmtiles"],
            "land_use": ["https://tiles.example.org/land_use.pmtiles"],
        },
        tile_filters={"lots": {"min_area": 300}, "land_use": {"use_side": "hbu"}},
    ).get_root().render()
    assert "hbuFilter: function (p) { return (p.area_m2 === null" in html
    assert "hbuDecorate: function (p) { p.use_class = p.hbu_use; }" in html


def test_a_different_screen_is_a_different_map():
    """The screens are baked into the layer's options, so ticking one is a
    rebuild of the map's script - which is what `app.py`'s signature has to
    include for the pane to redraw."""
    plain = basemap.build_map(
        tile_layers={"lots": ["https://tiles.example.org/lots.pmtiles"]}
    ).get_root().render()
    screened = basemap.build_map(
        tile_layers={"lots": ["https://tiles.example.org/lots.pmtiles"]},
        tile_filters={"lots": {"max_area": 500}},
    ).get_root().render()
    assert plain != screened
