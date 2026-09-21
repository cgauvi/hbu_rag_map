# vendor

Third-party browser code the map needs, committed rather than fetched.

## `leaflet-vectorgrid-1.3.0.js`

[Leaflet.VectorGrid](https://github.com/Leaflet/Leaflet.VectorGrid) 1.3.0, the
`bundled` build (it carries its own `pbf` and `vector-tile` dependencies), taken
verbatim from
`https://unpkg.com/leaflet.vectorgrid@1.3.0/dist/Leaflet.VectorGrid.bundled.js`.

    sha256  144c59f4da8a82a8d85a9c18c1a8bb62fdaec26cb2f624228e7f70bcd61ba061
    bytes   105580

**Why it is committed instead of linked.** `streamlit_folium` loads a plugin's
JavaScript by handing every `default_js` URL to the browser and *awaiting* each
one before it renders anything — and it does not catch a failure. One
unreachable script therefore does not degrade the map, it deletes it: the
`<div>` the map would live in is only populated inside the `.then()` that a
rejected load never reaches. A CDN this app does not control was the difference
between a map and a blank pane, and it failed closed in the least diagnosable
way possible. Served off our own tile server, the library is reachable exactly
when the app is, which is the condition the vector renderer already depends on.

MIT licensed, © 2016 Iván Sánchez Ortega. Upgrading is a matter of replacing the
file and the digest above; the URL that serves it carries the version, so a
browser holding the old one asks for the new path rather than a stale body.

## `pmtiles-4.5.0.js`

[PMTiles](https://github.com/protomaps/PMTiles) 4.5.0, the browser build
(`dist/pmtiles.js`, a self-contained script exposing a global `pmtiles`), taken
verbatim from `https://unpkg.com/pmtiles@4.5.0/dist/pmtiles.js`.

    sha256  caf981bc46f6327ee7e65d5dc964d89d38a69f60edca2bd4c5c890c21b554c6c
    bytes   20229

The reader for the tile archives. The map's layers are one PMTiles file each,
written by the dataplatform and fetched by the browser with byte-range
requests; this is what turns `(z, x, y)` into the two or three ranges that
locate a tile in the file, caches the archive's directory, and gunzips what
comes back. `basemap._PMTILES_GRID_JS` binds it to VectorGrid: the bytes it
returns are handed to the vendored parser above through a `blob:` URL, so
VectorGrid's own styling, hover and click code runs unchanged over tiles it
never fetched itself.

Committed for the same reason as VectorGrid, and served from the same place.
BSD-3-Clause, © Protomaps LLC. Same upgrade rule: replace the file, update the
digest, and change `PMTILES_FILE` in `tiles.py` so the URL changes with it.
