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
when the tiles are, which is the condition the vector renderer already depends
on.

MIT licensed, © 2016 Iván Sánchez Ortega. Upgrading is a matter of replacing the
file and the digest above; the URL that serves it carries the version, so a
browser holding the old one asks for the new path rather than a stale body.
