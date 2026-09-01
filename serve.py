"""
serve.py — the entrypoint: the tile server first, then Streamlit.

`streamlit run app.py` still works and is still what the app is; this exists
for one reason, and it is a deployment reason rather than a development one.

**Streamlit runs the app script per session, not at boot.** Nothing in `app.py`
executes until a browser connects, `st.cache_resource` included — so the tile
server it starts comes up on the first page load and not before. On a laptop
that is invisible: the page that needs tiles is the page that starts them.

Behind the load balancer it is a deployment loop. `hbu_infra` registers each
task in two target groups, and the tile one health-checks `/tiles/healthz`. A
task that has not been visited yet fails that check, ECS reads the failure as
an unhealthy task, replaces it — and the replacement is never visited either.
The service never converges, and the symptom is a rollback with a healthy
image.

So the tile server is started here, before Streamlit's own server binds, and
`app.py`'s `_tile_port()` finds it already running. `tiles.start` is
idempotent, so nothing about the lazy path had to be removed: a direct
`streamlit run app.py` still gets a tile server, just later.

    python -m serve --server.port=8501 ...

Every argument is passed through to `streamlit run` untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: `streamlit run` resolves the script against the working directory, and the
#: entrypoint should not depend on where it was invoked from — the image runs
#: it from /app, `make run` from the repository root, and a developer may run
#: it from anywhere.
APP = Path(__file__).resolve().parent / "app.py"


def main() -> int:
    # Before the import of streamlit, so a slow first import does not sit
    # between the container starting and the tile port answering.
    sys.path.insert(0, str(APP.parent))
    from src.utils import tiles  # noqa: PLC0415

    if tiles.start() is None and os.environ.get("HBU_TILE_ENABLED", "1") not in {
        "0",
        "false",
        "no",
    }:
        # Not fatal, and deliberately so: the app draws its fallback renderer
        # and says why in the sidebar, which is more useful than a container
        # that will not start. The ALB takes this task out of the tile target
        # group by itself, and leaves it serving the app.
        print(
            f"hbu-rag-map: the tile server could not take port "
            f"{tiles.DEFAULT_PORT}; the map will fall back to its GeoJSON "
            "renderer.",
            file=sys.stderr,
        )

    from streamlit.web import cli as streamlit_cli  # noqa: PLC0415

    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    return streamlit_cli.main()


if __name__ == "__main__":
    sys.exit(main())
