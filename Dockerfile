# ── Build stage ───────────────────────────────────────────────────────────────
# Nothing in this image needs torch: the chat model and the query encoder are
# both served by the HuggingFace Inference API, which is why the runtime layer
# is a few hundred megabytes rather than several gigabytes.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off

WORKDIR /build
COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements-docker.txt

# ── Test stage ────────────────────────────────────────────────────────────────
FROM builder AS test

RUN cp -r /install/. /usr/local/

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

WORKDIR /app
COPY app.py serve.py ./
COPY src/ src/
COPY tests/ tests/
COPY pyproject.toml .

# The integration marker is deselected by pyproject's addopts, so this runs the
# unit suite and needs no database.
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

ARG BUILD_VERSION=local-dev
ENV BUILD_VERSION=${BUILD_VERSION}

RUN adduser --disabled-password --gecos "" appuser

# Amazon's RDS root bundle, so a deployed task can connect with
# sslmode=verify-full instead of dropping to `require`. `require` encrypts but
# authenticates nothing — it accepts any certificate presented — which is most
# of what TLS was for. The global bundle covers every region and rotates
# rarely; hbu_infra/ecs.tf points URBAN_RAG_PG_SSLROOTCERT at this exact path.
ADD --chmod=644 https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
    /etc/ssl/certs/rds-global-bundle.pem

ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    PYTHONUNBUFFERED=1 \
    # Written to on first use; kept under the app dir so a read-only mount is
    # the operator's choice rather than a crash. documents.py degrades to
    # fetching every time when this is not writable.
    HBU_PDF_CACHE_DIR=/app/data/cache/pdf

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app.py serve.py ./
COPY src/ src/
COPY .streamlit/ .streamlit/

RUN mkdir -p /app/data/cache/pdf && chown -R appuser:appuser /app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Two ports, one process. 8501 is Streamlit; 8502 is the map's vector tile
# server, which src/utils/tiles.py runs on a thread of the same interpreter
# because Streamlit serves no routes of its own and Leaflet has to fetch tiles
# over HTTP. hbu_infra's ecs.tf maps both and routes /tiles/* to the second.
#
# Not in the HEALTHCHECK above, deliberately: a task whose tile server could
# not bind should keep serving the app — it falls back to the GeoJSON renderer
# and says so in the sidebar — rather than be killed and replaced by another
# task that will meet the same port. The ALB polls /tiles/healthz separately,
# which is what takes an unhealthy tile server out of rotation without taking
# the app down with it.
EXPOSE 8501 8502

USER appuser

# `serve`, not `streamlit run`: it starts the tile server before Streamlit's
# own, so /tiles/healthz answers as soon as the container is up rather than on
# the first page load. Streamlit runs the app script per *session*, so the lazy
# start in app.py would leave a never-visited task failing the tile target
# group's health check — which ECS reads as an unhealthy task and replaces,
# forever. Everything after the module name is passed through to `streamlit run`.
CMD ["python", "-m", "serve", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
