# hbu_rag_map — an interactive zoning map over the hbu Postgres.
#
#   make install          create .venv and install the app
#   make db-up            a local postgis+pgvector container, schema applied
#   make check            why is nothing showing up
#   make run              streamlit, at http://localhost:8501
#
# Against RDS instead of the container, from hbu_infra:
#   eval "$(make -s db-app-env ENV=dev)"   # then `make run` here
# or with nothing set at all, the endpoint is discovered from SSM /hbu-dev/db/*.
#
# The dev RDS has no public endpoint, so from outside the VPC it is reached
# through an SSM tunnel. Leave `make db-tunnel ENV=dev LOCAL_PORT=5433` running
# in hbu_infra, then here:
#   make run-tunnel       natively
#   make docker-run-tunnel  in the image

PYTHON      ?= python3
VENV        ?= .venv
# Windows venvs put the entry points in Scripts/, not bin/, so every recipe
# below misses them on a native run. Same `uname -s` test WIN_HOME uses.
BIN          = $(VENV)/$(if $(findstring NT,$(shell uname -s)),Scripts,bin)
UV          := $(shell command -v uv 2>/dev/null)

# Sibling repos. Same convention as hbu_infra's DATAPLATFORM variable: the
# schema lives over there, and this repo runs it from here when it is present.
HBU_INFRA   ?= ../hbu_infra
ENV         ?= dev

LOCAL_PG_PORT ?= 5432
LOCAL_URL     ?= postgresql://urban_rag:urban_rag@localhost:$(LOCAL_PG_PORT)/urban_rag?sslmode=disable
TUNNEL_PORT   ?= 5433
TUNNEL_DB_HOST ?= $(shell sed -n 's/^URBAN_RAG_PG_HOST=//p' .env 2>/dev/null | tail -n 1 | tr -d '\r')

# The address a *native* run dials the tunnel on, and it is deliberately not
# `localhost`. The Session Manager plugin binds IPv4 only; Windows resolves
# localhost to ::1 first, and libpq spends the entire connect_timeout on that
# dead address before falling back to IPv4 and succeeding — so every connect
# takes exactly connect_timeout seconds and then works, which reads as a slow
# tunnel rather than as a misresolution. Worse here than elsewhere, because the
# pool's own timeout is 15s: through `localhost` it raises PoolTimeout while
# libpq is still waiting.
TUNNEL_HOST   ?= 127.0.0.1

IMAGE       ?= hbu-rag-map
IMAGE_TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo local-dev)

# The map's vector tiles come off a second port in the same process, because
# Streamlit serves no routes of its own and Leaflet fetches tiles over HTTP.
# The browser has to be able to reach it, so a container run has to publish it
# alongside 8501 — unpublished, the map draws a basemap and nothing else: the
# page loads, every tile request fails, and the only sign of it is in the
# browser console. Deployed, hbu_infra's ALB routes /tiles/* here instead, and
# the URLs come out relative rather than naming a port at all.
TILE_PORT   ?= 8502

COMPOSE     ?= docker compose
AWS_PROFILE ?= charles_gauvin_east_1
AWS_REGION  ?= us-east-1

# Every host path below is mounted into a container, so it has to be one the
# Docker daemon can open — and on Windows neither $(HOME) nor $(USERPROFILE)
# reliably names it. make here is an msys2 binary from a different installation
# than the shell that invokes it, so $(HOME) can be /home/<user>, and
# USERPROFILE may not survive into make at all. Either way `docker run -v`
# accepts the path and mounts an empty directory, which is the worst shape of
# this failure: the container starts, finds no credentials, and the UI says
# "Not connected to the database" with nothing about a missing mount.
# `cygpath -m -D` is answered by the msys runtime itself and returns the
# C:/Users/<user> form the daemon wants. Empty off Windows, where $(HOME) is
# correct.
WIN_HOME := $(if $(findstring NT,$(shell uname -s)),$(patsubst %/Desktop,%,$(shell cygpath -m -D 2>/dev/null)))
AWS_DIR     ?= $(firstword $(wildcard $(WIN_HOME)/.aws $(USERPROFILE)/.aws $(HOME)/.aws) $(HOME)/.aws)
DOCKER_AWS_CA_BUNDLE_PATH ?= /etc/ssl/certs/aws-ca-bundle.pem

# The same mismatch again, one layer in: msys2 make hands a recipe
# HOME=/home/<user> and drops USERPROFILE entirely, so the *Windows* Python in
# .venv cannot resolve `~`. Path.home() raises "Could not determine home
# directory" and Streamlit dies on it before reading a line of config — and
# db.py's DEFAULT_CA_BUNDLE is built from the same call. WIN_HOME is already
# the C:/Users/<user> form; hand it over under the name Windows expanduser()
# checks first. Empty off Windows, where HOME is already right.
NATIVE_HOME_ENV = $(if $(WIN_HOME),USERPROFILE="$(WIN_HOME)" HOME="$(WIN_HOME)")

# The docker targets pass the profile explicitly and the native ones did not,
# which left `make run` reading whatever default credentials the shell happens
# to hold. Those are an account that cannot see the app-role secret, so the run
# dies on a cross-account AccessDenied naming secretsmanager — a much less
# obvious message than "wrong profile". `?=` above means an AWS_PROFILE already
# exported by the caller still wins.
NATIVE_AWS_ENV = AWS_PROFILE="$(AWS_PROFILE)" AWS_REGION="$(AWS_REGION)" \
	AWS_DEFAULT_REGION="$(AWS_REGION)"

# Behind a TLS-inspecting proxy the image has no way to verify AWS: botocore
# ships its own CA list and reads neither the host trust store nor anything the
# container was not given. The very first thing the app does is read the
# app-role password out of Secrets Manager, so without this the run dies on
# CERTIFICATE_VERIFY_FAILED and the UI reports it as "Not connected to the
# database" — the proxy nowhere in sight.
#
# Requiring the caller to export AWS_CA_BUNDLE made that the normal outcome,
# since `make docker-run-tunnel` in a fresh shell has none. .env already names
# the combined corporate+certifi bundle as SSL_CERT_FILE, for the HuggingFace
# client that only honours that variable, so fall back to it — and then to the
# conventional location, for a .env that predates that line.
ifeq (,$(strip $(AWS_CA_BUNDLE)))
AWS_CA_BUNDLE := $(firstword \
  $(shell sed -n 's/^SSL_CERT_FILE=//p' .env 2>/dev/null | tail -n 1 | tr -d '\r') \
  $(wildcard $(WIN_HOME)/.certs/zscaler-plus-certifi.pem $(HOME)/.certs/zscaler-plus-certifi.pem))
endif

# SSL_CERT_FILE is overridden either way, never merely passed through: the value
# in .env is a *host* path, and OpenSSL handed a filename that does not exist
# does not fall back to the system roots — it verifies against nothing.
ifneq (,$(strip $(AWS_CA_BUNDLE)))
DOCKER_AWS_CA_ARGS = -v "$(AWS_CA_BUNDLE):$(DOCKER_AWS_CA_BUNDLE_PATH):ro" \
	-e AWS_CA_BUNDLE="$(DOCKER_AWS_CA_BUNDLE_PATH)" \
	-e SSL_CERT_FILE="$(DOCKER_AWS_CA_BUNDLE_PATH)"
else
DOCKER_AWS_CA_ARGS = -e SSL_CERT_FILE=
endif

.PHONY: help install run run-tunnel check test lint fmt \
        db-up db-down db-init db-shell db-url db-logs \
        docker-build docker-run docker-run-tunnel docker-test clean

help: ## This list
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

install: ## Create the venv and install the app (uv when available, pip otherwise)
ifdef UV
	uv venv $(VENV) --python 3.12
	uv pip install --python $(BIN)/python -e ".[dev]"
else
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
endif
	@test -f .env || (cp .env.example .env && echo "Created .env — fill in HUGGINGFACE_API_TOKEN")

# `serve`, not `streamlit run app.py`: it brings the map's tile server up
# before Streamlit's, rather than on the first page load. See serve.py — the
# difference only matters behind a load balancer, but running the two the same
# way locally is what keeps that path exercised.
run: ## Start the app at http://localhost:8501 (tiles on $(TILE_PORT))
	$(NATIVE_HOME_ENV) $(NATIVE_AWS_ENV) HBU_TILE_PORT=$(TILE_PORT) \
	$(BIN)/python -m serve

# The native counterpart of docker-run-tunnel, and it now holds the same TLS
# posture. That target keeps `verify-full` by mapping the RDS hostname onto the
# host gateway with --add-host, so the name the certificate carries still
# resolves. A native run cannot rewrite its own resolver, so it splits the two
# halves instead: URBAN_RAG_PG_HOST goes on naming the endpoint, which is what
# verify-full matches the certificate against, while URBAN_RAG_PG_HOSTADDR
# carries the loopback address the socket actually goes to. Until db.py
# understood hostaddr this had to drop to `require` — encrypted but
# authenticating nothing, leaning on the SSM session as the only authenticated
# hop. That trade is no longer necessary, so the guards below now match
# docker-run-tunnel's: TUNNEL_DB_HOST has to be the endpoint, never a loopback
# address, or verify-full would fail the hostname check it exists to enforce.
#
# DATABASE_URL and URBAN_RAG_PG_DSN are cleared rather than left alone because
# they outrank URBAN_RAG_PG_* in db.py's resolution order, and either one in
# .env would silently win over everything set here. python-dotenv does not
# override a variable already present in the environment, empty included, so
# clearing them is enough — the same thing docker-run-tunnel does with `-e`.
run-tunnel: ## Start Streamlit against an open hbu_infra db-tunnel
	@test -n "$(TUNNEL_DB_HOST)" || { \
	  echo "TUNNEL_DB_HOST is empty; set it to the RDS endpoint, or put URBAN_RAG_PG_HOST in .env"; \
	  exit 1; \
	}
	@case "$(TUNNEL_DB_HOST)" in \
	  localhost|127.*|::1) \
	    echo "TUNNEL_DB_HOST must be the RDS endpoint, not $(TUNNEL_DB_HOST), when sslmode=verify-full"; \
	    exit 1 ;; \
	esac
	@$(BIN)/python -c "import socket; socket.create_connection(('$(TUNNEL_HOST)', $(TUNNEL_PORT)), 2)" 2>/dev/null \
	  || { \
	    echo "Nothing is listening on $(TUNNEL_HOST):$(TUNNEL_PORT)."; \
	    echo "  cd $(HBU_INFRA) && make db-tunnel ENV=$(ENV) LOCAL_PORT=$(TUNNEL_PORT)"; \
	    echo "  (leave it running, then re-run this in another terminal)"; \
	    exit 1; \
	  }
	$(NATIVE_HOME_ENV) $(NATIVE_AWS_ENV) \
	DATABASE_URL= URBAN_RAG_PG_DSN= \
	URBAN_RAG_PG_HOST=$(TUNNEL_DB_HOST) \
	URBAN_RAG_PG_HOSTADDR=$(TUNNEL_HOST) \
	URBAN_RAG_PG_PORT=$(TUNNEL_PORT) \
	URBAN_RAG_PG_SSLMODE=verify-full \
	HBU_TILE_PORT=$(TILE_PORT) \
	$(BIN)/python -m serve

check: ## Report what is and is not loaded, and how to fix each gap
	$(BIN)/python scripts/doctor.py

test: ## Unit tests (the integration marker is deselected by default)
	$(BIN)/python -m pytest tests/ -v

test-integration: ## Also run the tests that need a reachable database
	$(BIN)/python -m pytest tests/ -v -m integration

lint: ## ruff
	$(BIN)/ruff check src/ app.py serve.py scripts/ tests/

fmt: ## ruff --fix
	$(BIN)/ruff check --fix src/ app.py serve.py scripts/ tests/

# ---------------------------------------------------------------------------
# The local database
#
# A stand-in for RDS: same extensions, same schema, no AWS account. The schema
# itself comes from hbu_infra, because that is the repo that owns it — this
# only runs it, the way hbu_infra runs the dataplatform's bootstrap file.
# ---------------------------------------------------------------------------

db-up: ## Start the local postgis+pgvector container and apply the schema
	$(COMPOSE) up -d postgres
	@echo "Waiting for the server…"
	@until $(COMPOSE) exec -T postgres pg_isready -U urban_rag -d urban_rag >/dev/null 2>&1; \
	  do sleep 1; done
	@$(MAKE) db-init
	@echo
	@echo "Ready. Put this in .env:"
	@echo "  DATABASE_URL=$(LOCAL_URL)"

# Every table this app reads is created by hbu_infra — this only runs its SQL
# against the local container, the way hbu_infra runs the dataplatform's
# bootstrap file. 003_spatial_search.sql legitimately fails until rag.chunks
# exists, which is why errors here are reported rather than fatal.
db-init: ## Apply hbu_infra's sql/*.sql to the local container
	@test -d "$(HBU_INFRA)/sql" || { \
	  echo "hbu_infra not found at $(HBU_INFRA) — set HBU_INFRA=/path/to/hbu_infra"; \
	  exit 1; }
	@for f in $(HBU_INFRA)/sql/*.sql; do \
	  printf '→ %s' "$$f"; \
	  if $(COMPOSE) exec -T postgres psql -q -v ON_ERROR_STOP=1 -U urban_rag \
	       -d urban_rag < "$$f" >/dev/null 2>&1; then echo "  ok"; \
	  else echo "  skipped (needs a table that is not loaded yet)"; fi; \
	done

db-shell: ## psql into the local container
	$(COMPOSE) exec postgres psql -U urban_rag -d urban_rag

db-url: ## Print the local connection URL
	@echo "$(LOCAL_URL)"

db-logs: ## Tail the container's logs
	$(COMPOSE) logs -f postgres

db-down: ## Stop the container (keeps the volume)
	$(COMPOSE) down

db-reset: ## Stop the container AND delete its data
	$(COMPOSE) down -v

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build the runtime image
	docker build --target runtime -t $(IMAGE):$(IMAGE_TAG) \
	  --build-arg BUILD_VERSION=$(IMAGE_TAG) .
	docker tag $(IMAGE):$(IMAGE_TAG) $(IMAGE):latest

docker-run: docker-build ## Run the image against whatever .env points at
	docker run --rm -p 8501:8501 -p $(TILE_PORT):$(TILE_PORT) --env-file .env \
	  -v "$(AWS_DIR):/home/appuser/.aws:ro" \
	  -e AWS_PROFILE="$(AWS_PROFILE)" \
	  -e AWS_REGION="$(AWS_REGION)" \
	  -e AWS_DEFAULT_REGION="$(AWS_REGION)" \
	  -e HBU_TILE_PORT="$(TILE_PORT)" \
	  $(DOCKER_AWS_CA_ARGS) \
	  --add-host=host.docker.internal:host-gateway \
	  $(IMAGE):$(IMAGE_TAG)

docker-run-tunnel: docker-build ## Run the image through an open hbu_infra db-tunnel
	@test -n "$(TUNNEL_DB_HOST)" || { \
	  echo "TUNNEL_DB_HOST is empty; set it to the RDS endpoint, or put URBAN_RAG_PG_HOST in .env"; \
	  exit 1; \
	}
	@case "$(TUNNEL_DB_HOST)" in \
	  localhost|127.*|::1) \
	    echo "TUNNEL_DB_HOST must be the RDS endpoint, not $(TUNNEL_DB_HOST), when sslmode=verify-full"; \
	    exit 1 ;; \
	esac
	docker run --rm -p 8501:8501 -p $(TILE_PORT):$(TILE_PORT) --env-file .env \
	  -v "$(AWS_DIR):/home/appuser/.aws:ro" \
	  -e AWS_PROFILE="$(AWS_PROFILE)" \
	  -e AWS_REGION="$(AWS_REGION)" \
	  -e AWS_DEFAULT_REGION="$(AWS_REGION)" \
	  -e DATABASE_URL= \
	  -e URBAN_RAG_PG_DSN= \
	  -e URBAN_RAG_PG_HOST="$(TUNNEL_DB_HOST)" \
	  -e URBAN_RAG_PG_PORT="$(TUNNEL_PORT)" \
	  -e URBAN_RAG_PG_SSLMODE=verify-full \
	  -e URBAN_RAG_PG_SSLROOTCERT=/etc/ssl/certs/rds-global-bundle.pem \
	  -e URBAN_RAG_PG_IAM_AUTH= \
	  -e HBU_TILE_PORT="$(TILE_PORT)" \
	  $(DOCKER_AWS_CA_ARGS) \
	  --add-host="$(TUNNEL_DB_HOST):host-gateway" \
	  --add-host=host.docker.internal:host-gateway \
	  $(IMAGE):$(IMAGE_TAG)

docker-test: ## Run the unit suite inside the image
	docker build --target test -t $(IMAGE):test .
	docker run --rm $(IMAGE):test

compose-up: ## Build and run app + database together
	$(COMPOSE) --profile app up --build

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
