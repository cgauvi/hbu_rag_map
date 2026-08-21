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

PYTHON      ?= python3
VENV        ?= .venv
BIN          = $(VENV)/bin
UV          := $(shell command -v uv 2>/dev/null)

# Sibling repos. Same convention as hbu_infra's DATAPLATFORM variable: the
# schema lives over there, and this repo runs it from here when it is present.
HBU_INFRA   ?= ../hbu_infra
ENV         ?= dev

LOCAL_PG_PORT ?= 5432
LOCAL_URL     ?= postgresql://urban_rag:urban_rag@localhost:$(LOCAL_PG_PORT)/urban_rag?sslmode=disable

IMAGE       ?= hbu-rag-map
IMAGE_TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo local-dev)

COMPOSE     ?= docker compose

.PHONY: help install run check test lint fmt \
        db-up db-down db-init db-shell db-url db-logs \
        docker-build docker-run docker-test clean

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

run: ## Start Streamlit at http://localhost:8501
	$(BIN)/streamlit run app.py

check: ## Report what is and is not loaded, and how to fix each gap
	$(BIN)/python scripts/doctor.py

test: ## Unit tests (the integration marker is deselected by default)
	$(BIN)/python -m pytest tests/ -v

test-integration: ## Also run the tests that need a reachable database
	$(BIN)/python -m pytest tests/ -v -m integration

lint: ## ruff
	$(BIN)/ruff check src/ app.py scripts/ tests/

fmt: ## ruff --fix
	$(BIN)/ruff check --fix src/ app.py scripts/ tests/

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
	docker run --rm -p 8501:8501 --env-file .env \
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
