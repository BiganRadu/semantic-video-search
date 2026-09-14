SHELL := /bin/bash

DB_CONTAINER ?= vs-postgres
DB_PORT      ?= 5433
DB_IMAGE     ?= pgvector/pgvector:pg16
DB_VOLUME    ?= vs-pgdata
# Local secrets: GEMINI_API_KEY for query planning, AIVEN_DATABASE_URL for the
# hosted database. Gitignored. The key is optional -- without it, search runs on
# the measured default weights and nothing breaks.
-include .env
export GEMINI_API_KEY

# Which database every target talks to. `make run DB=aiven` uses the hosted one,
# anything else the local container. Search against Aiven measured ~2x the local
# latency (median 566ms vs 285ms, network round-trips), so local stays the
# default for development and Aiven is what the deployment points at.
DB ?= local
ifeq ($(DB),aiven)
  DATABASE_URL ?= $(AIVEN_DATABASE_URL)
else
  DATABASE_URL ?= postgres://video:video@localhost:$(DB_PORT)/video?sslmode=disable
endif
export DATABASE_URL

# The project lives on an NTFS/FUSE mount, which is slow and unreliable for a
# venv's ~30k small files. Keep the environment on the native filesystem.
VENV ?= $(HOME)/.venvs/video-search
PY   := $(VENV)/bin/python

.DEFAULT_GOAL := help
.PHONY: help venv db-up db-down db-nuke psql migrate migrate-down status
.PHONY: run web web-dev build corpus

venv: ## create the python environment
	@uv venv --python 3.12 $(VENV)
	@uv pip install --python $(PY) --torch-backend=auto -r requirements.txt
	@echo "environment ready at $(VENV)"

help: ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

## ---- database -------------------------------------------------------------

db-up: ## start postgres + pgvector (idempotent)
	@docker start $(DB_CONTAINER) >/dev/null 2>&1 || \
	 docker run -d --name $(DB_CONTAINER) \
	   -e POSTGRES_USER=video -e POSTGRES_PASSWORD=video -e POSTGRES_DB=video \
	   -p 127.0.0.1:$(DB_PORT):5432 -v $(DB_VOLUME):/var/lib/postgresql/data \
	   $(DB_IMAGE) >/dev/null
	@printf 'waiting for postgres'; \
	for i in $$(seq 40); do \
	  if docker exec $(DB_CONTAINER) pg_isready -U video -q 2>/dev/null; then echo ' ready'; exit 0; fi; \
	  printf '.'; sleep 0.5; \
	done; echo ' TIMEOUT'; exit 1

db-down: ## stop postgres (keeps the data)
	@docker stop $(DB_CONTAINER) >/dev/null 2>&1 && echo stopped || echo "not running"

db-nuke: ## stop postgres and DELETE the index
	@docker rm -f $(DB_CONTAINER) >/dev/null 2>&1 || true
	@docker volume rm $(DB_VOLUME) >/dev/null 2>&1 || true
	@echo "database removed"

psql: ## open a psql shell
	@docker exec -it $(DB_CONTAINER) psql -U video -d video

migrate: ## apply schema migrations
	@go run ./server migrate up

migrate-down: ## roll back the last migration
	@go run ./server migrate down

status: ## migration status
	@go run ./server migrate status

## ---- the app --------------------------------------------------------------

run: ## start the API + built frontend on one port
	@go run ./server

web: ## build the frontend into web/dist
	@cd web && npm install --silent && npm run build

web-dev: ## frontend dev server with hot reload (proxies /api to :8080)
	@cd web && npm run dev

build: web ## build the frontend and the binaries
	@mkdir -p bin && go build -o bin/server ./server && ls bin/

corpus: ## index one demo video (NAME=travel_japan); no NAME lists them
	@$(PY) scripts/corpus.py $(NAME)
