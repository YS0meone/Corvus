.PHONY: dev dev-cloud worker worker-cloud frontend infra

# ── Local development (localhost Qdrant, Redis, Grobid) ─────────────────────
dev:
	cd backend && uv run langgraph dev --allow-blocking

worker:
	cd backend && uv run python -m celery -A app.celery_app worker --loglevel=info --pool=solo

# ── Cloud configuration ───────────────────────────────────────────────────────
dev-cloud:
	cd backend && ENV_FILE=.env.cloud uv run langgraph dev

worker-cloud:
	cd backend && ENV_FILE=.env.cloud uv run python -m celery -A app.celery_app worker --loglevel=info --pool=solo

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend:
	cd web && pnpm dev

# ── Infrastructure (local Docker services) ───────────────────────────────────
infra:
	docker compose up -d
