# AI Researcher Proto

An AI research assistant that helps you **find academic papers** through multi-agent search and **answer deep questions** about them with evidence-based retrieval.

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| API server | FastAPI |
| Async task queue | Celery + Redis |
| Frontend | React 19 + Vite |
| Vector database | Qdrant |
| Relational / checkpoints | Postgres |
| PDF extraction | Grobid |
| Paper search | Semantic Scholar API |
| Web search | Tavily |
| Reranking | Cohere |
| Auth | Clerk (optional) |

## Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Node.js 20+ with [pnpm](https://pnpm.io/)
- Docker & Docker Compose

## Quick Start

### 1. Clone

```bash
git clone <your-repo-url>
cd ai-researcher-proto
```

### 2. Configure environment

```bash
cp backend/.env_example backend/.env
```

Edit `backend/.env` and set at minimum:
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `S2_API_KEY`

The root `.env.example` is informational only — `docker-compose.yml` uses hardcoded defaults.

### 3. Start infrastructure

```bash
docker compose up -d
```

This starts Postgres (5432), Redis (6379), Qdrant (6333), and Grobid (8070).

### 4. Start the Celery worker

```bash
cd backend && uv run python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

> `--pool=solo` is required on Windows.

### 5. Start the backend

```bash
cd backend && uv run langgraph dev
```

The LangGraph API is available at `http://localhost:2024`.

### 6. Start the frontend

```bash
cd web && pnpm install && pnpm dev
```

The UI is available at `http://localhost:5173`.

## Agent Architecture

The system uses a supervisor graph that routes between two subgraphs:

```
User message
     ↓
Supervisor (graph.py)
     ├── find_papers tool → Paper Finder subgraph (paper_finder.py)
     │       planner → executor → replan_agent  (up to 3 iterations)
     │       Tools: s2_search_papers, tavily_research_overview,
     │              forward_snowball, backward_snowball
     │       Cohere reranking after each executor step
     │
     └── [interrupt] → user selects papers
             ↓
         Q&A subgraph (qa.py)
             qa_retrieve → qa_evaluate → qa_answer  (up to 2 refinements)
             Tool: retrieve_evidence_from_selected_papers (scoped to selection)
```

Papers are ingested asynchronously: the Celery worker downloads the PDF, parses it with Grobid, embeds chunks with `allenai-specter`, and stores them in Qdrant.

## UI Features

- **Real-time Research Agent card** — live status updates as the paper finder runs (planning, searching, evaluating)
- **Paper selection interrupt** — after search completes, select which papers to use for Q&A
- **Ingestion progress** — track PDF download and processing status per paper
- **Streaming AI responses** — token-by-token streaming from the LangGraph backend
- **Thread history** — persistent conversation threads via the sidebar

## Detailed Documentation

- Backend: [`backend/README.md`](./backend/README.md) — environment variables, architecture, project structure
- Frontend: [`web/README.md`](./web/README.md) — setup, env vars, UI features
