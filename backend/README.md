# Corvus — Backend

A LangGraph-powered research assistant with two modes: **paper finding** (Semantic Scholar + Tavily + Cohere reranking) and **Q&A** (evidence retrieval from ingested papers via Qdrant).

## Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Docker & Docker Compose

## Quick Start

```bash
# 1. Copy and configure environment
cp backend/.env_example backend/.env
# Edit backend/.env — at minimum set OPENAI_API_KEY, COHERE_API_KEY, S2_API_KEY

# 2. Start all infrastructure
# Postgres (5432), Redis (6379), Qdrant (6333), Grobid (8070)
docker compose up -d

# 3. Start the Celery worker (in a separate terminal)
cd backend && uv run python -m celery -A app.celery_app worker --loglevel=info --pool=solo

# 4. Start the LangGraph dev server (in a separate terminal)
cd backend && uv run langgraph dev

# 5. Start the frontend (in a separate terminal, from project root)
cd web && pnpm dev
```

The backend API will be available at `http://localhost:2024`.

## Environment Variables

### LLM API

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for all LLM calls |

### Model Names

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/allenai-specter` | Sentence-transformers embedding model |
| `SUPERVISOR_MODEL_NAME` | `gpt-4o-mini` | Main supervisor / router |
| `PF_AGENT_MODEL_NAME` | `gpt-4o-mini` | Paper finder planner & executor |
| `PF_FILTER_MODEL_NAME` | `gpt-4o-mini` | Paper finder replan agent |
| `QA_AGENT_MODEL_NAME` | `gpt-4o-mini` | Q&A answer generation |
| `QA_EVALUATION_MODEL_NAME` | `gpt-4o-mini` | Q&A evidence evaluation |
| `QA_EVALUATOR_MODEL_NAME` | `gpt-4o-mini` | Q&A evaluator |
| `QA_BASELINE_MODEL_NAME` | `gpt-4o-mini` | Q&A baseline |

### LangSmith Tracing (optional)

| Variable | Default | Description |
|---|---|---|
| `LANGSMITH_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | `ai-researcher-proto` | LangSmith project name |

### External APIs

| Variable | Required | Description |
|---|---|---|
| `COHERE_API_KEY` | Yes | Cohere API key for reranking |
| `S2_API_KEY` | Yes | Semantic Scholar API key for paper search |

### Qdrant Vector Database

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_API_KEY` | _(empty)_ | Leave blank for local |
| `QDRANT_VECTOR_SIZE` | `768` | Embedding dimensions |
| `QDRANT_COLLECTION` | `papers` | Collection name |
| `QDRANT_DISTANCE` | `COSINE` | Distance metric |

### Redis / Celery

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | Celery result backend |

### Paper Processing

| Variable | Default | Description |
|---|---|---|
| `PDF_DOWNLOAD_DIR` | `./papers` | Local directory for downloaded PDFs |
| `GROBID_SERVER_URL` | `http://localhost:8070` | Grobid REST endpoint |

### Auth (optional)

| Variable | Default | Description |
|---|---|---|
| `CLERK_JWKS_URL` | _(empty)_ | Leave blank to disable auth |
| `DISABLE_AUTH` | `false` | Set `true` to bypass auth entirely |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |

## Architecture

### Supervisor Graph (`graph.py`)

Entry point that orchestrates the full workflow:

```
query_evaluation → planner → executor → supervisor_tools → post_tool
                                                                ↓
                                                    [replanner interrupt]
```

The supervisor decides whether to invoke the paper finder subgraph, the Q&A subgraph, or respond directly. After paper finding, an interrupt allows the user to select papers before Q&A proceeds.

### Paper Finder Subgraph (`paper_finder.py`)

Iterative search loop (max 3 iterations):

```
planner → executor → replan_agent
   ↑_____________________________|  (if goal not achieved)
```

- **planner**: Generates a list of search steps from the user query.
- **executor**: Runs one search step using available tools, then reranks results with Cohere.
- **replan_agent**: Evaluates whether the goal is achieved; if not, produces new plan steps.

**Tools available to the paper finder:**

| Tool | Description |
|---|---|
| `s2_search_papers` | Semantic Scholar keyword and metadata search |
| `tavily_research_overview` | Tavily web search for broader context |
| `forward_snowball` | Find papers that cite a given paper |
| `backward_snowball` | Find papers cited by a given paper |

### Q&A Subgraph (`qa.py`)

Evidence-based retrieval loop (max 2 refinement iterations):

```
qa_retrieve → qa_evaluate → qa_answer
                   ↓
              (refine queries and retry if evidence insufficient)
```

- **qa_retrieve**: Generates focused queries and retrieves evidence from Qdrant, scoped to the user-selected paper IDs.
- **qa_evaluate**: Determines whether the retrieved evidence is sufficient to answer.
- **qa_answer**: Produces a grounded answer with segment-level citations.

**Tools available to Q&A:**

| Tool | Description |
|---|---|
| `retrieve_evidence_from_selected_papers` | Vector search within selected paper IDs only |

### Additional Tools

| Tool | Description |
|---|---|
| `get_paper_details` | Fetch full metadata for a paper by Semantic Scholar ID |

### Paper Ingestion (Celery)

When a user adds a paper, a Celery task:
1. Downloads the PDF from the S2 URL.
2. Parses it with Grobid to extract structured text.
3. Chunks the text and embeds each chunk using `allenai-specter`.
4. Stores the vectors in Qdrant under the `papers` collection.

## Project Structure

```
backend/
├── app/
│   ├── agent/
│   │   ├── graph.py             # Supervisor graph — main entry point
│   │   ├── paper_finder.py      # Paper finder subgraph
│   │   ├── paper_finder_fast.py # Simplified paper finder variant
│   │   ├── qa.py                # Q&A subgraph
│   │   ├── qa_baseline.py       # Q&A baseline for evaluation
│   │   ├── states.py            # Shared TypedDict state definitions
│   │   ├── prompts.py           # System prompts for all agents
│   │   ├── ui_manager.py        # push_ui_message helper for streaming UI
│   │   ├── RedisDocumentStore.py # Redis-backed document store
│   │   └── utils.py             # Shared utilities
│   ├── auth.py                  # Clerk JWT verification middleware
│   ├── celery_app.py            # Celery app configuration
│   ├── core/
│   │   ├── config.py            # Pydantic Settings — all env vars
│   │   ├── logging_config.py    # Logging setup
│   │   └── schema.py            # Shared Pydantic models
│   ├── services/
│   │   ├── qdrant.py            # Qdrant client wrapper
│   │   └── s2_client.py         # Semantic Scholar API client
│   ├── tasks/
│   │   └── ingest.py            # Celery task: download → Grobid → embed → store
│   ├── tools/
│   │   └── search.py            # LangGraph tool definitions
│   └── webapp/
│       └── app.py               # FastAPI app (paper ingestion endpoints)
├── .env                         # Environment variables (create from .env_example)
├── .env_example                 # Template with all variables
├── langgraph.json               # LangGraph server configuration
└── pyproject.toml               # Python dependencies (managed by uv)
```
