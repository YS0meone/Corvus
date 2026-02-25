# Corvus

An AI-powered research assistant that combines **multi-agent paper discovery** with **evidence-based question answering** over academic literature. Ask Corvus to find papers on any research topic, then drill into them with precise, citation-grounded answers.

**[Live demo → corvus-agent.vercel.app](https://corvus-agent.vercel.app)** | **[MCP server](./mcp/)** — use Corvus's search tools directly in Claude Desktop

## Stack


| Layer                    | Technology                                                      |
| ------------------------ | --------------------------------------------------------------- |
| Agent orchestration      | LangGraph (supervisor + subgraph pattern)                       |
| LLM inference            | Google Gemini 3 Flash (preview) via LangChain                   |
| API server               | FastAPI                                                         |
| Async task queue         | Celery + Redis                                                  |
| Frontend                 | React 19 + Vite + TypeScript                                    |
| Vector database          | Qdrant                                                          |
| Relational / checkpoints | Postgres                                                        |
| PDF extraction           | Grobid                                                          |
| Paper search             | Semantic Scholar API (200M+ papers)                             |
| Web search               | Tavily                                                          |
| Reranking                | Cohere                                                          |
| Embeddings               | OpenAI `text-embedding-3-small` (1536-dim)                      |
| Auth                     | Clerk (optional)                                                |


## Agent Architecture

Corvus is built around a **three-tier LangGraph agent hierarchy**: a supervisor that plans and routes, a paper finder subgraph that searches, and a Q&A subgraph that retrieves and answers.

### Supervisor (`graph.py`)

Every user message enters the supervisor, which runs a four-node pipeline before invoking any tool:

```
query_evaluation → planner → executor → supervisor_tools → post_tool
                                                                ↓
                                                    replanner (interrupt)
                                                                ↓
                                                           executor …
```

`**query_evaluation**` — Classifies the query before any search is attempted. Uses structured output to return one of five decisions: `clear`, `needs_clarification`, `unselected_paper`, `irrelevant`, or `inappropriate`. Only `clear` queries proceed; all others generate an inline response explaining what Corvus needs.

`**planner**` — Chooses one of three execution plans based on intent:

- `find_only` — discover new papers
- `qa_only` — answer a question using papers already in context
- `find_then_qa` — find papers first, then answer a question about them

`**executor**` — Pops the next step from the plan and issues the corresponding tool call (`find_papers` or `retrieve_and_answer_question`) as a structured AIMessage with a tool call.

`**post_tool**` — Runs after every tool result to commit side effects (paper list UI, final answer) to state before any interrupt is issued, ensuring the checkpoint is consistent.

`**replanner**` — Issues a LangGraph `interrupt("select_papers")` so the user can choose which papers to carry into Q&A. On resume, the selected paper IDs and an optional follow-up message are extracted from the resume payload and committed to state.

---

### Paper Finder Subgraph (`paper_finder.py`)

Invoked by the supervisor's `find_papers` tool. Runs an iterative search loop (max 3 iterations) to retrieve papers from Semantic Scholar:

```
planner ──→ executor ──→ replan_agent
   ↑                          │
   └──── (goal not met) ──────┘
                              │
                        (goal met) ──→ END
```

`**planner**` — Analyses the search task and generates a concrete multi-step search plan. Each step must be an actionable search operation (web search, academic database query, or citation chasing). The planner is given today's date so temporal constraints like "recent" resolve correctly.

`**executor**` — Runs a nested ReAct-style search agent for the current plan step. The search agent has access to four tools:


| Tool                       | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `s2_search_papers`         | Keyword and filter search over Semantic Scholar (year, venue, citation count, etc.) |
| `tavily_research_overview` | Web search for broader context, seminal papers, and author profiles                 |
| `forward_snowball`         | Find papers that *cite* a given seed paper (surfaces recent follow-on work)         |
| `backward_snowball`        | Find papers *cited by* a given seed paper (surfaces foundational references)        |


After the search agent finishes its step, all accumulated papers are **reranked with Cohere** against a keyword-optimised query derived from the original task. Only the top-35 most relevant papers are kept, preventing the list from growing unboundedly across iterations.

`**replan_agent*`* — Evaluates whether the goal is achieved by inspecting the *actual paper list* (not just what appeared in web search summaries). If the goal is not yet met, it rewrites the remaining plan steps with explicit retrieval targets, then loops back to the executor.

All three nodes emit `finder_status` custom events that the frontend renders as a real-time **Research Agent** status card.

---

### Q&A Subgraph (`qa.py`)

Invoked after paper selection. Performs iterative evidence retrieval scoped to the user-selected paper IDs (max 3 iterations):

```
qa_retrieve ──→ tools ──→ qa_evaluate ──→ qa_answer
                               │
                   (insufficient evidence)
                               │
                          qa_retrieve …
```

`**qa_retrieve**` — Generates focused retrieval queries from the user question and calls `retrieve_evidence_from_selected_papers`, which performs vector search over the Qdrant collection filtered to the selected paper IDs only. This scoping ensures answers are grounded exclusively in the papers the user chose.

`**qa_evaluate**` — Uses structured output to decide between two outcomes: `AnswerQuestion` (evidence is sufficient) or `AskForMoreEvidence` (evidence is weak — includes a `limitation` string that seeds the next retrieval attempt with a better query strategy).

`**qa_answer**` — Synthesises a final answer from all accumulated evidence chunks, cites sources at the segment level, and explicitly acknowledges any gaps when the maximum iteration limit is reached.

All three nodes emit `qa_status` events rendered as a real-time **Q&A Agent** status card.

---

### Paper Ingestion Pipeline (Celery)

Papers are ingested asynchronously via a Celery task so the UI never blocks:

1. **arXiv lookup** — searches arXiv by title to find a PDF URL
2. **PDF download** — fetches the PDF to a temp file (`PDF_DOWNLOAD_DIR`)
3. **Grobid parsing** — sends the PDF to a local Grobid container for structured full-text extraction
4. **Chunking & embedding** — splits the extracted text into overlapping chunks and embeds each with OpenAI `text-embedding-3-small`
5. **Qdrant upsert** — stores vectors with paper ID and section metadata for scoped retrieval
6. **Cleanup** — the temp PDF is deleted regardless of success or failure

If no arXiv PDF is available, the pipeline falls back to abstract-only ingestion.

---

## UI Features

- **Real-time subagent status cards** — live in-place updates as the Research Agent and Q&A Agent run (planning, searching, evaluating, generating)
- **Paper selection interrupt** — after search completes, a structured interrupt pauses execution so the user can choose which papers to bring into Q&A
- **Ingestion progress panel** — per-paper status tracking with a progress bar as the Celery worker processes each PDF in the background
- **Streaming AI responses** — token-by-token streaming via the LangGraph SDK
- **Persistent thread history** — conversation threads are stored in Postgres and restored across sessions

## Quick Start

### 1. Clone

```bash
git clone https://github.com/YS0meone/Corvus.git
cd Corvus
```

### 2. Configure environment

```bash
cp backend/.env_example backend/.env.local
```

Edit `backend/.env.local` and fill in at minimum:

- `OPENAI_API_KEY` — for embeddings
- `GEMINI_API_KEY` — for all agent LLM calls
- `COHERE_API_KEY` — for reranking
- `S2_API_KEY` — for Semantic Scholar search

Local infrastructure variables (Qdrant, Redis, Grobid) are pre-filled with `localhost` defaults.

### 3. Start infrastructure

```bash
make infra
```

Starts Redis (6379), Qdrant (6333), and Grobid (8070) via Docker Compose.

### 4. Start the Celery worker, backend, and frontend

Open three terminals and run:

```bash
make worker    # Celery worker
make dev       # LangGraph backend  (http://localhost:2024)
make frontend  # React frontend     (http://localhost:5173)
```

Open `http://localhost:5173`.

### Switching to cloud services

To run against your deployed Qdrant, Redis, and Grobid instead:

```bash
cp backend/.env_example backend/.env.cloud
# fill in cloud service URLs and API keys

make dev-cloud
make worker-cloud
```

## Documentation

- `[backend/README.md](./backend/README.md)` — environment variables, project structure, full architecture reference
- `[web/README.md](./web/README.md)` — frontend setup and env vars

