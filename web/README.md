# Corvus — Frontend

A React 19 chat interface for Corvus. Built with Vite, TypeScript, and Tailwind CSS, with real-time streaming from a LangGraph backend.

## Prerequisites

- Node.js 20+
- pnpm

## Quick Start

```bash
cd web
pnpm install
pnpm dev
```

The app will be available at `http://localhost:5173`.

## Environment Variables

Copy `web/.env.example` to `web/.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:2024` | LangGraph backend URL |
| `VITE_ASSISTANT_ID` | `agent` | LangGraph graph / assistant ID |
| `VITE_LANGSMITH_API_KEY` | _(empty)_ | LangSmith API key (leave blank for local dev) |
| `VITE_CLERK_PUBLISHABLE_KEY` | _(empty)_ | Clerk publishable key (leave blank to disable auth) |

## Connecting to the Backend

Make sure the backend is running (`uv run langgraph dev` in `backend/`), then open `http://localhost:5173`. The app reads `VITE_API_URL` and `VITE_ASSISTANT_ID` from the environment — no manual connection dialog needed.

## Available Scripts

- `pnpm dev` — Start development server with HMR
- `pnpm build` — Build for production
- `pnpm preview` — Preview production build
- `pnpm lint` — Run ESLint
- `pnpm format` — Format with Prettier

## Features

- **Real-time Research Agent status card** — live updates as the paper finder subgraph runs (planning, searching, evaluating results)
- **Paper selection interrupt** — after the agent finds papers, select which ones to use before Q&A proceeds
- **Ingestion progress tracking** — per-paper status as PDFs are downloaded and processed by the Celery worker
- **Streaming AI responses** — token-by-token streaming via the LangGraph SDK
- **Thread history sidebar** — persistent conversation threads with a loading skeleton while history fetches
- **Rich markdown rendering** — syntax-highlighted code blocks, tables, and inline formatting
- **Paper citations** — paper cards with metadata displayed inline in the conversation
