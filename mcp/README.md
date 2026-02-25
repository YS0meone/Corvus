# Corvus MCP Server

Exposes Corvus's academic search capabilities as MCP tools, making them available in Claude Desktop, Cursor, and any other MCP-compatible client.

## Tools

| Tool | Description |
|---|---|
| `search_papers` | Keyword + filter search over Semantic Scholar's 200M+ paper database |
| `get_paper` | Fetch full metadata for a paper by Semantic Scholar ID |
| `forward_snowball` | Find papers that *cite* a given paper (recent follow-on work) |
| `backward_snowball` | Find papers *cited by* a given paper (foundational references) |

## Setup

### 1. Install dependencies

```bash
cd mcp
uv sync
```

### 2. Set your Semantic Scholar API key

The server reads from `backend/.env` automatically if you run it from the project root. Alternatively, set the environment variable directly:

```bash
export S2_API_KEY=your_key_here
```

A free S2 API key can be obtained at [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api). The server works without a key but at lower rate limits.

### 3. Connect to Claude Desktop

Add the following to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "corvus": {
      "command": "uv",
      "args": ["run", "python", "server.py"],
      "cwd": "/absolute/path/to/corvus/mcp",
      "env": {
        "S2_API_KEY": "your_key_here"
      }
    }
  }
}
```

Replace `/absolute/path/to/corvus/mcp` with the actual path on your machine.

### 4. Restart Claude Desktop

The Corvus tools will appear in Claude Desktop's tool list. You can now ask Claude things like:

- *"Find recent papers on test-time compute scaling"*
- *"What papers cite 'Attention Is All You Need'?"*
- *"What are the foundational references in this paper: `<paper_id>`?"*

## Running manually

```bash
cd mcp
uv run python server.py
```

Or with the MCP CLI inspector for debugging:

```bash
uv run mcp dev server.py
```
