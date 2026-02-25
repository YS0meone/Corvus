"""
Corvus MCP Server

Exposes Semantic Scholar academic search tools via the Model Context Protocol.
Connect this to Claude Desktop, Cursor, or any MCP-compatible client to give
your AI assistant access to 200M+ academic papers and citation graph traversal.

Tools:
  search_papers      — keyword + filter search over Semantic Scholar
  get_paper          — fetch full metadata for a paper by S2 ID
  forward_snowball   — find papers that cite a given paper (recent follow-on work)
  backward_snowball  — find papers cited by a given paper (foundational references)
"""

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load from backend/.env if running from project root, else fall back to env vars
_env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(_env_path)

S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_BASE = "https://api.semanticscholar.org/graph/v1"
_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

_PAPER_FIELDS = (
    "paperId,title,abstract,year,authors,citationCount,"
    "influentialCitationCount,venue,isOpenAccess,openAccessPdf,url"
)

mcp = FastMCP(
    "Corvus Academic Search",
    instructions=(
        "Tools for searching academic papers on Semantic Scholar (200M+ papers). "
        "Use search_papers to find papers by keyword, get_paper to fetch details "
        "by ID, forward_snowball to find citing papers, and backward_snowball to "
        "find referenced papers."
    ),
)


def _format_papers(papers: list[dict]) -> str:
    """Return a compact JSON string, stripping null fields."""
    cleaned = [{k: v for k, v in p.items() if v is not None} for p in papers]
    return json.dumps(cleaned, default=str, indent=2)


@mcp.tool()
async def search_papers(
    query: str,
    year: str | None = None,
    min_citation_count: int | None = None,
    fields_of_study: str | None = None,
    venue: str | None = None,
    open_access_only: bool = False,
    limit: int = 10,
) -> str:
    """Search Semantic Scholar's 200M+ paper database by keyword.

    Args:
        query: Search query string (keywords, title, author name, etc.)
        year: Year filter. Single year "2023" or range "2020-2024" or open range "2022-".
        min_citation_count: Only return papers with at least this many citations.
        fields_of_study: Comma-separated fields, e.g. "Computer Science,Mathematics".
        venue: Comma-separated venue names, e.g. "NeurIPS,ICML".
        open_access_only: If true, only return papers with a free PDF.
        limit: Number of results to return (max 50).

    Returns:
        JSON array of matching papers with title, abstract, authors, year, citation count.
    """
    params: dict = {
        "query": query,
        "fields": _PAPER_FIELDS,
        "limit": min(limit, 50),
    }
    if year:
        params["year"] = year
    if min_citation_count is not None:
        params["minCitationCount"] = min_citation_count
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study
    if venue:
        params["venue"] = venue
    if open_access_only:
        params["openAccessPdf"] = ""

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{S2_BASE}/paper/search", params=params, headers=_HEADERS
        )
        resp.raise_for_status()

    papers = resp.json().get("data", [])
    return _format_papers(papers)


@mcp.tool()
async def get_paper(paper_id: str) -> str:
    """Fetch full metadata for a single paper by its Semantic Scholar paper ID.

    Args:
        paper_id: The Semantic Scholar paper ID (e.g. "649def34f8be52c8b66281af98ae884c09aef38a").

    Returns:
        JSON object with full paper metadata including abstract, authors, venue, citation count.
    """
    fields = _PAPER_FIELDS + ",tldr,publicationDate,publicationTypes,externalIds"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{S2_BASE}/paper/{paper_id}",
            params={"fields": fields},
            headers=_HEADERS,
        )
        resp.raise_for_status()

    data = resp.json()
    return json.dumps({k: v for k, v in data.items() if v is not None}, default=str, indent=2)


@mcp.tool()
async def forward_snowball(paper_id: str, limit: int = 50) -> str:
    """Find papers that cite a given paper (surfaces recent follow-on work).

    Useful for discovering how a foundational paper has been extended, critiqued,
    or applied in subsequent research.

    Args:
        paper_id: The Semantic Scholar paper ID of the seed paper.
        limit: Maximum number of citing papers to return (max 500).

    Returns:
        JSON array of papers that cite the given paper, sorted by Semantic Scholar's default order.
    """
    fields = "paperId,title,abstract,year,authors,citationCount,influentialCitationCount"
    params = {
        "fields": ",".join(f"citingPaper.{f}" for f in fields.split(",")),
        "limit": min(limit, 500),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{S2_BASE}/paper/{paper_id}/citations", params=params, headers=_HEADERS
        )
        resp.raise_for_status()

    papers = [
        item["citingPaper"]
        for item in resp.json().get("data", [])
        if item.get("citingPaper") and item["citingPaper"].get("paperId")
    ]
    return _format_papers(papers)


@mcp.tool()
async def backward_snowball(paper_id: str, limit: int = 50) -> str:
    """Find papers cited by a given paper (surfaces foundational references).

    Useful for tracing the intellectual lineage of a paper — what prior work it
    builds on, which methods it adapts, and which datasets it uses.

    Args:
        paper_id: The Semantic Scholar paper ID of the seed paper.
        limit: Maximum number of referenced papers to return (max 500).

    Returns:
        JSON array of papers referenced by the given paper.
    """
    fields = "paperId,title,abstract,year,authors,citationCount,influentialCitationCount"
    params = {
        "fields": ",".join(f"citedPaper.{f}" for f in fields.split(",")),
        "limit": min(limit, 500),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{S2_BASE}/paper/{paper_id}/references", params=params, headers=_HEADERS
        )
        resp.raise_for_status()

    papers = [
        item["citedPaper"]
        for item in resp.json().get("data", [])
        if item.get("citedPaper") and item["citedPaper"].get("paperId")
    ]
    return _format_papers(papers)


if __name__ == "__main__":
    mcp.run()
