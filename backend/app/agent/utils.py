from app.core.config import settings
import os
import time
from typing import List, Dict
from app.core.schema import S2Paper
from langchain_core.documents import Document


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return result
    return wrapper

def get_user_query(messages: list) -> str:
    """Extract the original user query from the message history."""
    user_msg = ""
    for m in messages:
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
        elif isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            user_msg = m["content"]
    return user_msg 
    
def get_paper_info_text(papers: list[S2Paper], include_abstract: bool = True) -> str:
    """Get the text of the papers."""
    if not papers:
        return "No papers found yet"
    lines = []
    for paper in papers:
        author_list = paper.authors or []
        names = [a.get("name", "") for a in author_list[:3] if isinstance(a, dict) and a.get("name")]
        authors_str = ", ".join(names) + (" et al." if len(author_list) > 3 else "")
        parts = [
            f"- {paper.title or 'Untitled'}",
            f"  Authors: {authors_str or 'Unknown'}",
            f"  Published: {paper.publicationDate or 'N/A'}",
        ]
        if include_abstract and paper.abstract:
            parts.append(f"  Abstract: {paper.abstract}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)

def get_paper_abstract(papers: List[S2Paper], selected_paper_ids: List[str]) -> Dict[str, str]:
    abstracts = {}
    for paper in papers:
        if paper.paperId in selected_paper_ids:
            abstracts[paper.paperId] = paper.abstract
    return abstracts


def remove_duplicated_evidence(existing_evds: List[Document], new_evds: List[Document]) -> List[Document]:
    exists_evds_ids = set()
    non_duplicated_evds = []
    for evd in existing_evds:
        exists_evds_ids.add(str(evd.metadata.get("id", "")) + "_" + str(evd.metadata.get("para", "")))
    for evd in new_evds:
        if str(evd.metadata.get("id", "")) + "_" + str(evd.metadata.get("para", "")) not in exists_evds_ids:
            non_duplicated_evds.append(evd)
    return non_duplicated_evds


def merge_evidences(existing: List[Document], new: List[Document]) -> List[Document]:
    """LangGraph reducer for evidences: merges and deduplicates by (id, para)."""
    return existing + remove_duplicated_evidence(existing, new)
