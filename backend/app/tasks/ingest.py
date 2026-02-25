import logging
from app.celery_app import celery_app
from app.core.schema import S2Paper
from app.core.config import settings
from app.services.qdrant import QdrantService
from qdrant_client.http.exceptions import ResponseHandlingException
import arxiv
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_qdrant_service: QdrantService | None = None


def _get_qdrant_service() -> QdrantService:
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService(settings.qdrant_config)
    return _qdrant_service


@celery_app.task(
    bind=True,
    name="ingest_paper",
    max_retries=3,
    autoretry_for=(ResponseHandlingException,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
def ingest_paper_task(self, paper_dict: dict) -> dict:
    """Celery task that ingests a single S2Paper into Qdrant.

    Attempts full-PDF ingestion first.  If no open-access PDF is available,
    falls back to abstract-only ingestion.

    Args:
        paper_dict: JSON-serializable dict representing an S2Paper.

    Returns:
        Status dict with paperId, method used, chunk_count, and success flag.
    """
    paper = S2Paper(**paper_dict)
    qdrant = _get_qdrant_service()
    logger.info(f"[ingest] paperId={paper.paperId!r} title={paper.title!r}")

    if qdrant.check_paper_exists(paper.paperId):
        logger.info(f"Paper {paper.paperId} already exists in Qdrant, skipping ingestion")
        return {
            "paperId": paper.paperId,
            "method": "skipped",
            "chunk_count": 0,
            "success": True,
        }
    
    # Try to search the paper in arxiv for downloading
    arxiv_paper = None
    try:
        client = arxiv.Client(
            delay_seconds=3.0,
            num_retries=3
        )
        search = arxiv.Search(
            query=f'ti:"{paper.title}"',
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        results = client.results(search)
        for result in results:
            arxiv_paper = result
            break
    except Exception as e:
        logger.warning(f"arXiv search failed for {paper.paperId}: {e}")
        return {
            "paperId": paper.paperId,
            "method": "failed",
            "chunk_count": 0,
            "success": False,
            "error": f"arXiv search failed — {e}",
        }

    if arxiv_paper is None:
        # Paper not found on arXiv — do not fall back to abstract-only.
        # The abstract is already passed directly to the QA agent via paper metadata;
        # storing it in Qdrant would pollute the vector store and block future re-ingestion.
        logger.info(f"Paper {paper.paperId} not found on arXiv")
        return {
            "paperId": paper.paperId,
            "method": "failed",
            "chunk_count": 0,
            "success": False,
            "error": "Not available on arXiv — upload the PDF manually to enable Q&A",
        }

    try:
        file_name = re.sub(r'[<>:"/\\|?*]', '_', arxiv_paper.title.replace(" ", "_"))
        pdf_path = Path(settings.PDF_DOWNLOAD_DIR) / (file_name + ".pdf")
        Path(settings.PDF_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
        arxiv_paper.download_pdf(dirpath=settings.PDF_DOWNLOAD_DIR, filename=file_name+".pdf")

        try:
            chunk_count = qdrant.add_s2_paper(file_name, paper.paperId)
            logger.info(f"Ingested paper {paper.paperId} via PDF ({chunk_count} chunks)")
            return {
                "paperId": paper.paperId,
                "method": "full_pdf",
                "chunk_count": chunk_count,
                "success": True,
            }
        finally:
            pdf_path.unlink(missing_ok=True)
            logger.debug(f"Deleted temp PDF {pdf_path}")
    except Exception as e:
        logger.warning(f"PDF ingestion failed for {paper.paperId}: {e}")
        return {
            "paperId": paper.paperId,
            "method": "failed",
            "chunk_count": 0,
            "success": False,
            "error": f"PDF processing failed — {e}",
        }
