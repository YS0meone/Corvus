from pydantic import BaseModel, Field
from typing import List, Optional, Dict, TypedDict
from datetime import datetime
from enum import Enum


class ArxivPaper(BaseModel):
    """Pydantic model for ArXiv paper data matching the exact schema."""
    
    # Required fields
    id: str = Field(..., description="ArXiv ID (e.g., '0704.0001')")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    
    # Optional fields
    submitter: Optional[str] = Field(None, description="Name of the submitter")
    authors: Optional[str] = Field(None, description="Authors as a string (e.g., 'C. Bal\\'azs, E. L. Berger')")
    comments: Optional[str] = Field(None, description="Additional comments")
    journal_ref: Optional[str] = Field(None, alias="journal-ref", description="Journal reference")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    report_no: Optional[str] = Field(None, alias="report-no", description="Report number")
    categories: Optional[str] = Field(None, description="Space-separated categories (e.g., 'hep-ph')")
    license: Optional[str] = Field(None, description="License information")
    supporting_detail: Optional[str] = Field(None, description="Supporting detail")
    
    # Additional fields for our system
    submission_date: Optional[datetime] = Field(None, description="Date of submission")
    update_date: Optional[datetime] = Field(None, description="Date of last update")
    
    class Config:
        populate_by_name = True  # Allow both field names and aliases
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def to_elasticsearch_doc(self) -> dict:
        """Convert to dictionary format for Elasticsearch indexing."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        
        # Ensure proper field names for Elasticsearch
        if 'journal_ref' in doc:
            doc['journal-ref'] = doc.pop('journal_ref')
        if 'report_no' in doc:
            doc['report-no'] = doc.pop('report_no')
            
        return doc

class ArxivPaperBatch(BaseModel):
    """Model for batch operations on multiple papers."""
    papers: List[ArxivPaper] = Field(..., description="List of ArXiv papers")
    
    def to_elasticsearch_docs(self) -> List[dict]:
        """Convert all papers to Elasticsearch document format."""
        return [paper.to_elasticsearch_doc() for paper in self.papers]

# ── Semantic Scholar ──────────────────────────────────────────────────────────

class S2Journal(BaseModel):
    name: Optional[str] = Field(None, description="Journal name")
    pages: Optional[str] = Field(None, description="Pages")
    volume: Optional[str] = Field(None, description="Volume")


class S2PublicationVenue(BaseModel):
    alternate_names: Optional[list] = Field(None, description="Alternate names")
    alternate_urls: Optional[list] = Field(None, description="Alternate URLs")
    id: Optional[str] = Field(None, description="ID")
    issn: Optional[str] = Field(None, description="ISSN")
    name: Optional[str] = Field(None, description="Name")
    type: Optional[str] = Field(None, description="Type")
    url: Optional[str] = Field(None, description="URL")


class S2Paper(BaseModel):
    """Pydantic model for Semantic Scholar paper data matching the official schema."""
    abstract: Optional[str] = Field(None, description="Paper abstract")
    authors: Optional[list] = Field(None, description="List of authors")
    citationCount: Optional[int] = Field(None, description="Number of citations")
    citationStyles: Optional[dict] = Field(None, description="Citation styles")
    corpusId: Optional[int] = Field(None, description="Corpus ID")
    externalIds: Optional[dict] = Field(None, description="External IDs such as DOI, ArXivId, etc.")
    fieldsOfStudy: Optional[list] = Field(None, description="Fields of study")
    influentialCitationCount: Optional[int] = Field(None, description="Influential citation count")
    isOpenAccess: Optional[bool] = Field(None, description="Whether paper is open access")
    journal: Optional[S2Journal] = Field(None, description="Journal information")
    openAccessPdf: Optional[Dict] = Field(None, description="Open access PDF information")
    paperId: str = Field(..., description="Semantic Scholar paper ID")
    publicationDate: Optional[datetime] = Field(None, description="Publication date (ISO 8601 string)")
    publicationTypes: Optional[list] = Field(None, description="Types of publication")
    publicationVenue: Optional[S2PublicationVenue] = Field(None, description="Publication venue information")
    referenceCount: Optional[int] = Field(None, description="Reference count")
    s2FieldsOfStudy: Optional[list] = Field(None, description="S2 fields of study")
    title: Optional[str] = Field(None, description="Paper title")
    url: Optional[str] = Field(None, description="Semantic Scholar paper URL")
    venue: Optional[str] = Field(None, description="Venue display name")
    year: Optional[int] = Field(None, description="Publication year")
    tldr: Optional[dict] = Field(None, description="Short summary")


# ── UI types ──────────────────────────────────────────────────────────────────

class Step(TypedDict):
    id: str
    label: str
    status: str
    description: str


class StepName(Enum):
    QUERY_EVALUATION = "Query evaluation"
    QUERY_OPTIMIZATION = "Query optimization"
    PLAN = "Plan"
    FIND_PAPERS = "Find papers"
    RETRIEVE_AND_ANSWER_QUESTION = "Answer question"
    REPLANNING = "Replanning"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

