"""Public MCP evidence contracts. Unknown bibliographic fields stay null."""
from pydantic import BaseModel, Field


class Bibliography(BaseModel):
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    version: str | None = None


class Location(BaseModel):
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


class Provenance(BaseModel):
    document_id: str | None = None
    source: str | None = None
    content_sha256: str | None = None
    bibliography: Bibliography
    location: Location
    locator_status: str
    source_pdf: str | None = None
    completeness: str = "not_assessed"


class Scores(BaseModel):
    rerank: float | None
    fusion: float


class EvidenceHit(BaseModel):
    chunk_id: str
    provenance: Provenance
    excerpt: str
    excerpt_truncated: bool
    scores: Scores


class SearchResponse(BaseModel):
    schema_version: int = 1
    query: str
    queries: list[str] = Field(default_factory=list)
    index_revision: str | None = None
    retrieval: list[dict]
    reranking: str
    warnings: list[str]
    score_note: str = "Scores rank relevance; they are not probabilities of scientific truth."
    hits: list[EvidenceHit]


class PassageResponse(BaseModel):
    chunk_id: str
    provenance: Provenance
    text: str
    canonical_verified: bool
    warnings: list[str]
