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
    section_id: str | None = None
    section_level: int | None = None
    section_path: str | None = None
    section_types: list[str] = Field(default_factory=list)
    section_overlap: bool = False
    section_start: int | None = None
    section_end: int | None = None
    structure_version: str | None = None
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
    matched_queries: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    schema_version: int = 1
    query: str
    queries: list[str] = Field(default_factory=list)
    index_revision: str | None = None
    retrieval: list[dict]
    reranking: str
    warnings: list[str]
    retrieval_strategy: str = "passages"
    selected_articles: list[str] = Field(default_factory=list)
    score_note: str = "Scores rank relevance; they are not probabilities of scientific truth."
    hits: list[EvidenceHit]


class PassageContext(BaseModel):
    text: str
    char_start: int
    char_end: int
    passage_start: int
    passage_end: int
    truncated: bool


class PassageResponse(BaseModel):
    chunk_id: str
    provenance: Provenance
    text: str
    canonical_verified: bool
    warnings: list[str]
    context: PassageContext | None = None
