# Scientific library reliability

This change adds canonical source snapshots, explicit citation provenance, safer
index replacement, structured evidence tools and an annotation-gated benchmark.
It does not annotate the existing library or demonstrate retrieval gains on it.

## Storage and migration

Keep a backup of the existing Chroma database and Markdown sources before a first
production migration. Install the updated requirements in an isolated environment.
The implementation is tested with FastMCP 2.14.7, Chroma 1.5.9 and Chonkie 1.7.0.
The semantic chunker's model still needs its existing optional dependencies and
model download; tests do not download models or call paid APIs.

`RAGDOC_LIBRARY_DIR` selects the canonical store. By default it is
`<Chroma parent>/ragdoc_library/<collection name>`. Both the indexer and MCP reader
must see this same directory. Sharing only a remote Chroma connection is not
enough for canonical reads. Back up this directory along with Chroma and sources.
Snapshots are immutable UTF-8 Markdown addressed by SHA-256; the SQLite journal
records ingestion attempts. A successful journal event does not replace the index
integrity checks. The canonical snapshots include complete article text: treat
them with the same access and copyright controls as the original library.

Run the normal incremental indexer to migrate legacy articles. An unchanged
legacy article is reindexed once because its pipeline/provenance metadata differ.
It still incurs embedding costs. Existing tools remain available; a legacy full
document read explicitly warns that overlapping chunks are reconstructed.
Canonical reads require a migrated article and a matching snapshot.

## Ingestion behavior

The indexer validates UTF-8, rejects conversions marked partial, stores the new
source snapshot, and prepares all chunks and embeddings before any index mutation.
It then backs up the old indexed article, writes the new chunks, reads them back,
and finally removes obsolete chunks. New chunk IDs contain content hashes so an
old citation cannot silently resolve to different text after reindexing.

An ordinary storage error triggers restoration of the old article. A failed
rollback stops the run and leaves an explicit blocked write state. Any failed
article increments the error counter and makes the CLI exit nonzero.

This is **not a Chroma transaction** and does not guarantee rollback after power
loss or process termination. MCP reads check the shared write state and revision
before and after reading. On an interrupted write, inspect the ingestion journal
and storage, then repair using `--force`. During repair, reads remain blocked until
the complete run succeeds. Restore missing source files first, or explicitly use
`--delete-missing` if their removal is intended. That option deletes indexed data.

Only one indexer may write a collection at a time. The file lock coordinates
indexers sharing its filesystem, not independent writers on different hosts.
Other writers must use the same protocol or be stopped during indexing. Old
clients that do not check the write state must also be stopped during migration.

## Bibliography and PDF provenance

Place optional curated metadata beside an article: `paper.md` can have
`paper.metadata.json`. For example:

```json
{
  "title": "A verified article title",
  "authors": ["An author as recorded in the article"],
  "year": 2020,
  "version": "published",
  "collection": "glacier-albedo"
}
```

Add `doi` only when verified. Changing a sidecar triggers reindexing even if the
Markdown is unchanged. A filename is retained as a display fallback, never
presented as verified bibliographic metadata. Document IDs identify source files;
DOI/content matches are reported as candidate duplicates, not merged automatically.
Renaming a file is therefore an addition unless the old source is explicitly removed.

Docling conversion additionally saves `paper.docling.json` with its structured
representation and a sidecar with parser version and exact page spans for
unambiguous text. A page span is a Unicode-character interval `[start,end)` in the
canonical Markdown, tied to its `content_sha256`, with a one-based PDF page number.
Pages are not necessarily the printed journal page numbers. Figure and table
structure is preserved in the raw export, but dedicated figure/table readers are
not implemented. LlamaParse conversion currently leaves page spans empty.

Chunk locations are assigned only when their text matches an unambiguous exact
substring. Transformations during chunking can leave a locator unresolved.
Unknown authors, DOI and pages are null. A verified canonical substring establishes
textual provenance, not that OCR/transcription is correct or the claim is true.
Conversion completeness is `not_assessed` unless explicitly marked partial; the
pipeline does not claim a comprehensive PDF extraction quality assessment.

## MCP tools

| Tool | Contract |
|---|---|
| `search_evidence` | Typed hits, bibliography, positions, excerpt truncation, index revision, channel state and reranking status. Optional source/year/collection filters. Default at most two hits per source. |
| `get_passage` | Full indexed passage and verification against its canonical substring. Supply `expected_content_sha256` from the search result to pin a version. |
| `read_document` | Canonical Markdown with bounded character pagination; follow `next_offset` and pin the returned hash. |
| `search_documents` | Paginated catalogue search over supplied title/authors/DOI and filenames, with year bounds. |
| `audit_library` | Ingestion failures, write/repair state, incomplete chunks, mixed versions, missing snapshots, incomplete bibliography and candidate duplicates. |

`search_evidence` preserves null rerank scores when Cohere is unavailable and
returns fusion-ranked candidates with a warning. Missing/failing embeddings use
local lexical retrieval when available. `alpha=0` does not call the embedding API;
the MCP may still call Cohere for reranking if configured. Scores are rankings,
not probabilities of truth. Absence of hits does not establish absence from the
scientific literature. Tool errors are sent as MCP execution errors.

BM25 excludes candidates without lexical token overlap (rather than discarding
all nonpositive scores). The first hybrid search builds BM25 synchronously and
rebuilds after an index revision change. This prioritizes consistent retrieval;
first-search latency on a large production corpus remains to be measured. Builds
and searches are serialized within a retriever. Bulk Chroma reads use batches of
500 rows to stay below backend SQL parameter limits. Catalogue/audit calls still
scan all metadata; bounded batches do not make those full scans constant-time.

## Evaluation

`tests/test_datasets/scientific_questions_draft.json` contains 50 proposed research
questions, **zero reviewed judgments and no measured performance**. Review each
question against the actual library and add its judgment keyed by question ID:

```json
{
  "reviewed": true,
  "answerable": true,
  "sources": ["paper.md"],
  "chunks": ["an_actual_chunk_id_returned_by_search"],
  "content_sha256_by_source": {"paper.md": "the_actual_64_character_sha256"}
}
```

For a reviewed question without an answer in this corpus, set `answerable=false`,
both lists empty, and `content_sha256_by_source={}`. Include genuine negative cases;
do not infer answerability merely because a search found nothing. Maintain separate
development and held-out questions before tuning retrieval parameters.

Offline validation:

```sh
python scripts/benchmark_scientific.py tests/test_datasets/scientific_questions_draft.json --validate-only
```

Scoring is refused until every question is reviewed. A run uses the actual MCP
search and passage tools and thus may incur embedding and reranking API costs:

```sh
python scripts/benchmark_scientific.py reviewed_questions.json --output benchmark.json
```

The runner checks pinned source versions and the index revision, measures article
and passage recall against the annotated sets, first relevant rank and canonical
verification. It reports candidate returns for unanswerable questions separately;
it does not measure generated-answer factuality or provide calibrated abstention.
Retrieval channel and fallback state are recorded to expose degraded runs.
The old self-retrieval dataset remains a separate recognition regression test.

## Local verification

```sh
python -m pytest -q tests/unit/test_library_reliability.py tests/test_rag_metrics.py
```

These tests use synthetic articles, simulated failures, a real in-process FastMCP
client, and a temporary real Chroma database. They do not contact production Chroma
or paid embedding/reranking services. Do not run the entire historical `tests/`
directory blindly: several legacy scripts connect to configured real services.
