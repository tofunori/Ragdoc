# Scientific library reliability

This change adds canonical source snapshots, explicit citation provenance, safer
index replacement, structured evidence tools and an annotation-gated benchmark.
It does not annotate the existing library or demonstrate retrieval gains on it.

## Storage and migration

Keep a backup of the existing Chroma database and Markdown sources before a first
production migration. Install the updated requirements in an isolated environment.
The committed `uv.lock` defines a reproducible dependency set. Use an isolated
Python 3.12 environment (`uv sync --locked --python 3.12 --extra dev`), install NLTK
stopwords into `.venv/nltk_data`, set `NLTK_DATA` to that directory, and run
`uv run --locked python src/server.py --check-runtime` as in the
[installation guide](../INSTALLATION.md#prepare-an-isolated-environment).
The offline diagnostic verifies that the Voyage SDK exposes `contextualized_embed`
and that advanced tokenization is available; it does not call either external API.
The current indexer uses deterministic token chunking. Its tokenizer assets may
need downloading before first use; tests do not download models or call paid APIs.

`RAGDOC_LIBRARY_DIR` selects the canonical store. By default it is
`<Chroma parent>/ragdoc_library/<collection name>`. Both the indexer and MCP reader
must see this same directory. Sharing only a remote Chroma connection is not
enough for canonical reads. Back up this directory along with Chroma and sources.
Snapshots are immutable UTF-8 Markdown addressed by SHA-256; the SQLite journal
records ingestion attempts. A successful journal event does not replace the index
integrity checks. The canonical snapshots include complete article text: treat
them with the same access and copyright controls as the original library.

For a collection already tagged with the configured embedding model, run the normal
incremental indexer to backfill provenance. Unchanged legacy articles without a
metadata sidecar are skipped; a deliberate `--force --source paper.md` run may be
needed to backfill a reviewed source. Back up first and expect embedding costs.
Articles with sidecars are reindexed when their pipeline/provenance metadata differ.
A collection with a missing
or different model tag is refused; use a separately planned model migration into a
new collection instead (see [installation](../INSTALLATION.md#configuration)).
It still incurs embedding costs. Existing tools remain available; a legacy full
document read explicitly warns that overlapping chunks are reconstructed.
Canonical reads require a migrated article and a matching snapshot.

Section enrichment is a separate, embedding-preserving migration. First inspect
the proposed coverage, then apply it while all other writers are stopped:

```sh
uv run --locked python scripts/enrich_section_metadata.py
uv run --locked python scripts/enrich_section_metadata.py --apply
```

The migration keeps chunk IDs, text and vectors unchanged, derives hierarchical
section metadata only for exact canonical locators, verifies every rewritten
source and synchronizes the revision-pinned lexical index. It leaves ambiguous
locations unresolved. Repeat `--source paper.md` to migrate a reviewed subset
before the complete collection.

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
structure is preserved in the raw Docling export. Ragdrop's Mistral/MinerU
converters can additionally produce artifact manifests; after `index_artifacts.py`,
MCP tools `find_document_artifacts`, `get_artifact` and `get_artifact_image` read those
indexed artifacts. A raw Docling export is not automatically an artifact manifest.
LlamaParse conversion currently leaves page spans empty.

Chunk locations are assigned only when their text matches an unambiguous exact
substring. Transformations during chunking can leave a locator unresolved.
Unknown authors, DOI and pages are null. A verified canonical substring establishes
textual provenance, not that OCR/transcription is correct or the claim is true.
Conversion completeness is `not_assessed` unless explicitly marked partial; the
pipeline does not claim a comprehensive PDF extraction quality assessment.

## MCP tools

| Tool | Contract |
|---|---|
| `search_evidence` | Typed hits, bibliography, positions, excerpt truncation, index revision, channel state and reranking status. Optional source/year/collection/section filters, explicit subqueries and an opt-in article-first strategy. Default at most two hits per source. |
| `get_passage` | Full indexed passage and verification against its canonical substring. Supply `expected_content_sha256` from the search result to pin a version. Optional paragraph or section context is read directly from the canonical snapshot. |
| `read_document` | Canonical Markdown with bounded character pagination; follow `next_offset` and pin the returned hash. |
| `search_documents` | Paginated catalogue search over supplied title/authors/DOI and filenames, with year bounds. |
| `audit_library` | Compact integrity summary by default. Paginated `findings`, `duplicates`, and latest-per-source `events` views expose details without overflowing MCP clients. The event view is not a complete ingestion history. |
| `audit_citation_readiness` | Paginated per-document audit of readable canonical snapshots, exact passage locators and valid PDF page ranges. A DOI alone never marks a passage page-verifiable. |
| `get_runtime_status` | Offline dependency/capability checks, configured key presence and requested/active Chroma connection. No database opening or API calls. |

`search_evidence` preserves null rerank scores when Cohere is unavailable and
returns fusion-ranked candidates with a warning. Missing/failing embeddings use
local lexical retrieval when available. `alpha=0` does not call the embedding API;
the MCP may still call Cohere for reranking if configured. Scores are rankings,
not probabilities of truth. Absence of hits does not establish absence from the
scientific literature. Tool errors are sent as MCP execution errors.

Section filters use normalized categories (`abstract`, `introduction`, `methods`,
`results`, `discussion`, `conclusion`, `references`, `supplementary`, and
`acknowledgements`). `section_mode=strict` excludes unknown or unmatched sections;
`prefer` keeps them eligible and reports that the preference is heuristic. A
section category describes where text occurs, not whether a claim is original,
true, or supported by the article's data.

`retrieval_strategy=articles_then_passages` first aggregates passage evidence by
source using at most three contributions per article, then searches within a
bounded article shortlist. Explicit `subqueries` are fused with the original
question and each hit reports which queries retrieved it. These options improve
coverage mechanisms but are not claims of better scientific recall until the
reviewed benchmark demonstrates a gain.

Lexical retrieval uses a revision-pinned SQLite FTS5 sidecar. Its fielded BM25
weights body, title, authors, identifiers and source separately; identifier-like
queries prefer catalogue identifiers over incidental citations in article bodies.
Candidate generation admits at most ten passages per source and excludes macOS
`._` sidecars, preventing a long article from occupying the complete lexical pool.
The sidecar is rebuilt atomically when its schema, revision or passage count differs
from Chroma, and is synchronized after incremental writes. Builds read Chroma in
batches of 500 rows to stay below backend SQL parameter limits. The catalogue scans
metadata once per revision and retains one record per document. Revisionless legacy
indexes are not cached, and active writes still block catalogue reads. Audit calls
continue to scan the complete metadata set and verify canonical snapshots.

The public search limit is 100 hits. Source diversity can trigger larger retrieval
pools up to 1,000 candidates per query; the reranking input remains bounded to 100.
If the expansion cap prevents filling the requested number of diverse hits, the
response reports `candidate_limit_reached`. Successful query embeddings are cached
in memory (128 queries per retriever), including across expansion attempts.

Both the server and indexer use `RAGDOC_CHROMA_MODE`: `persistent` never probes an
HTTP server, `http` never falls back to disk, and `auto` retains the legacy HTTP-first
behavior. Set `RAGDOC_CHROMA_HOST` and `RAGDOC_CHROMA_PORT` for HTTP connections.
Changing these settings requires starting a new server process.

## Evaluation

`tests/test_datasets/scientific_questions_draft.json` contains 50 proposed research
questions, **zero reviewed judgments and no measured performance**. Review each
question against the actual library and add its judgment keyed by question ID:

```json
{
  "reviewed": true,
  "review_provenance": "human",
  "reviewer": "your reviewer identifier",
  "answerable": true,
  "sources": ["paper.md"],
  "chunks": ["an_actual_chunk_id_returned_by_search"],
  "content_sha256_by_source": {"paper.md": "the_actual_64_character_sha256"}
}
```

For a reviewed question without canonical evidence in this corpus, set
`answerable=false`, both lists empty, and `content_sha256_by_source={}`. Record
`corpus_status=present_but_unverifiable` when a relevant legacy document exists but
cannot provide a canonical passage, or `not_established_after_targeted_search` when
targeted review found no qualifying passage. Neither status establishes that the
scientific answer is negative. Maintain separate development and blind questions
before tuning retrieval parameters.

Offline validation (assistant drafts remain invalid for official scoring):

```sh
uv run --locked python scripts/benchmark_scientific.py tests/test_datasets/scientific_questions_draft.json --validate-only
```

Scoring is refused until every question is reviewed. A run uses the actual MCP
search and passage tools and thus may incur embedding and reranking API costs:

```sh
uv run --locked python scripts/benchmark_scientific.py reviewed_questions.json --output benchmark.json
```

The runner checks pinned source versions and the index revision, measures article
and passage recall against the annotated sets, first relevant rank and canonical
verification. It reports candidate returns when canonical evidence is unavailable;
it does not measure generated-answer factuality or provide calibrated abstention.
Retrieval channel and fallback state are recorded to expose degraded runs.
The old self-retrieval dataset remains a separate recognition regression test.

### Review status and provisional diagnostics

Official scoring requires human-reviewed judgments, reviewer provenance and source
SHA-256 pins. Assistant-selected evidence can be inspected in a provisional run
using `--provisional-assistant`; that status must remain visible in any report.
An evaluation set exposed during tuning is not a blind test. Use a fresh unseen
set for an independent final estimate. This repository does not claim a measured
general retrieval-quality score from a private corpus.

## Local verification

```sh
uv run --locked python -m pytest -q tests/unit tests/test_rag_metrics.py
```

These tests use synthetic articles, simulated failures, a real in-process FastMCP
client, and a temporary real Chroma database. They do not contact production Chroma
or paid embedding/reranking services. Do not run the entire historical `tests/`
directory blindly: several legacy scripts connect to configured real services.
