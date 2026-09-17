# Scientific library reliability

This change adds canonical source snapshots, explicit citation provenance, safer
index replacement, structured evidence tools and an annotation-gated benchmark.
It does not annotate the existing library or demonstrate retrieval gains on it.

## Storage and migration

Keep a backup of the existing Chroma database and Markdown sources before a first
production migration. Install the updated requirements in an isolated environment.
The committed `uv.lock` defines a reproducible dependency set. Use an isolated
Python 3.12 environment (`uv sync --locked --python 3.12 --extra dev`), install NLTK
stopwords into `.venv/nltk_data`, and run `python src/server.py --check-runtime`.
The offline diagnostic verifies that the Voyage SDK exposes `contextualized_embed`
and that advanced tokenization is available; it does not call either external API.
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
python scripts/benchmark_scientific.py tests/test_datasets/scientific_questions_draft.json --validate-only
```

Scoring is refused until every question is reviewed. A run uses the actual MCP
search and passage tools and thus may incur embedding and reranking API costs:

```sh
python scripts/benchmark_scientific.py reviewed_questions.json --output benchmark.json
```

The runner checks pinned source versions and the index revision, measures article
and passage recall against the annotated sets, first relevant rank and canonical
verification. It reports candidate returns when canonical evidence is unavailable;
it does not measure generated-answer factuality or provide calibrated abstention.
Retrieval channel and fallback state are recorded to expose degraded runs.
The old self-retrieval dataset remains a separate recognition regression test.

### Assisted annotation workflow

`tests/test_datasets/scientific_questions_v1.json` contains 35 development questions
and 15 questions labelled `heldout`. The latter were already exposed during
candidate preparation and an initial all-question baseline, so they are a locked
non-blind evaluation split rather than a final blind test. A fresh unseen question
set is required for an independent final estimate. Its English `lexical_query`
fields are recorded query aids, not gold answers.

Build candidate packets from the production MCP through three retrieval channels:

```sh
python scripts/prepare_scientific_annotations.py \
  tests/test_datasets/scientific_questions_v1.json \
  --url "$RAGDOC_MCP_URL" \
  --output-dir output/benchmarks/scientific_annotation_v1 \
  --top-k 20 --max-passages 20
```

After recording proposed chunk selections, verify and pin them while generating a
human-readable checklist:

```sh
python scripts/materialize_scientific_review.py \
  tests/test_datasets/scientific_questions_v1.json \
  output/benchmarks/scientific_annotation_v1/assistant_selections.json \
  --url "$RAGDOC_MCP_URL" \
  --dataset-output output/benchmarks/scientific_annotation_v1/scientific_questions_v1_assistant_draft.json \
  --review-output output/benchmarks/scientific_annotation_v1/HUMAN_REVIEW.md
```

Materialization verifies every selected chunk against its canonical snapshot and
pins the source SHA-256, but deliberately writes `reviewed=false`. The benchmark
therefore still refuses to score until a human has checked all 50 decisions.
Assistant selections can be run only with `--provisional-assistant`; the resulting
file is explicitly labelled `provisional_assistant_adjudicated` and cannot be
mistaken for the official human-reviewed score.

```sh
python scripts/benchmark_scientific.py \
  output/benchmarks/scientific_annotation_v1/scientific_questions_v1_assistant_draft.json \
  --provisional-assistant --url "$RAGDOC_MCP_URL" --split development \
  --use-lexical-query --output provisional-development.json
```

The remote runner accepts `--url`, `--split`, the existing retrieval strategies,
and either automatic multi-query expansion or a recorded English lexical subquery.
Those two expansion modes are alternatives because the MCP contract does not allow
`multi_query` and explicit `subqueries` in the same call.

### Provisional diagnostic (2026-09-16)

An assistant-adjudicated diagnostic was run against index revision
`7afa4f95a63e44bcb0bd192494b49093`. It is not a human-reviewed benchmark result.
At top 10, the simple French-query baseline reached 31.3% mean article recall and
17.2% mean exact-passage recall on the development split. A recorded English
lexical subquery improved these to 75.0% and 48.4%, respectively. Automatic
multi-query expansion reached 37.5% and 20.3%; article-then-passage retrieval with
the lexical subquery reached 56.3% and 39.1%.

After development comparison, the lexical-subquery configuration was applied to
the locked non-blind evaluation split. It reached 54.5% mean article recall, 40.9%
mean exact-passage recall, and a 44.7% canonical-verification rate among returned
passages. These
numbers expose real work remaining: multilingual query formulation and legacy
duplicates materially affect retrieval. Candidate returns when no canonical
evidence was established do not measure abstention or scientific answerability;
generated-answer and abstention evaluation remains a separate end-to-end task.
Because the selected relevant passages came from the same candidate-generating
configurations being compared, recall is measured against a small adjudicated set,
not exhaustive relevance across the library.

## Local verification

```sh
python -m pytest -q tests/unit/test_library_reliability.py tests/test_rag_metrics.py
```

These tests use synthetic articles, simulated failures, a real in-process FastMCP
client, and a temporary real Chroma database. They do not contact production Chroma
or paid embedding/reranking services. Do not run the entire historical `tests/`
directory blindly: several legacy scripts connect to configured real services.
