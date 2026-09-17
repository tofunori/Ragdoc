# Ragdoc + Ragdrop

**Search your own scientific papers from Claude, Codex or another MCP-compatible assistant.**

**Ragdoc** is a server for the **Model Context Protocol (MCP)**, a standard way for
assistants to call external tools. Once connected, your assistant can use Ragdoc to
search the articles in your personal literature library, read their content and
retrieve supporting passages with source context and provenance information.

**Ragdrop** is the companion macOS application that prepares that library: import
PDFs from Finder or Zotero, compare extracted text with the original, then approve
documents for indexing. Ragdoc searches the articles you have added to this library;
you ask questions through your connected assistant.

For example: “Which papers in my library compare field measurements with satellite
estimates?” Then: “Show me the source passages describing their limitations.”
Traceable passages help you check an answer; they do not guarantee scientific accuracy.

![Ragdrop in light mode: PDF and Zotero import actions above a synthetic example library](docs/assets/ragdrop/home-light.png)

*Actual Ragdrop interface, shown with original synthetic examples. The current app
uses French labels. The demonstration controls along the bottom are not part of
the normal app.*

## What you can do

- **Bring in PDFs from Finder or Zotero.** Queue several articles and detect
  duplicates using PDF fingerprints.
- **Review before indexing.** Compare the original PDF with rendered extraction,
  Markdown source, and extracted tables or figures. Follow page links when the
  conversion provides matching locators.
- **Keep track of each article.** Conversion, human review, transfer, indexing and
  verification remain distinct; failures and pending decisions stay visible.
- **Search beyond exact wording.** Ragdoc combines lexical and vector retrieval,
  with optional reranking, source filters and structured evidence results.
- **Read the supporting context.** Retrieve passages and canonical document
  snapshots with version hashes and provenance coverage rather than relying on a
  search snippet alone.

## See the workflow

![Ragdrop review in light mode: original synthetic PDF alongside rendered extraction and an approval button](docs/assets/ragdrop/review-light.png)

*Human review is a separate step. Approval makes an extraction eligible for
indexing; successful indexing does not certify its scientific accuracy.*

<details>
<summary>Dark appearance</summary>

![Ragdrop home in dark mode with the same synthetic example library](docs/assets/ragdrop/home-dark.png)

![Ragdrop review in dark mode; the original PDF retains its white page](docs/assets/ragdrop/review-dark.png)

</details>

Ragdrop offers **light, dark and system** appearance under **Réglages → Apparence**.
See [screenshot provenance and reproduction](docs/assets/ragdrop/README.md).

**Prepare your library with Ragdrop**

```mermaid
flowchart LR
    A[PDF / Zotero] --> B[OCR conversion]
    B --> C[Human review in Ragdrop]
    C -->|Approve and add| D[Your indexed article library]
```

**Search it from your assistant**

```mermaid
flowchart LR
    A[Claude / Codex / compatible client] -->|MCP request| B[Ragdoc server]
    B -->|Search and read| C[Your indexed article library]
    C -->|Passages and provenance| B
    B -->|MCP results| A
```

## Try it

| Your goal | Start here |
|---|---|
| Explore the macOS interface without a backend or API keys | [Build the isolated synthetic demo](Ragdrop/README.md#try-the-interface) |
| Build Ragdrop and connect your own backend | [macOS application guide](Ragdrop/README.md) |
| Run the search backend or connect an MCP client | [Backend installation](INSTALLATION.md) |
| Understand provenance, migration and evaluation | [Scientific reliability guide](docs/SCIENTIFIC_RELIABILITY.md) |

**This is currently a source-build project, not a one-click installation.** Ragdrop
requires macOS 14 or later and Swift 6.2 to build. Real imports need a configured
SSH-accessible Linux backend and an OCR service credential. Build from source; a notarized macOS download is not available yet. The demo works without those services.

## What runs where

Ragdrop reads local PDFs and the Zotero Desktop local API. Mistral OCR is the default
converter; MinerU is an alternative. These converters upload selected PDFs to their
respective services **before** the human review step. Approved Markdown, metadata
and visual artifacts are transferred to your backend over SSH.

Ragdoc stores Chroma vectors, a SQLite FTS5 lexical index and versioned Markdown
snapshots on your infrastructure. Voyage AI receives document text during embedding
and queries during semantic search. Cohere, when configured, receives the query and
candidate passages for reranking. Your MCP client receives the retrieved content;
its own model and data handling are separate from Ragdoc.

Lexical retrieval and canonical reads can run locally once the library is indexed.
With `alpha=0`, search skips Voyage; to avoid reranking API calls, leave Cohere
unconfigured too. Building a new vector index uses Voyage. Model identifiers and
configuration are described in the [backend guide](INSTALLATION.md#configuration).
Storage and indexes stay on your infrastructure; OCR, embeddings and optional
reranking use the services listed above.

## Trust and limitations

- OCR can lose or misread equations, table structure, units and reading order.
  Check the PDF before relying on extracted content.
- An exact match to a canonical snapshot establishes textual provenance, not
  scientific truth or correct OCR. PDF page links depend on available, verified
  locator coverage and may be missing.
- Search scores rank candidates; they are not confidence probabilities. No result
  does not establish that evidence is absent from the literature.
- Evaluation tooling separates human-reviewed judgments from provisional assistant
  diagnostics. Current diagnostics do not establish a general retrieval-quality
  score. See the
  [evaluation requirements](docs/SCIENTIFIC_RELIABILITY.md#evaluation).
- Back up source Markdown, canonical snapshots and the index before migration.
  Index replacement has recovery checks, but is not a database-wide transaction.

## Development and contributions

See [CONTRIBUTING.md](CONTRIBUTING.md) for offline checks and contribution scope.
Useful next steps include simpler backend onboarding, broader extraction tests,
accessibility, and language support. Use original synthetic documents in issues
and tests; do not commit articles, personal libraries, credentials or generated
indexes.

[MIT license](LICENSE). Service credentials and any rights needed to process your
own documents remain your responsibility.
