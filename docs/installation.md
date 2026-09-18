# Install Ragdoc

For the desktop application, start with the [Ragdrop macOS guide](../Ragdrop/README.md).
This guide covers the backend from source. It does not provision a NAS, an SSH
account, or a public service. The Ragdrop integration currently expects Linux;
the standalone Python backend can also be run on macOS or Windows with equivalent
paths and commands. The examples below use a POSIX shell.

## Prepare an isolated environment

The package declares Python 3.10 or later. The committed lockfile provides the
recommended Python 3.12 environment. Install Git, Python/uv and then:

```sh
git clone https://github.com/tofunori/Ragdoc.git
cd Ragdoc
uv sync --locked --python 3.12 --extra dev
uv run --locked python -m nltk.downloader -d .venv/nltk_data stopwords
export NLTK_DATA="$PWD/.venv/nltk_data"
uv run --locked python src/server.py --check-runtime
```

For the released version, run `git checkout v1.8.0` before building. Dependency/data downloads need
network access. `--check-runtime` itself does not open Chroma or call embedding or
reranking APIs. It checks local SDK capabilities and tokenizer data; configured
key presence is not proof that a service accepts the key. New indexing may also
need the tokenizer assets used by Chonkie. Offline tests avoid these downloads.

If you do not use uv, `python3.12 -m venv .venv` followed by
`.venv/bin/python -m pip install -r requirements.txt` is an alternative with
version ranges rather than the exact locked dependency set.

## Configuration

Copy `.env.example` to `.env` and enter your own keys. Do not commit that file.
The app's remote index command sources `.env` as a shell file as well as Python
reading it through dotenv: use simple `NAME=value` assignments and shell-safe
quoting, not commands.

```sh
cp .env.example .env
```

| Setting | Meaning |
|---|---|
| `VOYAGE_API_KEY` | Required to embed new documents and for semantic query embeddings |
| `COHERE_API_KEY` | Optional reranking; remove/leave unset to use fusion ranking without Cohere |
| `RAGDOC_EMBEDDING_MODEL` | Defaults to `voyage-context-4`; must match the collection metadata |
| `RAGDOC_CHROMA_MODE` | Set `persistent` for a local on-disk index, `http` for a Chroma server; legacy `auto` probes HTTP then disk |
| `CHROMA_DB_PATH` | Local Chroma directory; default `chroma_db_new` under this checkout |
| `COLLECTION_NAME` | Default `ragdoc_contextualized_v1` |
| `RAGDOC_LIBRARY_DIR` | Canonical Markdown snapshots and journal; default next to Chroma under `ragdoc_library/<collection>` |
| `RAGDOC_ARTIFACTS_DIR` | Extracted visual artifacts; default `ragdoc_artifacts` under this checkout |
| `NLTK_DATA` | Directory holding the downloaded tokenizer data |

The current reranking default is `rerank-v4.0-pro`. These are identifiers used by
the source code, not a promise of model availability, pricing or benchmark quality.
`src/config.py` and environment variables govern this pipeline; the YAML files in
`config/` are reference material, not a complete live configuration interface.

Use identical storage, model and collection settings for the MCP server and
indexer. **Changing the model name does not migrate existing vectors.** For an
existing library, back up Chroma, source Markdown, metadata and canonical snapshots
and read [storage and migration](SCIENTIFIC_RELIABILITY.md#storage-and-migration)
first. Model migration tooling is available in `scripts/migrate_embedding_collection.py`;
inspect its `--help` and plan a separate collection before attempting migration.

## First index and MCP connection

For a new standalone backend, place your own reviewed Markdown in
`articles_markdown/`. Optional `paper.metadata.json` files carry bibliography and
conversion provenance. This command **writes the index and calls Voyage**, with
possible API costs:

```sh
export RAGDOC_CHROMA_MODE=persistent
uv run --locked python scripts/index_incremental.py --source paper.md
```

Replace `paper.md` with the actual filename. Index only documents you intend to
process. Conversion and indexing are distinct; the backend can ingest existing
Markdown without Ragdrop. Do not start with a full private library before checking
a small, reviewed example.

The standard server entry point uses MCP stdio. In an MCP client, configure the
absolute path to the environment interpreter and server file, for example:

```json
{
  "mcpServers": {
    "ragdoc": {
      "command": "/path/to/Ragdoc/.venv/bin/python",
      "args": ["/path/to/Ragdoc/src/server.py"],
      "env": {
        "RAGDOC_CHROMA_MODE": "persistent",
        "CHROMA_DB_PATH": "/path/to/Ragdoc/chroma_db_new",
        "RAGDOC_LIBRARY_DIR": "/path/to/Ragdoc/ragdoc_library/ragdoc_contextualized_v1",
        "COLLECTION_NAME": "ragdoc_contextualized_v1",
        "RAGDOC_EMBEDDING_MODEL": "voyage-context-4",
        "NLTK_DATA": "/path/to/Ragdoc/.venv/nltk_data"
      }
    }
  }
}
```

Replace every example path. Keep API keys in the backend `.env` or supply them
through your client's private environment configuration. Connection setup differs
by MCP client. Use `get_runtime_status` for offline capability diagnostics and
`search_evidence`, `get_passage` and `read_document` for search and version-pinned
reading. Canonical reads require both the index and the matching snapshot directory.

## Ragdrop backend layout

Ragdrop currently expects this layout on an SSH-accessible Linux host:

```text
/path/to/Ragdoc/
  .env
  ragdoc-env-new/bin/python3
  articles_markdown/
  chroma_db_new/
  ragdoc_library/ragdoc_contextualized_v1/
  ragdoc_artifacts/
  scripts/index_incremental.py
  scripts/index_artifacts.py
```

Use your own absolute backend root in the application settings. It must be writable
by the SSH user. Install the OS commands `/usr/bin/timeout`, `ss`, `tar`, `find`,
and an SSH server. On that host, from the checkout:

```sh
UV_PROJECT_ENVIRONMENT=ragdoc-env-new uv sync --locked --python 3.12 --extra dev
./ragdoc-env-new/bin/python3 -m nltk.downloader -d ragdoc-env-new/nltk_data stopwords
```

In `.env`, set your keys and `NLTK_DATA` to the absolute `ragdoc-env-new/nltk_data`
path. Configure the fixed Ragdrop contract: `RAGDOC_CHROMA_MODE=persistent`,
`COLLECTION_NAME=ragdoc_contextualized_v1`, `RAGDOC_EMBEDDING_MODEL=voyage-context-4`,
`CHROMA_DB_PATH=<backend-root>/chroma_db_new`, and
`RAGDOC_LIBRARY_DIR=<backend-root>/ragdoc_library/ragdoc_contextualized_v1`.
The app currently pins those values when it indexes; they are not all configurable
from the GUI. Keep the artifact directory at the default `<backend-root>/ragdoc_artifacts`.

A fresh backend needs a collection before the app's catalogue/duplicate checks can
succeed. Create an empty collection once, after configuring `.env`; this opens local
storage and makes no embedding request:

```sh
./ragdoc-env-new/bin/python3 - <<'PY'
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, COLLECTION_CONTEXTUALIZED_METADATA
import chromadb
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
client.get_or_create_collection(COLLECTION_NAME, metadata=COLLECTION_CONTEXTUALIZED_METADATA)
PY
mkdir -p articles_markdown
```

This bootstrap is for a **new** backend, not a migration of existing data. The first
approved import produces the lexical index and canonical snapshots. Before that,
the library is empty and the live search diagnostic cannot succeed.

For the desktop status screen, run the canonical MCP launcher on that same host:

```sh
RAGDOC_TRANSPORT=http RAGDOC_PORT=8484 ./ragdoc-env-new/bin/python3 ragdoc-launch.py
```

It uses the pinned layout above, requires the collection to exist and checks its
embedding model. It binds to `127.0.0.1` by default, including for the app's check
at `http://127.0.0.1:8484/mcp` over SSH. Running it persistently requires your own
service manager. Do not expose this unauthenticated listener to the public internet;
use SSH or a separately secured deployment. `RAGDOC_HOST` can override the bind
address when that deployment is configured. No server service is installed by the
macOS build script.

The macOS side still needs its own Python converter dependencies and OCR credential:
see [real-import setup](../Ragdrop/README.md#set-up-real-imports).

## Data handling and troubleshooting

- Mistral/MinerU receive PDF content during conversion; Voyage receives document
  or query text; optional Cohere receives queries and candidate passages. API
  processing can incur costs. Do not treat self-hosted storage as offline processing.
- If a new index fails, inspect the error and write state. Do not delete a database
  as a routine troubleshooting step. Follow the recovery guidance in the scientific
  reliability guide and keep backups.
- If the app cannot find Python or `requests`, check the exact interpreter on its
  fixed PATH; activating a terminal virtual environment does not change Finder's
  launch environment.
- If catalogue reads fail, check SSH, the backend layout, collection, permissions
  and Chroma version before trying an import. The desktop catalogue reads Chroma's
  SQLite storage directly and is coupled to that schema.
- Lexical-only queries use `alpha=0`; Cohere may still run unless its key is absent.
  Canonical reads do not call OCR or embeddings. Your MCP client's model calls are
  outside the backend's control.

For supported offline tests, see [CONTRIBUTING.md](../CONTRIBUTING.md). Several legacy
scripts under `tests/` contact configured services; do not run all of them blindly.
