# Documentation

| Start with | What it covers |
|---|---|
| [Product overview](../README.md) | Search your own papers from Claude, Codex and compatible MCP clients |
| [Local setup preview](local-setup.md) | Guided Mac library, managed engine and Claude Desktop extension |
| [Ragdrop for macOS](../Ragdrop/README.md) | Demo, build, settings and PDF review |
| [Backend installation](installation.md) | Python, storage, SSH and first index |
| [MCP connection](mcp-setup.md) | Connect an assistant to the library |
| [Scientific reliability](SCIENTIFIC_RELIABILITY.md) | Provenance, migration, recovery and evaluation limits |
| [Contributing](../CONTRIBUTING.md) | Offline checks and development scope |
| [Release 1.8.0](releases/v1.8.0.md) | Changes, downloads, requirements and validation |
| [Screenshots](assets/ragdrop/README.md) | Original synthetic demo content and reproduction |

## Repository layout

- `Ragdrop/` — macOS application, native tests and build scripts.
- `src/` — MCP server, retrieval, canonical storage and evidence handling.
- `scripts/` — conversion, indexing and maintenance commands.
- `tests/` — backend checks and synthetic fixtures; follow the offline test guide.
- `docs/` — current guides, examples and release notes.
- `config/` — reference configuration and historical loader; current runtime
  configuration is documented in the backend guide.

The root `ragdoc-launch.py` is the canonical MCP launcher. The older
`ragdoc-cli.py`, `ragdoc-menu.py` and `chromadb_server_manager.py` remain in place
for compatibility with earlier workflows; they are not the supported onboarding
path for this release. Some historical commands reference tools no longer shipped.
Windows helper scripts are grouped under `scripts/legacy/windows/`.

Older material under [`archive/`](archive/) records previous behavior and may
contain obsolete models, paths or instructions. Use the guides above for current
installation. Local libraries, generated indexes, benchmark output and personal
MCP client settings are intentionally excluded from version control.
