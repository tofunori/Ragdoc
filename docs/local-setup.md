# Ragdrop local setup — 0.7 development build

This development build adds a guided **This Mac** library and a Claude Desktop
extension. It is separate from the published Ragdoc 1.8.0 / Ragdrop 0.6 release.
The app currently targets Apple silicon and macOS 14 or later.

## First launch

1. Put Ragdrop in Applications and open it.
2. Select a library folder, or keep the suggested folder. Click **Prepare this Mac**.
   Internet access is required. Ragdrop downloads a private Python 3.12 runtime,
   the locked engine dependencies, PDF tools and tokenizer resources. This may take
   several minutes. No system Python, terminal commands, SSH account or NAS are needed.
3. Add a **Mistral OCR** key and a **Voyage AI** key. They are saved in macOS Keychain.
   Keys may be added later, but importing and semantic search need the corresponding
   service. A Claude subscription does not supply these API credits. Saving a key
   does not prove that the service will accept it.
4. Click **Install Claude extension…** and confirm the installation in Claude Desktop.
   If macOS does not open the installer, choose **Settings → Extensions → Advanced
   settings → Install Extension** in Claude Desktop and select the supplied
   `Ragdoc.mcpb` file.
5. Open the library, choose a PDF, review the extraction, then approve it for indexing.
   Wait for indexing and verification to finish. In Claude, enable Ragdoc and ask
   it to search your library with `search_evidence`.

The extension is for **Claude Desktop on this Mac**. It is not a hosted connector
for Claude Web or mobile. Claude starts the prepared engine through its extension;
Ragdrop does not need to remain open. The extension installation and permissions are
confirmed in Claude; the application does not edit an existing Claude configuration.
See [Anthropic's extension instructions](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop).

## Storage and service boundaries

- Original PDFs stay in their existing locations. Moving one can break its review
  link; preserve it until the import is complete.
- The selected library folder holds reviewed Markdown, visual artifacts, Chroma,
  canonical snapshots, the local queue and the local Zotero monitoring ledger.
- The engine, private Python and download cache live under
  `~/Library/Application Support/Ragdrop/`. A connection file there contains paths,
  not API keys. Keep the app out of Trash while installing the extension.
- Mistral receives selected PDFs for OCR **before review**. Voyage receives approved
  text for indexing and queries for semantic search. The initial local setup does
  not upload documents or test paid APIs.
- Local mode leaves Cohere reranking disabled. Lexical queries (`alpha=0`) and
  canonical reads do not need a Voyage call. The connected assistant receives
  retrieved passages when you use its tools.
- MinerU remains an advanced alternative requiring its own token; Mistral is the
  guided default. PDF splitting and Markdown rendering dependencies are included
  in the managed desktop environment.

## Existing server installations

Existing saved server/converter settings or a legacy queue select **Your server**
on upgrade. The app does not migrate the server library. To try local mode, finish
or remove pending queue items, then use **Settings → Set up a library on this Mac**.
Each destination retains its own queue and Zotero monitoring state. Switching back
does not delete or move articles. The SSH backend remains available as an advanced
option; see [server setup](installation.md#ragdrop-backend-layout).

## Recovery

- A failed download can be retried with **Prepare this Mac**. Cancel stops the
  installer process group; existing article files remain in place.
- **Check / repair engine** reinstalls missing runtime components while retaining
  the recorded library folder, including custom folders with spaces or accents.
- If an index write was interrupted, Ragdrop offers **Back up and rebuild index**.
  The confirmation explains that rebuilding uses Voyage and may incur charges.
  Ragdrop snapshots managed library files under `Backups/` while holding the writer
  lock, then rebuilds approved Markdown. A failed backup aborts the rebuild.
  Missing source files are not automatically deleted from the index; restore them
  before retrying. Original PDFs are not moved.
- Setup and recovery controls are disabled while an import is running. Close other
  indexing tools before rebuilding. Do not manually delete an index to clear an error.

## Build and validation

Developers still need Swift 6.2, Python and network access to build the app:

```sh
cd Ragdrop
./script/build_and_run.sh build
```

The build bundles immutable engine sources, the pinned and checksum-verified
`uv` bootstrapper, its licenses, and the Claude extension. Runtime dependencies use
`uv.lock` and the `desktop` extra. They are installed at first setup, not on a user's
system Python. A source digest separates engine environments across code updates.

Useful checks from the repository root:

```sh
swift test --package-path Ragdrop
node --test Ragdrop/Integrations/ClaudeDesktop/server/index.test.js
npx --yes @anthropic-ai/mcpb@2.1.2 validate Ragdrop/Integrations/ClaudeDesktop
```

The opt-in Swift integration test uses `RAGDROP_INSTALL_TEST_ROOT` and
`RAGDROP_INSTALL_TEST_RESOURCES` to install and repair a real managed environment in
a disposable folder. It checks the empty catalogue, private interpreter, PDF parser
and bundled table renderer. The native setup validation app uses a separate bundle
identifier, preferences suite and temporary library; it never imports real papers.

Validation distinguishes local engine readiness from successful provider access or
an indexed article. A protocol smoke test covers MCP initialization, tool discovery
and the offline runtime diagnostic. It does not establish a live OCR import, a paid
semantic search, a completed Claude installation or a test on a physically clean Mac.

This development build is ad-hoc signed and **not notarized**. Apple Developer
signing and clean-machine distribution testing remain necessary before describing
it as a polished public one-click installer.
