# Ragdrop source publication

This change brings the existing macOS application into the public repository and
presents the app and backend together. It includes:

- Ragdrop 0.5 source, icon, Swift tests and normal/isolated-demo build scripts.
- Human review with PDF/rendered/source/artifact views, explicit batch stages,
  Zotero selection and optional monitoring, and light/dark/system appearance.
- The compatible converter, artifact, canonical-provenance and lexical-retrieval
  backend components and their offline tests. They are required to describe and
  build the current application coherently from this branch.
- Four unaltered screenshots of original synthetic examples and updated app,
  backend, MCP and contribution guides.

Publication-specific changes replace installation-specific default SSH host/root
values with `ragdoc-server` and `/srv/ragdoc` examples. Saved user preferences still
win. The canonical HTTP launcher now binds to loopback unless `RAGDOC_HOST` is
explicitly supplied. Neither change modifies an already installed application.

Private libraries, operational results, internal design/install reports and old
mockups are excluded. This is a source proposal: it does not publish a binary,
notarize an app, migrate an index, change the license, or install a backend service.

Remaining onboarding work includes provisioning the remote Python/SSH layout and
selecting a macOS converter interpreter. The app still expects a specific Chroma
collection/storage layout, and its interface is currently French. These constraints
are documented rather than presented as a plug-and-play setup.

## Verification of this proposal

- 40 Swift tests pass; normal and synthetic-demo bundles build and pass local
  signature verification. Builds do not install or launch the app.
- 127 Python tests pass in the locked environment (`tests/unit` and
  `tests/test_rag_metrics.py`), with external socket connections disabled.
- Offline runtime diagnostics find the expected SDK/tokenizer capabilities and
  report no issues. No paid API, live index or model-quality benchmark was run.
- Relative documentation links, image targets and whitespace checks pass.
- Independent reviews checked setup commands against source, dependency scope,
  public claims and all four screenshots. Reported documentation issues were
  corrected and rechecked; no blocking findings remain in those review scopes.

The source builds were checked on Apple silicon with Swift 6.4 and the macOS SDK;
Python tests used 3.12. The package targets macOS 14 and Swift tools 6.2. This does
not establish testing on every supported OS/toolchain or a fresh Linux deployment.
