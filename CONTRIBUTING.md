# Contributing

Use a focused branch and explain the behavior changed, its limitations and how you
checked it. A useful pull request includes original synthetic fixtures and only
the files needed for the change. Do not commit research PDFs, full article text,
Zotero databases, credentials, private hostnames, local paths, indexes or private
benchmark output. Keep the MIT license unchanged unless separately discussed.

## Local checks

Set up the [isolated Python environment](INSTALLATION.md) first. These tests use
synthetic data, temporary databases and mocked providers; they do not require an
index, paid service requests or a personal corpus:

```sh
uv run --locked python -m pytest -q tests/unit tests/test_rag_metrics.py
```

Do not run the entire historical `tests/` directory by default: several scripts
contact configured services. Integration or retrieval-quality experiments require
a separately prepared library and explicit API use. Provisional assistant judgments
are not official human-reviewed benchmark results.

On macOS with Swift 6.2+ and the SDK:

```sh
cd Ragdrop
swift test
./script/build_and_run.sh build
./script/build_theme_demo.sh
```

These build commands do not launch or replace the installed app. Open the separate
demo manually when checking UI behavior. Test light and dark modes, review states,
errors and a mixed batch. Explain whether a result is simulated or observed on a
real backend. For documentation screenshots, use the generated synthetic fixtures
and retain an explicit example label.

## Useful contributions

Simplifying the Python/SSH setup, reducing the desktop/backend layout coupling,
improving accessibility and language support, and testing extraction/provenance
across document layouts are useful directions. Discuss wider architecture changes
in an issue before replacing working ingestion or retrieval behavior.
