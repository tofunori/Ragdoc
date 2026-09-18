# Connect Ragdoc through MCP

Follow the [current backend setup](installation.md#first-index-and-mcp-connection)
for a stdio configuration with absolute interpreter, server and data paths.
The server reads the same collection, model and canonical snapshots as the indexer.

For Ragdrop's server-side HTTP diagnostic, see the
[required backend layout](installation.md#ragdrop-backend-layout). It checks a
loopback endpoint on the backend host; it is separate from a client's stdio process.

Use `get_runtime_status` for offline dependency diagnostics. Start research with
`search_evidence`, then use `get_passage` or `read_document` with the returned content
hash to read the source context. See the [tool contracts](SCIENTIFIC_RELIABILITY.md#mcp-tools)
for provenance coverage, retrieval fallbacks and version checks.

An MCP client may send retrieved text to its model provider. Check that client's
configuration separately; running the index on your own machine does not determine
where the client's model processes its input.
