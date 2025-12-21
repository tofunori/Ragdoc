# MCP Server Configuration Guide

## Quick Setup for Claude Desktop

Add this configuration to your `~/.claude.json` file (Windows: `C:\Users\<username>\.claude.json`):

```json
{
  "mcpServers": {
    "ragdoc": {
      "type": "stdio",
      "command": "C:\\Users\\<username>\\miniforge3\\envs\\ragdoc-env\\python.exe",
      "args": [
        "D:/Claude Code/ragdoc-mcp/src/server.py"
      ],
      "cwd": "D:/Claude Code/ragdoc-mcp",
      "env": {
        "CHROMA_DB_PATH": "D:/Claude Code/ragdoc-mcp/chroma_db_new",
        "COLLECTION_NAME": "zotero_research_context_hybrid_v3"
      }
    }
  }
}
```

## Important Path Corrections

### ✅ Correct Python Path
```
C:\Users\<username>\miniforge3\envs\ragdoc-env\python.exe
```

### ❌ Common Mistake
```
C:\Users\<username>\miniforge3\envs\ragdoc-env\Scripts\python.exe
```

**Note**: In conda environments, `python.exe` is at the root, NOT in `Scripts\`!

## Environment Variables

- `CHROMA_DB_PATH`: Path to ChromaDB database directory
- `COLLECTION_NAME`: Name of the collection (use `zotero_research_context_hybrid_v3` for hybrid search)

## Verification

After restarting Claude Desktop:

1. Type `/mcp` in Claude Desktop
2. Verify `ragdoc` shows as `✓ connected`
3. Available tools:
   - `semantic_search_hybrid` - Hybrid search with BM25 + Semantic
   - `list_documents` - List all indexed documents
   - `get_indexation_status` - Database statistics

## Troubleshooting

### "Failed to reconnect to ragdoc"

**Cause**: Incorrect Python path

**Solution**:
1. Verify conda environment exists: `conda env list`
2. Check Python path: `where python` (when ragdoc-env is activated)
3. Update `.claude.json` with correct path
4. Restart Claude Desktop

### "Command not found"

**Cause**: Python path has backslashes instead of forward slashes, or wrong directory

**Solution**: Use double backslashes `\\` in JSON paths for Windows

## Server Features

The RAGDOC MCP server includes:

- **Hybrid Search**: BM25 (lexical) + Voyage-3-Large (semantic)
- **Reciprocal Rank Fusion**: Combines BM25 and semantic rankings
- **Cohere v3.5 Reranking**: Final relevance refinement
- **Context Window Expansion**: Shows adjacent chunks for context
- **24,884+ Indexed Chunks**: Full glacier research corpus

## Testing the Connection

Test the server manually:
```bash
cd "D:\Claude Code\ragdoc-mcp"
C:\Users\<username>\miniforge3\envs\ragdoc-env\python.exe src/server.py
```

**Note:** On recent versions we run with `show_banner=False` to keep stdout strictly JSON-RPC (some MCP clients disconnect if any banner/text is printed on stdout).
So you may see **no banner output** — that's expected.

## Performance

- **Indexation**: 24,884 chunks across 100+ research papers
- **Search Speed**: ~2-3 seconds for hybrid search + reranking
- **Alpha Parameter**: 0.7 (70% semantic, 30% BM25) - adjustable
- **Quality**: +67% diversity improvement vs semantic-only search
