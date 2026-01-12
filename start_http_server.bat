@echo off
echo ============================================================
echo RAGDOC MCP HTTP SERVER
echo ============================================================
echo.
set PYTHONUNBUFFERED=1
set CHROMA_DB_PATH=D:/Claude Code/ragdoc-mcp/chroma_db_new
set COLLECTION_NAME=zotero_research_context_hybrid_v3
set VOYAGE_API_KEY=pa-XZ_Sb--dxVbRgc1XMssjlBoHO-wpgMQjBl6G0x36Gq2
set COHERE_API_KEY=1OepbeLBADNAZdLPcwDdc5IqDQXm4Hn6p6HoJi2V
cd /d "D:\Claude Code\ragdoc-mcp"
"C:\Users\thier\miniforge3\envs\ragdoc-env\python.exe" src/server_http.py
pause
