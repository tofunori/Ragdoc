#!/bin/bash
set -euo pipefail

root=/volume1/Services/mcp/ragdoc
python="$root/ragdoc-env-new/bin/python3"

cd "$root"
set -a
source "$root/.env"
set +a

export CHROMA_DB_PATH="$root/chroma_db_new"
export RAGDOC_LIBRARY_DIR="$root/ragdoc_library/ragdoc_contextualized_v1"
export COLLECTION_NAME=ragdoc_contextualized_v1
export RAGDOC_TRANSPORT=http
export RAGDOC_PORT=8484
export RAGDOC_EMBEDDING_MODEL=voyage-context-4

exec "$python" "$root/ragdoc-launch.py"
