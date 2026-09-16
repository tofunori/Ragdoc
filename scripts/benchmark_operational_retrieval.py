#!/usr/bin/env python3
"""Measure live MCP retrieval on deterministic catalogue-derived probes.

This is an operational recognition benchmark.  It verifies that known article
identifiers and titles resolve to their catalogue source; it does not replace a
human-reviewed benchmark of independent scientific questions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import Client


async def _catalogue(client: Client) -> list[dict]:
    documents = []
    offset = 0
    while True:
        result = await client.call_tool("search_documents", {"offset": offset, "limit": 100})
        page = result.structured_content
        documents.extend(page["documents"])
        if page["next_offset"] is None:
            return documents
        offset = page["next_offset"]


def _sample(documents: list[dict], field: str, count: int, seed: int) -> list[dict]:
    candidates = [
        document for document in documents
        if isinstance(document["bibliography"].get(field), str)
        and len(document["bibliography"][field].strip()) >= 8
    ]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} catalogue entries have a usable {field}")
    return random.Random(seed).sample(sorted(candidates, key=lambda row: row["source"]), count)


def _summary(rows: list[dict]) -> dict:
    ranks = [row["rank"] for row in rows]
    return {
        "queries": len(rows),
        "hit_at_1": sum(rank == 1 for rank in ranks) / len(rows),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(rows),
        "mrr_at_5": sum(0 if rank is None else 1 / rank for rank in ranks) / len(rows),
        "rerank_active_rate": sum(row["reranking"] == "cohere" for row in rows) / len(rows),
        "warning_queries": sum(bool(row["warnings"]) for row in rows),
        "retrieval_modes": dict(Counter(row["retrieval_mode"] for row in rows)),
    }


async def benchmark(url: str, count_per_kind: int, seed: int) -> dict:
    async with Client(url) as client:
        status = (await client.call_tool("get_server_status", {})).structured_content
        documents = await _catalogue(client)
        probes = []
        for document in _sample(documents, "doi", count_per_kind, seed):
            probes.append(("doi_lexical", document["bibliography"]["doi"], 0.0, document))
        title_documents = _sample(documents, "title", count_per_kind, seed + 1)
        for document in title_documents:
            probes.append(("title_lexical", document["bibliography"]["title"], 0.0, document))
        for document in title_documents:
            probes.append(("title_semantic", document["bibliography"]["title"], 1.0, document))

        rows = []
        for kind, query, alpha, expected in probes:
            result = await client.call_tool("search_evidence", {
                "query": query,
                "top_k": 5,
                "alpha": alpha,
                "multi_query": False,
            })
            data = result.structured_content
            sources = [hit["provenance"]["source"] for hit in data["hits"]]
            rank = sources.index(expected["source"]) + 1 if expected["source"] in sources else None
            rows.append({
                "kind": kind,
                "query": query,
                "expected_source": expected["source"],
                "rank": rank,
                "returned_sources": sources,
                "retrieval_mode": data["retrieval"][0]["mode"] if data["retrieval"] else "none",
                "reranking": data["reranking"],
                "warnings": data["warnings"],
            })

    by_kind = {
        kind: _summary([row for row in rows if row["kind"] == kind])
        for kind in sorted({row["kind"] for row in rows})
    }
    canonical = json.dumps(probes, ensure_ascii=False, default=str, sort_keys=True)
    return {
        "name": "Ragdoc operational retrieval recognition",
        "scope": (
            "Catalogue-derived DOI, lexical-title and semantic-title recognition through the live MCP. "
            "This does not measure independent scientific-question relevance."
        ),
        "at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "probe_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "server_status": status,
        "summary": {"overall": _summary(rows), **by_kind},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://rorqual.tail02163.ts.net:8484/mcp")
    parser.add_argument("--count-per-kind", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(benchmark(args.url, args.count_per_kind, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
