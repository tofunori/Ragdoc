#!/usr/bin/env python3
"""Export exact and conservative fuzzy duplicate candidates for human review."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from fastmcp import Client


STOPWORDS = {
    "a", "an", "and", "article", "de", "des", "du", "et", "for", "in", "la", "le",
    "les", "of", "on", "part", "the", "to", "un", "une", "with", "without", "using",
}


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char)).casefold()
    return {
        token for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 3 and token not in STOPWORDS and not re.fullmatch(r"(?:19|20)\d{2}", token)
    }


def _source_features(source: str) -> tuple[set[str], int | None]:
    stem = Path(source.removeprefix("._")).stem
    year_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", stem)
    return _tokens(stem), int(year_match.group(1)) if year_match else None


def _preferred_document(documents: list[dict]) -> dict:
    def score(document: dict) -> tuple:
        bibliography = document.get("bibliography", {})
        source = document.get("source", "")
        return (
            bool(bibliography.get("doi")),
            bool(bibliography.get("title")),
            bool(bibliography.get("authors")),
            bibliography.get("year") is not None,
            source.casefold().startswith("zotero_"),
            not source.startswith("._"),
            source,
        )
    return max(documents, key=score)


def exact_rows(documents: list[dict], duplicate_groups: list[dict]) -> tuple[list[dict], set[frozenset[str]]]:
    by_source = {document["source"]: document for document in documents}
    sources_by_key: dict[str, set[str]] = defaultdict(set)
    for group in duplicate_groups:
        key = group.get("key", "")
        sources_by_key[key].update(group.get("sources", []))
    keys_by_sources: dict[frozenset[str], list[str]] = defaultdict(list)
    for key, key_sources in sources_by_key.items():
        sources = frozenset(key_sources)
        if len(sources) > 1:
            keys_by_sources[sources].append(key)

    rows = []
    exact_pairs: set[frozenset[str]] = set()
    for source in sorted(by_source):
        if not source.startswith("._") or source.removeprefix("._") not in by_source:
            continue
        preferred = source.removeprefix("._")
        pair = frozenset({source, preferred})
        exact_pairs.add(pair)
        rows.append({
            "candidate_source": source,
            "preferred_source": preferred,
            "confidence": "high",
            "match_basis": "macos_dot_underscore_filename",
            "shared_identifier": preferred,
            "candidate_year": "",
            "preferred_year": by_source[preferred].get("bibliography", {}).get("year") or "",
            "preferred_title": by_source[preferred].get("bibliography", {}).get("title") or "",
            "title_token_overlap": "",
            "ambiguous_candidates": "",
            "proposed_action": "remove_dot_underscore_after_original_check",
        })
    for sources, keys in sorted(keys_by_sources.items(), key=lambda item: sorted(item[0])):
        group_documents = [by_source[source] for source in sources if source in by_source]
        if len(group_documents) < 2:
            continue
        preferred = _preferred_document(group_documents)
        for candidate in sorted(group_documents, key=lambda item: item["source"]):
            if candidate["source"] == preferred["source"]:
                continue
            pair = frozenset({candidate["source"], preferred["source"]})
            if pair in exact_pairs:
                continue
            exact_pairs.add(pair)
            basis = "+".join(sorted(key.split(":", 1)[0] for key in keys))
            rows.append({
                "candidate_source": candidate["source"],
                "preferred_source": preferred["source"],
                "confidence": "high",
                "match_basis": basis,
                "shared_identifier": " | ".join(keys),
                "candidate_year": candidate.get("bibliography", {}).get("year") or "",
                "preferred_year": preferred.get("bibliography", {}).get("year") or "",
                "preferred_title": preferred.get("bibliography", {}).get("title") or "",
                "title_token_overlap": "",
                "ambiguous_candidates": "",
                "proposed_action": "review_then_archive_nonpreferred",
            })
    return rows, exact_pairs


def fuzzy_rows(documents: list[dict], exact_pairs: set[frozenset[str]]) -> list[dict]:
    rich_documents = [
        document for document in documents
        if document.get("bibliography", {}).get("title")
        and document.get("bibliography", {}).get("authors")
        and document.get("bibliography", {}).get("year")
    ]
    legacy_documents = [
        document for document in documents
        if not document["source"].casefold().startswith("zotero_")
        and not document.get("bibliography", {}).get("doi")
    ]

    rows = []
    for legacy in sorted(legacy_documents, key=lambda item: item["source"]):
        source = legacy["source"]
        source_tokens, source_year = _source_features(source)
        if source_year is None or not source_tokens:
            continue
        ranked = []
        for rich in rich_documents:
            bibliography = rich["bibliography"]
            if bibliography["year"] != source_year or rich["source"] == source:
                continue
            title_tokens = _tokens(bibliography["title"])
            overlap = len(source_tokens & title_tokens)
            surnames = {
                next(iter(reversed(author.split())), "").casefold()
                for author in bibliography.get("authors", []) if author.split()
            }
            author_match = bool(source_tokens & surnames)
            if overlap < 2 and not (author_match and overlap >= 1):
                continue
            union = len(source_tokens | title_tokens) or 1
            score = overlap / union + (0.35 if author_match else 0.0) + min(overlap, 5) * 0.04
            ranked.append((score, overlap, author_match, rich))
        if not ranked:
            continue
        ranked.sort(key=lambda item: (-item[0], -item[1], item[3]["source"]))
        best_score, overlap, author_match, preferred = ranked[0]
        pair = frozenset({source, preferred["source"]})
        if pair in exact_pairs:
            continue
        close = [item for item in ranked[1:] if best_score - item[0] <= 0.08]
        if close:
            confidence = "review"
        elif author_match and overlap >= 2:
            confidence = "medium"
        elif overlap >= 4:
            confidence = "medium"
        else:
            continue
        action = "remove_dot_underscore_after_original_check" if source.startswith("._") else "review_then_archive_legacy"
        rows.append({
            "candidate_source": source,
            "preferred_source": preferred["source"],
            "confidence": confidence,
            "match_basis": "filename_author_year_title" if author_match else "filename_year_title",
            "shared_identifier": "",
            "candidate_year": source_year,
            "preferred_year": preferred["bibliography"]["year"],
            "preferred_title": preferred["bibliography"]["title"],
            "title_token_overlap": overlap,
            "ambiguous_candidates": " | ".join(item[3]["source"] for item in close),
            "proposed_action": action,
        })
    return rows


async def _catalogue(client: Client) -> list[dict]:
    documents = []
    offset = 0
    while True:
        page = (await client.call_tool("search_documents", {"offset": offset, "limit": 100})).structured_content
        documents.extend(page["documents"])
        if page["next_offset"] is None:
            return documents
        offset = page["next_offset"]


async def _duplicates(client: Client) -> list[dict]:
    try:
        first = (await client.call_tool(
            "audit_library", {"view": "duplicates", "limit": 100}
        )).structured_content
    except Exception as exc:
        message = str(exc).casefold()
        legacy_schema_error = (
            "unexpected keyword argument" in message
            or ("validation" in message and "view" in message)
        )
        if not legacy_schema_error:
            raise
        first = (await client.call_tool("audit_library", {})).structured_content
    if "items" not in first:  # Compatibility with Ragdoc before bounded audit views.
        if "candidate_duplicates" not in first:
            raise RuntimeError("audit_library returned neither paginated nor legacy duplicate details")
        return first["candidate_duplicates"]
    groups = list(first["items"])
    offset = first.get("next_offset")
    while offset is not None:
        page = (await client.call_tool(
            "audit_library", {"view": "duplicates", "offset": offset, "limit": 100}
        )).structured_content
        groups.extend(page["items"])
        offset = page.get("next_offset")
    return groups


async def export(url: str, output: Path) -> dict:
    async with Client(url) as client:
        documents = await _catalogue(client)
        duplicate_groups = await _duplicates(client)
    rows, exact_pairs = exact_rows(documents, duplicate_groups)
    rows.extend(fuzzy_rows(documents, exact_pairs))
    confidence_order = {"high": 0, "medium": 1, "review": 2}
    rows.sort(key=lambda row: (confidence_order[row["confidence"]], row["candidate_source"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "candidate_source", "preferred_source", "confidence", "match_basis", "shared_identifier",
        "candidate_year", "preferred_year", "preferred_title", "title_token_overlap",
        "ambiguous_candidates", "proposed_action",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "documents": len(documents),
        "exact_duplicate_groups": len({group.get("key") for group in duplicate_groups}),
        "review_rows": len(rows),
        "confidence_counts": {
            confidence: sum(row["confidence"] == confidence for row in rows)
            for confidence in confidence_order
        },
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://rorqual.tail02163.ts.net:8484/mcp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(export(args.url, args.output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
