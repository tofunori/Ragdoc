"""Deterministic Markdown section structure and canonical context extraction."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass


STRUCTURE_VERSION = "markdown-sections-v1"
SECTION_TYPES = (
    "abstract", "introduction", "methods", "results", "discussion",
    "conclusion", "references", "supplementary", "acknowledgements",
)


@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    level: int
    path: tuple[str, ...]
    start: int
    content_start: int
    end: int
    types: tuple[str, ...]


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    plain = " ".join(re.sub(r"[^a-z0-9]+", " ", normalized.encode("ascii", "ignore").decode()).split())
    tokens = plain.split()
    if len(tokens) >= 4 and all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    return plain


def classify_section(title: str) -> tuple[str, ...]:
    """Map explicit heading words to broad scientific section types."""
    value = _plain(title)
    matches: list[str] = []
    rules = {
        "abstract": (r"\babstract\b", r"\bresume\b"),
        "introduction": (r"\bintroduction\b", r"\bbackground\b", r"\bcontexte\b"),
        "methods": (r"\bmethods?\b", r"\bmethodology\b", r"\bmethodologie\b", r"\bmateriels? et methodes?\b",
                    r"\bdata and methods?\b", r"\bexperimental setup\b"),
        "results": (r"\bresults?\b", r"\bresultats?\b", r"\bfindings?\b"),
        "discussion": (r"\bdiscussion\b", r"\binterpretation\b"),
        "conclusion": (r"\bconclusions?\b", r"\bsummary and conclusions?\b"),
        "references": (r"\breferences?\b", r"\bbibliograph(?:y|ie)\b", r"\bliterature cited\b"),
        "supplementary": (r"\bsupplement(?:ary|al)?\b", r"\bappendi(?:x|ces)\b", r"\bannexes?\b"),
        "acknowledgements": (r"\backnowledg(?:e)?ments?\b", r"\bremerciements?\b"),
    }
    for section_type, patterns in rules.items():
        if any(re.search(pattern, value) for pattern in patterns):
            matches.append(section_type)
    return tuple(matches)


def parse_sections(markdown: str) -> list[Section]:
    """Parse ATX and Setext headings while ignoring fenced code blocks."""
    lines = list(re.finditer(r".*(?:\n|\Z)", markdown))
    headings: list[tuple[int, int, int, str]] = []
    fence: str | None = None
    index = 0
    while index < len(lines):
        match = lines[index]
        raw = match.group(0)
        line = raw.rstrip("\r\n")
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            index += 1
            continue
        if fence is not None:
            index += 1
            continue

        atx = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if atx:
            title = atx.group(2).strip()
            if title:
                headings.append((match.start(), match.end(), len(atx.group(1)), title))
            index += 1
            continue

        if index + 1 < len(lines) and line.strip():
            underline = lines[index + 1].group(0).rstrip("\r\n")
            setext = re.match(r"^\s{0,3}(=+|-+)\s*$", underline)
            if setext:
                level = 1 if setext.group(1).startswith("=") else 2
                headings.append((match.start(), lines[index + 1].end(), level, line.strip()))
                index += 2
                continue
        index += 1

    stack: list[tuple[int, str, tuple[str, ...]]] = []
    sections: list[Section] = []
    canonical_digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    for position, (start, content_start, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        direct_types = classify_section(title)
        first_heading_is_title = (
            position == 0 and level == 1 and not markdown[:start].strip()
            and len(headings) > 1 and len(_plain(title).split()) >= 5
        )
        if first_heading_is_title:
            effective_types: tuple[str, ...] = ()
        elif direct_types:
            effective_types = direct_types
        else:
            effective_types = next((item[2] for item in reversed(stack) if item[2]), ())
        stack.append((level, title, effective_types))
        path = tuple(item[1] for item in stack)
        end = headings[position + 1][0] if position + 1 < len(headings) else len(markdown)
        raw_id = f"{canonical_digest}:{start}:{level}:{title}".encode("utf-8")
        sections.append(Section(
            section_id=hashlib.sha256(raw_id).hexdigest()[:16], title=title, level=level,
            path=path, start=start, content_start=content_start, end=end,
            types=effective_types,
        ))
    return sections


def section_metadata(markdown: str, start: int, end: int,
                     sections: list[Section] | None = None) -> dict:
    """Return scalar Chroma metadata for every section intersecting a passage."""
    if not 0 <= start < end <= len(markdown):
        raise ValueError("Passage bounds are outside canonical Markdown")
    overlaps = []
    for section in sections if sections is not None else parse_sections(markdown):
        size = max(0, min(end, section.end) - max(start, section.start))
        if size:
            overlaps.append((size, section))
    if not overlaps:
        return {"structure_version": STRUCTURE_VERSION}
    overlaps.sort(key=lambda item: (-item[0], item[1].start))
    primary = overlaps[0][1]
    types = sorted({section_type for _, section in overlaps for section_type in section.types})
    metadata = {
        "section": primary.title,
        "section_id": primary.section_id,
        "section_level": primary.level,
        "section_path": " > ".join(primary.path),
        "section_start": primary.start,
        "section_end": primary.end,
        "section_types_json": json.dumps(types, separators=(",", ":")),
        "section_overlap": len(overlaps) > 1,
        "structure_version": STRUCTURE_VERSION,
    }
    for section_type in SECTION_TYPES:
        metadata[f"section_is_{section_type}"] = section_type in types
    return metadata


def paragraph_context(markdown: str, start: int, end: int, before: int, after: int) -> tuple[int, int]:
    """Find paragraph-aligned context bounds around an exact passage."""
    if before < 0 or after < 0:
        raise ValueError("Paragraph context counts must be non-negative")
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r"(?s)(?:^|(?<=\n\n))\s*\S.*?(?=\n\s*\n|\Z)", markdown)]
    touched = [i for i, (left, right) in enumerate(paragraphs) if left < end and right > start]
    if not touched:
        return start, end
    first = max(0, touched[0] - before)
    last = min(len(paragraphs) - 1, touched[-1] + after)
    return paragraphs[first][0], paragraphs[last][1]


def bounded_context(markdown: str, passage_start: int, passage_end: int, *, mode: str,
                    paragraphs_before: int, paragraphs_after: int,
                    max_chars: int) -> dict | None:
    """Extract canonical context while guaranteeing that the passage remains included."""
    if mode == "none":
        return None
    if mode == "paragraphs":
        context_start, context_end = paragraph_context(
            markdown, passage_start, passage_end, paragraphs_before, paragraphs_after
        )
    elif mode == "section":
        intersecting = [section for section in parse_sections(markdown)
                        if section.start < passage_end and section.end > passage_start]
        if not intersecting:
            context_start, context_end = paragraph_context(
                markdown, passage_start, passage_end, paragraphs_before, paragraphs_after
            )
        else:
            context_start = min(passage_start, *(section.start for section in intersecting))
            context_end = max(passage_end, *(section.end for section in intersecting))
    else:
        raise ValueError("context must be one of: none, paragraphs, section")

    if max_chars < passage_end - passage_start:
        raise ValueError("max_context_chars is smaller than the requested passage")
    truncated = context_end - context_start > max_chars
    if truncated:
        spare = max_chars - (passage_end - passage_start)
        left = min(passage_start - context_start, spare // 2)
        right = min(context_end - passage_end, spare - left)
        remaining = spare - left - right
        if remaining:
            extra_left = min(passage_start - context_start - left, remaining)
            left += extra_left
            right += min(context_end - passage_end - right, remaining - extra_left)
        context_start, context_end = passage_start - left, passage_end + right

    return {
        "text": markdown[context_start:context_end],
        "char_start": context_start,
        "char_end": context_end,
        "passage_start": passage_start,
        "passage_end": passage_end,
        "truncated": truncated,
    }
