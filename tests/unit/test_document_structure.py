import json

from src.document_structure import bounded_context, parse_sections, section_metadata


def test_sections_preserve_hierarchy_setext_and_ignore_fenced_headings():
    text = (
        "Article title\n=============\n\n# Methods\nText.\n\n"
        "```python\n# Not a heading\n```\n\n## Sampling\nSample text.\n\n"
        "# Results and Discussion\nFinding."
    )
    sections = parse_sections(text)
    assert [section.title for section in sections] == [
        "Article title", "Methods", "Sampling", "Results and Discussion"
    ]
    assert sections[2].path == ("Methods", "Sampling")
    assert sections[2].types == ("methods",)
    assert sections[-1].types == ("results", "discussion")
    assert all(section.start < section.end for section in sections)


def test_article_title_does_not_contaminate_explicit_sections():
    text = (
        "# Assessment of methods for mapping snow albedo from MODIS\n\n"
        "## A B S T R A C T\nSummary.\n\n## I N T R O D U C T I O N\nBackground.\n\n"
        "## Methodology\nApproach.\n\n### Study area\nSite.\n\n## Results\nFinding."
    )
    sections = parse_sections(text)
    by_title = {section.title: section for section in sections}
    assert by_title["A B S T R A C T"].types == ("abstract",)
    assert by_title["I N T R O D U C T I O N"].types == ("introduction",)
    assert by_title["Methodology"].types == ("methods",)
    assert by_title["Study area"].types == ("methods",)
    assert by_title["Results"].types == ("results",)


def test_section_ids_are_pinned_to_canonical_document_version():
    first = parse_sections("# Results\nA")[0]
    second = parse_sections("# Results\nB")[0]
    assert first.start == second.start
    assert first.section_id != second.section_id


def test_chunk_metadata_records_all_intersected_section_types():
    text = "# Methods\nMethod.\n\n# Results\nFinding."
    start = text.index("Method")
    end = len(text)
    metadata = section_metadata(text, start, end)
    assert json.loads(metadata["section_types_json"]) == ["methods", "results"]
    assert metadata["section_is_methods"] is True
    assert metadata["section_is_results"] is True
    assert metadata["section_overlap"] is True


def test_canonical_context_keeps_exact_passage_when_truncated():
    text = "# Results\n\nFirst paragraph.\n\nTarget evidence here.\n\nLast paragraph."
    start = text.index("Target")
    end = start + len("Target evidence here.")
    paragraphs = bounded_context(
        text, start, end, mode="paragraphs", paragraphs_before=1,
        paragraphs_after=1, max_chars=40,
    )
    assert paragraphs["text"][start - paragraphs["char_start"]:end - paragraphs["char_start"]] == text[start:end]
    assert paragraphs["truncated"] is True
    section = bounded_context(
        text, start, end, mode="section", paragraphs_before=0,
        paragraphs_after=0, max_chars=len(text),
    )
    assert section["text"].startswith("# Results")


def test_section_context_contains_chunk_that_crosses_heading_boundary():
    text = "# Methods\nMethod tail.\n\n# Results\nFinding."
    start = text.index("Method tail")
    end = len(text)
    context = bounded_context(
        text, start, end, mode="section", paragraphs_before=0,
        paragraphs_after=0, max_chars=len(text),
    )
    relative_start = start - context["char_start"]
    relative_end = end - context["char_start"]
    assert context["text"][relative_start:relative_end] == text[start:end]


def test_section_context_keeps_preamble_before_first_heading():
    text = "Preamble evidence.\n\n# Results\nFinding."
    start = 0
    end = len(text)
    context = bounded_context(
        text, start, end, mode="section", paragraphs_before=0,
        paragraphs_after=0, max_chars=len(text),
    )
    assert context["text"] == text
