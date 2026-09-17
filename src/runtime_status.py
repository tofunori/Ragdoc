"""Offline capability checks: no database access or external API requests."""
import importlib
from importlib.metadata import PackageNotFoundError, version
import os
import sys


def runtime_status():
    versions = {}
    for name in ("fastmcp", "chromadb", "voyageai", "cohere", "chonkie", "nltk"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    issues = []
    try:
        voyage = importlib.import_module("voyageai")
        contextualized = hasattr(voyage.Client, "contextualized_embed")
    except ImportError:
        contextualized = False
    if not contextualized:
        issues.append("voyage_contextualized_api_unavailable")
    try:
        from src.bm25_tokenizers.advanced_tokenizer import AdvancedTokenizer
        AdvancedTokenizer()
        tokenizer = "advanced"
    except (ImportError, LookupError):
        tokenizer = "simple"
        issues.append("advanced_tokenizer_unavailable: install nltk and its stopwords data")
    return {"python": sys.executable, "versions": versions,
            "voyage_contextualized_api": contextualized, "tokenizer": tokenizer,
            "api_keys_configured": {"voyage": bool(os.getenv("VOYAGE_API_KEY")),
                                    "cohere": bool(os.getenv("COHERE_API_KEY"))},
            "issues": issues, "offline_check_only": True}
