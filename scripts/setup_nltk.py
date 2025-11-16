#!/usr/bin/env python3
"""
Download required NLTK data for advanced tokenization.

This script downloads:
- stopwords: For stopwords removal
- punkt: For sentence tokenization (if needed in future)

Usage:
    python scripts/setup_nltk.py
"""

import nltk
import sys


def download_nltk_data():
    """Download required NLTK data packages."""
    packages = [
        ('stopwords', 'Stopwords corpus'),
        ('punkt', 'Punkt tokenizer'),
    ]

    print("=" * 70)
    print("NLTK Data Setup for RAGDOC Advanced Tokenization")
    print("=" * 70)
    print()

    all_success = True

    for package_name, description in packages:
        try:
            print(f"Downloading {description} ({package_name})...", end=" ")
            nltk.download(package_name, quiet=True)
            print("✓ Success")
        except Exception as e:
            print(f"✗ Failed: {e}")
            all_success = False

    print()
    print("=" * 70)

    if all_success:
        print("✓ All NLTK data downloaded successfully!")
        print()
        print("Advanced tokenization is now ready to use.")
        print("=" * 70)
        return 0
    else:
        print("✗ Some downloads failed. Please check your internet connection")
        print("  and try again.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(download_nltk_data())
