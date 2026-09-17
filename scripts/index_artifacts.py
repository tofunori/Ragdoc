#!/usr/bin/env python3
"""Incrementally index MinerU visual artifacts copied by Ragdrop."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.artifacts import ArtifactIndex
from src.config import ARTIFACTS_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", dest="sources")
    args = parser.parse_args()
    result = ArtifactIndex(ARTIFACTS_PATH).index(set(args.sources) if args.sources else None)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
