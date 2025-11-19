#!/usr/bin/env python3
"""
Configuration centralisée pour l'indexation incrémentale Chroma
DEPRECATED: This file is kept for backward compatibility.
Please use src.config instead.
"""

import sys
from pathlib import Path

# Add src to path to import new config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *

# Re-export all variables from config
# This ensures that 'from indexing_config import *' still works
