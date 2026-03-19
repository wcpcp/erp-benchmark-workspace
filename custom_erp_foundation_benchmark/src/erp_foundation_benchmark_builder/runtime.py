from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_GEN_SRC = REPO_ROOT / "data_generation" / "src"

if str(DATA_GEN_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_SRC))
