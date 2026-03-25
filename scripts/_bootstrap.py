from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_local_src(script_path: str) -> tuple[Path, Path]:
    root = Path(script_path).resolve().parents[1]
    src = root / "src"
    package_init = src / "erp_benchmarks" / "__init__.py"
    if not package_init.exists():
        raise RuntimeError(f"Local package not found: {package_init}")

    src_str = str(src)
    sys.path = [src_str] + [entry for entry in sys.path if entry != src_str]

    # Remove preloaded modules with the same top-level name to avoid mixing an
    # installed package with this workspace checkout.
    for name in list(sys.modules):
        if name == "erp_benchmarks" or name.startswith("erp_benchmarks."):
            sys.modules.pop(name, None)

    return root, src
