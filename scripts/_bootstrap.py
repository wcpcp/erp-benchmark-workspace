from __future__ import annotations

import importlib.util
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

    spec = importlib.util.spec_from_file_location(
        "erp_benchmarks",
        package_init,
        submodule_search_locations=[str(src / "erp_benchmarks")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for local package: {package_init}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["erp_benchmarks"] = module
    spec.loader.exec_module(module)

    return root, src
