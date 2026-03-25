#!/usr/bin/env python3

from _bootstrap import bootstrap_local_src

ROOT, SRC = bootstrap_local_src(__file__)

from erp_benchmarks.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
