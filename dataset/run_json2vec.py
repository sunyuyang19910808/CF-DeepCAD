"""
Entry point for json2vec.

- Default / no --one: same as ``cd dataset && python json2vec.py`` (full train/val/test).
- ``--one DATA_ID``: only process one sample (e.g. ``0000/00000007`` for ``00000007.json``).
  Loads json2vec as a module (no Parallel), so breakpoints in ``process_one`` always hit.

Debugging notes
---------------
- Use ``--one 0000/00000007`` with launch config **Python: run_json2vec (single 00000007)**.
- Full run still uses joblib Parallel(n_jobs=10); use direct ``python json2vec.py`` or this script
  without ``--one`` for that path.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path


def _load_json2vec_module(here: Path):
    """Load json2vec.py without executing its ``if __name__ == '__main__'`` block."""
    path = here / "json2vec.py"
    spec = importlib.util.spec_from_file_location("deepcad_json2vec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_json2vec(script: Path) -> None:
    """Loads and executes json2vec.py as __main__ (same process)."""
    runpy.run_path(str(script), run_name="__main__")


def main() -> None:
    here = Path(__file__).resolve().parent
    os.chdir(here)

    parser = argparse.ArgumentParser(description="Run json2vec (full split or single id).")
    parser.add_argument(
        "--one",
        metavar="DATA_ID",
        help="Single sample id as in split file, e.g. 0000/00000007",
    )
    args, _unknown = parser.parse_known_args()

    if args.one:
        j2v = _load_json2vec_module(here)
        j2v.process_one(args.one)
        return

    script = here / "json2vec.py"
    sys.argv = [str(script)]
    _run_json2vec(script)


if __name__ == "__main__":
    main()
