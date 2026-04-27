"""Console-script entry points for the egrefine package.

`egrefine-refine` and `egrefine-eval` are exposed via [project.scripts] in
pyproject.toml. They locate and execute the user-facing scripts/run_*.py
files using runpy. Editable install (`pip install -e .`) is the supported
installation mode — non-editable wheels would need scripts/ shipped as
package data, which is intentionally out of scope for v0.1.0.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _scripts_dir() -> Path:
    """Locate the repository's scripts/ directory by walking up from this file."""
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "scripts"
        if (candidate / "run_refine.py").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate scripts/ directory. EGRefine v0.1.0 supports editable installs only "
        "(`pip install -e .` from the repository root)."
    )


def refine_main() -> None:
    """Entry point for the `egrefine-refine` console script."""
    script = _scripts_dir() / "run_refine.py"
    sys.argv[0] = "egrefine-refine"
    runpy.run_path(str(script), run_name="__main__")


def eval_main() -> None:
    """Entry point for the `egrefine-eval` console script."""
    script = _scripts_dir() / "run_eval.py"
    sys.argv[0] = "egrefine-eval"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    raise SystemExit("Use `egrefine-refine` or `egrefine-eval` after `pip install -e .`")
