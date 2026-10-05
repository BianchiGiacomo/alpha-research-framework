"""
Machine-generated result files.

Every number quoted in the documentation should come from a file written
here, together with the configuration, data hashes, code version and
library versions that produced it.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def environment() -> Dict[str, str]:
    import pandas, scipy, sklearn  # noqa: E401
    from . import __version__
    return {
        "alpharesearch": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def save_results(name: str, payload: Dict[str, Any], out_dir: str = "results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "study": name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        **payload,
    }
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(_jsonable(doc), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path
