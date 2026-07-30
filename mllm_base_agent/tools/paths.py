"""Repository/output path resolution shared by SpatialWorld tools.

Mirrors the small path-resolution helpers used by other agentic tool
implementations (e.g. RPent's ``rpent/utils/config.py`` /
``rpent/utils/logging.py``) but scoped to what SpatialWorld actually needs:
resolving relative paths against a repo root and knowing the "current run"
output directory so tools default to something sensible without an explicit
``path`` argument.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# mllm_base_agent/tools/paths.py -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_output_dir: Optional[Path] = None


def get_repo_root() -> Path:
    """Return the SpatialWorld repository root directory.

    Resolution order: ``SPATIALWORLD_REPO_ROOT`` env var, then the parent of
    the ``mllm_base_agent/`` package directory.
    """
    env = os.environ.get("SPATIALWORLD_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _REPO_ROOT


def set_output_dir(path: Optional[str]) -> Optional[Path]:
    """Record the current run's output directory for tool defaults.

    Called by runners/scripts once ``run_output_dir`` is known so that
    ``list_dir``/``glob_search`` etc. can default to "this run's" directory
    instead of requiring every call site to pass an explicit path.
    """
    global _output_dir
    _output_dir = Path(path).expanduser().resolve() if path else None
    return _output_dir


def get_output_dir() -> Optional[Path]:
    """Return the output directory set by the last :func:`set_output_dir` call."""
    return _output_dir


def resolve_path(path: str) -> Path:
    """Resolve *path* against the repo root when it is not already absolute."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = get_repo_root() / p
    return p


__all__ = [
    "get_repo_root",
    "get_output_dir",
    "set_output_dir",
    "resolve_path",
]
