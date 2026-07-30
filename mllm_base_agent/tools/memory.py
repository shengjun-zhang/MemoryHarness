"""Generic skill-memory library reader, mirroring RPent's memory pattern.

RPent (see ``rpent/utils/resources.py`` + ``robots/libero/guides/*``) keeps a
reviewed, mostly-static knowledge base per environment: a ``MEMORY.md`` index
file plus a handful of leaf ``feedback_*.md`` notes, read by the agent at the
start of a run and consulted on demand via generic file tools
(``read_text_file`` / ``list_dir``) rather than a bespoke retrieval API.

This module provides that same capability for SpatialWorld's multi-agent
environments, built entirely on top of the existing common tool layer
(:mod:`mllm_base_agent.tools.common`) so it needs no new dependency and no
new tool-calling protocol: it just resolves a "memory root" directory (e.g.
``mllm_base_agent/dual_agent/ai2thor/core/memory/``) and re-uses
``read_text_file`` / ``list_dir`` / ``search_in_files`` scoped to it.

Usage from a runner/loop::

    from mllm_base_agent.tools.memory import MemoryLibrary

    memory = MemoryLibrary.for_env("ai2thor", agent_mode="dual")
    index_text = memory.read_index()          # MEMORY.md contents (or "").
    result = memory.read_entry("feedback_blocking_agents.md")
    # -> {"path": ..., "content": ..., ...} or {"error": ...}

Any tool-calling agent that already uses :class:`mllm_base_agent.tools.toolkit.Toolkit`
can instead call the plain ``read_text_file`` / ``list_dir`` tools directly
with an absolute path returned by :meth:`MemoryLibrary.root_dir` /
:meth:`MemoryLibrary.index_path` -- ``MemoryLibrary`` itself is just a thin,
convenience-oriented wrapper for callers that are not going through the full
Toolkit tool-calling loop (e.g. the AI2-THOR dual-agent text-protocol loop in
``mllm_base_agent/dual_agent/ai2thor/main.py``, which parses a lightweight
``<ACTION>`` grammar rather than OpenAI-style function calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from mllm_base_agent.tools.common import list_dir as _list_dir
from mllm_base_agent.tools.common import read_text_file as _read_text_file
from mllm_base_agent.tools.common import search_in_files as _search_in_files
from mllm_base_agent.tools.paths import get_repo_root

DEFAULT_INDEX_FILENAME = "MEMORY.md"
DEFAULT_MAX_CHARS = 20000


def _default_memory_root(env: str, agent_mode: str = "dual") -> Path:
    """Best-effort default memory root for a given env/agent_mode.

    Layout convention: ``mllm_base_agent/<agent_mode>_agent/<env>/core/memory/``
    (this is where the AI2-THOR dual-agent memory library lives today). Callers
    with a different layout should pass an explicit ``root_dir`` to
    :class:`MemoryLibrary` instead of relying on this helper.
    """
    repo_root = get_repo_root()
    return repo_root / "mllm_base_agent" / f"{agent_mode}_agent" / env / "core" / "memory"


@dataclass
class MemoryLibrary:
    """Thin convenience wrapper around the common file tools, scoped to a memory dir.

    Args:
        root_dir: Absolute path to the memory directory (contains ``MEMORY.md``
            plus leaf note files). Use :meth:`for_env` to derive this from an
            environment name using the standard SpatialWorld layout.
        index_filename: Name of the index file inside ``root_dir`` (default
            ``MEMORY.md``, matching RPent's convention).
    """

    root_dir: Path
    index_filename: str = DEFAULT_INDEX_FILENAME

    @classmethod
    def for_env(cls, env: str, agent_mode: str = "dual") -> "MemoryLibrary":
        """Build a :class:`MemoryLibrary` for ``mllm_base_agent/<agent_mode>_agent/<env>``."""
        return cls(root_dir=_default_memory_root(env, agent_mode))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self.root_dir.is_dir()

    def index_path(self) -> Path:
        return self.root_dir / self.index_filename

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_index(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """Return the raw text of ``MEMORY.md``, or ``""`` if missing/unreadable.

        Never raises: a missing memory library (e.g. a new environment that
        has not accumulated any notes yet) degrades gracefully to "no memory
        available", matching RPent's "memory is optional" behavior
        (``rpent/utils/resources.py::ensure_resources``).
        """
        if not self.exists():
            return ""
        result = _read_text_file(str(self.index_path()), max_chars=max_chars)
        return result.get("content", "") if "error" not in result else ""

    def list_entries(self) -> List[str]:
        """Return the file names available in this memory library (best-effort)."""
        if not self.exists():
            return []
        result = _list_dir(str(self.root_dir), recursive=False, max_entries=500)
        entries = result.get("entries", []) or []
        return [e for e in entries if not e.startswith(".")]

    def read_entry(self, file_name: str, max_chars: int = DEFAULT_MAX_CHARS) -> Dict[str, Any]:
        """Read one memory file by name (or relative path) within the library.

        Returns the same shape as :func:`mllm_base_agent.tools.common.read_text_file`
        (``{"path", "size"/"total_lines", "content"}`` or ``{"error": ...}``).
        Guards against path traversal outside ``root_dir`` (a memory file name
        should never need ``..``).
        """
        if not self.exists():
            return {"error": f"memory library not found: {self.root_dir}"}

        cleaned = (file_name or "").strip()
        if not cleaned:
            return {"error": "file_name is required"}
        if ".." in Path(cleaned).parts:
            return {"error": f"invalid memory file name: {file_name}"}

        candidate = (self.root_dir / cleaned).resolve()
        try:
            candidate.relative_to(self.root_dir.resolve())
        except ValueError:
            return {"error": f"invalid memory file name: {file_name}"}

        return _read_text_file(str(candidate), max_chars=max_chars)

    def search(self, pattern: str, max_results: int = 100) -> Dict[str, Any]:
        """Grep across every memory file for ``pattern`` (case-insensitive)."""
        if not self.exists():
            return {"error": f"memory library not found: {self.root_dir}"}
        return _search_in_files(
            pattern=pattern,
            path=str(self.root_dir),
            case_insensitive=True,
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # Prompt-building helpers
    # ------------------------------------------------------------------

    def index_prompt_block(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """Return a ready-to-embed prompt block with the memory index, or "" if none.

        Intended to be spliced into a system prompt so the index (one-line
        hooks per entry) is always visible, while the full leaf notes are
        fetched on demand via ``ReadMemory(<file_name>)`` (kept out of the
        system prompt to avoid bloating every request).

        The header is deliberately written as a directive ("you MUST"), not a
        casual FYI: past runs showed models very rarely call
        ``ReadMemory(...)`` when the library is framed as merely available.
        Naming the concrete trigger conditions up front (repeated failure /
        about to give up / about to claim DONE) gives the model an explicit,
        checkable rule to follow instead of a vague "use if helpful" hint,
        which in practice gets skipped.
        """
        text = self.read_index(max_chars=max_chars)
        if not text:
            return ""
        return (
            "**Memory Library -- MANDATORY CONSULTATION, not optional reading:**\n"
            "This is a reviewed library of concrete lessons from past runs of "
            "this exact task type. `ReadMemory(<file_name>)` is a **free** "
            "lookup (zero step-budget cost, zero downside) into it. "
            "You MUST call `ReadMemory(<file_name>)` for the matching entry "
            "below, BEFORE your next action, whenever ANY of these are true:\n"
            "  1. Your last action failed, or the same/similar action has now "
            "failed 2+ times in a row.\n"
            "  2. You are about to output DONE or FAIL.\n"
            "  3. You are genuinely unsure what to do next.\n"
            "Skipping this when one of the above applies is a mistake, not a "
            "neutral choice -- these entries exist specifically because past "
            "agents repeated the same error without consulting them.\n"
            f"{text.strip()}"
        )


__all__ = ["MemoryLibrary", "DEFAULT_INDEX_FILENAME"]
