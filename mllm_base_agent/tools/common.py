"""Common agent tools: file read/write/list/find/search + finish.

This mirrors the "generic Toolkit" layer described in RPent
(``rpent/tools/common.py``): a small set of environment-agnostic tools that
give an LLM-driven agent basic filesystem access (read/write/list/find/grep),
plus an explicit ``finish`` signal tool to terminate a tool-calling loop.

Every tool here:
  * has an OpenAI-style ``function calling`` JSON schema entry in
    ``TOOLS_SPEC`` (``{"type": "function", "function": {...}}``), and
  * has a plain-Python handler in ``TOOL_HANDLERS`` that takes keyword
    arguments matching the schema's ``parameters`` and returns a JSON-able
    dict result (``{"error": ...}`` on failure, never raises for expected
    failure modes).

These are registered onto every :class:`mllm_base_agent.tools.toolkit.Toolkit`
by default; environment-specific subclasses can add more tools on top (e.g. a
future ``segment``/``back_project`` perception tool, mirroring RPent's
``LiberoToolkit``).
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mllm_base_agent.tools.paths import get_output_dir, resolve_path

# Directory names that are almost never useful to search/list into and are
# expensive to walk (VCS metadata, virtualenvs, caches, model checkpoints).
DEFAULT_EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
}

DEFAULT_MAX_CHARS = 40000
DEFAULT_FIND_LIMIT = 200
DEFAULT_GREP_LIMIT = 200
DEFAULT_GREP_MAX_FILES = 5000


TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_text_file",
            "description": (
                "Read a UTF-8 text file from disk. Use to inspect configs, logs, "
                "trajectories, or any other text artifact. Large files are "
                "truncated (see max_chars)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path, or path relative to the repo root.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 40000).",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-based line number to start reading from (optional).",
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": "Number of lines to read starting at start_line (optional).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_text_file",
            "description": (
                "Write a UTF-8 text file to disk (creates parent directories as "
                "needed). Use to save trajectory summaries, notes, reports, or "
                "any other text artifact produced during the run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or repo-relative path"},
                    "content": {"type": "string", "description": "Full file content to write"},
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to the file instead of overwriting (default false).",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List entries in a directory. Non-recursive by default; set "
                "recursive=true to walk subdirectories too. Defaults to the "
                "current run's output directory when path is omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list. Default: current run output dir.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recurse into subdirectories (default false).",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum number of entries to return (default 500).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": (
                "Find files by glob pattern (e.g. '*.py', '**/*.json', "
                "'episode_*.json') under a directory, recursively. Use this "
                "to locate configs, episode logs, or checkpoints by name "
                "pattern without knowing the exact path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '*.py' or '**/episode_*.json'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Root directory to search under. Default: repo root.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default 200).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": (
                "Search for a text/regex pattern inside files under a directory "
                "(like grep -r). Returns matching file paths with line numbers "
                "and the matching line content. Use this to locate where a "
                "symbol, error message, or config key is used/defined."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to search for."},
                    "path": {
                        "type": "string",
                        "description": "Root directory to search under. Default: repo root.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Only search files matching this glob, e.g. '*.py' (default: all files).",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default false).",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat pattern as a regex (default false = literal substring).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return (default 200).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call when the task is complete or unrecoverable. Halts the "
                "tool-calling agent loop. Save any artifacts (via "
                "write_text_file) BEFORE calling finish."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Outcome, e.g. 'success', 'failure', or 'stuck'.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short natural-language summary of the run.",
                    },
                },
                "required": ["status", "summary"],
            },
        },
    },
]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[TRUNCATED - file is {len(text)} chars, showed first {max_chars}]"


def read_text_file(
    path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    start_line: Optional[int] = None,
    num_lines: Optional[int] = None,
) -> Dict[str, Any]:
    p = resolve_path(path)
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory, not a file: {p}"}
    try:
        text = p.read_text(errors="replace")
    except Exception as exc:
        return {"error": str(exc)}

    if start_line is not None:
        lines = text.splitlines()
        start_idx = max(0, int(start_line) - 1)
        end_idx = start_idx + int(num_lines) if num_lines else len(lines)
        selected = lines[start_idx:end_idx]
        content = "\n".join(selected)
        return {
            "path": str(p),
            "total_lines": len(lines),
            "start_line": start_idx + 1,
            "end_line": min(end_idx, len(lines)),
            "content": _truncate(content, max_chars),
        }

    return {"path": str(p), "size": len(text), "content": _truncate(text, max_chars)}


def write_text_file(path: str, content: str, append: bool = False) -> Dict[str, Any]:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(p, mode, encoding="utf-8") as handle:
        handle.write(content)
    return {
        "path": str(p),
        "bytes_written": len(content.encode("utf-8")),
        "appended": bool(append),
    }


def list_dir(path: str = "", recursive: bool = False, max_entries: int = 500) -> Dict[str, Any]:
    if path:
        p = resolve_path(path)
    else:
        p = get_output_dir() or resolve_path(".")
    if not p.exists():
        return {"error": f"directory not found: {p}"}
    if not p.is_dir():
        return {"error": f"not a directory: {p}"}

    entries: List[str] = []
    if recursive:
        for root, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDED_DIRS]
            rel_root = os.path.relpath(root, p)
            for name in sorted(dirnames) + sorted(filenames):
                rel_path = name if rel_root == "." else os.path.join(rel_root, name)
                entries.append(rel_path)
                if len(entries) >= max_entries:
                    break
            if len(entries) >= max_entries:
                break
    else:
        entries = sorted(os.listdir(p))[:max_entries]

    return {
        "path": str(p),
        "count": len(entries),
        "entries": entries,
        "truncated": len(entries) >= max_entries,
    }


def find_files(pattern: str, path: str = "", max_results: int = DEFAULT_FIND_LIMIT) -> Dict[str, Any]:
    root = resolve_path(path) if path else get_repo_root_for_find()
    if not root.exists():
        return {"error": f"directory not found: {root}"}

    matches: List[str] = []
    truncated = False
    # Support both simple ("*.py") and recursive ("**/*.py") glob patterns:
    # Path.rglob already recurses, so a plain "*.py" pattern still matches
    # files in subdirectories; an explicit "**/" prefix is just stripped.
    try:
        iterator = root.glob(pattern) if "**" in pattern else root.rglob(pattern)
    except Exception as exc:
        return {"error": f"invalid pattern: {exc}"}

    for match in iterator:
        parts = set(match.relative_to(root).parts[:-1])
        if parts & DEFAULT_EXCLUDED_DIRS:
            continue
        if match.is_file():
            matches.append(str(match))
        if len(matches) >= max_results:
            truncated = True
            break

    matches.sort()
    return {"path": str(root), "pattern": pattern, "count": len(matches), "files": matches, "truncated": truncated}


def get_repo_root_for_find() -> Path:
    from mllm_base_agent.tools.paths import get_repo_root

    return get_repo_root()


def search_in_files(
    pattern: str,
    path: str = "",
    glob: str = "",
    case_insensitive: bool = False,
    regex: bool = False,
    max_results: int = DEFAULT_GREP_LIMIT,
) -> Dict[str, Any]:
    import re as re_module

    root = resolve_path(path) if path else get_repo_root_for_find()
    if not root.exists():
        return {"error": f"directory not found: {root}"}

    if regex:
        try:
            flags = re_module.IGNORECASE if case_insensitive else 0
            compiled = re_module.compile(pattern, flags)
        except Exception as exc:
            return {"error": f"invalid regex: {exc}"}

        def matches_line(line: str) -> bool:
            return compiled.search(line) is not None
    else:
        needle = pattern.lower() if case_insensitive else pattern

        def matches_line(line: str) -> bool:
            haystack = line.lower() if case_insensitive else line
            return needle in haystack

    results: List[Dict[str, Any]] = []
    files_scanned = 0
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDED_DIRS]
        for filename in sorted(filenames):
            if glob and not fnmatch.fnmatch(filename, glob):
                continue
            file_path = Path(dirpath) / filename
            files_scanned += 1
            if files_scanned > DEFAULT_GREP_MAX_FILES:
                truncated = True
                break
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        if matches_line(line):
                            results.append({
                                "path": str(file_path),
                                "line": line_no,
                                "text": line.rstrip("\n")[:500],
                            })
                            if len(results) >= max_results:
                                truncated = True
                                break
            except (UnicodeDecodeError, OSError, IsADirectoryError):
                continue
            if len(results) >= max_results:
                break
        if len(results) >= max_results or truncated:
            break

    return {
        "path": str(root),
        "pattern": pattern,
        "count": len(results),
        "matches": results,
        "truncated": truncated,
    }


def finish(status: str, summary: str) -> Dict[str, Any]:
    """Signal that the tool-calling run is complete. Halts the agent loop.

    The ``_finish`` sentinel is what :class:`mllm_base_agent.tools.toolkit.ToolResult`
    and any tool-calling loop check to stop iterating (mirrors RPent's
    ``rpent/tools/common.py::finish``).
    """
    return {"_finish": True, "status": status, "summary": summary}


TOOL_HANDLERS: Dict[str, Any] = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_dir": list_dir,
    "find_files": find_files,
    "search_in_files": search_in_files,
    "finish": finish,
}


# ----------------------------------------------------------------------------
# Model-facing usage guidance
# ----------------------------------------------------------------------------
#
# The per-tool ``description``/``parameters`` text in TOOLS_SPEC is enough for
# the model to know *what each tool does*, but not necessarily *when/how to
# use them as a workflow* (e.g. "call finish when done", "prefer find_files
# over guessing a path"). Rather than a separate doc file, that guidance lives
# here as a plain string constant so it ships together with the tool
# definitions and can be embedded directly into a system prompt (see
# Toolkit.build_system_prompt / ToolCallingAgent's default system_prompt).
COMMON_TOOLS_USAGE_GUIDE = """\
You have access to tools for reading/writing/finding files. Guidelines:
- Prefer `find_files` (glob) or `search_in_files` (grep-like) to locate a \
file/symbol before guessing its exact path.
- `list_dir` defaults to the current run's output directory when no path is \
given.
- Paths may be absolute or relative to the repository root.
- Save any artifacts you want to keep (e.g. via `write_text_file`) BEFORE \
calling `finish`.
- Call `finish` exactly once, when the task is complete or you determine it \
cannot be completed, with a concise `summary` of what you did.
- Only call tools that are listed for you; do not invent tool names or \
arguments not in their schema."""


__all__ = [
    "TOOLS_SPEC",
    "TOOL_HANDLERS",
    "COMMON_TOOLS_USAGE_GUIDE",
    "read_text_file",
    "write_text_file",
    "list_dir",
    "find_files",
    "search_in_files",
    "finish",
]
