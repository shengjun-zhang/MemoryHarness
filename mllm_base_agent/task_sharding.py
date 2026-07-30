"""Deterministic task sharding helpers for multi-machine benchmark runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ShardConfig:
    """Resolved zero-based shard coordinates."""

    num_shards: int = 1
    shard_index: int = 0
    source: str = "default"

    @property
    def enabled(self) -> bool:
        return self.num_shards > 1


def _as_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _extract_cluster_size(spec: Mapping[str, Any]) -> Optional[int]:
    for key in ("world_size", "num_workers", "worker_num", "workers", "worker", "size"):
        value = spec.get(key)
        if isinstance(value, (list, tuple, Mapping)):
            return len(value) or None
        parsed = _as_non_negative_int(value)
        if parsed and parsed > 0:
            return parsed

    cluster = spec.get("cluster")
    if isinstance(cluster, Mapping):
        workers = cluster.get("worker") or cluster.get("workers")
        if isinstance(workers, (list, tuple, Mapping)):
            return len(workers) or None
        return _extract_cluster_size(cluster)
    return None


def _extract_rank(spec: Mapping[str, Any]) -> Optional[int]:
    for key in ("index", "rank", "worker_index", "task_index"):
        parsed = _as_non_negative_int(spec.get(key))
        if parsed is not None:
            return parsed

    task = spec.get("task")
    if isinstance(task, Mapping):
        return _extract_rank(task)
    return None


def _cluster_spec_shard() -> Tuple[Optional[int], Optional[int]]:
    raw = os.environ.get("AFO_ENV_CLUSTER_SPEC")
    if not raw:
        return None, None
    try:
        spec = json.loads(raw)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(spec, Mapping):
        return None, None
    return _extract_cluster_size(spec), _extract_rank(spec)


def resolve_shard_config(
    num_shards: Optional[int] = None,
    shard_index: Optional[int] = None,
) -> ShardConfig:
    """Resolve shard coordinates from CLI, common env vars, or AFO cluster spec.

    Explicit arguments have highest priority. Environment variables supported are
    ``TASK_NUM_SHARDS``/``WORLD_SIZE`` and ``TASK_SHARD_INDEX``/``RANK``.
    AFO's ``AFO_ENV_CLUSTER_SPEC`` is used as a final cluster-specific fallback.
    """

    if (num_shards is None) != (shard_index is None):
        raise ValueError("--num-shards and --shard-index must be provided together")

    candidates = [
        (num_shards, shard_index, "cli"),
        (
            _as_non_negative_int(os.environ.get("TASK_NUM_SHARDS")),
            _as_non_negative_int(os.environ.get("TASK_SHARD_INDEX")),
            "TASK_NUM_SHARDS/TASK_SHARD_INDEX",
        ),
        (
            _as_non_negative_int(os.environ.get("WORLD_SIZE")),
            _as_non_negative_int(os.environ.get("RANK")),
            "WORLD_SIZE/RANK",
        ),
    ]
    afo_num, afo_index = _cluster_spec_shard()
    candidates.append((afo_num, afo_index, "AFO_ENV_CLUSTER_SPEC"))

    resolved_num: Optional[int] = None
    resolved_index: Optional[int] = None
    source = "default"
    for candidate_num, candidate_index, candidate_source in candidates:
        if candidate_num is not None and candidate_index is not None:
            resolved_num = candidate_num
            resolved_index = candidate_index
            source = candidate_source
            break

    if resolved_num is None or resolved_index is None:
        return ShardConfig()
    if resolved_num < 1:
        raise ValueError(f"num_shards must be >= 1, got {resolved_num}")
    if not 0 <= resolved_index < resolved_num:
        raise ValueError(
            f"shard_index must be in [0, {resolved_num}), got {resolved_index}"
        )
    return ShardConfig(resolved_num, resolved_index, source)


def shard_tasks(
    tasks: Sequence[T],
    config: ShardConfig,
    strategy: str = "round_robin",
) -> List[T]:
    """Return one deterministic, disjoint shard while preserving task order."""

    if strategy == "round_robin":
        return list(tasks[config.shard_index :: config.num_shards])
    if strategy == "contiguous":
        start = len(tasks) * config.shard_index // config.num_shards
        end = len(tasks) * (config.shard_index + 1) // config.num_shards
        return list(tasks[start:end])
    raise ValueError(f"unsupported shard strategy: {strategy}")
