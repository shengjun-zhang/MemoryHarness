#!/usr/bin/env python3
"""
     benchmark     

       `run_csv_benchmark.py`：
-    CSV     
-     /    
-    skip-completed save-name outputs-completed   
-         CSV

        ：
-         `python -m mllm_base_agent.dual_agent.ai2thor.main`
-    JSON    `mllm_base_agent/dual_agent/ai2thor/main.py`     `dual_episode_*.json`
- benchmark           `mllm_base_agent/dual_agent/ai2thor/benchmark_outputs/.../<task_id>/`（   --output-dir   ）
-             （`env.agent_count: 2`）；          setdefault，         YAML   `env.agent_count: 1`
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set

import yaml
from dotenv import load_dotenv

#           ，            ，
#                      `config`  ，
#        `dual_agent/config` 
project_root = str(Path(__file__).resolve().parents[3])
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)
dual_agent_dir = Path(__file__).resolve().parent
#        dual_agent    （    ，        cwd）
_DEFAULT_BENCHMARK_OUTPUT_DIR = str(dual_agent_dir / "benchmark_outputs")
_DEFAULT_OUTPUTS_COMPLETED_DIR = str(dual_agent_dir / "outputs_completed")
_DEFAULT_DUAL_AGENT_CONFIG = str((Path(__file__).resolve().parents[3] / "configs" / "ai2thor" / "dual" / "config.yaml").resolve())
_DEFAULT_DUAL_CSV_DIR = Path(project_root) / "experiments" / "csv" / "ai2thor" / "dual"
os.environ.setdefault(
    "AI2THOR_TASKS_ROOT",
    str(Path(project_root) / "data" / "ai2thor" / "dual" / "tasks"),
)

from .config import load_config
from mllm_base_agent.task_sharding import resolve_shard_config, shard_tasks
from mllm_base_agent.subprocess_streaming import run_task_subprocess_streaming
from scripts.ai2thor.run_benchmark import (  #        benchmark
    build_csv_extra_fields,
    count_csv_status,
    deduplicate_task_ids,
    extract_token_stats_from_text,
    load_task_metadata,
    read_task_ids_from_csv,
    save_task_log,
    update_csv_task_record,
    write_missing_result_diagnostic,
)

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed, using simple progress. Install: pip install tqdm")


INFLIGHT_TASKS_LOCK = Lock()
INFLIGHT_TASKS: Set[str] = set()


def normalize_task_id(task_id: str) -> str:
    """    task id，   ai2thor_04000 / ai2thor04000      """
    if "_" in task_id and task_id.startswith("ai2thor_"):
        return task_id.replace("ai2thor_", "ai2thor", 1)
    return task_id


def find_result_json(task_output_dir: str) -> Optional[Path]:
    """               JSON """
    task_path = Path(task_output_dir)
    if not task_path.exists():
        return None

    direct_candidates = sorted(task_path.glob("dual_episode_*.json"))
    if direct_candidates:
        return direct_candidates[-1]

    nested_candidates = sorted(task_path.rglob("dual_episode_*.json"))
    if nested_candidates:
        return nested_candidates[-1]

    return None


def resolve_dual_agent_csv_path(csv_arg: Optional[str]) -> Optional[Path]:
    """Resolve dual AI2-THOR CSVs, preferring experiments/csv/ai2thor/dual."""
    if not csv_arg:
        return None

    raw = Path(csv_arg)
    basename = raw.name
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.extend([raw, _DEFAULT_DUAL_CSV_DIR / basename])
    else:
        candidates.extend(
            [
                _DEFAULT_DUAL_CSV_DIR / basename,
                Path(project_root) / raw,
                dual_agent_dir / raw,
                raw,
            ]
        )

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def find_completed_tasks(output_dir: str, save_name: Optional[str] = None) -> set:
    """    dual benchmark            （        shard-XXX      ）"""
    completed = set()
    output_path = Path(output_dir)

    if not output_path.exists():
        return completed

    prefixes = ["dual_benchmark_", "dual_benchmark_sequential_"]
    if save_name:
        prefixes.append(f"{save_name}_")

    benchmark_dirs = sorted(
        [
            d
            for d in output_path.iterdir()
            if d.is_dir() and any(d.name.startswith(prefix) for prefix in prefixes)
        ],
        key=lambda x: x.name,
        reverse=True,
    )

    def _scan_task_dirs(run_dir: Path) -> None:
        for task_dir in run_dir.iterdir():
            if not task_dir.is_dir():
                continue
            if task_dir.name in {"task_logs", "failed_logs"}:
                continue
            if find_result_json(str(task_dir)):
                completed.add(normalize_task_id(task_dir.name))

    for benchmark_dir in benchmark_dirs:
        shard_dirs = sorted(
            (
                d
                for d in benchmark_dir.iterdir()
                if d.is_dir() and d.name.startswith("shard-")
            ),
            key=lambda x: x.name,
            reverse=True,
        )
        # 新版嵌套布局：<save_name>_<time>/shard-XXX/<task_id>/
        # 旧版平铺布局：<save_name>_shard-XXX-of-YYY_<time>/<task_id>/
        run_dirs = shard_dirs or [benchmark_dir]
        for run_dir in run_dirs:
            _scan_task_dirs(run_dir)

    return completed


def read_result_status_info(result_json: Optional[Path]) -> Dict[str, Any]:
    """         JSON        """
    info = {
        "has_result_json": bool(result_json and result_json.exists()),
        "task_result": None,
        "failure_type": None,
        "fail_reason": None,
        "agent_1_steps": 0,
        "agent_2_steps": 0,
        "communication_events": 0,
        "turn_count": 0,
    }

    if not result_json or not result_json.exists():
        return info

    try:
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        success = data.get("success")
        if success is True:
            info["task_result"] = "success"
        elif success is False:
            info["task_result"] = "failure"

        info["failure_type"] = data.get("failure_type")
        info["fail_reason"] = data.get("fail_reason")
        info["agent_1_steps"] = data.get("agent_1_steps", 0) or 0
        info["agent_2_steps"] = data.get("agent_2_steps", 0) or 0
        info["turn_count"] = data.get("turn_count", 0) or 0
        info["communication_events"] = len(data.get("communication_history", []) or [])
    except Exception as e:
        info["fail_reason"] = f"Failed to parse result JSON: {e}"

    return info


def _sanitized_log_for_classification(task_log_content: str) -> str:
    """                   API/         """
    if not task_log_content:
        return ""
    kept: List[str] = []
    for line in task_log_content.splitlines():
        lo = line.lower()
        if "parsed action" in lo and ("✓" in line or "success" in lo):
            continue
        if "api request attempt" in lo:
            continue
        if "using custom platform parameter" in lo and "cloudrendering" in lo:
            continue
        kept.append(line)
    return "\n".join(kept)


def _fail_reason_indicates_model_failure(fr_lower: str) -> bool:
    """fail_reason                      →   /    """
    if not fr_lower or "failed to parse result json" in fr_lower:
        return False
    if "consecutive" in fr_lower and "action failure" in fr_lower:
        return True
    patterns = (
        "reached maximum",
        "maximum global step",
        "max_steps",
        "max steps",
        "max step",
        "step limit",
        "reached max steps",
        "agent_1 reached max",
        "agent_2 reached max",
        "agent_1    ",
        "agent_2    ",
        "      ",
        "    ",
        "    ",
        "task incomplete",
        "blocked or exhausted",
        "both agents exhausted",
        "cannot continue",
        "premature done",
        "done but",
        "model claimed done",
        "model output invalid action",
        "no action available",
        "final evaluation on terminal state failed",
        "exceeded maximum",
        "maximum step",
        "refused to continue",
        "model determined",
    )
    if any(p in fr_lower for p in patterns):
        return True
    if "final evaluation" in fr_lower and "fail" in fr_lower:
        return True
    return False


def _fail_reason_indicates_external_api(fr_lower: str) -> bool:
    """fail_reason          API /    /    /    """
    if not fr_lower:
        return False
    patterns = (
        "rate limit",
        "too many requests",
        "resource exhausted",
        "resourceexhausted",
        "quota exceeded",
        "invalid api key",
        "api key invalid",
        "authentication failed",
        "unauthorized",
        "forbidden",
        "connection refused",
        "connection reset",
        "econnrefused",
        "network is unreachable",
        "name or service not known",
        "read timed out",
        "request timed out",
        "handshake failure",
        "ssl error",
        "certificate",
        "google.api_core.exceptions",
        "internal server error",
        "httpstatuserror",
        "bad gateway",
        "service unavailable",
        "not available in your region",
    )
    if any(p in fr_lower for p in patterns):
        return True
    for token in (" 429", " 502", " 503", " 401", " 403", ": 429", ": 502", ": 503", ": 401", ": 403"):
        if token in fr_lower:
            return True
    if fr_lower.strip() in ("429", "502", "503", "401", "403"):
        return True
    if "timeout" in fr_lower and any(
        x in fr_lower for x in ("request", "read", "connect", "api", "http", "socket")
    ):
        return True
    return False


def _fail_reason_indicates_external_env(fr_lower: str) -> bool:
    """fail_reason        /    /      """
    if not fr_lower:
        return False
    patterns = (
        "env_crash",
        "environment exception",
        "unity crashed",
        "unity crash",
        "gpu process",
        "segmentation fault",
        "cuda error",
        "vulkan error",
        "could not create",
        "could not connect to display",
    )
    if any(p in fr_lower for p in patterns):
        return True
    if "cloudrendering" in fr_lower and any(
        x in fr_lower for x in ("error", "fail", "exception", "unable", "crash")
    ):
        return True
    return False


def _log_indicates_external_api(log_lower: str) -> bool:
    if not log_lower:
        return False
    patterns = (
        "rate limit",
        "too many requests",
        "resourceexhausted",
        "quota exceeded",
        "429 too many",
        "502 bad gateway",
        "503 service unavailable",
        "401 unauthorized",
        "403 forbidden",
        "connection refused",
        "connection reset",
        "econnrefused",
        "read timed out",
        "request timed out",
        "api error",
        "invalid api key",
        "authentication failed",
        "google.api_core.exceptions",
        "internal server error",
        "httpstatuserror",
        "not available in your region",
    )
    if any(p in log_lower for p in patterns):
        return True
    if "timeout" in log_lower and any(
        x in log_lower for x in ("request", "read", "connect", "api", "http", "socket", "urllib", "httpx")
    ):
        return True
    return False


def _log_indicates_external_env(log_lower: str) -> bool:
    if not log_lower:
        return False
    patterns = (
        "env_crash",
        "environment exception",
        "unity crashed",
        "segmentation fault",
        "gpu process crashed",
        "could not connect to display",
    )
    if any(p in log_lower for p in patterns):
        return True
    if "cloudrendering" in log_lower and any(
        x in log_lower for x in ("error", "fail", "exception", "unable", "crash")
    ):
        return True
    return False


def _log_indicates_parse_or_action_error(log_lower: str) -> Optional[str]:
    if not log_lower:
        return None
    parse_markers = (
        "failed to parse",
        "parse error",
        "json decode error",
        "jsondecodeerror",
        "invalid json",
        "could not parse model output",
        "malformed json",
    )
    if any(p in log_lower for p in parse_markers):
        return "parse_error"
    action_markers = (
        "invalid action",
        "no action available",
        "action error",
    )
    if any(p in log_lower for p in action_markers):
        return "action_error"
    return None


def determine_failure_reason(task_log_content: str = "", result_json_path: Optional[Path] = None) -> str:
    """      ：JSON.failure_type > fail_reason    >          >      """
    result_info = read_result_status_info(result_json_path)
    failure_type = result_info.get("failure_type")
    if failure_type:
        return failure_type

    fail_reason = (result_info.get("fail_reason") or "").strip()
    fr_low = fail_reason.lower()
    task_result = result_info.get("task_result")

    if fr_low:
        if "failed to parse result json" in fr_low:
            return "external_error"
        if _fail_reason_indicates_external_api(fr_low):
            return "api_error"
        if _fail_reason_indicates_external_env(fr_low):
            return "env_error"
        if _fail_reason_indicates_model_failure(fr_low):
            return "model_error"

    san = _sanitized_log_for_classification(task_log_content)
    log_low = san.lower()

    if log_low:
        if "graphrecursionerror" in log_low or "recursion limit" in log_low:
            return "recursion_limit"
        if _log_indicates_external_api(log_low):
            return "api_error"
        if _log_indicates_external_env(log_low):
            return "env_error"
        pa = _log_indicates_parse_or_action_error(log_low)
        if pa:
            return pa

    if task_result == "failure":
        return "model_error"

    return "external_error"


def decide_csv_status_from_result(result_info: Dict[str, Any], fallback_failure_type: str) -> Optional[str]:
    """             CSV：true / false / None( ，   ) 

         fail_reason    ，                    failure_type        null 
    """
    task_result = result_info.get("task_result")
    fail_reason_raw = result_info.get("fail_reason") or ""
    fail_reason = fail_reason_raw.lower()
    json_ft = result_info.get("failure_type")

    if task_result == "success":
        return "true"

    #  ：         （  run     env_error）→    /    false；    env_error→false
    if fail_reason and (
        (
            "called with invalid argument" in fail_reason
            and "expected arguments" in fail_reason
        )
        or "invalid action parameters:" in fail_reason
    ):
        return "false"

    #        JSON           ，              
    if json_ft in ("api_error", "env_error", "external_error"):
        return None

    if fail_reason:
        if "failed to parse result json" in fail_reason:
            return None
        if _fail_reason_indicates_model_failure(fail_reason):
            return "false"
        if _fail_reason_indicates_external_api(fail_reason) or _fail_reason_indicates_external_env(
            fail_reason
        ):
            return None

    failure_type = json_ft or fallback_failure_type

    if failure_type in ("api_error", "env_error", "external_error", "recursion_limit"):
        return None
    if failure_type in ("parse_error", "action_error", "model_error"):
        return "false"

    if task_result == "failure":
        return "false"

    return None


def extract_actual_actions(result_json: Optional[Path]) -> Dict[str, Any]:
    """        JSON           """
    empty = {"actual_action_count": None, "actual_action_text": None}
    if not result_json or not result_json.exists():
        return empty

    try:
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectory = data.get("trajectory", [])
        if not isinstance(trajectory, list):
            return empty

        actions = []
        for entry in trajectory:
            if not isinstance(entry, dict):
                continue
            action_string = (entry.get("action_string") or "").strip()
            if not action_string:
                continue
            actions.append(action_string)

        return {
            "actual_action_count": len(actions) if actions else None,
            "actual_action_text": " | ".join(actions) if actions else None,
        }
    except Exception as e:
        print(f"  ⚠️  Failed to extract actual actions: {e}")
        return empty


def snapshot_task_run_dirs(task_id: str) -> Set[str]:
    """     task        dual_agent      """
    outputs_root = Path(project_root) / "dual_agent" / "outputs"
    normalized = normalize_task_id(task_id)
    if not outputs_root.exists():
        return set()
    return {str(path.resolve()) for path in outputs_root.glob(f"task_{normalized}_*") if path.is_dir()}


def find_new_task_run_dir(task_id: str, before_dirs: Set[str], started_at: float) -> Optional[Path]:
    """         task          """
    outputs_root = Path(project_root) / "dual_agent" / "outputs"
    normalized = normalize_task_id(task_id)
    if not outputs_root.exists():
        return None

    candidates = [path for path in outputs_root.glob(f"task_{normalized}_*") if path.is_dir()]
    if not candidates:
        return None

    new_dirs = [path for path in candidates if str(path.resolve()) not in before_dirs]
    if new_dirs:
        return max(new_dirs, key=lambda x: x.stat().st_mtime)

    recent_dirs = [path for path in candidates if path.stat().st_mtime >= started_at - 1.0]
    if recent_dirs:
        return max(recent_dirs, key=lambda x: x.stat().st_mtime)

    return max(candidates, key=lambda x: x.stat().st_mtime)


def copytree_no_delete(source: Path, dest: Path) -> Path:
    """    ；              ，     """
    final_dest = dest
    if final_dest.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dest = dest.parent / f"{dest.name}_{suffix}"
    shutil.copytree(source, final_dest)
    return final_dest


def copy_to_outputs_completed(task_id: str, task_output_dir: str, outputs_completed_dir: str) -> bool:
    """       ，        """
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            print(f"  ⚠️  Source dir not found: {task_output_dir}")
            return False

        dest_path = Path(outputs_completed_dir) / task_id
        copytree_no_delete(source_path, dest_path)
        return True
    except Exception as e:
        print(f"  ❌ Copy to outputs_completed failed ({task_id}): {e}")
        return False


def save_failed_snapshot(task_id: str, task_output_dir: str, failed_logs_dir: str, attempt: int = 1):
    """          ，        """
    try:
        source_path = Path(task_output_dir)
        if not source_path.exists():
            return
        failed_logs_path = Path(failed_logs_dir)
        failed_logs_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = failed_logs_path / f"{task_id}_attempt_{attempt}_{timestamp}"
        copytree_no_delete(source_path, dest_path)
        print(f"  📝 Failed snapshot saved to: {dest_path}")
    except Exception as e:
        print(f"  ⚠️  Error saving failed snapshot: {e}")


def prepare_benchmark_config(
    config_path: Path,
    headless: bool,
    collaboration_mode: str,
) -> tuple[Path, Optional[str]]:
    """             ，   benchmark      """
    needs_temp = headless or collaboration_mode != "alternating"
    if not needs_temp:
        return config_path, None

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    config_data.setdefault("env", {})
    # benchmark         （  mllm_base_agent.dual_agent.ai2thor.main   ）；YAML    agent_count      
    config_data["env"].setdefault("agent_count", 2)

    if headless:
        config_data["env"]["platform"] = "CloudRendering"

    if "dual_agent" not in config_data:
        config_data["dual_agent"] = {}
    config_data["dual_agent"]["collaboration_mode"] = collaboration_mode

    temp_config_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    )
    yaml.dump(config_data, temp_config_file, default_flow_style=False, allow_unicode=True)
    temp_config_file.close()
    return Path(temp_config_file.name), temp_config_file.name


def main():
    """    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dual-agent CSV benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --csv "experiments/csv/ai2thor/dual/Spatial-Annotation-ai2thor-Gemini-2.5-pro.csv" --workers 4 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --csv "experiments/csv/ai2thor/dual/Spatial-Annotation-ai2thor-Gemini-2.5-pro.csv" --sequential --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --task ai2thor05002 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --tasks ai2thor05001 ai2thor05002 --workers 2 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --csv "experiments/csv/ai2thor/dual/Spatial-Annotation-ai2thor-Gemini-2.5-pro.csv" --collaboration-mode sequential --switch-interval 5
  python -m mllm_base_agent.dual_agent.ai2thor.run_benchmark --task ai2thor05002 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml --agent1 experiments/configs/ai2thor/dual/config_close_Gemini-3.1-Pro-Preview.yaml --agent2 experiments/configs/ai2thor/dual/config_kimi-a3b.yaml
""",
    )

    parser.add_argument("--csv", type=str, default=None, help="   CSV   ")
    parser.add_argument("--workers", type=int, default=4, help="   worker  ")
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=0,
        help="单个任务超时秒数；0 表示不限制（集群排障建议设置，例如 7200）",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="跨机器任务分片总数；默认从 TASK_NUM_SHARDS/WORLD_SIZE/AFO 集群信息读取",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="当前机器的分片编号（从 0 开始）；默认从 TASK_SHARD_INDEX/RANK/AFO 集群信息读取",
    )
    parser.add_argument(
        "--shard-strategy",
        choices=["round_robin", "contiguous"],
        default="round_robin",
        help="任务分片策略：轮询分配（默认，较均衡）或连续区间",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_DUAL_AGENT_CONFIG,
        help="        （   experiments/configs/ai2thor/dual/config_close_gpt-5.yaml，    agent_count=2）",
    )
    parser.add_argument("--task", type=str, default=None, help="      task_id")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="            ",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="              ")
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Override per-agent max steps (default: 10 + golden_actions.steps)",
    )
    parser.add_argument(
        "--switch-interval",
        type=int,
        default=None,
        help="          ",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_BENCHMARK_OUTPUT_DIR,
        help="benchmark      （   dual_agent/benchmark_outputs）",
    )
    parser.add_argument(
        "--collaboration-mode",
        type=str,
        default="alternating",
        choices=["alternating", "parallel", "sequential"],
        help="        ",
    )
    parser.add_argument("--headless", action="store_true", help="  CloudRendering     ")
    parser.add_argument("--sequential", action="store_true", help="      ")
    parser.add_argument("--skip-completed", action="store_true", help="       ")
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="重新运行 CSV 中所有任务（包括已标记为 true/false 的任务）；默认只运行 Completed 为空(null) 的任务",
    )
    parser.add_argument(
        "--history-feedback",
        action="store_true",
        help="开启历史反馈模式：将每步动作及执行结果（含距离错误）注入 agent 历史，使其能跨步推理（透传给 main.py）",
    )
    parser.add_argument(
        "--llm-history-feedback",
        action="store_true",
        help=(
            "开启 LLM 历史分析 Agent：额外生成一个与当前 agent 相同的大模型，"
            "对各 agent 近期 动作-结果 历史进行管理与分析，产出高信息量、"
            "可指导动作的逐条注记，用以替换 --history-feedback 注入的原始 "
            "action+error 文本。关闭时不影响 --history-feedback（透传给 main.py）"
        ),
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help=(
            "发送给 VLM 的图像降采样倍数（0 < scale <= 1.0，透传给 main.py）。"
            "例如 --image-scale 0.5 会把 800x600 缩到 400x300 再 base64 编码，"
            "显著减小请求体、避免长 episode 触发 HTTP 413。默认 1.0 = 不缩放（原行为）"
        ),
    )
    parser.add_argument(
        "--image-recent-steps",
        type=int,
        default=0,
        help=(
            "临近当前 step 的 K 个历史图片保持原始分辨率（800x600），更久远的"
            "历史图片使用 --image-scale 缩放（透传给 main.py）。当前观测图片始终"
            "为原始分辨率。例如 --image-recent-steps 3 保留最近 3 张历史图片"
            "原始大小，仅缩放更早的。默认 0 = 仅当前观测为原始分辨率，所有历史"
            "图片按 --image-scale 缩放（原行为）。"
        ),
    )
    parser.add_argument(
        "--partner-view",
        action="store_true",
        help=(
            "开启伙伴视角注入：每个决策步还额外给模型一张从伙伴身体当前相机"
            "实时渲染的第一视角图片（透传给 main.py）。这让当前 agent 能从不同"
            "角度观察共享场景，并核对伙伴汇报的发现。仅在 env 有 2 个实体 agent"
            "时生效。默认关闭（原行为：每个 agent 只看自己的视角）。"
        ),
    )
    parser.add_argument(
        "--partner-view-scale",
        type=float,
        default=None,
        help=(
            "伙伴视角图片的降采样倍数（0 < scale <= 1.0，透传给 main.py）。"
            "默认与 --image-scale 一致（伙伴图片作为辅助输入，与历史图片同样"
            "缩放）。设为 1.0 可保持伙伴图片为原始 800x600 分辨率。"
        ),
    )
    parser.add_argument(
        "--outputs-completed-dir",
        type=str,
        default=_DEFAULT_OUTPUTS_COMPLETED_DIR,
        help="         （   dual_agent/outputs_completed）",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="benchmark       ",
    )
    parser.add_argument(
        "--agent1",
        type=str,
        default=None,
        help="Agent 1            ；      agent config，       ",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default=None,
        help="Agent 2            ；   agent1   ，       ",
    )

    args = parser.parse_args()

    try:
        shard_config = resolve_shard_config(args.num_shards, args.shard_index)
    except ValueError as exc:
        parser.error(str(exc))

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    csv_path = resolve_dual_agent_csv_path(args.csv)
    if csv_path and not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    if args.task:
        task_ids = [args.task.strip()]
        print(f"📋 Single-task mode: {task_ids[0]}")
    elif args.tasks:
        task_ids = [task_id.strip() for task_id in args.tasks if task_id.strip()]
        print(f"📋 Explicit task list: {task_ids[:5]}")
    elif csv_path:
        print(f"📋 Reading task IDs from CSV: {csv_path}")
        only_null = not args.rerun_all
        task_ids = read_task_ids_from_csv(str(csv_path), only_null=only_null)
        if not task_ids:
            if only_null:
                print("❌ No task IDs with Completed=null in CSV")
            else:
                print("❌ No task IDs found in CSV")
            sys.exit(1)
        if only_null:
            print(f"✓ Found {len(task_ids)} tasks with Completed=null")
        else:
            print(f"✓ Found {len(task_ids)} tasks (rerun-all mode, including completed true/false)")
    else:
        print(f"📋 Loading tasks from config: {config_path}")
        config = load_config(str(config_path))
        task_ids = config.get_all_task_names()
        if not task_ids:
            print("❌ No tasks found in config")
            sys.exit(1)
        print(f"✓ Found {len(task_ids)} tasks from config")

    print(f"  First 5: {task_ids[:5]}")
    if len(task_ids) > 5:
        print(f"  ... (total {len(task_ids)} tasks)")

    unique_task_ids, duplicate_task_ids = deduplicate_task_ids(task_ids)
    if duplicate_task_ids:
        print(f"⚠️  Found {len(duplicate_task_ids)} duplicated task IDs; duplicates will be skipped")
    task_ids = unique_task_ids

    unsharded_task_count = len(task_ids)
    task_ids = shard_tasks(task_ids, shard_config, args.shard_strategy)
    if shard_config.enabled:
        print(
            f"🔀 Task shard {shard_config.shard_index}/{shard_config.num_shards} "
            f"({args.shard_strategy}, source={shard_config.source}): "
            f"selected {len(task_ids)}/{unsharded_task_count} tasks"
        )

    if not task_ids:
        print("✓ This shard has no tasks to run")
        sys.exit(0)

    if args.rerun_all and args.skip_completed:
        print("⚠️  --rerun-all 已启用，忽略 --skip-completed（强制重跑所有任务，不做断点续跑）")

    if args.skip_completed and not args.rerun_all:
        print("\n🔍 Checking completed tasks from previous dual benchmark dirs...")
        completed_tasks = find_completed_tasks(args.output_dir, args.save_name)
        if completed_tasks:
            original_count = len(task_ids)
            task_ids = [
                task_id
                for task_id in task_ids
                if normalize_task_id(task_id) not in completed_tasks
            ]
            skipped_count = original_count - len(task_ids)
            print(f"✓ Found {len(completed_tasks)} completed tasks")
            print(f"✓ Skipping {skipped_count} tasks")
            if not task_ids:
                print("✓ All tasks completed, nothing to run")
                sys.exit(0)
        else:
            print("✓ No completed tasks found, will run all")

    if not task_ids:
        print("✓ All tasks assigned to this shard are completed")
        sys.exit(0)

    actual_config_path, temp_config_path = prepare_benchmark_config(
        config_path=config_path,
        headless=args.headless,
        collaboration_mode=args.collaboration_mode,
    )

    # 多 shard 并行运行时，各 shard 进程由 launch_dual_benchmark.sh 注入同一个
    # BENCHMARK_RUN_TIMESTAMP，从而共享同一个实验目录 <save_name>_<timestamp>/，
    # 再各自写入其下的 shard-XXX/ 子目录，避免像过去那样各 shard 各自生成时间戳、
    # 平铺出多个互不相关的顶层目录。未设置该环境变量时（如本地单机调试）回退到
    # 当前时间戳，行为与之前一致。
    timestamp = os.environ.get("BENCHMARK_RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", timestamp):
        print(f"❌ Invalid BENCHMARK_RUN_TIMESTAMP: {timestamp} (expected YYYYMMDD_HHMMSS)")
        sys.exit(2)
    if args.save_name:
        prefix = args.save_name
    else:
        prefix = "dual_benchmark_sequential" if args.sequential else "dual_benchmark"
    experiment_output_dir = Path(args.output_dir) / f"{prefix}_{timestamp}"
    if shard_config.enabled:
        # 一个实验时间戳目录下按 shard 再分子目录，避免并行 shard 互相覆盖文件，
        # 同时保证所有 shard 都能被聚合到同一个 <save_name>_<timestamp>/ 目录中。
        shard_name = f"shard-{shard_config.shard_index:03d}-of-{shard_config.num_shards:03d}"
        benchmark_output_dir = experiment_output_dir / shard_name
    else:
        benchmark_output_dir = experiment_output_dir
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    task_logs_dir = benchmark_output_dir / "task_logs"
    task_logs_dir.mkdir(parents=True, exist_ok=True)
    failed_logs_dir = benchmark_output_dir / "failed_logs"
    failed_logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_completed_path = Path(args.outputs_completed_dir)
    outputs_completed_path.mkdir(parents=True, exist_ok=True)
    result_csv_path = csv_path
    if csv_path and shard_config.enabled:
        result_csv_path = benchmark_output_dir / f"{csv_path.stem}.shard-{shard_config.shard_index:03d}-of-{shard_config.num_shards:03d}.csv"
        shutil.copy2(csv_path, result_csv_path)
        print(f"Shard CSV (isolated writes): {result_csv_path}")
    csv_lock = Lock()

    print(f"\n{'=' * 80}")
    print(f"🚀 Starting {'sequential' if args.sequential else 'parallel'} dual-agent benchmark")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Task shard: {shard_config.shard_index}/{shard_config.num_shards} ({args.shard_strategy})")
    print(f"Workers: {1 if args.sequential else args.workers}")
    print(f"Config: {actual_config_path}")
    if args.agent1:
        print(f"Agent 1 config: {args.agent1}")
    if args.agent2:
        print(f"Agent 2 config: {args.agent2}")
    print(f"Output dir: {benchmark_output_dir}")
    print(f"Collaboration mode: {args.collaboration_mode}")
    if args.max_steps:
        print(f"Max steps: {args.max_steps}")
    if args.switch_interval:
        print(f"Switch interval: {args.switch_interval}")
    if args.headless:
        print("Headless: enabled (CloudRendering)")
    print(f"{'=' * 80}\n")

    def execute_task(task_id: str) -> Dict[str, Any]:
        # 整个函数体用最外层 try/except 兜底：无论内部（包括 load_task_metadata、
        # mkdir 等在“正常”try 块之外的调用）抛出什么异常，都绝不让它冒泡到
        # ThreadPoolExecutor / future.result()。否则单个任务的意外异常会导致
        # 整个 shard 主进程崩溃退出（跳过写 benchmark_summary.json 的收尾逻辑），
        # 让其余已经跑完/正在跑的任务的结果全部丢失且难以定位原因。
        try:
            return _execute_task_inner(task_id)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  ❌ {task_id} unhandled exception in execute_task: {exc}\n{tb}")
            try:
                (benchmark_output_dir / task_id).mkdir(parents=True, exist_ok=True)
                write_missing_result_diagnostic(
                    str(benchmark_output_dir / task_id),
                    task_id,
                    None,
                    "",
                    "",
                    error_text=f"Unhandled exception in execute_task: {exc}\n{tb}",
                )
            except Exception:
                pass
            return {
                "task_id": task_id,
                "status": "failed_external",
                # 未被 _execute_task_inner 自身捕获的异常（例如 load_task_metadata/
                # mkdir 在 IO 抖动的网络文件系统上出错）比普通 API 波动更值得警惕，
                # 归类为 external_error 而非 api_error。
                "failure_type": "external_error",
                "attempts": 1,
                "duration": 0.0,
                "success": False,
                "golden_actions_count": None,
                "actual_actions_count": None,
                "golden_action": None,
                "actual_actions": None,
                "instruction": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "failure_reason": f"Unhandled exception in execute_task: {exc}",
                "task_result": None,
                "agent_1_steps": 0,
                "agent_2_steps": 0,
                "communication_events": 0,
                "turn_count": 0,
                "copied_to_completed": False,
            }
        finally:
            with INFLIGHT_TASKS_LOCK:
                INFLIGHT_TASKS.discard(normalize_task_id(task_id))

    def _execute_task_inner(task_id: str) -> Dict[str, Any]:
        task_start_time = time.time()
        task_log_parts: List[str] = []
        task_status = "failed_external"
        # failure_type_detail 记录更细粒度的失败分类，用于区分“API 波动”（瞬时、
        # 不代表本机环境有问题，不应让整个 shard 非零退出）与“环境/未知异常”
        # （更可能意味着本机/容器环境本身出了问题，值得让集群感知并重启）。
        failure_type_detail: Optional[str] = None
        task_success = False
        copied_to_completed = False
        task_metadata = load_task_metadata(task_id)
        token_stats = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        actual_actions_info = {"actual_action_count": None, "actual_action_text": None}
        failure_reason_detail = None
        result_info = {
            "has_result_json": False,
            "task_result": None,
            "failure_type": None,
            "fail_reason": None,
            "agent_1_steps": 0,
            "agent_2_steps": 0,
            "communication_events": 0,
            "turn_count": 0,
        }

        normalized_task_id = normalize_task_id(task_id)
        with INFLIGHT_TASKS_LOCK:
            if normalized_task_id in INFLIGHT_TASKS:
                return {
                    "task_id": task_id,
                    "status": "failed_external",
                    "attempts": 1,
                    "duration": time.time() - task_start_time,
                    "success": False,
                    "golden_actions_count": task_metadata.get("golden_action_count"),
                    "actual_actions_count": actual_actions_info.get("actual_action_count"),
                    "golden_action": task_metadata.get("golden_action_text"),
                    "actual_actions": actual_actions_info.get("actual_action_text"),
                    "instruction": task_metadata.get("instruction"),
                    "prompt_tokens": token_stats["prompt_tokens"],
                    "completion_tokens": token_stats["completion_tokens"],
                    "total_tokens": token_stats["total_tokens"],
                    "failure_reason": "Task already running (dedupe protection)",
                    "failure_type": "external_error",
                    "task_result": None,
                    "agent_1_steps": 0,
                    "agent_2_steps": 0,
                    "communication_events": 0,
                    "turn_count": 0,
                    "copied_to_completed": False,
                }
            INFLIGHT_TASKS.add(normalized_task_id)

        benchmark_task_output_dir = benchmark_output_dir / task_id
        benchmark_task_output_dir.mkdir(parents=True, exist_ok=True)
        stdout_text = ""
        stderr_text = ""
        result_json = None

        try:
            before_dirs = snapshot_task_run_dirs(normalized_task_id)
            cmd = [
                sys.executable,
                "-m",
                "mllm_base_agent.dual_agent.ai2thor.main",
                "--config",
                str(actual_config_path),
                "--task",
                normalized_task_id,
                "--output-dir",
                str(benchmark_task_output_dir),
            ]
            if args.agent1:
                cmd.extend(["--agent1", args.agent1])
            if args.agent2:
                cmd.extend(["--agent2", args.agent2])
            if args.max_steps:
                cmd.extend(["--max-steps", str(args.max_steps)])
            if getattr(args, "recursion_limit", None):
                cmd.extend(["--recursion-limit", str(args.recursion_limit)])
            if args.switch_interval:
                cmd.extend(["--switch-interval", str(args.switch_interval)])
            if getattr(args, "history_feedback", False):
                cmd.append("--history-feedback")
            if getattr(args, "llm_history_feedback", False):
                cmd.append("--llm-history-feedback")
            _img_scale = float(getattr(args, "image_scale", 1.0) or 1.0)
            if 0.0 < _img_scale < 1.0:
                cmd.extend(["--image-scale", str(_img_scale)])
            _img_recent_steps = int(getattr(args, "image_recent_steps", 0) or 0)
            if _img_recent_steps > 0:
                cmd.extend(["--image-recent-steps", str(_img_recent_steps)])
            if getattr(args, "partner_view", False):
                cmd.append("--partner-view")
            _partner_view_scale = getattr(args, "partner_view_scale", None)
            if _partner_view_scale is not None:
                cmd.extend(["--partner-view-scale", str(_partner_view_scale)])

            execution_start_time = time.time()
            return_code, stdout_text, execution_duration = run_task_subprocess_streaming(
                cmd=cmd,
                cwd=Path(project_root),
                task_id=task_id,
                task_output_dir=benchmark_task_output_dir,
                timeout_seconds=args.task_timeout_seconds or None,
            )
            stderr_text = ""

            if stdout_text:
                task_log_parts.append("=== STDOUT ===\n")
                task_log_parts.append(stdout_text)
                task_log_parts.append("\n")
            if stderr_text:
                task_log_parts.append("=== STDERR ===\n")
                task_log_parts.append(stderr_text)
                task_log_parts.append("\n")

            produced_run_dir = None
            result_json = find_result_json(str(benchmark_task_output_dir))
            if result_json is None:
                produced_run_dir = find_new_task_run_dir(
                    normalized_task_id,
                    before_dirs,
                    execution_start_time,
                )
                if produced_run_dir:
                    copied_dir = copytree_no_delete(produced_run_dir, benchmark_task_output_dir)
                    benchmark_task_output_dir = copied_dir

            task_log_parts.append("=== Output dir ===\n")
            if produced_run_dir:
                task_log_parts.append(f"Produced run dir: {produced_run_dir}\n")
                task_log_parts.append(f"Copied to: {benchmark_task_output_dir}\n\n")
            else:
                task_log_parts.append(f"Task output dir: {benchmark_task_output_dir}\n\n")

            task_log_parts.append("=== Run info ===\n")
            task_log_parts.append(f"Command: {' '.join(cmd)}\n")
            task_log_parts.append(f"Exit code: {return_code}\n")
            task_log_parts.append(f"Duration: {execution_duration:.2f}s\n")
            task_log_parts.append(f"{'=' * 80}\n\n")

            task_log = "".join(task_log_parts)
            token_stats = extract_token_stats_from_text(task_log)
            result_json = find_result_json(str(benchmark_task_output_dir))
            result_info = read_result_status_info(result_json)
            actual_actions_info = extract_actual_actions(result_json)
            failure_reason_detail = result_info.get("fail_reason")

            if result_info.get("task_result") == "success":
                task_success = True
                task_status = "success"
                save_task_log(task_id, task_log, task_logs_dir, 1, "success")
                copied_to_completed = copy_to_outputs_completed(
                    task_id,
                    str(benchmark_task_output_dir),
                    str(outputs_completed_path),
                )
                if csv_path:
                    csv_extra_fields = build_csv_extra_fields(
                        task_metadata,
                        actual_actions_info,
                        token_stats,
                        failure_reason_detail,
                    )
                    update_csv_task_record(
                        result_csv_path,
                        task_id,
                        status="true",
                        extra_fields=csv_extra_fields,
                        lock=csv_lock,
                    )
                print(f"  ✅ {task_id} success")
            else:
                failure_reason = determine_failure_reason(task_log, result_json)
                csv_status = decide_csv_status_from_result(result_info, failure_reason)
                failure_reason_detail = result_info.get("fail_reason") or failure_reason
                csv_extra_fields = build_csv_extra_fields(
                    task_metadata,
                    actual_actions_info,
                    token_stats,
                    failure_reason_detail,
                )

                save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
                if benchmark_task_output_dir.exists():
                    save_failed_snapshot(task_id, str(benchmark_task_output_dir), str(failed_logs_dir), 1)

                if result_json:
                    if failure_reason in ("api_error", "env_error", "external_error") and csv_status != "false":
                        task_status = "failed_external"
                        failure_type_detail = failure_reason
                        print(f"  ⚠️  {task_id} external failure ({failure_reason}) -> null")
                    else:
                        task_status = "failed_model"
                        print(f"  ❌ {task_id} model failure -> {csv_status}")

                    if csv_status == "false":
                        copied_to_completed = copy_to_outputs_completed(
                            task_id,
                            str(benchmark_task_output_dir),
                            str(outputs_completed_path),
                        )

                    if csv_path:
                        update_csv_task_record(
                            result_csv_path,
                            task_id,
                            status=csv_status,
                            extra_fields=csv_extra_fields,
                            lock=csv_lock,
                        )
                else:
                    task_status = "failed_external"
                    failure_type_detail = "external_error"
                    failure_reason_detail = "No result JSON produced"
                    err_blob = f"{stderr_text}\n{stdout_text}".lower()
                    if "graphrecursionerror" in err_blob or "recursion limit" in err_blob:
                        failure_type_detail = "recursion_limit"
                        failure_reason_detail = (
                            "GraphRecursionError: graph recursion_limit exceeded "
                            "(no dual_episode JSON saved; pull latest mllm_base_agent/dual_agent/ai2thor/main.py)"
                        )
                    write_missing_result_diagnostic(
                        str(benchmark_task_output_dir),
                        task_id,
                        return_code,
                        stdout_text,
                        stderr_text,
                        error_text=failure_reason_detail,
                    )
                    if csv_path:
                        update_csv_task_record(
                            result_csv_path,
                            task_id,
                            status=None,
                            extra_fields=csv_extra_fields,
                            lock=csv_lock,
                        )
                    print(f"  ⚠️  {task_id} no result JSON -> null")

        except KeyboardInterrupt:
            task_status = "interrupted"
            task_log = "".join(task_log_parts)
            save_task_log(task_id, task_log, task_logs_dir, 1, "interrupted")
            raise
        except Exception as e:
            error_msg = str(e)
            task_log_parts.append("=== Exception ===\n")
            task_log_parts.append(f"Error: {error_msg}\n")
            task_log_parts.append(f"{'=' * 80}\n\n")
            task_log = "".join(task_log_parts)
            save_task_log(task_id, task_log, task_logs_dir, 1, "failed")
            if benchmark_task_output_dir.exists():
                save_failed_snapshot(task_id, str(benchmark_task_output_dir), str(failed_logs_dir), 1)

            result_json = find_result_json(str(benchmark_task_output_dir))
            result_info = read_result_status_info(result_json)
            token_stats = extract_token_stats_from_text(task_log)
            actual_actions_info = extract_actual_actions(result_json)
            failure_reason = determine_failure_reason(task_log, result_json)
            failure_reason_detail = result_info.get("fail_reason") or error_msg
            csv_status = decide_csv_status_from_result(result_info, failure_reason)
            csv_extra_fields = build_csv_extra_fields(
                task_metadata,
                actual_actions_info,
                token_stats,
                failure_reason_detail,
            )

            if failure_reason in ("parse_error", "action_error", "model_error"):
                task_status = "failed_model"
                if csv_status == "false":
                    copied_to_completed = copy_to_outputs_completed(
                        task_id,
                        str(benchmark_task_output_dir),
                        str(outputs_completed_path),
                    )
            else:
                task_status = "failed_external"
                # 主线程未捕获的异常（代码 bug、IO 异常等）比 API 错误更值得警惕，
                # 归入 external_error 而不是 api_error，以便下面的 exit_code 判定仍然能
                # 感知到这类非典型 API 波动的问题。若 failure_reason 本身已经是
                # api_error/env_error 则保留原分类。
                failure_type_detail = failure_reason if failure_reason in ("api_error", "env_error") else "external_error"
                if not result_json:
                    write_missing_result_diagnostic(
                        str(benchmark_task_output_dir),
                        task_id,
                        None,
                        "",
                        "",
                        error_text=error_msg,
                    )

            if csv_path:
                update_csv_task_record(
                    result_csv_path,
                    task_id,
                    status=csv_status,
                    extra_fields=csv_extra_fields,
                    lock=csv_lock,
                )
            print(f"  ❌ {task_id} exception: {error_msg}")
        finally:
            with INFLIGHT_TASKS_LOCK:
                INFLIGHT_TASKS.discard(normalized_task_id)

        task_duration = time.time() - task_start_time
        return {
            "task_id": task_id,
            "status": task_status,
            "failure_type": failure_type_detail,
            "attempts": 1,
            "duration": task_duration,
            "success": task_success,
            "golden_actions_count": task_metadata.get("golden_action_count"),
            "actual_actions_count": actual_actions_info.get("actual_action_count"),
            "golden_action": task_metadata.get("golden_action_text"),
            "actual_actions": actual_actions_info.get("actual_action_text"),
            "instruction": task_metadata.get("instruction"),
            "prompt_tokens": token_stats["prompt_tokens"],
            "completion_tokens": token_stats["completion_tokens"],
            "total_tokens": token_stats["total_tokens"],
            "failure_reason": failure_reason_detail,
            "task_result": result_info["task_result"],
            "agent_1_steps": result_info.get("agent_1_steps", 0),
            "agent_2_steps": result_info.get("agent_2_steps", 0),
            "communication_events": result_info.get("communication_events", 0),
            "turn_count": result_info.get("turn_count", 0),
            "copied_to_completed": copied_to_completed,
        }

    task_records: List[Dict[str, Any]] = []
    successful = 0
    failed_model = 0
    failed_external = 0
    # failed_external 再细分：failed_api 是纯 API 波动（限流/超时/413 等瞬时问题，
    # 任务已在 CSV 中标记为 null 供后续重跑，不代表本机/容器环境有问题）；
    # failed_infra 是环境崩溃（env_error）、结果缺失、未捕获异常等更严重的问题，
    # 更可能意味着当前机器/容器本身出了故障，值得让 exit_code 非零、触发集群感知重启。
    failed_api = 0
    failed_infra = 0
    copied_to_completed_count = 0
    exit_code = 0

    def _bump_failed_external_counters(result: Dict[str, Any]) -> None:
        nonlocal failed_external, failed_api, failed_infra
        failed_external += 1
        if result.get("failure_type") == "api_error":
            failed_api += 1
        else:
            failed_infra += 1

    try:
        if args.sequential:
            if HAS_TQDM:
                task_iterator = tqdm(task_ids, desc="Tasks", unit="task", ncols=100)
            else:
                task_iterator = task_ids

            for idx, task_id in enumerate(task_iterator, 1):
                if not HAS_TQDM:
                    print(f"\n{'=' * 80}")
                    print(f"📋 Task {idx}/{len(task_ids)}: {task_id}")
                    print(f"{'=' * 80}")
                result = execute_task(task_id)
                task_records.append(result)
                if result["success"]:
                    successful += 1
                elif result["status"] == "failed_model":
                    failed_model += 1
                else:
                    _bump_failed_external_counters(result)
                if result.get("copied_to_completed"):
                    copied_to_completed_count += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {
                    executor.submit(execute_task, task_id): task_id for task_id in task_ids
                }
                if HAS_TQDM:
                    task_iterator = tqdm(
                        as_completed(future_to_task),
                        total=len(task_ids),
                        desc="Tasks",
                        unit="task",
                        ncols=100,
                    )
                else:
                    task_iterator = as_completed(future_to_task)

                for future in task_iterator:
                    result = future.result()
                    task_records.append(result)
                    if result["success"]:
                        successful += 1
                    elif result["status"] == "failed_model":
                        failed_model += 1
                    else:
                        _bump_failed_external_counters(result)
                    if result.get("copied_to_completed"):
                        copied_to_completed_count += 1
                    if HAS_TQDM:
                        task_iterator.set_postfix(
                            {
                                "ok": successful,
                                "model": failed_model,
                                "api": failed_api,
                                "infra": failed_infra,
                                "copied": copied_to_completed_count,
                            }
                        )
    except KeyboardInterrupt:
        print("\n⚠️ User interrupt")
        exit_code = 1
    except Exception as exc:
        # 兜底：execute_task 本身已经不会再抛出非 KeyboardInterrupt 异常，但
        # ThreadPoolExecutor/tqdm 等调度层代码理论上仍可能出问题。宁可在这里
        # 记录清晰的异常堆栈并仍然写出已收集到的 task_records 的 summary，
        # 也不要让主进程直接崩溃、丢失所有已完成任务的结果且无从排查原因。
        print(f"\n❌ Unexpected error in task scheduling loop: {exc}")
        traceback.print_exc()
        exit_code = 1
    else:
        # `failed_model` 是评测层面的正常负样本（例如模型误判 DONE / 达到最大步数），
        # 属于预期内的 benchmark 结果，不应视为进程运行错误。
        # `failed_api`（限流/超时/413 等 API 波动）同样不应让整个 shard 非零退出：
        # 这类失败通常只是瞬时的、与当前机器/容器环境无关，任务已写回 CSV 为
        # null，后续重跑（--skip-completed）会自动补跑，没必要因为个别任务撞上
        # 一次 API 抖动就让 AFO 判定整个 worker task 失败、杀掉同一 worker 上其它
        # 正常运行的任务。
        # 只有 `failed_infra`（环境崩溃 env_error、结果缺失、未捕获异常等）才代表
        # 本次 shard 运行本身/所在机器出了问题，才需要让进程以非零码退出（供集群
        # 调度层据此重试/告警/重启容器）。
        exit_code = 0 if failed_infra == 0 else 1
    finally:
        if temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
                print("✓ Temp config removed")
            except Exception as e:
                print(f"⚠️ Failed to remove temp config: {e}")

    total_duration = sum(record["duration"] for record in task_records)
    avg_duration = total_duration / len(task_records) if task_records else 0
    aggregate_agent_stats = {
        "agent_1_steps": sum(record.get("agent_1_steps", 0) for record in task_records),
        "agent_2_steps": sum(record.get("agent_2_steps", 0) for record in task_records),
        "communication_events": sum(record.get("communication_events", 0) for record in task_records),
        "turn_count": sum(record.get("turn_count", 0) for record in task_records),
    }

    summary_log_path = task_logs_dir / f"summary_{timestamp}.log"
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write("Dual benchmark run summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if csv_path:
            f.write(f"CSV: {csv_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Actual config: {actual_config_path}\n")
        if args.agent1:
            f.write(f"Agent 1 config: {args.agent1}\n")
        if args.agent2:
            f.write(f"Agent 2 config: {args.agent2}\n")
        f.write(f"Output dir: {benchmark_output_dir}\n")
        f.write(f"Mode: {'sequential' if args.sequential else f'parallel (workers: {args.workers})'}\n")
        f.write(f"Collaboration mode: {args.collaboration_mode}\n")
        if args.max_steps:
            f.write(f"Max steps: {args.max_steps}\n")
        if args.switch_interval:
            f.write(f"Switch interval: {args.switch_interval}\n")
        if args.headless:
            f.write("Headless: enabled\n")
        f.write(f"\n{'=' * 80}\n")
        f.write("Summary\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Total tasks: {len(task_ids)}\n")
        if task_ids:
            f.write(f"Success: {successful} ({successful / len(task_ids) * 100:.1f}%)\n")
            f.write(f"Model failure: {failed_model} ({failed_model / len(task_ids) * 100:.1f}%)\n")
            f.write(
                f"External failure: {failed_external} ({failed_external / len(task_ids) * 100:.1f}%) "
                f"[api: {failed_api}, infra: {failed_infra}]\n"
            )
        f.write(f"Copied to {args.outputs_completed_dir}: {copied_to_completed_count}\n")
        f.write(f"Total time: {total_duration:.2f}s ({total_duration / 60:.2f} min)\n")
        f.write(f"Avg time: {avg_duration:.2f}s\n")
        f.write("\nDual-agent aggregate stat.")
        f.write(f"  Agent 1 steps: {aggregate_agent_stats['agent_1_steps']}\n")
        f.write(f"  Agent 2 steps: {aggregate_agent_stats['agent_2_steps']}\n")
        f.write(f"  Communication events: {aggregate_agent_stats['communication_events']}\n")
        f.write(f"  Turn count: {aggregate_agent_stats['turn_count']}\n")
        f.write(f"\n{'=' * 80}\n")
        f.write("Task details\n")
        f.write(f"{'=' * 80}\n\n")
        for i, record in enumerate(task_records, 1):
            status_icon = "✅" if record["success"] else "❌"
            fail_reason = record.get("failure_reason") or "N/A"
            token_str = str(record.get("total_tokens")) if record.get("total_tokens") is not None else "N/A"
            actual_action_str = (
                str(record.get("actual_actions_count"))
                if record.get("actual_actions_count") is not None
                else "N/A"
            )
            golden_action_str = (
                str(record.get("golden_actions_count"))
                if record.get("golden_actions_count") is not None
                else "N/A"
            )
            f.write(
                f"{i:4d}. {status_icon} {record['task_id']:20s} | "
                f"status: {record['status']:15s} | "
                f"duration: {record['duration']:8.2f}s | "
                f"golden_actions: {golden_action_str:>4s} | "
                f"actual_actions: {actual_action_str:>4s} | "
                f"tokens: {token_str:>8s}"
            )
            if record.get("failure_reason") and not record["success"]:
                f.write(f" | reason: {fail_reason}")
            f.write("\n")

    summary_json = {
        "timestamp": timestamp,
        "csv": str(csv_path) if csv_path else None,
        "result_csv": str(result_csv_path) if result_csv_path else None,
        "config": str(config_path),
        "actual_config": str(actual_config_path),
        "agent1_config": args.agent1,
        "agent2_config": args.agent2,
        "output_dir": str(benchmark_output_dir),
        "mode": "sequential" if args.sequential else "parallel",
        "workers": 1 if args.sequential else args.workers,
        "collaboration_mode": args.collaboration_mode,
        "max_steps": args.max_steps,
        "task_timeout_seconds": args.task_timeout_seconds,
        "switch_interval": args.switch_interval,
        "headless": args.headless,
        "history_feedback": getattr(args, "history_feedback", False),
        "llm_history_feedback": getattr(args, "llm_history_feedback", False),
        "image_scale": float(getattr(args, "image_scale", 1.0) or 1.0),
        "image_recent_steps": int(getattr(args, "image_recent_steps", 0) or 0),
        "partner_view": getattr(args, "partner_view", False),
        "partner_view_scale": getattr(args, "partner_view_scale", None),
        "num_shards": shard_config.num_shards,
        "shard_index": shard_config.shard_index,
        "shard_strategy": args.shard_strategy,
        "unsharded_task_count": unsharded_task_count,
        "total_tasks": len(task_ids),
        "successful": successful,
        "failed_model": failed_model,
        "failed_external": failed_external,
        "failed_api": failed_api,
        "failed_infra": failed_infra,
        "copied_to_completed": copied_to_completed_count,
        "duration_seconds": total_duration,
        "avg_duration_seconds": avg_duration,
        "agent_statistics": aggregate_agent_stats,
        "task_records": task_records,
    }
    summary_json_path = benchmark_output_dir / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print("🎉 Dual benchmark complete")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(task_ids)}")
    print(f"Success: {successful}")
    print(f"Model failure: {failed_model}")
    print(f"External failure: {failed_external} (api: {failed_api}, infra: {failed_infra})")
    print(f"Copied to {args.outputs_completed_dir}: {copied_to_completed_count}")
    print(f"Agent 1 steps: {aggregate_agent_stats['agent_1_steps']}")
    print(f"Agent 2 steps: {aggregate_agent_stats['agent_2_steps']}")
    print(f"Communication events: {aggregate_agent_stats['communication_events']}")
    print(f"Turn count: {aggregate_agent_stats['turn_count']}")
    print(f"Output dir: {benchmark_output_dir}")
    print(f"Task logs: {task_logs_dir}")
    print(f"Summary log: {summary_log_path}")
    print(f"Summary json: {summary_json_path}")

    if result_csv_path:
        csv_stats = count_csv_status(result_csv_path)
        print(f"\n{'=' * 80}")
        print("📊 CSV status")
        print(f"{'=' * 80}")
        total_csv = csv_stats["total"]
        if total_csv > 0:
            true_count = csv_stats["true"]
            false_count = csv_stats["false"]
            null_count = csv_stats["null"]
            print(f"Total: {total_csv}")
            print(f"true: {true_count} ({true_count / total_csv * 100:.1f}%)")
            print(f"false: {false_count} ({false_count / total_csv * 100:.1f}%)")
            print(f"null: {null_count} ({null_count / total_csv * 100:.1f}%)")

    print(f"{'=' * 80}\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
