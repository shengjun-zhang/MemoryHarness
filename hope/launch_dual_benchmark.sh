#!/usr/bin/env bash
# HOPE/AFO multi-worker launcher for SpatialWorld dual-agent benchmarks.

set -euo pipefail

BENCHMARK_ENV="${BENCHMARK_ENV:-ai2thor}"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    BENCHMARK_ENV="$1"
    shift
fi

IMAGE_SCALE="${IMAGE_SCALE:-}"
IMAGE_RECENT_STEPS="${IMAGE_RECENT_STEPS:-}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --image-scale|-s)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            IMAGE_SCALE="$2"
            shift 2
            ;;
        --image-recent-steps|-k)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            IMAGE_RECENT_STEPS="$2"
            shift 2
            ;;
        --rerun-all)
            RERUN_ALL=1
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: launch_dual_benchmark.sh [ai2thor|procthor] [options]

Options:
  -s, --image-scale SCALE          Historical image scale (0 < SCALE <= 1)
  -k, --image-recent-steps STEPS   Recent historical images kept full-resolution
      --rerun-all                  Re-run every CSV task, including historical results
  -h, --help                       Show this help

The same values can be supplied with IMAGE_SCALE and IMAGE_RECENT_STEPS.
Set RERUN_ALL=1 (or the legacy RERUN=1) for a full re-run.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

REPO_DIR="${REPO_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/worldmodel/SpatialWorld}"
AI2THOR_ENV_DIR="${AI2THOR_ENV_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-ai2thor}"
PROCTHOR_ENV_DIR="${PROCTHOR_ENV_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-procthor}"
MESA_DIR="${MESA_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/mesa-vulkan}"
# NOTE: AI2-THOR guards its Unity build cache (~/.ai2thor/releases/<commit>) with
# fcntl.lockf() advisory locks (ai2thor/util/lock.py) taken on every Controller()
# start (LockEx while downloading/pruning, LockSh while a Unity process is running
# against that release). fcntl/flock semantics are NOT reliably shared across
# nodes/containers on network filesystems such as DolphinFS: if a process holding
# the lock gets SIGKILL'ed (e.g. TASK_TIMEOUT_SECONDS above killing a hung task),
# the lock can be left "stuck" from the NFS/FUSE client's point of view, and every
# other worker/task across the whole cluster that later calls lock_sh()/lock_ex()
# on that same shared release blocks forever inside the (uninterruptible, no
# timeout) fcntl syscall -- this is exactly the "stuck right after
# AI2ThorEnvWrapper.__init__ starting, heartbeat forever, no further log lines"
# symptom. Fix: keep the actual Unity build + its lock files on *local* disk
# (private per node/container), and only use the shared DolphinFS path
# (AI2THOR_SHARED_HOME) as a read-only source to seed that local cache once, via a
# plain file copy (no locking involved).
AI2THOR_SHARED_HOME="${AI2THOR_SHARED_HOME:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/.ai2thor-cache}"
# Forced (not "${AI2THOR_HOME:-...}"): AI2THOR_HOME must always be a *local* disk
# path, never inherited from the outer environment/docker image/previous manual
# `export`, otherwise it can silently end up pointing back at the shared
# DolphinFS cache above (AI2THOR_SHARED_HOME) -- which reintroduces the exact
# cross-node fcntl-lock deadlock this script exists to avoid, and additionally
# makes seed_local_ai2thor_cache() below try to `cp -r` AI2THOR_SHARED_HOME/.ai2thor
# into a temp dir *inside itself* ("cannot create directory
# .../.ai2thor.tmp/.ai2thor/releases/...: No such file or directory" / "Remote I/O
# error" on DolphinFS). If you need a different local cache location, set
# AI2THOR_LOCAL_HOME instead.
AI2THOR_HOME="${AI2THOR_LOCAL_HOME:-/tmp/ai2thor-local-cache-${UID}}"
PROCTHOR_DATASET_DIR="${PROCTHOR_DATASET_DIR:-$REPO_DIR/data/procthor-10k}"
LOCAL_WORKERS="${LOCAL_WORKERS:-1}"
BENCHMARK_TIMEOUT_SECONDS="${BENCHMARK_TIMEOUT_SECONDS:-86400}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-7200}"
SHARD_STRATEGY="${SHARD_STRATEGY:-round_robin}"
# Accept RERUN as a backwards-compatible alias. Command-line --rerun-all takes
# precedence because it is parsed above and sets RERUN_ALL explicitly.
RERUN_ALL="${RERUN_ALL:-${RERUN:-0}}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Seed the local AI2-THOR cache (Unity build + cuda-vulkan-mapping.json) from the
# shared DolphinFS path exactly once per node/container. Guarded by a *local*
# flock (reliable on local disk, unlike the shared network path) so that multiple
# LOCAL_WORKERS processes on the same node don't race to copy concurrently.
seed_local_ai2thor_cache() {
    # Safety net: refuse to run if AI2THOR_HOME somehow still resolves onto the
    # shared path (or is a parent/child of it) -- copying a directory into itself
    # produces exactly the "No such file or directory" / "Remote I/O error" mess
    # seen when AI2THOR_HOME leaked in from an outer environment variable.
    local shared_real local_real
    shared_real="$(readlink -f -- "$AI2THOR_SHARED_HOME" 2>/dev/null || echo "$AI2THOR_SHARED_HOME")"
    mkdir -p "$AI2THOR_HOME"
    local_real="$(readlink -f -- "$AI2THOR_HOME" 2>/dev/null || echo "$AI2THOR_HOME")"
    case "$local_real/" in
        "$shared_real/"*|"$shared_real")
            log "FATAL: AI2THOR_HOME ($AI2THOR_HOME -> $local_real) resolves inside the shared AI2THOR_SHARED_HOME ($AI2THOR_SHARED_HOME -> $shared_real); refusing to self-copy. Unset any pre-existing AI2THOR_HOME env var, or set AI2THOR_LOCAL_HOME to a private local-disk path instead."
            exit 1
            ;;
    esac
    if [ ! -d "$AI2THOR_SHARED_HOME/.ai2thor" ]; then
        log "No shared AI2-THOR cache found at $AI2THOR_SHARED_HOME/.ai2thor, skipping local seed (will download fresh)"
        return 0
    fi
    local seed_lock="$AI2THOR_HOME/.seed.lock"
    local marker="$AI2THOR_HOME/.ai2thor/.seeded_from_shared"
    (
        flock -w 900 200 || { log "Timed out waiting for local AI2-THOR cache seed lock: $seed_lock"; exit 1; }
        if [ -f "$marker" ]; then
            log "Local AI2-THOR cache already seeded: $AI2THOR_HOME/.ai2thor"
        else
            log "Seeding local AI2-THOR cache: $AI2THOR_SHARED_HOME/.ai2thor -> $AI2THOR_HOME/.ai2thor (one-time local disk copy)"
            rm -rf "$AI2THOR_HOME/.ai2thor.tmp"
            cp -r "$AI2THOR_SHARED_HOME/.ai2thor" "$AI2THOR_HOME/.ai2thor.tmp"
            # Drop any lock files copied from the shared cache: they are meaningless
            # (and potentially "stuck") outside of the original network-fs context,
            # and ai2thor recreates them on demand via os.open(..., O_CREAT).
            find "$AI2THOR_HOME/.ai2thor.tmp" -name '*.lock' -delete
            rm -rf "$AI2THOR_HOME/.ai2thor"
            mv "$AI2THOR_HOME/.ai2thor.tmp" "$AI2THOR_HOME/.ai2thor"
            touch "$marker"
            log "Local AI2-THOR cache seeded successfully"
        fi
    ) 200>"$seed_lock"
}

case "$BENCHMARK_ENV" in
    ai2thor)
        ENV_DIR="$AI2THOR_ENV_DIR"
        MODULE="mllm_base_agent.dual_agent.ai2thor.run_benchmark"
        CSV_FILE="${CSV_FILE:-$REPO_DIR/experiments/csv/ai2thor/Spatial-Annotation-ai2thor-doubao-seed-2.0-lite.csv}"
        CONFIG_FILE="${CONFIG_FILE:-$REPO_DIR/experiments/configs/ai2thor/dual/config_close_doubao-2.yaml}"
        OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/mllm_base_agent/dual_agent/ai2thor/benchmark_outputs}"
        OUTPUTS_COMPLETED_DIR="${OUTPUTS_COMPLETED_DIR:-$REPO_DIR/mllm_base_agent/dual_agent/ai2thor/outputs_completed}"
        ;;
    procthor)
        ENV_DIR="$PROCTHOR_ENV_DIR"
        MODULE="mllm_base_agent.dual_agent.procthor.run_benchmark"
        CSV_FILE="${CSV_FILE:-$REPO_DIR/experiments/csv/procthor/dual/Spatial-Annotation-procthor-Gpt-5p4.csv}"
        CONFIG_FILE="${CONFIG_FILE:-$REPO_DIR/experiments/configs/procthor/dual/config_close_doubao-2.yaml}"
        OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/mllm_base_agent/dual_agent/procthor/benchmark_outputs}"
        OUTPUTS_COMPLETED_DIR="${OUTPUTS_COMPLETED_DIR:-$REPO_DIR/mllm_base_agent/dual_agent/procthor/outputs_completed}"
        ;;
    *)
        log "Unsupported BENCHMARK_ENV=$BENCHMARK_ENV (expected ai2thor or procthor)"
        exit 2
        ;;
esac

PYTHON_BIN="$ENV_DIR/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    log "Python interpreter not found: $PYTHON_BIN"
    exit 1
fi
if [ -n "$IMAGE_SCALE" ] && ! "$PYTHON_BIN" -c 'import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if 0.0 < value <= 1.0 else 1)' "$IMAGE_SCALE"; then
    log "Invalid image scale: $IMAGE_SCALE (expected 0 < scale <= 1)"
    exit 2
fi
if [ -n "$IMAGE_RECENT_STEPS" ] && ! [[ "$IMAGE_RECENT_STEPS" =~ ^[0-9]+$ ]]; then
    log "Invalid image recent steps: $IMAGE_RECENT_STEPS (expected a non-negative integer)"
    exit 2
fi
if [ ! -f "$CSV_FILE" ]; then
    log "CSV file not found: $CSV_FILE"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    log "Config file not found: $CONFIG_FILE"
    exit 1
fi

# AFO_ENV_CLUSTER_SPEC example:
# {"index":"0", "role":"worker", "worker":["host0:port", "host1:port"]}
read -r NUM_SHARDS SHARD_INDEX RUN_KEY < <(
    "$PYTHON_BIN" -c 'import hashlib, json, os
raw = os.environ.get("AFO_ENV_CLUSTER_SPEC", "")
if raw:
    data = json.loads(raw)
    workers = data.get("worker", [])
    stable_spec = {key: value for key, value in data.items() if key not in ("index", "rank", "worker_index", "task_index")}
    run_key = hashlib.sha256(json.dumps(stable_spec, sort_keys=True).encode()).hexdigest()[:16]
    print(len(workers) or 1, int(data.get("index", 0)), run_key)
else:
    print(1, 0, "local")'
)

export TASK_NUM_SHARDS="$NUM_SHARDS"
export TASK_SHARD_INDEX="$SHARD_INDEX"
export PROCTHOR_DATASET_DIR
export AI2THOR_HOME
export AI2THOR_VK_ICD_FILENAMES="$MESA_DIR/share/vulkan/icd.d/lvp_icd.x86_64.json"
export AI2THOR_LD_LIBRARY_PATH="$AI2THOR_ENV_DIR/lib:$MESA_DIR/lib"
export CUDA_VISIBLE_DEVICES=""
export VK_ICD_FILENAMES="$AI2THOR_VK_ICD_FILENAMES"
export LD_LIBRARY_PATH="$AI2THOR_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export PATH="$MESA_DIR/bin:$PATH"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export XDG_RUNTIME_DIR="/tmp/xdg-runtime-${UID}"
mkdir -p "$XDG_RUNTIME_DIR" "$OUTPUT_DIR" "$OUTPUTS_COMPLETED_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

seed_local_ai2thor_cache

SAVE_NAME="${SAVE_NAME:-${BENCHMARK_ENV}_dual_cluster}"

# All workers share one experiment timestamp, so their shard directories are
# grouped under benchmark_outputs/<save_name>_<time>/.
if [ "$NUM_SHARDS" -gt 1 ]; then
    RUN_TIMESTAMP_DIR="$OUTPUT_DIR/.run_timestamps"
    RUN_TIMESTAMP_FILE="$RUN_TIMESTAMP_DIR/${SAVE_NAME}_${RUN_KEY}"
    mkdir -p "$RUN_TIMESTAMP_DIR"
    if [ "$SHARD_INDEX" -eq 0 ]; then
        BENCHMARK_RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
        RUN_TIMESTAMP_TMP="${RUN_TIMESTAMP_FILE}.tmp.$$"
        printf '%s\n' "$BENCHMARK_RUN_TIMESTAMP" > "$RUN_TIMESTAMP_TMP"
        mv -f "$RUN_TIMESTAMP_TMP" "$RUN_TIMESTAMP_FILE"
    else
        for _ in $(seq 1 120); do
            if [ -s "$RUN_TIMESTAMP_FILE" ]; then
                break
            fi
            sleep 1
        done
        if [ ! -s "$RUN_TIMESTAMP_FILE" ]; then
            log "Timed out waiting for shared benchmark timestamp: $RUN_TIMESTAMP_FILE"
            exit 1
        fi
        BENCHMARK_RUN_TIMESTAMP="$(tr -d '[:space:]' < "$RUN_TIMESTAMP_FILE")"
    fi
else
    BENCHMARK_RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
fi
if [[ ! "$BENCHMARK_RUN_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    log "Invalid shared benchmark timestamp: $BENCHMARK_RUN_TIMESTAMP"
    exit 1
fi
export BENCHMARK_RUN_TIMESTAMP

log "===== SpatialWorld distributed dual-agent benchmark ====="
log "hostname=$(hostname) environment=$BENCHMARK_ENV"
log "cluster_spec=${AFO_ENV_CLUSTER_SPEC:-local}"
log "shard=$SHARD_INDEX/$NUM_SHARDS strategy=$SHARD_STRATEGY local_workers=$LOCAL_WORKERS"
log "python=$PYTHON_BIN"
log "csv=$CSV_FILE"
log "config=$CONFIG_FILE"
log "output_dir=$OUTPUT_DIR"
log "save_name=$SAVE_NAME"
log "benchmark_run_timestamp=$BENCHMARK_RUN_TIMESTAMP"
if [ "$RERUN_ALL" = "1" ]; then
    log "skip_completed=0 rerun_all=1"
else
    log "skip_completed=1 rerun_all=0"
fi
if [ -n "$IMAGE_SCALE" ]; then
    log "image_scale=$IMAGE_SCALE"
fi
if [ -n "$IMAGE_RECENT_STEPS" ]; then
    log "image_recent_steps=$IMAGE_RECENT_STEPS"
fi

if [ ! -f "$VK_ICD_FILENAMES" ]; then
    log "Vulkan ICD not found: $VK_ICD_FILENAMES"
    exit 1
fi
if [ "$BENCHMARK_ENV" = "procthor" ]; then
    for split in train val test; do
        if [ ! -s "$PROCTHOR_DATASET_DIR/$split.jsonl.gz" ]; then
            log "ProcTHOR dataset file missing: $PROCTHOR_DATASET_DIR/$split.jsonl.gz"
            exit 1
        fi
    done
fi

cmd=(
    "$PYTHON_BIN" -m "$MODULE"
    --csv "$CSV_FILE"
    --config "$CONFIG_FILE"
    --output-dir "$OUTPUT_DIR"
    --outputs-completed-dir "$OUTPUTS_COMPLETED_DIR"
    --save-name "$SAVE_NAME"
    --num-shards "$NUM_SHARDS"
    --shard-index "$SHARD_INDEX"
    --shard-strategy "$SHARD_STRATEGY"
    --workers "$LOCAL_WORKERS"
    --task-timeout-seconds "$TASK_TIMEOUT_SECONDS"
    --headless
)

# `--skip-completed` 会额外扫描 OUTPUT_DIR 下所有历史 benchmark 运行目录，只要
# 某个 task 在磁盘上任意一次历史跑批中已经产出过 dual_episode_*.json 结果文件，就
# 会被跳过。全量重跑时绝不能传这个参数：`--rerun-all` 会读取 CSV 的所有任务，
# 且不扫描历史结果。因此两个参数在启动器中互斥，避免依赖 runner 的隐式忽略逻辑。
if [ "$RERUN_ALL" = "1" ]; then
    cmd+=(--rerun-all)
else
    cmd+=(--skip-completed)
fi
if [ "${SEQUENTIAL:-0}" = "1" ]; then
    cmd+=(--sequential)
fi
if [ "${HISTORY_FEEDBACK:-0}" = "1" ]; then
    cmd+=(--history-feedback)
fi
if [ "${LLM_HISTORY_FEEDBACK:-0}" = "1" ]; then
    cmd+=(--llm-history-feedback)
fi
if [ -n "$IMAGE_SCALE" ]; then
    cmd+=(--image-scale "$IMAGE_SCALE")
fi
if [ -n "$IMAGE_RECENT_STEPS" ]; then
    cmd+=(--image-recent-steps "$IMAGE_RECENT_STEPS")
fi
if [ -n "${MAX_STEPS:-}" ]; then
    cmd+=(--max-steps "$MAX_STEPS")
fi

log "command: ${cmd[*]}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    log "DRY_RUN=1, validation completed without starting benchmark"
    exit 0
fi

set +e
timeout --signal=TERM --kill-after=120 "$BENCHMARK_TIMEOUT_SECONDS" "${cmd[@]}"
exit_code=$?
set -e

if [ "$exit_code" -eq 124 ] || [ "$exit_code" -eq 137 ]; then
    log "Benchmark shard timed out after ${BENCHMARK_TIMEOUT_SECONDS}s"
elif [ "$exit_code" -ne 0 ]; then
    log "Benchmark shard failed with exit_code=$exit_code"
else
    log "Benchmark shard completed successfully"
fi
exit "$exit_code"
