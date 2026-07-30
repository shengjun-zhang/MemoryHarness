#!/bin/bash
# SpatialWorld ProcTHOR 冒烟测试：本地 ProcTHOR-10K + CloudRendering + Mesa llvmpipe
set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

SECONDS=0
REPO_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/worldmodel/SpatialWorld
ENV_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-procthor
AI2THOR_ENV_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-ai2thor
MESA_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/mesa-vulkan
PROCTHOR_DATASET_DIR="$REPO_DIR/data/procthor-10k"
AI2THOR_HOME=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/.ai2thor-cache
CONFIG_FILE=experiments/configs/procthor/config_close_doubao-2.yaml
TASK_ID=procthor000
TASK_TIMEOUT_SECONDS=1800

log "===== ProcTHOR 冒烟测试启动 ====="
log "hostname: $(hostname)"
log "REPO_DIR: $REPO_DIR"
log "ENV_DIR: $ENV_DIR"
log "PROCTHOR_DATASET_DIR: $PROCTHOR_DATASET_DIR"

cd "$REPO_DIR"
source "$ENV_DIR/bin/activate"

export PROCTHOR_DATASET_DIR
export AI2THOR_HOME
export AI2THOR_VK_ICD_FILENAMES="$MESA_DIR/share/vulkan/icd.d/lvp_icd.x86_64.json"
export AI2THOR_LD_LIBRARY_PATH="$AI2THOR_ENV_DIR/lib:$MESA_DIR/lib"
export CUDA_VISIBLE_DEVICES=""
export VK_ICD_FILENAMES="$AI2THOR_VK_ICD_FILENAMES"
export LD_LIBRARY_PATH="$AI2THOR_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export PATH="$MESA_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export XDG_RUNTIME_DIR="/tmp/xdg-runtime-${UID}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

log "Python: $(which python)"
python --version 2>&1 | while read -r line; do log "$line"; done

log "===== 检查 ProcTHOR Python 依赖和本地数据集 ====="
python -c "import ai2thor, prior; print('ai2thor=' + ai2thor.__version__); print('prior=installed')"
for split in train val test; do
  dataset_file="$PROCTHOR_DATASET_DIR/${split}.jsonl.gz"
  if [ ! -s "$dataset_file" ]; then
    log "✗ ProcTHOR 数据文件不存在或为空: $dataset_file"
    exit 1
  fi
done
log "✓ ProcTHOR-10K 本地数据集完整"

if [ ! -f "$VK_ICD_FILENAMES" ]; then
  log "✗ Vulkan ICD 文件不存在: $VK_ICD_FILENAMES"
  exit 1
fi
if command -v vulkaninfo >/dev/null 2>&1; then
  log "===== Vulkan 预检 ====="
  timeout 60 vulkaninfo --summary 2>&1 | while read -r line; do log "  $line"; done
else
  log "✗ 找不到 vulkaninfo"
  exit 1
fi

log "===== 启动 ProcTHOR 任务 $TASK_ID（超时 ${TASK_TIMEOUT_SECONDS}s） ====="
set +e
timeout "$TASK_TIMEOUT_SECONDS" python \
  -m scripts.procthor.work.run_task \
  --config "$CONFIG_FILE" \
  --tasks "$TASK_ID" \
  --headless \
  --max-steps 1
exit_code=$?
set -e

log "ProcTHOR 任务进程退出，exit_code=$exit_code，累计耗时=${SECONDS}s"
if [ "$exit_code" -eq 124 ]; then
  log "✗ ProcTHOR 冒烟测试超时"
  PLAYER_LOG="$AI2THOR_HOME/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log"
  UNITY_LOG="$AI2THOR_HOME/.ai2thor/log/unity.log"
  for diagnostic_log in "$UNITY_LOG" "$PLAYER_LOG"; do
    if [ -f "$diagnostic_log" ]; then
      log "===== 诊断日志末尾: $diagnostic_log ====="
      tail -n 120 "$diagnostic_log"
    fi
  done
  exit 124
elif [ "$exit_code" -ne 0 ]; then
  log "✗ ProcTHOR 冒烟测试失败"
  exit "$exit_code"
fi

log "✓✓✓ ProcTHOR 冒烟测试完成：数据集、Unity、Vulkan、首帧和单步 Agent 链路可用 ✓✓✓"
