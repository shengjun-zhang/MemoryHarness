#!/bin/bash
# SpatialWorld（AI2-THOR，CloudRendering + Mesa lavapipe 软件Vulkan渲染）冒烟测试
# 集群运行脚本：单worker，验证conda环境/Mesa软件渲染库/AI2-THOR Unity build缓存
# 在CPU集群节点上是否可用（不依赖任何GPU，也不依赖提交所在本地机器的内容，
# 所有依赖均已放在下方引用的dolphinfs绝对路径下）。

set -e

# ---- 带时间戳的日志函数，方便在集群日志里定位"卡在哪个阶段" ----
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

SECONDS=0  # bash内置计时器，配合log_elapsed查看每阶段耗时
log_elapsed() {
  log "$* (累计耗时: ${SECONDS}s)"
}

log "===== 冒烟测试启动 ====="
log "hostname: $(hostname)"

# ================= 1. 基本路径配置 =================
REPO_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/worldmodel/SpatialWorld
ENV_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-ai2thor
# lavapipe (Mesa software Vulkan renderer)：真实 NVIDIA Vulkan ICD 会导致 ai2thor 的
# Unity 进程在 Vulkan 初始化阶段死循环挂起（根因未修复），因此固定使用纯CPU软件渲染。
MESA_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/mesa-vulkan

log "===== [阶段 1/5] 打印基本参数 ====="
log "REPO_DIR: $REPO_DIR"
log "ENV_DIR: $ENV_DIR"
log "MESA_DIR: $MESA_DIR"
log_elapsed "[阶段 1/5] 参数打印完成"

# ================= 2. 解析集群spec（便于日志区分节点，单worker场景下仅作记录） =================
log "===== [阶段 2/5] 解析集群信息 ====="
if [ -n "$AFO_ENV_CLUSTER_SPEC" ]; then
  cluster_spec=${AFO_ENV_CLUSTER_SPEC//\"/\\\"}
  log "cluster spec is $cluster_spec"

  node_rank=$("$ENV_DIR/bin/python" -c "import json; data = json.loads('$cluster_spec'); print(data['index'])" 2>/dev/null || echo 0)
  log "current node rank: $node_rank"
else
  log "未检测到 AFO_ENV_CLUSTER_SPEC（本地调试模式），跳过集群信息解析"
fi
log_elapsed "[阶段 2/5] 集群信息解析完成"

# ================= 3. 环境变量配置（纯CPU软件渲染，不依赖GPU/CUDA） =================
log "===== [阶段 3/5] 激活conda环境 & 设置软件渲染环境变量 ====="
cd "$REPO_DIR"
log "已切换到工作目录: $(pwd)"

source "$ENV_DIR/bin/activate"
log "conda环境已激活: $(which python)"
python --version 2>&1 | while read -r line; do log "python版本: $line"; done

# CloudRendering 平台仍会走 ai2thor 的 gpu_device->Vulkan UUID 映射逻辑，但当前配置未启用
# gpu_device（见 experiments/configs/ai2thor/config_close_doubao-2.yaml），实际渲染完全
# 由下面的 Mesa lavapipe 软件 Vulkan ICD 完成，无需机器上有真实GPU。
export CUDA_VISIBLE_DEVICES=""
export VK_ICD_FILENAMES="$MESA_DIR/share/vulkan/icd.d/lvp_icd.x86_64.json"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$MESA_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$MESA_DIR/bin:$PATH"
# 关闭Python stdout/stderr缓冲，确保print日志实时写出，不会因为缓冲导致
# "看起来卡住了但其实只是日志没flush出来"的误判。
export PYTHONUNBUFFERED=1
# Unity/AI2-THOR 的 FIFO 创建在 /tmp；显式使用容器本地临时盘，避免共享文件系统
# 对 named pipe / file lock 的语义或性能差异影响启动。
export TMPDIR=/tmp
# 消除 Mesa 在无桌面容器中的 XDG_RUNTIME_DIR 警告，并确保目录仅当前用户可访问。
export XDG_RUNTIME_DIR="/tmp/xdg-runtime-${UID}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

log "VK_ICD_FILENAMES: $VK_ICD_FILENAMES"
if [ -f "$VK_ICD_FILENAMES" ]; then
  log "✓ Vulkan ICD 文件存在"
else
  log "✗ 警告: Vulkan ICD 文件不存在，渲染可能会失败"
fi
log "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
log "PATH: $PATH"
log "TMPDIR: $TMPDIR"
log "XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"

# 在启动 Unity 前先验证 Vulkan loader + ICD。vulkaninfo 成功并不保证旧版 Unity
# 一定兼容，但能快速排除 ICD 文件损坏、动态库缺失等基础问题。
if command -v vulkaninfo >/dev/null 2>&1; then
  log "运行 vulkaninfo --summary 预检..."
  if timeout 60 vulkaninfo --summary 2>&1 | while read -r line; do log "  [vulkaninfo] $line"; done; then
    log "✓ Vulkan 预检成功"
  else
    log "✗ Vulkan 预检失败或超过 60s，停止冒烟测试"
    exit 1
  fi
else
  log "✗ 找不到 vulkaninfo，无法执行 Vulkan 预检"
  exit 1
fi
log_elapsed "[阶段 3/5] 环境变量设置完成"

# ================= 4. 检查AI2-THOR Unity build缓存是否命中 =================
log "===== [阶段 4/5] 检查AI2-THOR Unity build缓存 ====="
AI2THOR_CACHE_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/.ai2thor-cache/.ai2thor/releases
if [ -d "$AI2THOR_CACHE_DIR" ]; then
  log "✓ 检测到 Unity build 缓存目录: $AI2THOR_CACHE_DIR"
  ls -la "$AI2THOR_CACHE_DIR" 2>&1 | while read -r line; do log "  $line"; done
else
  log "✗ 警告: 未检测到 Unity build 缓存目录，若网络不通将无法下载 build，任务会在 Controller 初始化阶段卡住"
fi
log_elapsed "[阶段 4/5] 缓存检查完成"

# ================= 5. 运行冒烟测试任务 =================
# 单任务、单步验证：仅跑 open_fridge 一个内置任务预设，确保
#   - conda 环境 / python 解释器可用
#   - Mesa 软件 Vulkan 渲染库可正常加载
#   - AI2-THOR Unity build 缓存（conda-envs/.ai2thor-cache）可直接复用，无需联网下载
#   - 大模型网关（aigc.sankuai.com）网络可达
# 均在集群节点上工作正常。
#
# 用 timeout 包裹主任务：若长时间无响应（例如 Unity 子进程在 Vulkan 初始化阶段
# 死循环挂起），也能在给定时间后强制退出并报错，而不是让整个集群任务无限期挂起。
TASK_TIMEOUT_SECONDS=1800  # 30分钟，单任务冒烟测试的合理上限，可按需调整

log "===== [阶段 5/5] 启动冒烟测试任务 (open_fridge，超时时间: ${TASK_TIMEOUT_SECONDS}s) ====="
set +e
timeout "$TASK_TIMEOUT_SECONDS" python \
  -m scripts.ai2thor.work.run_task \
  --config experiments/configs/ai2thor/config_close_doubao-2.yaml \
  --tasks open_fridge \
  --headless
task_exit_code=$?
set -e

log_elapsed "[阶段 5/5] 冒烟测试任务进程已退出 (exit_code=${task_exit_code})"

if [ $task_exit_code -eq 124 ]; then
  log "✗✗✗ 冒烟测试超时（超过 ${TASK_TIMEOUT_SECONDS}s 未完成），很可能卡在 Unity/Controller 初始化或Vulkan渲染阶段 ✗✗✗"
  UNITY_LOG="$AI2THOR_CACHE_DIR/../log/unity.log"
  PLAYER_LOG="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/.ai2thor-cache/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log"
  for diagnostic_log in "$UNITY_LOG" "$PLAYER_LOG"; do
    if [ -f "$diagnostic_log" ]; then
      log "===== 诊断日志末尾: $diagnostic_log ====="
      tail -n 120 "$diagnostic_log" 2>&1 | while read -r line; do log "  $line"; done
    fi
  done
  exit 124
elif [ $task_exit_code -ne 0 ]; then
  log "✗✗✗ 冒烟测试失败，退出码: ${task_exit_code} ✗✗✗"
  exit $task_exit_code
fi

log "✓✓✓ 冒烟测试完成，全部阶段成功 ✓✓✓"
