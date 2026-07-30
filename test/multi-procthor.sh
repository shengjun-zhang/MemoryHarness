#!/bin/bash
set -e

ENV_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-procthor
AI2THOR_ENV_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-ai2thor
# lavapipe (Mesa software Vulkan renderer). The real NVIDIA Vulkan ICD is available at
# conda-envs/nvidia-vulkan-userspace (extracted from nvidia-driver-branch-535-libs-535.129.03),
# but the ai2thor Unity process hangs in a busy-wait loop during Vulkan init when using it
# (root cause not yet fixed), so we fall back to software rendering for now.
MESA_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/mesa-vulkan

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/worldmodel/SpatialWorld

source "$ENV_DIR/bin/activate"

export PROCTHOR_DATASET_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/worldmodel/SpatialWorld/data/procthor-10k
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES="$MESA_DIR/share/vulkan/icd.d/lvp_icd.x86_64.json"
export LD_LIBRARY_PATH="$AI2THOR_ENV_DIR/lib:$MESA_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$MESA_DIR/bin:$PATH"

python \
  -m mllm_base_agent.dual_agent.procthor.run_benchmark \
  --csv experiments/csv/procthor/dual/Spatial-Annotation-procthor-Gpt-5p4.csv \
  --config experiments/configs/procthor/dual/config_close_doubao-2.yaml \
  --headless \
  --workers 1 \
  --rerun-all \
  --collaboration-mode alternating \
  --history-feedback \
  --llm-history-feedback \
  --image-scale 0.5 \
  --partner-view \
  --save-name doubao_procthor_dual_history_feedback_llm_partner_view
