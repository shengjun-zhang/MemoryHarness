#!/bin/bash
# 【备选方案】在一台真正【有 docker 权限】的物理机/虚拟机上执行本脚本，本地构建并推送
# SpatialWorld AI2-THOR 冒烟测试/CPU集群运行镜像。
#
# 首选方案见 hope/docker/README_镜像平台构建流程：
#   直接在美团内部【镜像平台】(https://mlp.sankuai.com/ml/#/image/list) 网页上，用
#   Dockerfile 类型粘贴 hope/docker/Dockerfile 的内容发起构建（Kaniko在K8s上构建，
#   不需要本机有 docker 权限，构建完自动推送到 hulk 仓库）。CatPaw IDE 所在的开发机
#   本身是跑在K8s里的容器，缺少运行 dockerd 所需的内核权限，因此优先用镜像平台构建；
#   只有你确实找到了一台独立的、有 docker 权限的物理机/虚拟机时，才用本脚本。
#
# 前置条件：
#   1. 该机器已安装 docker，且当前用户有权限执行 `docker build`/`docker push`
#      （即 `docker info` 不报错；若需要 root，请用 sudo 运行本脚本）。
#   2. 该机器能访问内部镜像仓库 registry-offlinebiz.sankuai.com（pull 基础镜像 +
#      push 目标镜像；大数据镜像必须推到这个仓库，见 km.sankuai.com/page/133114854）。
#   3. 已用 `docker login registry-offlinebiz.sankuai.com` 登录过。
#
# 用法：
#   sh build_and_push.sh [镜像地址:tag]
#   # 例如：
#   sh build_and_push.sh registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/zhangshengjun02/spatialworld-ai2thor:v1
#
# 不传参数时使用下面的默认镜像地址（请按需修改为你自己的仓库路径）。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_IMAGE="registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/zhangshengjun02/spatialworld-ai2thor:v1"
IMAGE="${1:-$DEFAULT_IMAGE}"

echo "===== 构建参数 ====="
echo "Dockerfile 目录: $SCRIPT_DIR"
echo "目标镜像地址: $IMAGE"

if ! command -v docker >/dev/null 2>&1; then
  echo "❌ 当前机器未安装 docker，请先在有 docker 权限的机器上运行本脚本。"
  exit 1
fi

echo -e "\n===== 开始构建镜像 ====="
docker build \
  -t "$IMAGE" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$SCRIPT_DIR"

echo -e "\n===== 构建完成，本地验证容器能否正常启动 ====="
docker run --rm "$IMAGE" bash -c "
  echo '--- glibc ---'; ldd --version | head -1
  echo '--- 关键系统库 ---'
  for lib in libuuid.so.1 libstdc++.so.6 libm.so.6; do
    ldconfig -p 2>/dev/null | grep -q \"\$lib\" && echo \"✓ \$lib found\" || echo \"✗ \$lib NOT found\"
  done
"

echo -e "\n===== 推送镜像到仓库 ====="
docker push "$IMAGE"

echo -e "\n✓✓✓ 镜像已推送: $IMAGE"
echo "接下来在 hope/smoke_test.hope 的 [docker] afo.docker.image.name 中填入这个地址即可。"
