# SpatialWorld AI2-THOR 冒烟测试 / CPU 集群运行镜像
#
# 本 Dockerfile 按照美团内部【镜像平台】(https://mlp.sankuai.com/ml/#/image/list) 的
# 构建方式编写：https://km.sankuai.com/collabpage/2475964536
#   - 在镜像平台"新建镜像"时，第一步单独选择下面 FROM 指定的基础镜像；
#     如果平台要求 Dockerfile 内容里不含 FROM，把下面的 FROM 行删掉，
#     其余 RUN/ENV 指令原样粘贴到 Dockerfile 编辑框即可。
#   - 镜像平台默认用 Kaniko 在 K8s 上构建，构建机默认【无法访问外网】，
#     因此本文件只使用 yum 安装依赖（走内网源），不做任何 curl/wget 外网下载，
#     避免构建失败（如需要访问外网站点，需要在平台高级参数里加 --enable-network，
#     这里刻意不依赖它，保持构建稳定）。
#   - 构建空间限制 50GB（含中间层），本文件只装几个系统包，体积很小，不会触发限制。
#
# 目标：把"跑通 AI2-THOR + Mesa lavapipe 软件Vulkan渲染"所需的系统级依赖固化进镜像，
# 这样集群节点不再需要机器上恰好装好 libuuid 等系统库才能跑。
#
# 有意不打进镜像的内容（仍在运行时从 dolphinfs 挂载读取，保持和现有裸机运行方式一致）：
#   - Python venv 本身（conda-envs/spatialworld-ai2thor，由 uv 管理）
#   - Mesa lavapipe 软件 Vulkan 渲染库（conda-envs/mesa-vulkan）
#   - AI2-THOR 已下载好的 Unity build 缓存（conda-envs/.ai2thor-cache）
#   - 代码仓库本身（worldmodel/SpatialWorld）
# 这些都在 dolphinfs 共享挂载路径下，AFO 容器默认会挂载启动任务租户的 dolphinfs，
# 因此镜像内无需重复打包，保持镜像小、构建快，且和裸机验证过的产物完全一致。

FROM registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/camera3d/training_cuda12.4-nccl2.21.5-glibc2.28-prod_918d6595:v1.0.0

USER root

# ---------------------------------------------------------------------------
# 系统依赖（走 yum 内网源，不依赖外网）：
#   - libuuid, libstdc++: AI2-THOR Unity build（UnityPlayer.so / 可执行文件）运行时链接依赖
#     （已用 ldd 实测确认这两个是 Unity 可执行文件的动态库依赖）
#   - ca-certificates: 访问 aigc.sankuai.com（内部大模型网关）等 HTTPS 服务需要
# 基础镜像已是 glibc 2.28（与当前裸机验证环境一致）。libvulkan.so.1 本身不在此安装，
# 而是运行时通过 VK_ICD_FILENAMES / LD_LIBRARY_PATH 指向 dolphinfs 上的
# conda-envs/mesa-vulkan（软件 Vulkan ICD），与裸机跑法保持一致，避免镜像里
# 装的 libvulkan 版本和 Mesa lavapipe ICD 不匹配。
# ---------------------------------------------------------------------------
RUN yum install -y libuuid libstdc++ ca-certificates procps-ng \
    && yum clean all \
    && (update-ca-trust 2>/dev/null || true)

# 常用环境变量：ai2thor_ld_library_path / ai2thor_vk_icd_filenames 的默认值，
# 避免每个 worker.script 都要重复设置一遍。任务仍可通过配置文件/环境变量覆盖。
#
# 注意：这里刻意不再设置 AI2THOR_HOME 指向共享 dolphinfs 路径！
# ai2thor 用 fcntl.lockf() 保护它的 Unity build 缓存目录（ai2thor/util/lock.py，
# Controller() 内部 self._build.lock_sh()），而 fcntl/flock 语义在 dolphinfs 这类
# 网络文件系统上并不能保证跨节点/跨容器可靠：一旦某个持锁进程被信号强杀（例如任务超时被
# kill -9），锁可能从 NFS/FUSE 客户端角度"卡死"，导致集群里其他所有后续启动 AI2-THOR
# 的任务都永久阻塞在 Controller() 构造阶段（现象：日志停在
# "AI2ThorEnvWrapper.__init__ starting" 之后，只有 heartbeat，没有任何新日志）。
# hope/launch_dual_benchmark.sh 和 hope/launch_single_benchmark.sh 会在运行时强制把
# AI2THOR_HOME 指向每个 node/container 私有的本地磁盘路径（/tmp/ai2thor-local-cache-*），
# 并通过 seed_local_ai2thor_cache() 从共享缓存做一次性文件拷贝（无锁语义），而不是直接
# 复用这个共享路径。如果在这里重新把 AI2THOR_HOME 设成共享路径，会覆盖脚本里的本地路径
# 默认值（因为 shell 里 "${AI2THOR_LOCAL_HOME:-...}" 判断的是 AI2THOR_LOCAL_HOME 不是
# AI2THOR_HOME，本身不受这个 ENV 影响），但更要注意 experiments/configs/*/config_close_doubao-2.yaml
# 里的 env.ai2thor_home 若被误设置，其优先级高于 AI2THOR_HOME 环境变量
# （wrapper.py: env_config.get("ai2thor_home", os.environ.get("AI2THOR_HOME"))），
# 同样会绕过本地路径，请勿在 YAML 里恢复这个字段。
ENV AI2THOR_VK_ICD_FILENAMES=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/mesa-vulkan/share/vulkan/icd.d/lvp_icd.x86_64.json \
    AI2THOR_LD_LIBRARY_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/spatialworld-ai2thor/lib:/mnt/dolphinfs/ssd_pool/docker/user/hadoop-videogen-hl/hadoop-camera3d/zhangshengjun/conda-envs/mesa-vulkan/lib \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
