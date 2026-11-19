# =============================================================================
# ktrain 训练镜像（单一镜像，含全部依赖）
# -----------------------------------------------------------------------------
# Base: nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04（CUDA 12.8 + nvcc + py3.12）。
#
# 分层（利用缓存，大层在前、易变的应用依赖在尾部）：
#   1. 系统依赖 + Python venv
#   2. torch 2.8.0 cu128（PyTorch cu128 index）
#   3. Transformer Engine 2.16.0（源码构建，依赖 torch）
#   4. FlashAttention-2 预编译 wheel
#   5. AWS EFA net stack（libfabric/OpenMPI/aws-ofi-nccl + libnccl-net.so 软链）
#   6. AWS CLI v2（sync_artifact.sh 的 s3-direct 模式需要）
#   7. 应用依赖：`COPY requirements.txt` + `pip install -r`（单一事实源，
#      含 ms-swift / transformers / deepspeed / megatron-core / ray[default] /
#      liger-kernel / pylance / wandb 等）
#
# ray / liger 均在镜像内，是否启用由运行时 env 控制：
#   ray   → framework: ray 的 KubeRay RayJob 才用；
#   liger → 任务 yaml 里 USE_LIGER_KERNEL=true 才生效。
#
# 构建（context = repo 根目录）：
#   bash ktrain/image/build_image.sh Dockerfile <tag>
# =============================================================================

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.9;9.0" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /workspace

# ---- 系统依赖 + Python 3.12 venv ----
# wget 是 KubeRay head/worker 就绪探针所需（打 local_raylet_healthz/gcs_healthz）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates unzip \
        ffmpeg libsndfile1 sox pkg-config cmake rsync wget \
        python3.12 python3.12-venv python3.12-dev python3-pip \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:${PATH}

# ---- PyTorch 2.8（cu128）----
RUN pip install pybind11==3.0.1 ninja==1.11.1.4 packaging==25.0 \
    && pip install \
        torch==2.8.0+cu128 \
        torchaudio==2.8.0+cu128 \
        torchvision==0.23.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128

# ---- Transformer Engine（mcore-bridge 在 import 时无条件 import；源码构建）----
RUN MAX_JOBS=6 pip install --no-build-isolation \
        "transformer_engine[pytorch,core_cu12]==2.16.0"

# ---- FlashAttention 2（预编译 wheel，不源码编译）----
RUN pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# ---- AWS EFA（libfabric / OpenMPI / aws-ofi-nccl 插件）----
# 安装器自带的 rdma-core 版本（≥59）高于 ubuntu24.04 系统源（57），先用安装器
# 自带 deb 预装再跑安装器并跳过其 rdma-core。aws-ofi-nccl 插件名是
# libnccl-net-ofi.so，而 NCCL 默认找 libnccl-net.so，这里建软链（否则回退 TCP）。
RUN cd /tmp \
    && curl -fsSL -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz \
    && tar -xf aws-efa-installer-latest.tar.gz \
    && apt-get update \
    && RDMADIR=./aws-efa-installer/DEBS/UBUNTU2404/x86_64/rdma-core \
    && { [ -d "$RDMADIR" ] || RDMADIR=$(ls -d ./aws-efa-installer/DEBS/UBUNTU*/x86_64/rdma-core 2>/dev/null | head -1); } \
    && apt-get install -y --no-install-recommends "$RDMADIR"/*.deb \
    && apt-get remove -y libnccl-ofi || true \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --skip-rdma-core --no-verify \
    && OFI_DIR=/opt/amazon/ofi-nccl/lib \
    && if [ -f "${OFI_DIR}/libnccl-net-ofi.so" ] && [ ! -e "${OFI_DIR}/libnccl-net.so" ]; then ln -s libnccl-net-ofi.so "${OFI_DIR}/libnccl-net.so"; fi \
    && rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer-latest.tar.gz \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}

# ---- AWS CLI v2 ----
RUN cd /tmp \
    && curl -fsSL -o awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli \
    && rm -rf /tmp/aws /tmp/awscliv2.zip \
    && aws --version

# ---- 应用依赖（尾部；requirements.txt 为单一事实源）----
# context 为 repo 根目录（build_image.sh 已切到 repo 根）。
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# 验证关键依赖均可导入
RUN python -c "import torch, flash_attn, transformer_engine, transformer_engine.pytorch, ray, liger_kernel, lance; print('torch', torch.__version__, '| ray', ray.__version__, '| TE', transformer_engine.__version__)"

# No ENTRYPOINT / CMD — ktrain PyTorchJob / KubeRay RayJob supplies the command.
