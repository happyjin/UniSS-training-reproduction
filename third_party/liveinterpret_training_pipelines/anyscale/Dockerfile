# =============================================================================
# Qwen3-OmniSpeak training image for Anyscale on AWS SageMaker HyperPod
# Build: anyscale image build -n qwen3-omnispeak-d -f anyscale/Dockerfile
# Output: anyscale/image/qwen3-omnispeak-d:<auto-version>
#
# DEPENDENCY SOURCE-OF-TRUTH: anyscale/requirements.txt
# This Dockerfile mirrors that file byte-for-byte in the heredoc below.
# =============================================================================

# Layer 0: Anyscale-managed base. Ray 2.55.1 + Python 3.12 + CUDA 12.6.
FROM anyscale/ray:2.55.1-slim-py312-cu126

WORKDIR /home/ray

# Anyscale's container builder rejects SHELL and COPY from local build context.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

# Layer 1: System packages
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        unzip \
        ffmpeg \
        libsndfile1 \
        sox \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Layer 1b: AWS EFA installer (libfabric + aws-ofi-nccl)
RUN cd /tmp \
    && curl -fsSL -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz \
    && tar -xf aws-efa-installer-latest.tar.gz \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
           ./aws-efa-installer/DEBS/UBUNTU2204/x86_64/rdma-core/*.deb \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --skip-rdma-core --no-verify \
    && rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer-latest.tar.gz \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-} \
    PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:${PATH} \
    FI_PROVIDER=efa \
    NCCL_PROTO=simple

USER ray

# Layer 3: pip build prerequisites
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install pybind11 ninja packaging

# Layer 4: Project Python dependencies
RUN cat > /tmp/requirements.txt <<'EOF'
# ---- Torch (anchor) ----
torch==2.11.0
torchaudio==2.11.0
torchvision==0.26.0

# ---- CUDA toolchain ----
nvidia-cuda-nvcc==13.0.88
nvidia-nvvm==13.0.88
nvidia-cuda-crt==13.0.88
nvidia-cuda-cccl>=13.0,<13.1

# ---- Frameworks ----
ms-swift @ git+https://github.com/modelscope/ms-swift.git@main
transformers==4.57.3
deepspeed==0.18.9
ray==2.55.1
modelscope==1.37.1

# ---- Megatron path ----
megatron-core==0.17.0
mcore-bridge==1.4.1

# ---- Audio / Video IO ----
soundfile==0.13.1
librosa==0.11.0
decord==0.6.0
qwen_omni_utils==0.0.9

# ---- Codec tokenizer ----
qwen-tts==0.1.1

# ---- Dataset tooling ----
pandas==2.3.3
einops==0.8.2
huggingface_hub==0.36.2

# ---- Experiment logging ----
wandb==0.27.0
swanlab==0.7.19
tensorboard==2.20.0
EOF

# Sanity-check: heredoc must not be empty
RUN test "$(grep -c '==' /tmp/requirements.txt)" -gt 5 \
    && echo "[image-build] requirements.txt has $(grep -c '==' /tmp/requirements.txt) pinned packages"

RUN pip install --pre -r /tmp/requirements.txt

# Layer 4b: symlink /usr/local/cuda → pip-installed CUDA toolchain
USER root
RUN set -e \
    && NV_ROOT="/home/ray/anaconda3/lib/python3.12/site-packages/nvidia" \
    && test -d "${NV_ROOT}" \
    && mkdir -p /usr/local/cuda/bin /usr/local/cuda/lib /usr/local/cuda/include \
    && for d in "${NV_ROOT}"/*/bin; do \
           [ -d "$d" ] || continue; \
           for f in "$d"/*; do \
               [ -e "$f" ] || continue; \
               ln -sfn "$f" "/usr/local/cuda/bin/$(basename "$f")"; \
           done; \
       done \
    && for d in "${NV_ROOT}"/*/lib "${NV_ROOT}"/*/lib64; do \
           [ -d "$d" ] || continue; \
           for f in "$d"/*; do \
               [ -e "$f" ] || continue; \
               ln -sfn "$f" "/usr/local/cuda/lib/$(basename "$f")"; \
           done; \
       done \
    && ln -sfn lib /usr/local/cuda/lib64 \
    && for d in "${NV_ROOT}"/*/include; do \
           [ -d "$d" ] || continue; \
           for f in "$d"/*; do \
               [ -e "$f" ] || continue; \
               ln -sfn "$f" "/usr/local/cuda/include/$(basename "$f")"; \
           done; \
       done \
    && for d in "${NV_ROOT}"/*/nvvm; do \
           [ -d "$d" ] || continue; \
           ln -sfn "$d" /usr/local/cuda/nvvm; \
       done \
    && test -x /usr/local/cuda/bin/nvcc \
    && test -x /usr/local/cuda/bin/ptxas \
    && test -x /usr/local/cuda/nvvm/bin/cicc \
    && test -f /usr/local/cuda/nvvm/libdevice/libdevice.10.bc \
    && test -d /usr/local/cuda/lib64 \
    && /usr/local/cuda/bin/nvcc --version \
    && /usr/local/cuda/bin/ptxas --version \
    && if ! /usr/local/cuda/bin/ptxas --version | grep -q "release 13\."; then \
           echo "[image-build] FATAL: ptxas is not cu13"; exit 1; \
       fi

ENV PATH=/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda

# Layer 4c: aws-ofi-nccl plugin enablement
RUN test -f /usr/local/cuda/lib/libcudart.so.13 \
    && ln -sfn libcudart.so.13 /usr/local/cuda/lib/libcudart.so \
    && echo "[image-build] libcudart.so symlink created"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:${LD_LIBRARY_PATH}
USER ray

# Layer 5: FlashAttention 3 (Hopper) — compile from source
RUN git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/fa \
    && cd /tmp/fa/hopper \
    && MAX_JOBS=4 \
       NVCC_THREADS=2 \
       FLASH_ATTENTION_FORCE_BUILD=TRUE \
       FLASH_ATTENTION_DISABLE_SM80=TRUE \
       FLASH_ATTENTION_DISABLE_FP8=TRUE \
       FLASH_ATTENTION_DISABLE_FP16=TRUE \
       FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
       FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
       FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
       FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
       FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
       FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
       FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
       FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
       FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
       FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
       FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
       FLASH_ATTENTION_DISABLE_CLUSTER=TRUE \
       pip install --no-build-isolation -v . \
    && rm -rf /tmp/fa 

# Verify FA3 installation
RUN python -c "import importlib.util as u; \
spec_pkg=u.find_spec('flash_attn_3'); \
spec_mod=u.find_spec('flash_attn_interface'); \
assert spec_pkg is not None, 'flash_attn_3 package metadata missing'; \
assert spec_mod is not None, 'flash_attn_interface module missing'; \
print('[image-build] FA3 OK:', spec_pkg.origin or 'pkg-only', '|', spec_mod.origin)"

# No ENTRYPOINT / CMD — Anyscale Job provides entrypoint.