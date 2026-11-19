#!/usr/bin/env bash
# E10: build the dedicated venv `simuls2s` for the LiveInterpret S5 multi-speaker baseline.
# User-approved on 2026-11-19. Never touches any other environment.
# Refuses to run if the venv already exists (no overwrite); delete nothing.
#
#   setsid nohup bash e10_multispk/setup/setup_simuls2s.sh > e10_multispk/runs/setup_simuls2s.log 2>&1 &
set -euo pipefail
V=${SIMULS2S_VENV:-/opt/dlami/nvme/jasonleeeli/venvs/simuls2s}
BASE_PY=${BASE_PY:-/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python}   # Python 3.12
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -e "${V}" ]]; then
    echo "[setup] ${V} already exists; refusing to overwrite" >&2
    exit 1
fi
"${BASE_PY}" -m venv "${V}"
PIP=("${V}/bin/python" -m pip)
"${PIP[@]}" install -U pip wheel setuptools

# Pin the training stack (LiveInterpret requirements.txt, ms-swift 4.4.0 image).
cat > "${V}/constraints.txt" <<'EOF'
torch==2.8.0
torchaudio==2.8.0
transformers==4.57.3
tokenizers==0.22.1
accelerate==1.10.1
peft==0.15.2
numpy==2.2.6
EOF

"${PIP[@]}" install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

"${PIP[@]}" install -c "${V}/constraints.txt" \
    ms-swift==4.4.0 transformers==4.57.3 tokenizers==0.22.1 accelerate==1.10.1 modelscope==1.37.1 \
    datasets==3.6.0 peft==0.15.2 trl==0.17.0 deepspeed==0.18.9 soundfile==0.13.1 librosa==0.11.0 \
    einops==0.8.2 pandas==2.3.3 huggingface_hub==0.36.2 numpy==2.2.6 scipy==1.16.3 tensorboard==2.20.0 \
    liger-kernel==0.5.10 silero-vad==6.2.3 onnxruntime

# CosyVoice3 flow / HiFT / CAM++ / speech tokenizer (vocoder side only; no CosyVoice LLM).
"${PIP[@]}" install -c "${V}/constraints.txt" \
    HyperPyYAML==1.2.2 conformer==0.3.2 diffusers==0.29.0 omegaconf==2.3.0 hydra-core==1.3.2 \
    lightning==2.2.4 x-transformers inflect gdown matplotlib networkx openai-whisper pyworld wget \
    rich protobuf pydantic

# Qwen3.5 linear-attention fast path (optional: the code falls back to the torch kernels).
"${PIP[@]}" install -c "${V}/constraints.txt" flash-linear-attention \
    || echo "[setup][WARN] flash-linear-attention failed; torch fallback will be used"
"${PIP[@]}" install -c "${V}/constraints.txt" --no-build-isolation causal-conv1d \
    || echo "[setup][WARN] causal-conv1d failed; torch fallback will be used"

"${V}/bin/python" - <<'PY'
import importlib
for m in ['torch', 'transformers', 'peft', 'swift', 'deepspeed', 'soundfile', 'librosa', 'onnxruntime',
          'hyperpyyaml', 'conformer', 'diffusers', 'omegaconf', 'whisper', 'fla', 'causal_conv1d', 'silero_vad']:
    try:
        mod = importlib.import_module(m)
        print(f'[setup] OK {m} {getattr(mod, "__version__", "")}')
    except Exception as e:  # noqa: BLE001
        print(f'[setup] MISSING {m}: {type(e).__name__}: {e}')
import torch
print('[setup] cuda', torch.version.cuda, torch.cuda.is_available())
PY
"${PIP[@]}" freeze > "${HERE}/simuls2s.freeze.txt"
echo "[setup] done"
