#!/usr/bin/env bash
# E10: second half of the simuls2s setup. setup_simuls2s.sh stopped at `openai-whisper` (its sdist needs
# pkg_resources); the S5 code computes whisper log-mels itself (inference/source_prompt.py), so it is dropped.
# Installs into the existing venv only; never recreates or deletes it.
#
#   setsid nohup bash e10_multispk/setup/setup_simuls2s_part2.sh > e10_multispk/runs/setup_simuls2s_part2.log 2>&1 &
set -euo pipefail
V=${SIMULS2S_VENV:-/opt/dlami/nvme/jasonleeeli/venvs/simuls2s}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -x "${V}/bin/python" ]] || { echo "[setup2] ${V} missing; run setup_simuls2s.sh first" >&2; exit 1; }
PIP=("${V}/bin/python" -m pip)
C="${V}/constraints.txt"

# CosyVoice3 flow / HiFT / CAM++ side (attach_speech_decoder's hint pins ruamel.yaml 0.17.40 for HyperPyYAML)
"${PIP[@]}" install -c "${C}" \
    HyperPyYAML==1.2.2 'ruamel.yaml==0.17.40' conformer==0.3.2 diffusers==0.29.0 omegaconf==2.3.0 \
    hydra-core==1.3.2 lightning==2.2.4 x-transformers inflect gdown matplotlib networkx pyworld wget \
    rich protobuf pydantic

# Qwen3.5 linear-attention fast path (optional: the code falls back to the torch kernels).
"${PIP[@]}" install -c "${C}" flash-linear-attention \
    || echo "[setup2][WARN] flash-linear-attention failed; torch fallback will be used"
# fla 0.5.2 refuses its Triton backward of chunk_bwd_dqkwg on Hopper with Triton 3.4 (torch 2.8; wrong grads,
# fla #640) and points to tilelang; added 2026-11-19 after the tuner smoke hit it.
"${PIP[@]}" install -c "${C}" tilelang
"${PIP[@]}" install -c "${C}" --no-build-isolation causal-conv1d \
    || echo "[setup2][WARN] causal-conv1d failed; torch fallback will be used"

"${V}/bin/python" - <<'PY'
import importlib
for m in ['torch', 'transformers', 'peft', 'swift', 'deepspeed', 'soundfile', 'librosa', 'onnxruntime',
          'hyperpyyaml', 'conformer', 'diffusers', 'omegaconf', 'fla', 'causal_conv1d', 'silero_vad']:
    try:
        mod = importlib.import_module(m)
        print(f'[setup2] OK {m} {getattr(mod, "__version__", "")}')
    except Exception as e:  # noqa: BLE001
        print(f'[setup2] MISSING {m}: {type(e).__name__}: {e}')
import torch
print('[setup2] torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())
PY
"${PIP[@]}" freeze > "${HERE}/simuls2s.freeze.txt"
echo "[setup2] done"
