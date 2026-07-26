#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${USER_ROOT:-/opt/dlami/nvme/jasonleeeli}"
TRAIN_ENV="${TRAIN_ENV:-${USER_ROOT}/conda_envs/uniss-train}"
EVAL_ENV="${EVAL_ENV:-${USER_ROOT}/conda_envs/uniss-eval}"
CONDA_BIN="${CONDA_BIN:-${USER_ROOT}/softwares/miniforge3-recovered/bin/conda}"
STOPES_ROOT="${STOPES_ROOT:-${USER_ROOT}/evaluation_deps/stopes-a4e75e8}"
STOPES_COMMIT="a4e75e8cb5b8eed629ac0056f301d32d8f194db2"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${USER_ROOT}/conda_pkgs_eval}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_ROOT}/cache/pip}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/cache/huggingface}"

if [[ ! -d "${EVAL_ENV}" ]]; then
  "${CONDA_BIN}" create -y -p "${EVAL_ENV}" --clone "${TRAIN_ENV}"
fi

"${EVAL_ENV}/bin/python" -m pip install \
  imageio-ffmpeg==0.6.0 \
  sacrebleu==2.5.1 \
  opencc-python-reimplemented==0.1.7 \
  vllm==0.8.5.post1 \
  funasr==1.3.29 \
  modelscope==1.38.1

# vLLM 0.8.x pins protobuf 4.x through its OpenTelemetry stack, while the
# cloned training environment's TensorBoard requires protobuf >=6. Evaluation
# jobs do not use TensorBoard, so remove it only from the isolated eval env.
"${EVAL_ENV}/bin/python" -m pip uninstall -y tensorboard || true

if [[ ! -d "${STOPES_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${STOPES_ROOT}")"
  git clone https://github.com/facebookresearch/stopes.git "${STOPES_ROOT}"
fi
git -C "${STOPES_ROOT}" checkout "${STOPES_COMMIT}"
"${EVAL_ENV}/bin/python" -m pip install -e "${STOPES_ROOT}[auto_pcp]"

"${EVAL_ENV}/bin/python" - <<'PY'
import funasr
import imageio_ffmpeg
import modelscope
import stopes
import vllm
print("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
print("vllm", vllm.__version__)
print("funasr", funasr.__version__)
print("modelscope", modelscope.__version__)
print("stopes", stopes.__file__)
PY

# Keep the audit output. The cloned environment intentionally retains the
# already validated UniSS training stack, whose ModelOpt/Transformer-Engine
# package metadata reports known version/platform warnings despite working for
# checkpoint conversion.
"${EVAL_ENV}/bin/python" -m pip check || true
