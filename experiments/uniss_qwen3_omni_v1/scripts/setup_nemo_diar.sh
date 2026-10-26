#!/usr/bin/env bash
# The one environment the multi-speaker plan installs (docs/multi_speaker_si_plan_v5.zh-CN.md): NeMo for the
# Nemotron-3 streaming diarizer only -- posteriors and who-spoke-when segments. No existing env is touched.
# Steps follow the gstar-nemotron branch (nemotron/setup_env.sh): a venv (Python 3.12), torch, NeMo from the
# NeMo Speech checkout (PyPI nemo-toolkit[asr] if the editable install fails), the Nemotron-3-Diarization weights.
# Idempotent: a finished step is skipped.
#
#   bash experiments/uniss_qwen3_omni_v1/scripts/setup_nemo_diar.sh            (DRY_RUN=1 prints the steps)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python   # Python 3.12; the venv gets its own site-packages
ENV=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar
SRC=/opt/dlami/nvme/jasonleeeli/softwares/NeMo-Speech
PM="${REPO_ROOT}/pretrained_models/Nemotron-3-Diarization"
export PIP_CACHE_DIR=/opt/dlami/nvme/jasonleeeli/pip_cache CONDA_PKGS_DIRS=/opt/dlami/nvme/jasonleeeli/conda_pkgs
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }

if [ ! -x "${ENV}/bin/python" ]; then
  log "create ${ENV}"
  run "${BASE_PY}" -m venv "${ENV}"   # the conda launchers here have lost their exec bit; a venv needs none
fi
PY="${ENV}/bin/python"
if [ ! -d "${SRC}/.git" ]; then
  log "clone NeMo Speech -> ${SRC}"
  run git clone -q --depth 1 https://github.com/NVIDIA-NeMo/Speech.git "${SRC}"
fi
if [ -z "${DRY_RUN:-}" ] && "${PY}" -c "import nemo.collections.asr" 2>/dev/null; then
  log "NeMo already importable"
else
  # NeMo main wants torch >= 2.7 and pins 2.11.0 in its extras; torchaudio stops at 2.11.0, so both are pinned
  # there (an unpinned install pulled torch 2.14.1 against torchaudio 2.6.0: libtorchaudio failed to load)
  log "install NeMo, then torch / torchaudio 2.11.0 (cu128) as a matching pair"
  run "${PY}" -m pip install -q -U pip
  run "${PY}" -m pip install -q Cython packaging
  if ! run "${PY}" -m pip install -q -e "${SRC}[asr]"; then
    log "editable install failed; PyPI nemo-toolkit[asr]"
    run "${PY}" -m pip install -q "nemo-toolkit[asr]"
  fi
  run "${PY}" -m pip install -q torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
fi
mkdir -p "${PM}"
for f in Nemotron-3-Diarization.nemo config.json; do
  if [ ! -s "${PM}/${f}" ]; then
    log "download ${f}"
    run curl -sSL --fail --retry 3 -o "${PM}/${f}.part" "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/${f}"
    run command mv -f "${PM}/${f}.part" "${PM}/${f}"
  fi
done
[ -n "${DRY_RUN:-}" ] && exit 0
"${PY}" - <<'PY'
import nemo, torch
from nemo.collections.asr.models import SortformerEncLabelModel  # noqa: F401
print("[setup] nemo", nemo.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda, "| SortformerEncLabelModel OK")
PY
log "NEMO_DIAR_SETUP_DONE"
