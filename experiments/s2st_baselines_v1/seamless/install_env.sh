#!/usr/bin/env bash
# One new venv for the SeamlessStreaming baseline (R1). Interpreter: multi_speaker's python3.11 (coordinator-approved:
# fairseq2n 0.2.x ships cp38-cp311 wheels only). Installs ONLY into the new venv; refuses an existing venv dir.
set -euo pipefail
V=/opt/dlami/nvme/jasonleeeli/venvs/s2st_baselines
SRC=/opt/dlami/nvme/jasonleeeli/src_baselines/seamless_communication   # git clone, commit 9a081e935c29c6b1e0e05bcbf3fcb623ad306524
export PIP_CACHE_DIR=/opt/dlami/nvme/jasonleeeli/cache/pip TMPDIR=/opt/dlami/nvme/jasonleeeli/work/s2st_baselines_tmp
[ -e "$V" ] && { echo "refuse: $V exists"; exit 1; }
/opt/dlami/nvme/jasonleeeli/conda_envs/multi_speaker/bin/python3.11 -m venv "$V"
P=$V/bin/python
$P -m pip install -q --upgrade pip wheel "setuptools<70"
$P -m pip install torch==2.2.2 torchaudio==2.2.2 "numpy<2"
$P -m pip install fairseq2==0.2.1 "numpy<2"
$P -m pip install "$SRC" "numpy<2" "torch==2.2.2" "torchaudio==2.2.2" sacrebleu soundfile
$P -m pip freeze > /opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/s2st_baselines_v1/seamless/pip_freeze.txt
$P -c "import torch, fairseq2, simuleval, seamless_communication; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('fairseq2', fairseq2.__version__); import fairseq2n; print('ok')"
echo INSTALL_DONE
