#!/usr/bin/env bash
# Idea 2 e2e prereg §5, CPU part: export the CoVoST 2 non-test clips, build the cvt streams, their clean reference
# prompts and the teacher-target inputs (idea2_build_cvt.py export -> streams -> refs -> teacher). Marker
# data/open_voicecond_v1/idea2/cvt/CPU_DONE (or CPU_FAILED_<step>). No GPU.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_cvt_cpu.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
CV=data/open_voicecond_v1/idea2/cvt
mkdir -p $CV
for step in export streams refs teacher; do
  if [ -n "${DRY_RUN:-}" ]; then echo "[dry] $step"; continue; fi
  echo "$(TZ=Europe/Prague date '+%F %T %Z') $step"
  CUDA_VISIBLE_DEVICES= $PY -I experiments/mist_w3po_v1/voicecond/idea2_build_cvt.py $step --workers 48 || { date -u +%FT%TZ >> $CV/CPU_FAILED_$step; exit 1; }
done
[ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ >> $CV/CPU_DONE
