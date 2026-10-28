#!/usr/bin/env bash
# Chinese AMI after the one-voice-per-speaker re-speaking (the v6 audit's exclusion list): mix V (default v8: speed-up
#   <= 1.5x, slow-down <= 1.25x, failing clips left out) -> audit -> Nemotron posteriors (both presets, 8 cards) ->
#   diarizer audit A8 -> zh->en windows and rows.
# Stops there: training waits for a look at the A8 audit. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
D=data/ami_multispk_si_v1; RV=${D}/zh_revoice_v1/talker
M=experiments.uniss_qwen3_omni_v1.data_clean
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
V=${V:-v8}; TAG=zh${V#v}
for p in ${D}/zh_mix_${V} ${D}/e2e_gstar_zh2en_v1; do [ -e "$p" ] && { log "拒绝覆盖已有输出: $p"; exit 1; }; done
[ -s ${RV}/asr.jsonl ] || { log "no ASR of the re-spoken clips"; exit 1; }
# the exclusion list of the v6 audit (Paraformer CER of each re-spoken clip, inner silence, length) stays valid
PYTHONPATH=. ${PY} -m ${M}.build_ami_zh_mix mix --out ${D}/zh_mix_${V} --name zh_mix_${V} --clips-from ${RV}/results.jsonl \
  --tempo-max 1.5 --tempo-min 0.8 --exclude ${D}/zh_mix_v6/exclude.jsonl --procs 48 || { log "mix ${V} failed"; exit 1; }
PYTHONPATH=. ${PY} -m ${M}.build_ami_zh_mix audit --out ${D}/zh_mix_${V} --recheck ${RV}/asr.jsonl \
  > ${D}/zh_mix_${V}/audit.log 2>&1 || { log "audit ${V} failed"; exit 1; }
log "mix ${V} audited"
MIX=zh_mix_${V} TAG=${TAG} CARDS="0 1 2 3 4 5 6 7" bash reports/gstar_si_paper_v1/run_nemotron_zh.sh || { log "nemotron failed"; exit 1; }
PYTHONPATH=. ${PY} -m ${M}.audit_zh_diar --mix zh_mix_${V} --tag ${TAG} --out ${D}/zh_mix_${V}/audit_diar.json \
  > ${D}/zh_mix_${V}/audit_diar.log 2>&1 || { log "diar audit failed"; exit 1; }
log "A8: $(head -c 600 ${D}/zh_mix_${V}/audit_diar.log | tr '\n' ' ')"
PYTHONPATH=. ${PY} -m ${M}.build_ami_gstar_zh2en_rows --out ${D}/e2e_gstar_zh2en_v1 --mix zh_mix_${V} --recheck ${RV} \
  --diar "nemotron_{}_${TAG}" > logs/zh2en_rows_v1.log 2>&1 || { log "zh2en rows failed"; exit 1; }
log "ZH_DATA_DONE"
