#!/usr/bin/env bash
# Chinese AMI after the one-voice-per-speaker re-speaking: waits for run_revoice.sh, then
#   v6 mix of the new clips -> audit (Paraformer CER of every clip) -> exclusion list -> v7 mix without the failing
#   clips -> audit -> Nemotron posteriors on v7 (both presets, 8 cards) -> diarizer audit A8 -> zh->en windows and rows.
# Stops there: training waits for a look at the A8 audit. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
D=data/ami_multispk_si_v1; RV=${D}/zh_revoice_v1/talker
M=experiments.uniss_qwen3_omni_v1.data_clean
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
for p in ${D}/zh_mix_v6 ${D}/zh_mix_v7 ${D}/e2e_gstar_zh2en_v1; do [ -e "$p" ] && { log "拒绝覆盖已有输出: $p"; exit 1; }; done
until grep -q REVOICE_DONE logs/zh_revoice_v1_run.log; do sleep 30; done
cat ${RV}/asr_s*.jsonl > ${RV}/asr.jsonl
log "asr rows: $(wc -l < ${RV}/asr.jsonl)"
PYTHONPATH=. ${PY} -m ${M}.build_ami_zh_mix mix --out ${D}/zh_mix_v6 --name zh_mix_v6 --clips-from ${RV}/results.jsonl \
  --tempo-max 1.5 --procs 48 || { log "mix v6 failed"; exit 1; }
PYTHONPATH=. ${PY} -m ${M}.build_ami_zh_mix audit --out ${D}/zh_mix_v6 --recheck ${RV}/asr.jsonl --write-exclude exclude.jsonl \
  > ${D}/zh_mix_v6/audit.log 2>&1 || { log "audit v6 failed"; exit 1; }
PYTHONPATH=. ${PY} -m ${M}.build_ami_zh_mix mix --out ${D}/zh_mix_v7 --name zh_mix_v7 --clips-from ${RV}/results.jsonl \
  --tempo-max 1.5 --exclude ${D}/zh_mix_v6/exclude.jsonl --procs 48 || { log "mix v7 failed"; exit 1; }
PYTHONPATH=. ${PY} -m ${M}.build_ami_zh_mix audit --out ${D}/zh_mix_v7 --recheck ${RV}/asr.jsonl \
  > ${D}/zh_mix_v7/audit.log 2>&1 || { log "audit v7 failed"; exit 1; }
log "mix v7 audited"
MIX=zh_mix_v7 TAG=zh7 CARDS="0 1 2 3 4 5 6 7" bash reports/gstar_si_paper_v1/run_nemotron_zh.sh || { log "nemotron failed"; exit 1; }
PYTHONPATH=. ${PY} -m ${M}.audit_zh_diar --mix zh_mix_v7 --tag zh7 --out ${D}/zh_mix_v7/audit_diar.json \
  > ${D}/zh_mix_v7/audit_diar.log 2>&1 || { log "diar audit failed"; exit 1; }
log "A8: $(head -c 600 ${D}/zh_mix_v7/audit_diar.log | tr '\n' ' ')"
PYTHONPATH=. ${PY} -m ${M}.build_ami_gstar_zh2en_rows --out ${D}/e2e_gstar_zh2en_v1 --mix zh_mix_v7 --recheck ${RV} \
  --diar 'nemotron_{}_zh7' > logs/zh2en_rows_v1.log 2>&1 || { log "zh2en rows failed"; exit 1; }
log "ZH_DATA_DONE"
