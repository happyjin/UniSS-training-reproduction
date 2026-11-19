#!/usr/bin/env bash
# Wave 3 V group: the orchestrator's metric step (SQUIM PESQ + TitaNet similarity per run) in parallel for the runs given
# as "arm:dir" arguments, each written to a temporary name and renamed into wave3_combos/V/metrics/ (the orchestrator
# skips a non-empty final file, so the two never write the same name). GPU for similarity: card $GPU (card 3 by
# default, where the V group's jobs already run).
#   GPU=3 bash wave3_v_metrics_par.sh B:zh2en B:en2zh v3:zh2en ...
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
V=eval_outputs/mist_w3po_v1/wave3_combos/V
S1=eval_outputs/mist_w3po_v1/wave2_S1
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
GPU=${GPU:-3}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
one() {
  local arm=${1%%:*} d=${1##*:} r
  r=$V/$arm/talker_$d; [ "$arm" = B ] && r=$S1/talker_${d}_s20261021
  local f=$V/metrics/${arm}_${d}_squim.jsonl t=$V/metrics/.${arm}_${d}_squim.par.jsonl
  if [ ! -s "$f" ]; then
    PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $r --out $t --procs 24 \
      && { [ -s "$f" ] || command mv -f "$t" "$f"; } && log "pesq $arm $d"
  fi
  f=$V/metrics/${arm}_${d}_spksim.jsonl; t=$V/metrics/.${arm}_${d}_spksim.par.jsonl
  if [ ! -s "$f" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim --dir $d --run $r \
      --out $t --device cuda:0 && { [ -s "$f" ] || command mv -f "$t" "$f"; } && log "spksim $arm $d"
  fi
}
for x in "$@"; do one "$x" & done
wait
log "metrics_par done: $*"
