#!/usr/bin/env bash
# En->Zh Thinker checkpoint selection (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §3.2), while the training
# runs: for every checkpoint step_00500 ... step_03000 of RUN, as soon as its DONE appears, one detached en2zh_orch
# orchestrator on the NON-training cards (--exclude TCARDS; the training slot's cards are excluded globally as well):
#   dev5 text decode (1248 windows, 4 window shards, 1.0 s reads, the A' decode flags, --cap135) -> text tcpBLEU
#   all / ov / 1spk + text LAAL (en2zh_lat, current protocol), and the held-out C0 / CRF loss on the dev5 sched windows
#   (--heldout, recorded only).
# After the last one: the text LAAL under the chunk protocol for each (CPU, recorded), then the fixed rule
# (en2zh_select.py thinker) -> logs/mist_w3po_v1/en2zh/thinker_selection.json and the line EN2ZH_THINKER_SELECTED <step>.
# Stops with EN2ZH_PROXY_FAILED if the training exits without its checkpoint, or the step-3000 proxy has no outputs.
#   RUN=<thinker run id> TCARDS=0,1,2,3|4,5,6,7 [STEPS="500 1000 1500 2000 2500 3000"] [DRY_RUN=1] \
#     bash experiments/mist_w3po_v1/scripts/en2zh_ckpt_proxy.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/en2zh/ckpt_proxy.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=${RUN:?}; TCARDS=${TCARDS:?}
STEPS=${STEPS:-500 1000 1500 2000 2500 3000}
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
TR=eval_outputs/mist_w3po_v1/crf/train
E=logs/mist_w3po_v1/en2zh
DEC=eval_outputs/mist_w3po_v1/decode
TE=eval_outputs/mist_w3po_v1/crf/t_eval
SCHD=eval_outputs/mist_w3po_v1/crf/sched_en2zh_dev_v1
W5C=eval_outputs/mist_w3po_v1/data/windows_en2zh_c2g06_chunk_v1_dev5/windows_dev_online.jsonl
REF=data/ami_multispk_si_v1/e2e_gstar_v1/windows_validation_ref.jsonl
SEL=$E/thinker_selection.json
L=logs/mist_w3po_v1/$RUN.log
EX=$(echo "$TCARDS" | tr ',' ' ')
log() { echo "$(date -u +%FT%TZ) $*"; }
name() { printf 'dev5_%s_s%05d' "$RUN" "$1"; }
ocmd() { local s=$1; echo "$PY experiments/mist_w3po_v1/scripts/en2zh_orch.py --set dev5 --run $E/dev5_s$(printf %05d $s) \
--name $(name $s) --ckpt $TR/$RUN/ckpt/step_$(printf %05d $s) --talker none --shards 4 --shard-by window --window 1 \
--exclude $EX --cap135 --heldout $SCHD"; }
if [ -n "${DRY_RUN:-}" ]; then
  for s in $STEPS; do echo "step $s: $(ocmd $s)"; done
  $PY experiments/mist_w3po_v1/scripts/en2zh_orch.py --set dev5 --run /nonexistent/dry --name $(name 500) \
    --ckpt $TR/$RUN/ckpt/step_00500 --talker none --shards 4 --shard-by window --window 1 --exclude $EX --cap135 \
    --heldout $SCHD --dry | tail -1
  echo "selection: $PY experiments/mist_w3po_v1/scripts/en2zh_select.py thinker $(for s in $STEPS; do echo -n "--cand $s $(name $s) "; done)--out $SEL"
  exit 0
fi
[ -e $SEL ] && { log "拒绝覆盖已有输出: $SEL"; exit 1; }
mkdir -p $E
for s in $STEPS; do
  CK=$TR/$RUN/ckpt/step_$(printf %05d $s)
  log "waiting for $CK/DONE"
  until [ -e $CK/DONE ]; do
    if [ -e $L ] && grep -q CRF_RUN_EXIT $L && [ ! -e $CK/DONE ]; then
      log "EN2ZH_PROXY_FAILED: training exited ($(grep CRF_RUN_EXIT $L | tail -1)) without $CK"; exit 1
    fi
    sleep 60
  done
  C=$(ocmd $s)
  setsid nohup $C < /dev/null > $E/dev5_s$(printf %05d $s).launch.log 2>&1 &
  log "step $s: dev5 proxy orchestrator pid $!"
  sleep 90   # its first decodes start before the next orchestrator plans (the shared ledger orders the rest)
done
for s in $STEPS; do
  until [ -e $E/dev5_s$(printf %05d $s)/events/DONE ]; do sleep 30; done
  n=$(name $s)
  if [ -e $DEC/$n/score_text.json ] && [ ! -e $TE/$n/textlat_chunk_$n.json ]; then
    $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $TE/$n/textlat_chunk_$n.json.work \
      --run $DEC/$n --text-only --win $W5C --out $TE/$n/textlat_chunk_$n.json > $TE/$n/textlat_chunk.log 2>&1 \
      || log "chunk text LAAL failed for $n (recorded only)"
  fi
  log "step $s: dev5 proxy DONE"
done
$PY experiments/mist_w3po_v1/scripts/en2zh_select.py thinker $(for s in $STEPS; do echo -n "--cand $s $(name $s) "; done) \
  --out $SEL > $E/thinker_selection.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "EN2ZH_PROXY_FAILED: selection rc=$rc ($(tail -1 $E/thinker_selection.log | cut -c1-300))"; exit 1; fi
log "EN2ZH_THINKER_SELECTED $($PY -c "import json; print(json.load(open('$SEL'))['chosen'])")"
