#!/usr/bin/env bash
# R3 chain (reports/mist_w3po_v1/robust_r3_noisesub_prereg.zh-CN.md, section 6), detached, cards 0 and 6 only:
#   own watcher r3_noplaceholder.sh (POLL 1 s) -> r3_subspace.py on card 0 (encoder only on the GPU, ~5 GiB) ->
#   smoke (6 steps, world 2 on cards 0,6) at mb 16, else mb 8: peak max_memory_allocated must be <= 62 GiB (coordinator:
#   <= ~70 GB/card incl. allocator overhead, R4 stacks on these cards; global batch stays 64 via accumulation) ->
#   training 2300 steps -> WiSE 0.5 copy (alpha 16, hard-linked adapters) -> r3_merge.py (CPU) -> 8 Thinker jobs, lane
#   card 0 (en2zh) / lane card 6 (zh2en): r3_wise05 devU, r3_wise05 s1, base devU, enc_wise05 devU (each waits for other
#   jobs such as R4 to leave the card) -> touch STOP -> r3_score.py -> r3/CHAIN_DONE (any failure: r3/CHAIN_FAILED + STOP).
# Placeholders (scripts/gpu_load/) are stopped by PID only, after which the launch waits <= 120 s for them to leave;
# R4 / other jobs are never touched. Never touches guard 2726951, gpu_watch 52045, sync 1222453.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/r3_chain.sh > logs/mist_w3po_v1/r3/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n / DRY_RUN only (DRY_RUN=1 prints the plan and exits).
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/r3
LOGD=logs/mist_w3po_v1/r3
V=experiments/mist_w3po_v1/voicecond
export PIDS=$LOGD/r3_workers.pids
export STOP=$LOGD/r3_noplaceholder.stop
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
CARDS=0,6
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/CHAIN_FAILED"; touch "$STOP"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then
  log "[dry] subspace card 0 -> $M/subspace; smoke mb 16|8 on $CARDS; train $M/run_r3; wise $M/run_r3_wise05; merge $M/merged_r3_wise05"
  for s in r3_wise05:devU r3_wise05:s1 base:devU enc_wise05:devU; do
    for g in 0:en2zh 6:zh2en; do GPU=${g%%:*} SYS=${s%%:*} DSET=${s##*:} DIR=${g##*:} DRY_RUN=1 bash $V/r3_eval.sh | cut -c1-160; done
  done
  exit 0
fi
mkdir -p "$M" "$LOGD"
[ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
for f in CHAIN_DONE CHAIN_FAILED subspace run_r3 run_r3_wise05 merged_r3_wise05 summary.json; do [ -e "$M/$f" ] && { echo "exists: $M/$f" >&2; exit 1; }; done
clear_cards() {   # $1 = comma list
  local g k left pid args
  for g in ${1//,/ }; do
    for k in $(seq 1 24); do
      left=""
      while read -r pid args; do
        case " 2726951 52045 1222453 $$ " in *" $pid "*) continue;; esac
        case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g";; esac
      done < <(ps -eo pid=,args=)
      [ -z "$left" ] && break
      sleep 5
    done
  done
}
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }
log "chain $$"
POLL=1 setsid nohup bash $V/r3_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"
# ---- 1. subspace
clear_cards 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  setsid nohup "$PYQ" $V/r3_subspace.py --out "$M/subspace" > $LOGD/r3_subspace.log 2>&1 < /dev/null 8>&- 9>&- &
P=$!; echo "0 $P r3_subspace" >> "$PIDS"; log "subspace pid $P"
waitp $P
[ -s "$M/subspace/Q.pt" ] || fail "subspace (see $LOGD/r3_subspace.log)"
log "subspace: $(tail -n 1 $LOGD/r3_subspace.log | cut -c1-400)"
"$PYQ" -c "import json,sys; s=json.load(open('$M/subspace/stats.json')); sys.exit(0 if s['rank_total']>0 else 1)" || fail "all subspace ranks are 0"
# ---- 2. smoke + 3. train
train() {   # out mb [smoke steps]
  clear_cards $CARDS
  env CUDA_VISIBLE_DEVICES=$CARDS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
      HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=. \
      setsid nohup "$PYQ" -m torch.distributed.run --nproc-per-node 2 --master-port 29677 $V/r3_enc_train.py --arm erm \
      --steps 2300 --out "$1" --mb "$2" --workers 12 --subspace "$M/subspace/Q.pt" ${3:+--max-steps-run $3} \
      > "$LOGD/r3_$(basename $1).log" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!; echo "$CARDS $p r3_train_$(basename $1)" >> "$PIDS"; log "$(basename $1) (mb $2) torchrun pid $p"; TP=$p
}
MB=""
for mb in 16 8; do
  train "$M/smoke$mb" $mb 6; waitp $TP
  pk=$("$PYQ" -c "import json; r=[json.loads(l) for l in open('$M/smoke$mb/metrics_rank0.jsonl')]; print(max(x['mem_gib'] for x in r) if len(r)>=6 else 999)" 2>/dev/null || echo 999)
  log "smoke mb $mb: peak $pk GiB"
  "$PYQ" -c "import sys; sys.exit(0 if float('$pk') <= 62 else 1)" && { MB=$mb; break; }
done
[ -n "$MB" ] || fail "smoke failed or peak > 62 GiB at mb 16 and 8"
train "$M/run_r3" $MB; waitp $TP
R=$M/run_r3
[ -s "$R/adapters_final.pt" ] && [ -s "$R/adapters_final.json" ] || fail "trainer exited without adapters_final.pt (see $LOGD/r3_run_r3.log)"
log "training done (mb $MB)"
# ---- 4. WiSE copy + merge
W=$M/run_r3_wise05
mkdir -p "$W"; ln "$R/adapters_final.pt" "$W/adapters_final.pt" || fail "hard link"
"$PYQ" - "$R/adapters_final.json" "$W/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["arm"] == "r3" and m["module"] == "thinker.audio_tower" and m["alpha"] == 32.0 and m["rank"] == 16 and m["steps"] == 2300, m
m.update({"arm": "r3_wise05", "alpha": m["alpha"] / 2, "wise_ft": "W0 + 0.5*(W_r3 - W0): alpha halved"})
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
PY
PYTHONPATH=. "$PYQ" $V/r3_merge.py --adapters "$W/adapters_final.pt" --out "$M/merged_r3_wise05" > $LOGD/r3_merge.log 2>&1 \
  || fail "merge (see $LOGD/r3_merge.log)"
grep -q '"adapted"' $LOGD/r3_merge.log || fail "merge self-check"
log "merged: $(tail -n 1 $LOGD/r3_merge.log | cut -c1-300)"
# ---- 5. evaluation lanes
lane() {   # card dir
  local s p
  for s in r3_wise05:devU r3_wise05:s1 base:devU enc_wise05:devU; do
    p=$(GPU=$1 SYS=${s%%:*} DSET=${s##*:} DIR=$2 bash $V/r3_eval.sh 2>&1 | tee -a $LOGD/lanes.log | grep -o 'pid [0-9]*$' | awk '{print $2}')
    [ -n "$p" ] || { echo "$s $2 launch" >> "$M/EVAL_FAILED"; continue; }
    waitp $p
    n=$(wc -l < "$M/eval_${s%%:*}/${s##*:}/thinker_$2/blocks_$2_w1.jsonl" 2>/dev/null || echo 0)
    log "eval $s $2 done ($n rows)"
  done
}
lane 0 en2zh & la=$!
sleep 20
lane 6 zh2en & lb=$!
wait $la; wait $lb
touch "$STOP"
log "watcher STOP touched; cards 0, 6 back to the guard"
[ -e "$M/EVAL_FAILED" ] && fail "some evaluation launches failed: $(cat $M/EVAL_FAILED)"
PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.r3_score --out "$M/summary.json" > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
date > "$M/CHAIN_DONE"
log "chain done"
