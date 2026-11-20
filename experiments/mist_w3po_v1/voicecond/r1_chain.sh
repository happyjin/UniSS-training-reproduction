#!/usr/bin/env bash
# Robust R1 chain (reports/mist_w3po_v1/robust_r1_taskdiff_prereg.zh-CN.md), detached:
#   build r1/data (devU set u + S1 bench_dev set b) -> own watcher r1_noplaceholder.sh (POLL 1 s, own STOP) ->
#   wait until test_robust/UE and UW are finished (DONE, or FAILED) and none of their pids is alive (cards 1, 7) ->
#   r1_run.sh tau_clean on cards 1,2,3,7 (world 4, mb 16) -> wait -> r1_merge.py (tau_noisy = wave14/run_unist) ->
#   Thinker (si_thinker_stream la2 w1) for base / unist / unist_wise05 / r1 on r1/data, two systems at a time on the
#   4 cards -> touch STOP (cards back to the guard) -> r1_score.py -> r1/DONE (any failure: r1/FAILED + STOP).
# Never touches guard 2726951, gpu_watch 52045, sync 1222453; only cards 1, 2, 3, 7.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/r1_chain.sh > logs/mist_w3po_v1/r1/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
M=eval_outputs/mist_w3po_v1/r1
LOGD=logs/mist_w3po_v1/r1
export PIDS=$LOGD/r1_workers.pids
STOP=$LOGD/r1_noplaceholder.stop
TR=eval_outputs/mist_w3po_v1/test_robust
W14=eval_outputs/mist_w3po_v1/wave14
V=experiments/mist_w3po_v1/voicecond
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
CARDS=(1 2 3 7)
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; echo "$*" > "$M/FAILED"; touch "$STOP"; exit 1; }
for f in DONE FAILED run_clean merged_r1 eval_base eval_unist eval_unist_wise05 eval_r1 score.json data; do
  [ -e "$M/$f" ] && { log "refuse: $M/$f exists"; exit 1; }
done
[ -e "$STOP" ] && { log "refuse: $STOP exists"; exit 1; }

clear_card() {   # SIGTERM gpu_load placeholders on card $1 and wait (<= 120 s) until none is left
  local g=$1 k left pid args
  for k in $(seq 1 25); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && return 0
    [ "$k" = 1 ] && log "SIGTERM placeholders on card $g:$left"
    sleep 5
  done
  log "WARNING: placeholders still on card $g after 120 s:$left"
}
waitp() { for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }

log "chain $$"
# ------------------------------------------------------------------ data
mkdir -p "$M/data"
"$PYQ" - "$M/data" <<'PY' || fail "data"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
U, S = Path("eval_outputs/mist_w3po_v1/devU/data"), Path("eval_outputs/mist_w3po_v1/wave2_S1/data")
for d in ("en2zh", "zh2en"):
    mu = [json.loads(l) for l in open(U / f"meta_{d}.jsonl")]
    iu = [json.loads(l) for l in open(U / f"items_{d}.jsonl")]
    ms = [json.loads(l) for l in open(S / f"meta_{d}.jsonl")]
    keep = {m["id"] for m in ms if m["set"] == "b"}
    is_ = [json.loads(l) for l in open(S / f"items_{d}.jsonl")]
    meta = mu + [m for m in ms if m["id"] in keep]
    items = iu + [r for r in is_ if r["id"] in keep]
    assert len({m["id"] for m in meta}) == len(meta) == len(items), d
    assert {m["id"] for m in meta} == {r["id"] for r in items}, d
    for name, rows in (("meta", meta), ("items", items)):
        with open(out / f"{name}_{d}.jsonl", "x", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(d, len(items), "items (u", len(iu), "+ b", len(items) - len(iu), ")")
PY
STOP=$STOP setsid nohup bash $V/r1_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
log "watcher pid $!"

# ------------------------------------------------------------------ wait for the test arms on cards 1, 7
log "waiting for test_robust UE / UW (cards 1, 7)"
while :; do
  ok=1
  for a in UE UW; do
    { [ -e "$TR/$a/DONE" ] || [ -e "$TR/$a/FAILED" ]; } || ok=0
  done
  if [ "$ok" = 1 ]; then
    live=""
    for a in UE UW; do
      [ -s logs/mist_w3po_v1/test_robust/$a.pids ] || continue
      while read -r c p _; do
        [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null && live="$live $p"
      done < logs/mist_w3po_v1/test_robust/$a.pids
    done
    [ -z "$live" ] && break
  fi
  sleep 60
done
for a in UE UW; do [ -e "$TR/$a/FAILED" ] && log "note: test arm $a ended with FAILED (cards free anyway)"; done
log "test arms finished; cards 1, 7 free of them"

# ------------------------------------------------------------------ training (tau_clean)
GPUS=1,2,3,7 STEPS=2300 OUT=$M/run_clean NAME=clean MB=16 PORT=29691 WORKERS=12 bash $V/r1_run.sh > $LOGD/run_launch.log 2>&1 \
  || fail "r1_run.sh"
cat $LOGD/run_launch.log
TPID=$(awk '$3 == "clean" {p = $2} END {print p}' "$PIDS")
[ -n "$TPID" ] && kill -0 "$TPID" 2>/dev/null || fail "trainer pid not alive"
log "trainer torchrun pid $TPID"
waitp "$TPID"
[ -s "$M/run_clean/adapters_final.pt" ] && [ -s "$M/run_clean/adapters_final.json" ] || fail "trainer exited without adapters_final.pt"
log "training done; merging tau_noisy - tau_clean"
PYTHONPATH=. "$PYQ" $V/r1_merge.py --noisy $W14/run_unist/adapters_final.pt --clean $M/run_clean/adapters_final.pt \
  --out $M/merged_r1 > $LOGD/merge.log 2>&1 || fail "merge (see $LOGD/merge.log)"
cat $LOGD/merge.log

# ------------------------------------------------------------------ Thinker evaluation
QWEN3=pretrained_models/Qwen3-Omni-30B-A3B-Instruct
declare -A TK=([base]=$QWEN3 [unist]=$W14/merged_unist [unist_wise05]=$W14/merged_unist_wise05 [r1]=$M/merged_r1)
thinker() {   # sys dir card ; prints pid
  local s=$1 d=$2 g=$3 lf=$LOGD/thinker_${1}_$2.log
  [ -e "$lf" ] && fail "log exists: $lf"
  clear_card "$g" >&2
  setsid nohup "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream \
    --thinker "${TK[$s]}" --direction "$d" --windows 1 --policy la2 --gpu-mem 0.85 --out "$M/eval_$s/thinker_$d" \
    --items-file "$M/data/items_$d.jsonl" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$g $p thinker_${s}_$d" >> "$PIDS"
  log "thinker $s $d card $g pid $p" >&2
  echo $p
}
for pair in "base unist" "unist_wise05 r1"; do
  read -r s1 s2 <<< "$pair"
  p1=$(thinker $s1 en2zh ${CARDS[0]}); sleep 20; p2=$(thinker $s1 zh2en ${CARDS[1]}); sleep 20
  p3=$(thinker $s2 en2zh ${CARDS[2]}); sleep 20; p4=$(thinker $s2 zh2en ${CARDS[3]})
  waitp $p1 $p2 $p3 $p4
  for s in $s1 $s2; do for d in en2zh zh2en; do
    [ -s "$M/eval_$s/thinker_$d/blocks_${d}_w1.jsonl" ] || fail "no Thinker output $s $d (see $LOGD/thinker_${s}_$d.log)"
  done; done
done
touch "$STOP"
log "Thinker runs done; watcher STOP touched, cards back to the guard"
PYTHONPATH=. "$PYQ" -m experiments.mist_w3po_v1.voicecond.r1_score --out "$M/score.json" > $LOGD/score.log 2>&1 \
  || fail "score (see $LOGD/score.log)"
cat $LOGD/score.log
date > "$M/DONE"
log "chain done"
