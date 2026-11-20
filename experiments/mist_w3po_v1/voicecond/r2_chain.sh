#!/usr/bin/env bash
# R2 chain (reports/mist_w3po_v1/robust_r2_contrast_prereg.zh-CN.md), cards 4,5 only, detached:
#   own watcher r2_noplaceholder.sh (POLL 1 s, PIDS / STOP) -> smoke (6 steps, mb 32; on failure mb 16) -> training
#   (r2_contrast_train.py, torchrun world 2, 2300 steps) -> WiSE 0.5 copy (alpha 16) -> merge (wave17_merge.py, unchanged)
#   -> eval data (devU + wave2_S1 set b) -> Thinker LA2 (combo_si_thinker --second-view none) for base, enc_wise05
#   (wave17/merged_enc_wise05), r2_wise05 (en2zh card 4, zh2en card 5) -> STOP (cards back to the guard) -> r2_score.py
#   -> DONE (any failure: FAILED + STOP).
# Placeholders of a card are SIGTERMed by PID right before each launch, then we wait (<= 120 s) until no gpu_load
# process remains on that card. Never touches guard 2726951, gpu_watch 52045, sync 1222453 or cards 0-3, 6-7.
#   setsid nohup bash experiments/mist_w3po_v1/voicecond/r2_chain.sh > logs/mist_w3po_v1/r2/chain.log 2>&1 < /dev/null 8>&- 9>&- &
# DRY_RUN=1 prints the commands only. Checked with bash -n / DRY_RUN only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
M=eval_outputs/mist_w3po_v1/r2
LOGD=logs/mist_w3po_v1/r2
export PIDS=$LOGD/r2_workers.pids
STOP=$LOGD/r2_noplaceholder.stop
V=experiments/mist_w3po_v1/voicecond
QWEN3=pretrained_models/Qwen3-Omni-30B-A3B-Instruct
ENC=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
TENV=(env CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
      HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface PYTHONPATH=.)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fail() { log "FAILED: $*"; [ -n "$DRY" ] && return 0; echo "$*" > "$M/FAILED"; touch "$STOP"; exit 1; }
clear_card() {   # $1 card: SIGTERM placeholders, wait until none is left (<= 120 s)
  local g=$1 k left
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
launch() {   # cards(comma) name cmd... ; prints pid
  local gs=$1 name=$2; shift 2
  local lf=$LOGD/r2_$name.log
  if [ -n "$DRY" ]; then printf '[dry] cards %s %s: ' "$gs" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && fail "log exists: $lf"
  for g in ${gs//,/ }; do clear_card "$g"; done
  setsid nohup "$@" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$gs $p $name" >> "$PIDS"
  log "$name cards $gs pid $p" >&2
  echo $p
}
wait_free() {   # $1 card: wait (<= 6 h) until memory held by non-placeholder processes leaves >= 122000 MiB (vLLM 0.85)
  local g=$1 k tot used
  [ -n "$DRY" ] && return 0
  for k in $(seq 1 360); do
    tot=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    used=0
    while IFS=, read -r p m; do
      p=${p// /}; m=${m// /}; [ -z "$p" ] && continue
      case "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" in *scripts/gpu_load/*) continue;; esac
      used=$((used + m))
    done < <(nvidia-smi -i "$g" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    [ $((tot - used)) -ge 122000 ] && return 0
    [ "$k" = 1 ] && log "card $g: waiting for memory (other jobs hold $used MiB)"
    sleep 60
  done
  fail "card $g: not enough free memory after 6 h"
}
waitp() { for p in "$@"; do case "$p" in ''|*[!0-9]*) continue;; esac; while kill -0 "$p" 2>/dev/null; do sleep 20; done; done; }

if [ -z "$DRY" ]; then
  mkdir -p "$LOGD"
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  for f in DONE FAILED run merged_r2_wise05 run_wise05 data eval_base eval_enc_wise05 eval_r2_wise05 summary.json; do
    [ -e "$M/$f" ] && { echo "exists: $M/$f (refuse to overwrite)" >&2; exit 1; }
  done
  mkdir -p "$M"
  log "chain $$"
  POLL=1 STOP=$STOP setsid nohup bash $V/r2_noplaceholder.sh > $LOGD/noplaceholder.log 2>&1 < /dev/null 8>&- 9>&- &
  log "watcher pid $!"
fi

# ---------------------------------------------------------------- smoke
# coordinator 21:55 CET: R4's vLLM shares cards 4-5 -> training peak must stay <= ~70 GB per card: take the largest
# microbatch whose smoke peak (max_memory_allocated, rank 0) is <= 60 GiB (headroom for the allocator's reserve)
MB=""
for mb in 32 16 8; do
  p=$(launch 4,5 smoke$mb "${TENV[@]}" "$PYQ" -m torch.distributed.run --nproc-per-node 2 --master-port 29687 \
      $V/r2_contrast_train.py --steps 2300 --out $M/smoke$mb --mb $mb --max-steps-run 6 --workers 8)
  waitp $p
  if [ -n "$DRY" ]; then MB=$mb; break; fi
  if grep -q "smoke stop after 6 steps" $LOGD/r2_smoke$mb.log; then
    peak=$("$PYQ" -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print(max(x['mem_gib'] for x in r)); assert all(x['loss']==x['loss'] for x in r)" $M/smoke$mb/metrics_rank0.jsonl) \
      || { log "smoke mb $mb: NaN loss"; continue; }
    log "smoke mb $mb ok, peak ${peak} GiB"
    if "$PYQ" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 60 else 1)" "$peak"; then MB=$mb; break; fi
    log "smoke mb $mb: peak above 60 GiB, trying a smaller microbatch"
  else
    log "smoke mb $mb failed"
  fi
done
[ -n "$MB" ] || fail "no microbatch in 32/16/8 passed the smoke within 60 GiB"
log "smoke ok, mb $MB"

# ---------------------------------------------------------------- training
p=$(launch 4,5 train "${TENV[@]}" "$PYQ" -m torch.distributed.run --nproc-per-node 2 --master-port 29688 \
    $V/r2_contrast_train.py --steps 2300 --out $M/run --mb $MB --workers 12)
log "training pid $p"
waitp $p
if [ -z "$DRY" ]; then
  [ -s "$M/run/adapters_final.pt" ] && [ -s "$M/run/adapters_final.json" ] || fail "trainer exited without adapters_final.pt"
  mkdir -p "$M/run_wise05" && ln "$M/run/adapters_final.pt" "$M/run_wise05/adapters_final.pt" || fail "hard link"
  "$PYQ" - "$M/run/adapters_final.json" "$M/run_wise05/adapters_final.json" <<'PY' || fail "wise json"
import json, sys
m = json.load(open(sys.argv[1]))
assert m["arm"] == "r2" and m["module"] == "thinker.audio_tower" and m["alpha"] == 32.0 and m["rank"] == 16 and m["steps"] == 2300, m
m.update({"arm": "r2_wise05", "alpha": 16.0, "wise_ft": "W0 + 0.5*(W_r2 - W0): LoRA alpha halved"})
with open(sys.argv[2], "x") as fh:
    json.dump(m, fh)
PY
  log "merge r2_wise05"
  "$PYQ" $V/wave17_merge.py --adapters "$M/run_wise05/adapters_final.pt" --out "$M/merged_r2_wise05" > $LOGD/merge.log 2>&1 \
    && grep -q '"adapted"' $LOGD/merge.log || fail "merge (see $LOGD/merge.log)"
  "$PYQ" - "$M/data" <<'PY' || fail "eval data"
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=False)
U, S = Path("eval_outputs/mist_w3po_v1/devU/data"), Path("eval_outputs/mist_w3po_v1/wave2_S1/data")
for d in ("en2zh", "zh2en"):
    mu = [json.loads(l) for l in open(U / f"meta_{d}.jsonl")]
    ms = [m for m in map(json.loads, open(S / f"meta_{d}.jsonl")) if m["set"] == "b"]
    keep = {m["id"] for m in mu} | {m["id"] for m in ms}
    iu = [json.loads(l) for l in open(U / f"items_{d}.jsonl")]
    is_ = [r for r in map(json.loads, open(S / f"items_{d}.jsonl")) if r["id"] in keep]
    assert len(iu) == len(mu) and len(is_) == len(ms), (d, len(iu), len(mu), len(is_), len(ms))
    with open(out / f"meta_{d}.jsonl", "x") as fh:
        for m in mu + ms: fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(out / f"items_{d}.jsonl", "x") as fh:
        for r in iu + is_: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(d, len(iu), len(is_))
PY
fi

# ---------------------------------------------------------------- Thinker LA2 on devU + (b)
for sys_ in base enc_wise05 r2_wise05; do
  case $sys_ in base) TK=$QWEN3;; enc_wise05) TK=$ENC;; r2_wise05) TK=$M/merged_r2_wise05;; esac
  tp=()
  k=0
  for d in en2zh zh2en; do
    g=$((4 + k)); k=$((k + 1))
    wait_free $g
    tp+=($(launch $g thinker_${sys_}_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g "$PYV" -m experiments.mist_w3po_v1.voicecond.combo_si_thinker \
           --second-view none --thinker "$TK" --direction $d --gpu-mem 0.85 --out $M/eval_$sys_/thinker_$d \
           --items-file $M/data/items_$d.jsonl))
    [ -n "$DRY" ] || sleep 30
  done
  waitp "${tp[@]}"
  if [ -z "$DRY" ]; then
    for d in en2zh zh2en; do
      n=$(wc -l < $M/eval_$sys_/thinker_$d/blocks_${d}_w1.jsonl 2>/dev/null || echo 0)
      want=$(wc -l < $M/data/items_$d.jsonl)
      [ "$n" = "$want" ] || fail "thinker $sys_ $d: $n of $want rows"
    done
  fi
  log "thinker $sys_ done"
done
[ -n "$DRY" ] || touch "$STOP"
log "watcher STOP touched; cards 4,5 back to the guard"
if [ -z "$DRY" ]; then
  "$PYQ" $V/r2_score.py --root $M --out $M/summary.json > $LOGD/score.log 2>&1 || fail "score (see $LOGD/score.log)"
  date > "$M/DONE"
fi
log "chain done"
