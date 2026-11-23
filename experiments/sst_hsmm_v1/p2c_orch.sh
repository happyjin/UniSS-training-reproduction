#!/usr/bin/env bash
# sst_hsmm_v1 P2'' dev gate (prereg change 7): P4 model (p4/merged + p4/train/gate_final.pt), threshold rule, Pareto gate as change 6 (D2 <= +0.05 s, NP_in <= -25%, ASR >= -0.5); copy of p2b_orch.sh -> p2c. Waits for p4/DONE (stops on p4/FAILED).
#  1. items of wave2_S1/data split into 4 contiguous shards per direction -> e2e_si_thinker (P1a merged + gate_final), 8 jobs,
#     one per card, starts staggered 40 s -> merged in item order to p2/thinker_<d>/blocks_<d>_w1.jsonl
#  2. Talker T1c, source voice: si_infer exactly as wave17_eval.sh (window 1, file, block, gpu-mem 0.1, seed default), 8 jobs
#  3. speech-only ASR (cards 0 / 1)  4. s0_apply --policy dip on NEW and CTRL (CPU)  5. p2_score -> p2/summary.json, DONE
# Placeholders stopped by PID per card (lib.sh launch); watcher noplaceholder.sh. Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p2c_orch.sh > logs/sst_hsmm_v1/p2c_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
P1=eval_outputs/sst_hsmm_v1/p4
R=eval_outputs/sst_hsmm_v1/p2c
CTRL=eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
LOGD=logs/sst_hsmm_v1/p2c
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
declare -A TOKMS=([en2zh]=345.6 [zh2en]=332.5)

if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  mkdir -p $LOGD
  log "orch $$: waiting for $P1/DONE"
  until [ -e $P1/DONE ]; do [ -e $P1/FAILED ] && { log "P1 FAILED: no P2"; exit 1; }; sleep 60; done
  [ -s $P1/train/gate_final.pt ] && [ -d $P1/merged ] || { log "P1 artefacts missing"; exit 1; }
  mkdir -p $R/data $R/play
  git rev-parse HEAD > $R/commit.txt
  start_watcher
  for d in en2zh zh2en; do
    $PYQ -I - $DATA/items_$d.jsonl $R/data $d <<'PY' || die "split $d"
import json, sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows); b = [round(k * n / 4) for k in range(5)]
for k in range(4):
    with open(f"{sys.argv[2]}/items_{sys.argv[3]}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
print(sys.argv[3], n, b)
PY
  done
fi

# 1. HSMM Thinker, 8 shards
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g thinker_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g SST_RULE=threshold SST_GATE=$P1/train/gate_final.pt SST_TOKMS=${TOKMS[$d]} \
    $PYV -m experiments.sst_hsmm_v1.e2e_si_thinker --thinker $P1/merged --direction $d \
    --items-file $R/data/items_${d}_$k.jsonl --out $R/thinker_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do
    for k in 0 1 2 3; do [ -s $R/thinker_${d}_$k/blocks_${d}_w1.jsonl ] || die "thinker $d $k"; done
    mkdir -p $R/thinker_$d
    $PYQ -I - $DATA/items_$d.jsonl $R $d <<'PY' || die "merge blocks $d"
import json, sys
items, R, d = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [json.loads(l)["id"] for l in open(items, encoding="utf-8") if l.strip()]
rows = [l for k in range(4) for l in open(f"{R}/thinker_{d}_{k}/blocks_{d}_w1.jsonl", encoding="utf-8") if l.strip()]
assert [json.loads(l)["id"] for l in rows] == ids, (d, len(rows), len(ids))
with open(f"{R}/thinker_{d}/blocks_{d}_w1.jsonl", "x", encoding="utf-8") as fh:
    fh.writelines(rows)
print(d, len(rows))
PY
  done
fi
log "thinker done"

# 2. Talker T1c, source voice (as wave17_eval.sh), 8 jobs
ps_=(); g=0
for d in en2zh zh2en; do for s in 0 1 2 3; do
  ps_+=("$(launch $g talker_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
    -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model $TALKER --direction $d --window 1 \
    --output $R/talker_${d}_s20261021 --shard $s --num-shards 4 --text-policy file \
    --blocks-file $R/thinker_$d/blocks_${d}_w1.jsonl --speed-policy block --gpu-mem 0.1 --items-file $DATA/items_$d.jsonl)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 30
done; done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $R/talker_${d}_s20261021/part_$s.jsonl ] || die "talker $d $s"; done; done
log "talkers done"

# 3. speech-only ASR
ps_=(); g=0
for d in en2zh zh2en; do
  O=$R/talker_${d}_s20261021
  ps_+=("$(launch $g asr_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$O' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$O/results_speechonly.jsonl' --output '$O/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$O/asr_speech.jsonl'")")
  g=$((g + 1))
done
waitp "${ps_[@]}"
for d in en2zh zh2en; do [ -n "$DRY" ] || [ -s $R/talker_${d}_s20261021/asr_speech.jsonl ] || die "asr $d"; done
log "asr done"

# 4. dip player on NEW and CTRL (CPU, 1 thread per worker)
[ -n "$DRY" ] && { echo "[dry] s0_apply dip x4 + p2_score" >&2; exit 0; }
touch $STOP
for d in en2zh zh2en; do
  for x in "new $R" "ctrl $CTRL"; do
    read -r name root <<< "$x"
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.voice_onset_v1.s0_apply \
      --run $root/talker_${d}_s20261021 --dir $d --policy dip --out $R/play/${name}_$d.jsonl --procs 32 \
      > $LOGD/play_${name}_$d.log 2>&1 || die "play $name $d"
  done
done
log "play done"

# 5. score
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p2_score --root p2c --rules pareto --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "p2 done"
