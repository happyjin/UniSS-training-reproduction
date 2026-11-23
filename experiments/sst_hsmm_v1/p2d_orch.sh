#!/usr/bin/env bash
# sst_hsmm_v1 option A (post-hoc diagnostic, NOT a gate; report P2c_NOGO_20261123 section 3): P4 model (p4/merged +
# p4/train/gate_final.pt), threshold rule with write threshold TAU in {0.70, 0.85, 0.95} (SST_TAU; 0.5 = P2'') on the
# P2 dev items -> latency/quality curve. Per TAU root p2d_tXXX laid out as p2c, scored by p2_score --rules pareto.
#  1. Thinker: per TAU 8 shards (one per card, 40 s stagger), TAUs one after another
#  2. Talker T1c as p2c, all TAUs at once (3 jobs per card, fixed KV, 10 s stagger)   3. speech-only ASR, 6 jobs on cards 0-5
#  4. dip player NEW + CTRL per TAU (CPU)   5. p2_score per TAU; p2d/DONE
#   setsid nohup bash experiments/sst_hsmm_v1/p2d_orch.sh > logs/sst_hsmm_v1/p2d_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
P1=eval_outputs/sst_hsmm_v1/p4
R=eval_outputs/sst_hsmm_v1/p2d
CTRL=eval_outputs/mist_w3po_v1/wave17/eval_enc_wise05
DATA=eval_outputs/mist_w3po_v1/wave2_S1/data
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
LOGD=logs/sst_hsmm_v1/p2d
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
declare -A TOKMS=([en2zh]=345.6 [zh2en]=332.5)
TAUS="0.70 0.85 0.95"
root() { echo eval_outputs/sst_hsmm_v1/p2d_t${1/./}; }

if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  for t in $TAUS; do [ -e "$(root $t)" ] && { echo "exists: $(root $t)" >&2; exit 1; }; done
  mkdir -p $LOGD $R
  git rev-parse HEAD > $R/commit.txt
  start_watcher
  for t in $TAUS; do
    RT=$(root $t); mkdir -p $RT/data $RT/play
    for d in en2zh zh2en; do
      $PYQ -I - $DATA/items_$d.jsonl $RT/data $d <<'PY' || die "split $d"
import json, sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows); b = [round(k * n / 4) for k in range(5)]
for k in range(4):
    with open(f"{sys.argv[2]}/items_{sys.argv[3]}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
PY
    done
  done
fi

# 1. Thinker per TAU
for t in $TAUS; do
  RT=$(root $t); ps_=(); g=0
  for d in en2zh zh2en; do for k in 0 1 2 3; do
    ps_+=("$(launch $g thinker_t${t/./}_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g SST_RULE=threshold SST_TAU=$t \
      SST_GATE=$P1/train/gate_final.pt SST_TOKMS=${TOKMS[$d]} \
      $PYV -m experiments.sst_hsmm_v1.e2e_si_thinker --thinker $P1/merged --direction $d \
      --items-file $RT/data/items_${d}_$k.jsonl --out $RT/thinker_${d}_$k)")
    g=$((g + 1)); [ -n "$DRY" ] || sleep 40
  done; done
  waitp "${ps_[@]}"
  if [ -z "$DRY" ]; then
    for d in en2zh zh2en; do
      for k in 0 1 2 3; do [ -s $RT/thinker_${d}_$k/blocks_${d}_w1.jsonl ] || die "thinker $t $d $k"; done
      mkdir -p $RT/thinker_$d
      $PYQ -I - $DATA/items_$d.jsonl $RT $d <<'PY' || die "merge blocks $t $d"
import json, sys
items, R, d = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [json.loads(l)["id"] for l in open(items, encoding="utf-8") if l.strip()]
rows = [l for k in range(4) for l in open(f"{R}/thinker_{d}_{k}/blocks_{d}_w1.jsonl", encoding="utf-8") if l.strip()]
assert [json.loads(l)["id"] for l in rows] == ids, (d, len(rows), len(ids))
with open(f"{R}/thinker_{d}/blocks_{d}_w1.jsonl", "x", encoding="utf-8") as fh:
    fh.writelines(rows)
print(R, d, len(rows))
PY
    done
  fi
  log "thinker $t done"
done

# 2. Talker, all TAUs stacked
ps_=()
for t in $TAUS; do
  RT=$(root $t); g=0
  for d in en2zh zh2en; do for s in 0 1 2 3; do
    ps_+=("$(launch $g talker_t${t/./}_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
      -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model $TALKER --direction $d --window 1 \
      --output $RT/talker_${d}_s20261021 --shard $s --num-shards 4 --text-policy file \
      --blocks-file $RT/thinker_$d/blocks_${d}_w1.jsonl --speed-policy block --gpu-mem 0.1 --items-file $DATA/items_$d.jsonl)")
    g=$((g + 1)); [ -n "$DRY" ] || sleep 10
  done; done
done
waitp "${ps_[@]}"
for t in $TAUS; do for d in en2zh zh2en; do for s in 0 1 2 3; do
  [ -n "$DRY" ] || [ -s $(root $t)/talker_${d}_s20261021/part_$s.jsonl ] || die "talker $t $d $s"; done; done; done
log "talkers done"

# 3. speech-only ASR, 6 jobs
ps_=(); g=0
for t in $TAUS; do for d in en2zh zh2en; do
  O=$(root $t)/talker_${d}_s20261021
  ps_+=("$(launch $g asr_t${t/./}_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$O' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$O/results_speechonly.jsonl' --output '$O/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$O/asr_speech.jsonl'")")
  g=$((g + 1))
done; done
waitp "${ps_[@]}"
for t in $TAUS; do for d in en2zh zh2en; do [ -n "$DRY" ] || [ -s $(root $t)/talker_${d}_s20261021/asr_speech.jsonl ] || die "asr $t $d"; done; done
log "asr done"

# 4-5. dip player + score per TAU
[ -n "$DRY" ] && { echo "[dry] s0_apply dip x12 + p2_score x3" >&2; exit 0; }
touch $STOP
for t in $TAUS; do
  RT=$(root $t)
  for d in en2zh zh2en; do
    for x in "new $RT" "ctrl $CTRL"; do
      read -r name rr <<< "$x"
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.voice_onset_v1.s0_apply \
        --run $rr/talker_${d}_s20261021 --dir $d --policy dip --out $RT/play/${name}_$d.jsonl --procs 32 \
        > $LOGD/play_t${t/./}_${name}_$d.log 2>&1 || die "play $t $name $d"
    done
  done
  PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p2_score --root $(basename $RT) --rules pareto --out $RT/summary.json \
    > $LOGD/score_t${t/./}.log 2>&1 || die "score $t"
  touch $RT/DONE
  log "score $t done"
done
touch $R/DONE
log "p2d done"
