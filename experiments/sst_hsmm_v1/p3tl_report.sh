#!/usr/bin/env bash
# prereg change 40-F4: lookahead talker variant of p3h8 = p3ts_report.sh with TALKER (export dir) from the environment and
# LOOK=1 -> the thinker blocks get "look" (p3tl_look.py, $LOOKU units, default 3) and the talker runs with --lookahead.
# Was: prereg change 39: talker speed variant of p3h8 (steps 4-7 of p3h8_report.sh only): same thinker blocks and searched voice tokens
# (p3h8/thinker_<d>, p3h8/tokens), Talker T1c with --speed-policy queue --queue-shift-per-s 0 + $XFLAGS (e.g. "--final-shift 2"),
# 8 talker jobs on cards $CARDS (two jobs per listed card pair allowed), speech-only ASR, TitaNet / ECAPA2 / PESQ, dip player on NEW
# (N / B_b play files copied from p3h8: identical inputs), p3_score -> eval_outputs/sst_hsmm_v1/$OUTN/summary.json. Refuses outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${OUTN:?}" "${XFLAGS:?}"
SRC=eval_outputs/sst_hsmm_v1/p3h8
R=eval_outputs/sst_hsmm_v1/$OUTN
T=eval_outputs/mist_w3po_v1/test_robust
TALKER=${TALKER:?}
LOOK=${LOOK:-0}
LF=(); [ "$LOOK" = 1 ] && LF=(--lookahead)
export S1_DATA=/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/mist_w3po_v1/test_robust/data
LOGD=logs/sst_hsmm_v1/$OUTN
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
NENV=(env PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface S1_DATA=$S1_DATA)
CA=(${CARDS:-0 1 2 3 4 5 6 7})
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  [ -e $SRC/DONE ] || { echo "p3h8 missing" >&2; exit 1; }
  mkdir -p $LOGD $R/metrics $R/play $R/tokens; git rev-parse HEAD > $R/commit.txt
  for d in en2zh zh2en; do
    mkdir -p $R/thinker_$d; cp -n $SRC/tokens/tokens_new_$d.jsonl $R/tokens/
    if [ "$LOOK" = 1 ]; then
      $PYE -m experiments.sst_hsmm_v1.p3tl_look --blocks $SRC/thinker_$d/blocks_${d}_w1.jsonl --direction $d --units ${LOOKU:-3} \
        --out $R/thinker_$d/blocks_${d}_w1.jsonl > $R/thinker_$d/look_stats.json || die "look $d"
    else cp -n $SRC/thinker_$d/blocks_${d}_w1.jsonl $R/thinker_$d/; fi
    cp -n $SRC/play/N_$d.jsonl $SRC/play/B_b_$d.jsonl $R/play/
  done
  echo "$XFLAGS" > $R/xflags.txt; echo "TALKER=$TALKER LOOK=$LOOK LOOKU=${LOOKU:-3}" > $R/talker.txt
  start_watcher
fi
COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy queue --queue-shift-per-s 0 $XFLAGS "${LF[@]}" --gpu-mem 0.1 --seed 20261021)
ps_=(); ci=0
for d in en2zh zh2en; do for s in 0 1 2 3; do
  g=${CA[$((ci % ${#CA[@]}))]}
  ps_+=("$(launch $g talk_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
    -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $R/tokens/tokens_new_$d.jsonl \
    "${COMMON[@]}" --direction $d --output $R/new/talker_$d --shard $s --num-shards 4 \
    --blocks-file $R/thinker_$d/blocks_${d}_w1.jsonl --items-file $T/UE_abs/data/items_$d.jsonl)")
  ci=$((ci + 1)); [ -n "$DRY" ] || sleep 30
done; done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $R/new/talker_$d/part_$s.jsonl ] || die "talker $d $s"; done; done
log "talkers done"
ps_=(); ci=0
for d in en2zh zh2en; do
  O=$R/new/talker_$d; g=${CA[$ci]}
  ps_+=("$(launch $g asr_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$O' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$O/results_speechonly.jsonl' --output '$O/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$O/asr_speech.jsonl'")")
  ci=$((ci + 1))
done
waitp "${ps_[@]}"
ps_=(); ci=0
for d in en2zh zh2en; do
  O=$R/new/talker_$d; [ -n "$DRY" ] || [ -s $O/asr_speech.jsonl ] || die "asr $d"
  g=${CA[$ci]}; g2=${CA[$((ci + 1))]}
  ps_+=("$(launch $g spk_$d env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. S1_DATA=$S1_DATA $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $O --out $R/metrics/new_${d}_titanet.jsonl --device cuda:0)")
  ps_+=("$(launch $g2 ecapa2_$d env CUDA_VISIBLE_DEVICES=$g2 "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.test_abs_ecapa2 \
    --dir $d --run $O --out $R/metrics/ecapa2_new_${d}_spksim.jsonl --device cuda:0)")
  if [ -z "$DRY" ]; then env OMP_NUM_THREADS=1 PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $O \
    --out $R/metrics/new_${d}_squim.jsonl --procs 16 > $LOGD/pesq_$d.log 2>&1 & ps_+=($!); fi
  ci=$((ci + 2))
done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for f in new_${d}_titanet new_${d}_squim ecapa2_new_${d}_spksim; do [ -n "$DRY" ] || [ -s $R/metrics/$f.jsonl ] || die "no $f"; done; done
log "metrics done"
[ -n "$DRY" ] && { echo "[dry] s0_apply dip NEW x2 + p3_score" >&2; exit 0; }
touch $STOP
for d in en2zh zh2en; do
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.voice_onset_v1.s0_apply \
    --run $R/new/talker_$d --dir $d --policy dip --out $R/play/NEW_$d.jsonl --procs 32 > $LOGD/play_NEW_$d.log 2>&1 || die "play NEW $d"
done
log "play done"
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p3_score --root $OUTN --rules pareto --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "$OUTN done"
