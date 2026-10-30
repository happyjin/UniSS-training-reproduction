#!/usr/bin/env bash
# Paper v5 on the new server, after the Zh->En S2-dl 3k training (run_v5_train.sh's training child; its parent was stopped
# so that the mixed-interval run waits for the decoding). Order:
#   A  wait for thinker_gstar_s2dl_zh2en_3k_20261030 to be merged; touch logs/gpu_coord/TRAIN1_DONE
#   B  decode (make_v5_runs.py, worker count per card from its free memory): p2 = Zh->En S2-dl 3k at 1 / 0.5 / 2 s on AMI
#      and AliMeeting at 1 s; p5 = the diarizer-defined En->Zh windows for O2, O2 without speaker stream and the cascade
#   C  score every system (gstar_score; the cascade assembled first)
#   D  speech of zh2en_S2dl3k_d1 (gstar_talker prep -> si_infer, 8 shards on cards with >= 60 GB free, 60 s apart)
#   E  train the mixed-interval En->Zh model (S2-dl-m, 3k, 8 GPUs) while the speech is transcribed (capped Whisper)
#   F  decode S2-dl-m at 0.5 / 1 / 2 s (p4) and score; touch logs/gpu_coord/M_GPU_DONE
# Cards with a colleague's process are kept under 126 GB (136 GB otherwise); never touch other users' jobs.
# Every output is new (gstar_v5_20261030, gstar_v5_speech_20261030); nothing existing is overwritten.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
exec >> logs/multispk/v5_after.log 2>&1
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; OUT=${EV}/gstar_v5_20261030; SPO=${EV}/gstar_v5_speech_20261030
D=data/ami_multispk_si_v1; C=logs/gpu_coord; JR=reports/gstar_si_paper_v1/jobs/v5_runs
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python; PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python; M=experiments.uniss_qwen3_omni_v1.evaluation
T1C=${EV}/si_t1c_20261021T181440Z/autoext/export_1500/uniss
L1=logs/multispk/thinker_gstar_s2dl_zh2en_3k_20261030.log
S2DLZ=${EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged
RUN4=thinker_gstar_s2dlm_3k_20261030; S2DLM=${EV}/${RUN4}/thinker_merged; L4=logs/multispk/${RUN4}.log
WZ=${D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl; WE=${D}/e2e_gstar_stream_v1/windows_test_ref.jsonl
WA=data/alimeeting_si_v1/windows_test_ref.jsonl; WD=${D}/e2e_gstar_v1/windows_test_diar_offline.jsonl
mkdir -p $C logs/multispk/v5
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1; }
limit() {   # $1 card: 136 GB, or 126 GB when another user's process runs on it (room for it to grow)
  python3 -c "import sys; sys.path.insert(0, 'reports/gstar_si_paper_v1'); from make_v5_runs import foreign; print(126000 if foreign($1) else 136000)"
}
by_free() { for c in 0 1 2 3 4 5 6 7; do echo "$(used $c) $c"; done | sort -n | awk '{print $2}'; }   # emptiest first

multi() {   # $1 phase, $2 thinker: one gstar_multi per card file of the phase, pids appended to the global MPIDS
  local f c
  for f in ${JR}/$1_card*.json; do
    [ -e "$f" ] || continue
    c=${f##*_card}; c=${c%.json}
    CUDA_VISIBLE_DEVICES=$c PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m ${M}.gstar_multi --thinker $2 $f \
      > logs/multispk/v5/$1_card$c.log 2>&1 &
    MPIDS+=($!); sleep 10
  done
}

score() {   # $1 system, $2 windows, $3 post key
  local d=${OUT}/$1 n
  n=$(cat $d/hyp_s*.jsonl 2>/dev/null | wc -l)
  [ "$n" -eq "$(wc -l < $2)" ] || { log "$1: $n / $(wc -l < $2) windows, NOT scored"; return; }
  [ -e $d/hyp.jsonl ] && { log "$1: hyp.jsonl exists, not rescored"; return; }
  cat $d/hyp_s*.jsonl > $d/hyp.jsonl
  PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows $2 --hyp $d/hyp.jsonl --post-key $3 --out $d/score.json > $d/score.log 2>&1
  log "$1: $(grep -m1 cpBLEU_session $d/score.json) $(grep -m1 LAAL_s $d/score.json)"
}

# A
log "A: wait for ${S2DLZ}"
until grep -qE "THINKER_GSTAR_(DONE|FAILED|MERGE_FAILED)" ${L1} 2>/dev/null; do sleep 60; done
grep -q THINKER_GSTAR_DONE ${L1} || { log "training 1 failed: $(tail -2 ${L1})"; exit 1; }
touch $C/TRAIN1_DONE; log "A: trained"; sleep 30

# B
[ -e ${OUT} ] && { log "refuse: ${OUT} exists"; exit 1; }
mkdir -p ${OUT}
PYTHONPATH=. ${PYE} -m ${M}.gstar_cascade segment --windows ${WD} --out ${OUT}/en2zh_D-segments
cards=($(by_free))
log "B: cards by free memory: ${cards[*]} ($(for c in ${cards[@]}; do echo -n "$c:$(used $c) "; done))"
pids5=(); p2cards=("${cards[@]:2}")
if ! python3 reports/gstar_si_paper_v1/make_v5_runs.py p5 ${cards[0]},${cards[1]}; then
  log "B: p5 does not fit now; its cards join p2, p5 runs after p2"; p2cards=("${cards[@]}")
fi
for f in ${JR}/p5_card*.json; do
  [ -e "$f" ] || continue
  c=${f##*_card}; c=${c%.json}
  th=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[0]['argv'][1])" $f)
  CUDA_VISIBLE_DEVICES=$c PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m ${M}.gstar_multi --thinker $th $f \
    > logs/multispk/v5/p5_card$c.log 2>&1 &
  pids5+=($!); sleep 10
done
sleep 120   # let the p5 Thinkers load before reading the remaining cards' free memory
rest=$(IFS=,; echo "${p2cards[*]}")
python3 reports/gstar_si_paper_v1/make_v5_runs.py p2 ${rest}
MPIDS=(); multi p2 ${S2DLZ}; pids2=("${MPIDS[@]}")
log "B: p5 pids ${pids5[*]}, p2 pids ${pids2[*]}"
for p in ${pids5[@]} ${pids2[@]}; do wait $p || log "a gstar_multi run failed (pid $p)"; done
if ! ls ${JR}/p5_card*.json > /dev/null 2>&1; then   # p5 did not fit beside p2: run it now
  cards=($(by_free))
  if python3 reports/gstar_si_paper_v1/make_v5_runs.py p5 ${cards[0]},${cards[1]}; then
    pids5=()
    for f in ${JR}/p5_card*.json; do
      c=${f##*_card}; c=${c%.json}
      th=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[0]['argv'][1])" $f)
      CUDA_VISIBLE_DEVICES=$c PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m ${M}.gstar_multi --thinker $th $f \
        > logs/multispk/v5/p5_card$c.log 2>&1 &
      pids5+=($!); sleep 10
    done
    for p in ${pids5[@]}; do wait $p || log "a p5 run failed (pid $p)"; done
  else log "B: p5 still does not fit; diarizer windows not decoded"; fi
fi

# C
for w in 1 0.5 2; do score zh2en_S2dl3k_d$w ${WZ} lowlat; done
score ali_S2dl3k_d1 ${WA} lowlat
score en2zh_D-O2 ${WD} offline
score en2zh_D-O2-nosd ${WD} offline
d=${OUT}/en2zh_D-C-ft2_seg
if [ "$(cat $d/hyp_s*.jsonl | wc -l)" -eq "$(wc -l < ${OUT}/en2zh_D-segments/segments.jsonl)" ]; then
  cat $d/hyp_s*.jsonl > $d/hyp.jsonl; mkdir -p ${OUT}/en2zh_D-C-ft2
  PYTHONPATH=. ${PYE} -m ${M}.gstar_cascade assemble --seg ${OUT}/en2zh_D-segments/segments.jsonl --hyp $d/hyp.jsonl \
    --windows ${WD} --out ${OUT}/en2zh_D-C-ft2/hyp.jsonl
  PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows ${WD} --hyp ${OUT}/en2zh_D-C-ft2/hyp.jsonl --post-key offline \
    --out ${OUT}/en2zh_D-C-ft2/score.json > ${OUT}/en2zh_D-C-ft2/score.log 2>&1
  log "en2zh_D-C-ft2: $(grep -m1 cpBLEU_session ${OUT}/en2zh_D-C-ft2/score.json)"
else log "en2zh_D-C-ft2: segment decode incomplete, not scored"; fi

# D
o=${SPO}/zh2en_S2dl3k_d1; h=${OUT}/zh2en_S2dl3k_d1
if [ -e ${h}/hyp.jsonl ] && [ ! -e ${o} ]; then
  mkdir -p ${o}
  PYTHONPATH=. ${PYE} -m ${M}.gstar_talker prep --hyp ${h}/hyp.jsonl --windows ${WZ} --causal --blocks --preset lowlat \
    --session zh_mix_v8 --diar nemotron_lowlat_zh8 --sep ' ' --out ${o}/talker_in
  tp=(); k=0
  for c in $(by_free); do
    [ $k -ge 8 ] && break
    [ $(used $c) -le $(( $(limit $c) - 60000 )) ] || continue   # talker at gpu-mem 0.3 holds up to ~55 GB
    CUDA_VISIBLE_DEVICES=$c PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=16 ${PYV} -m ${M}.si_infer \
      --model ${T1C} --direction zh2en --window 1 --output ${o}/talker --text-policy file \
      --blocks-file ${o}/talker_in/blocks.jsonl --items-file ${o}/talker_in/items.jsonl --speed-policy fixed --history-ban 8 \
      --max-model-len 32768 --gpu-mem 0.3 --shard $k --num-shards 8 > ${o}/talk$k.log 2>&1 &
    tp+=($!); log "D: talker shard $k on card $c (pid $!)"; k=$((k+1)); sleep 60
  done
  [ $k -eq 8 ] || log "D: only $k cards had room; shards $k..7 run after the first ones"
  while [ $k -lt 8 ]; do   # remaining shards on the first free card, one at a time
    wait ${tp[0]}; tp=("${tp[@]:1}")
    c=$(by_free | head -1)
    CUDA_VISIBLE_DEVICES=$c PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=16 ${PYV} -m ${M}.si_infer \
      --model ${T1C} --direction zh2en --window 1 --output ${o}/talker --text-policy file \
      --blocks-file ${o}/talker_in/blocks.jsonl --items-file ${o}/talker_in/items.jsonl --speed-policy fixed --history-ban 8 \
      --max-model-len 32768 --gpu-mem 0.3 --shard $k --num-shards 8 > ${o}/talk$k.log 2>&1 &
    tp+=($!); log "D: talker shard $k on card $c (pid $!)"; k=$((k+1)); sleep 60
  done
  for p in ${tp[@]}; do wait $p || log "D: a talker shard failed (pid $p)"; done
  cat ${o}/talker/part_*.jsonl > ${o}/talker/results.jsonl
  ${PYE} scripts/gpu_queue/p80_speechonly.py ${o}/talker
  log "D: $(wc -l < ${o}/talker/results_speechonly.jsonl) clips to transcribe"
else log "D: skipped (no hyp, or ${o} exists)"; fi

# E: training 4 in the background, the speech ASR and the rest of D's post-processing meanwhile
(
  (until grep -qE "iteration +30/" ${L4} 2>/dev/null; do grep -qE "THINKER_GSTAR_(FAILED|DONE)|拒绝覆盖" ${L4} 2>/dev/null && exit; sleep 20; done
   { echo "[$(date -u +%F' '%T) UTC] 4 stable: per-card used MiB"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
     grep -E "iteration +30/" ${L4} | grep -oE "elapsed time per iteration \(ms\): [0-9.]+"; } >> $C/train_mem.txt
   touch $C/TRAIN4_STABLE) &
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 MBS=8 GBS=64 RUN_ID=${RUN4} BASE=${EV}/thinker_gstar_o2_20261029b/thinker_merged \
    INIT_MERGED=1 ROWS=${D}/e2e_gstar_stream_multid_v1/rows_train.jsonl TRAIN_ITERS=3000 MASTER_PORT=29862 \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > ${L4} 2>&1
  touch $C/TRAIN4_DONE
) &
tr4=$!
log "E: training ${RUN4} (subshell $tr4)"
if [ -e ${o}/talker/results_speechonly.jsonl ] && [ ! -e ${o}/talker/asr_speech.jsonl ]; then
  sleep 600   # past the training's start-up peak
  ap=(); k=0
  for c in 0 1 2 4 5 6; do
    CUDA_VISIBLE_DEVICES=$c PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ${PYE} -c "import runpy, sys, torch; torch.cuda.set_per_process_memory_fraction(0.15); sys.argv = ['asr_transcribe'] + sys.argv[1:]; runpy.run_module('evaluation.asr_transcribe', run_name='__main__')" \
      --input ${o}/talker/results_speechonly.jsonl --output ${o}/talker/asr_speech.part$k.jsonl --num-shards 6 --shard-index $k \
      --device cuda:0 --batch-size 8 --resume >> ${o}/asr.part$k.log 2>&1 &
    ap+=($!); k=$((k+1)); sleep 5
  done
  fail=0; for p in ${ap[@]}; do wait $p || fail=1; done
  if [ $fail -eq 0 ]; then
    cat ${o}/talker/asr_speech.part*.jsonl > ${o}/talker/asr_speech.jsonl
    PYTHONPATH=. ${PYE} -m ${M}.gstar_talker stitch --results ${o}/talker/results.jsonl --prep ${o}/talker_in --windows ${WZ} \
      --mode stream --out ${o}/stitched
    PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows ${WZ} --hyp ${h}/hyp.jsonl --asr ${o}/talker/asr_speech.jsonl \
      --post-key lowlat --out ${o}/score.json > ${o}/score.log 2>&1
    PYTHONPATH=. ${PYE} -m ${M}.gstar_listen --transcript ${o}/stitched/transcript.jsonl --out ${o}/listen.json
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. ${PYE} -m ${M}.gstar_voice --windows ${WZ} --hyp ${h}/hyp.jsonl \
      --results ${o}/talker/results.jsonl --out ${o}/voice.json
    TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. ${PYE} -m evaluation.utmos_metrics \
      --input ${o}/talker/results_speechonly.jsonl --output-dir ${o}/utmos --device cuda:0 > ${o}/utmos.log 2>&1 || log "utmos failed"
    log "E: speech zh2en_S2dl3k_d1: $(grep -m1 cpASR_BLEU_session ${o}/score.json)"
  else log "E: speech ASR incomplete; rerun the ASR block (shards resume)"; fi
fi
wait $tr4
grep -q THINKER_GSTAR_DONE ${L4} || { log "training 4 failed: $(tail -2 ${L4})"; touch $C/M_GPU_DONE; exit 1; }

# F
sleep 30
python3 reports/gstar_si_paper_v1/make_v5_runs.py p4 $(IFS=,; echo "$(by_free | tr '\n' ',' | sed 's/,$//')")
MPIDS=(); multi p4 ${S2DLM}; pids4=("${MPIDS[@]}")
for p in ${pids4[@]}; do wait $p || log "a p4 run failed (pid $p)"; done
for w in 0.5 1 2; do score en2zh_S2dlm_d$w ${WE} lowlat; done
touch $C/M_GPU_DONE
log "V5_AFTER_DONE"
