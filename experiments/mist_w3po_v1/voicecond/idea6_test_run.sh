#!/usr/bin/env bash
# Idea 6 full-system test (reports/mist_w3po_v1/voicecond_idea6_bayes_prereg.zh-CN.md "完整系统测试预登记"), both arms
# (c = C_post, e = E2E_post), 9 sets each = 18 orchestrators, all in parallel, stacked on ALL 8 cards (user 2026-11-19:
# "8块卡都可以叠加跑"; --exclude none for bench, no --exclude for OOD-3; the shared exclusion file of
# sync_train_exclude.sh still applies; the orchestrators (YieldOrch) stop placeholder workers by PID before placing).
# Template: experiments/mist_w3po_v1/scripts/idea2_e2e_test_run_v3.sh steps 3-4 (token tables, dev selftest are done
# before: idea6_test_tokens.py -> eval_outputs/mist_w3po_v1/idea6_test/tokens/i6{c,e}_<set>.{jsonl,sidecar.json}).
#   bench  voicecond/idea6_test_orch.py bench (= scripts/voicecond_bench_orch.py, --voice-override <sidecar>,
#          --speech-from obsi_b_<c>_<d>, --talker t1c --window 1 --cap135) -> obsi_i6<arm>_<c>_<d>__t1c{,_tts2};
#          then bench_eval (at most EVAL_SLOTS at once, CPU)
#   ood3   voicecond/idea6_test_orch.py ood3 (= voicecond/idea2_ood3_orch.py --voice-override <jsonl>, --speech-from
#          idea2_ood3_<ss>_txt) -> idea2_ood3_i6<arm>_<ss>__t1c{,_tts2}; then chunk LAAL, idea1_speech_metrics measure,
#          compare vs B / Idea 2 frozen (sv) / Idea 1 C (i1c) [and e vs c], TitaNet similarity (M2's slot refs)
# Orchestrator starts go through one flock (8>&- 9>&- on the children), START_GAP s apart. Env as M2: SI_KV_CACHE_GIB=7.73
# ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10. Markers logs/mist_w3po_v1/voicecond/idea6/test/I6_<unit>_{DONE,FAILED}
# and I6_TEST_DONE. Existing run dirs / outputs are never relaunched or rewritten.
#   [START_GAP=20] [EVAL_SLOTS=3] [DRY_RUN=1] bash experiments/mist_w3po_v1/voicecond/idea6_test_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea6/test_run.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
J=/opt/dlami/nvme/jasonleeeli
export PATH=$J/venvs/vllm_omni_030/bin:$PATH HF_HOME=$J/cache/huggingface TMPDIR=$J/work/mist_w3po_v1/tmp
export SI_KV_CACHE_GIB=7.73 ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10
export TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton VLLM_CACHE_ROOT=$J/cache/vllm_omni_030 \
  TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPY=$J/venvs/nemo_diar/bin/python
EPY=$J/conda_envs/uniss-eval/bin/python
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
TOK=eval_outputs/mist_w3po_v1/idea6_test/tokens
OUT=eval_outputs/mist_w3po_v1/idea6_test/ood3
I2O=eval_outputs/mist_w3po_v1/voicecond_idea2/ood3
I1O=eval_outputs/mist_w3po_v1/voicecond_idea1/m3/ood3
GL=logs/mist_w3po_v1/voicecond/idea6/test
OD=data/open_voicecond_v1/idea2/ood3
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech
START_GAP=${START_GAP:-20}
EVAL_SLOTS=${EVAL_SLOTS:-3}
mkdir -p $GL $OUT
dry() { [ -n "${DRY_RUN:-}" ]; }
freest() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1; }

start_orch() {   # <run dir> <command...>: one start per START_GAP s across all units
  local R=$1; shift
  if dry; then log "[dry] $*"; return 0; fi
  (
    flock 8
    [ -e $R ] || { setsid nohup "$@" < /dev/null > $R.launch.log 2>&1 8>&- 9>&- & log "$(basename $R) orchestrator pid $!"; sleep $START_GAP; }
  ) 8> $GL/launch.lock
}
with_eval_slot() {   # run "$@" holding one of EVAL_SLOTS CPU eval slots
  local i
  while :; do
    for i in $(seq 1 $EVAL_SLOTS); do
      ( flock -n 7 || exit 99; "$@" ) 7> $GL/eval_slot_$i.lock
      local rc=$?
      [ $rc != 99 ] && return $rc
    done
    sleep 30
  done
}

bench_unit() {   # <arm c|e> <cond> <dir>
  local arm=$1 c=$2 d=$3 N R
  N=obsi_i6${arm}_${c}_${d}__t1c; R=$GL/$N
  [ -e $GL/I6_${arm}_${c}_${d}_DONE ] && return 0
  start_orch $R $PY experiments/mist_w3po_v1/voicecond/idea6_test_orch.py bench --cond $c --dir $d \
    --voice-override $TOK/i6${arm}_${c}_${d}.sidecar.json --base $B2 --run $R --name $N --ckpt - --speech-from obsi_b_${c}_${d} \
    --talker t1c --window 1 --cap135 --exclude none
  dry && return 0
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$N failed"; date -u +%FT%TZ >> $GL/I6_${arm}_${c}_${d}_FAILED; return 1; }
  log "$N orchestrator done; bench_eval"
  with_eval_slot env PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_eval --cond $c --dir $d --name $N \
    > $GL/eval_${arm}_${c}_${d}.log 2>&1 || { log "bench_eval failed $N"; date -u +%FT%TZ >> $GL/I6_${arm}_${c}_${d}_FAILED; return 1; }
  date -u +%FT%TZ >> $GL/I6_${arm}_${c}_${d}_DONE; log "$N DONE"
}

ood3_args() {
  case $1 in
    ami) D=zh2en; S=test; DR=zh2en;;
    en)  D=en2zh; S=test; DR="";;
    ali) D=zh2en; S=ali;  DR=ali;;
  esac
}
cmp_pair() {   # <ref measure prefix> <new measure prefix> <ss> <out>
  [ -e $4 ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics compare \
    --pair $1.jsonl:$2.jsonl:${3}_s1 ${1}_tts2.jsonl:${2}_tts2.jsonl:${3}_s2 --primary ${3}_s1 ${3}_s2 --out $4 > $4.log 2>&1
}
ood3_unit() {   # <arm> <ss>
  local arm=$1 ss=$2 N R rc=0 n f sfx
  N=idea2_ood3_i6${arm}_${ss}__t1c; R=$GL/$N
  [ -e $GL/I6_${arm}_ood3_${ss}_DONE ] && return 0
  ood3_args $ss
  start_orch $R $PY experiments/mist_w3po_v1/voicecond/idea6_test_orch.py ood3 --subset $ss --voice-override $TOK/i6${arm}_${ss}.jsonl \
    --dir $D --set $S --base $B2 --run $R --name $N --ckpt - --speech-from idea2_ood3_${ss}_txt --talker t1c --window 1 --cap135
  dry && return 0
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$N failed"; date -u +%FT%TZ >> $GL/I6_${arm}_ood3_${ss}_FAILED; return 1; }
  log "$N orchestrator done; OOD-3 metrics"
  for sfx in "" _tts2; do
    n=$N$sfx; f=$OUT/lat_chunk_$n.json
    if [ ! -e $f ]; then
      if [ $D = zh2en ]; then
        CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $OD/${ss}_ref.jsonl --dr $DR \
          experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $OD/${ss}_chunkwin.jsonl \
          --out $f > $f.log 2>&1 || rc=1
      else
        CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $OD/${ss}_ref.jsonl --work $f.work \
          --run $DEC/$n --talker $SPD/$n/talker --win $OD/${ss}_chunkwin.jsonl --out $f > $f.log 2>&1 || rc=1
      fi
    fi
    [ -e $OUT/$n.jsonl ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics measure --talker $SPD/$n/talker \
      --out $OUT/$n.jsonl --procs 32 > $OUT/$n.measure.log 2>&1 || rc=1
  done
  cmp_pair $I2O/idea2_ood3_b_${ss}__t1c $OUT/$N $ss $OUT/compare_i6${arm}_b_${ss}.json || rc=1
  cmp_pair $I2O/idea2_ood3_sv_${ss}__t1c $OUT/$N $ss $OUT/compare_i6${arm}_sv_${ss}.json || rc=1
  cmp_pair $I1O/idea1_ood3_c_${ss}__t1c $OUT/$N $ss $OUT/compare_i6${arm}_i1c_${ss}.json || rc=1
  CUDA_VISIBLE_DEVICES=$(freest) $NPY experiments/mist_w3po_v1/voicecond/idea2_spksim_generic.py sim --tag ood3_$ss --device cuda:0 \
    --runs $N ${N}_tts2 > $OUT/spksim_i6${arm}_${ss}.log 2>&1 || rc=1
  [ $rc = 0 ] && { date -u +%FT%TZ >> $GL/I6_${arm}_ood3_${ss}_DONE; log "$N DONE"; } || { date -u +%FT%TZ >> $GL/I6_${arm}_ood3_${ss}_FAILED; log "$N post failed"; }
  return $rc
}

log "18 units: bench c/e x {id2,ood1,clean} x {zh2en,en2zh}, OOD-3 c/e x {ami,en,ali}"
pids=""
for arm in c e; do
  for ss in ami en ali; do ood3_unit $arm $ss & pids="$pids $!"; dry || sleep 2; done
  for c in id2 ood1 clean; do for d in zh2en en2zh; do bench_unit $arm $c $d & pids="$pids $!"; dry || sleep 2; done; done
done
rc=0; for p in $pids; do wait $p || rc=1; done
dry && exit 0
# cross-arm OOD-3 comparisons (E2E vs C_post) and Idea 1 C similarity (for the vs-i1c similarity rows)
for ss in ami en ali; do
  if [ -e $GL/I6_c_ood3_${ss}_DONE ] && [ -e $GL/I6_e_ood3_${ss}_DONE ]; then
    cmp_pair $OUT/idea2_ood3_i6c_${ss}__t1c $OUT/idea2_ood3_i6e_${ss}__t1c $ss $OUT/compare_i6e_i6c_${ss}.json || rc=1
  fi
  CUDA_VISIBLE_DEVICES=$(freest) $NPY experiments/mist_w3po_v1/voicecond/idea2_spksim_generic.py sim --tag ood3_$ss --device cuda:0 \
    --runs idea1_ood3_c_${ss}__t1c idea1_ood3_c_${ss}__t1c_tts2 > $OUT/spksim_i1c_${ss}.log 2>&1 || rc=1
done
[ $rc = 0 ] && date -u +%FT%TZ >> $GL/I6_TEST_DONE || date -u +%FT%TZ >> $GL/I6_TEST_FAILED
log "Idea 6 test runner exit rc=$rc"
