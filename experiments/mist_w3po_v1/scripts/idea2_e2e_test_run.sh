#!/usr/bin/env bash
# Idea 2 e2e TEST of main_v2's final checkpoint (reports/mist_w3po_v1/voicecond_idea2_e2e_prereg.zh-CN.md, test addendum
# 2026-11-19): waiter + runner, armed before main_v2 finishes.
#   0. waits for eval_outputs/.../e2e/idea2_e2e_main_v2/DONE (LAUNCHER_RC without DONE -> FAILED) and for main_v2's
#      launcher to release slot B (cards 4-7); then stops the guard placeholders on cards 4-7 BY PID (cmdline checked).
#   1. dev-only checks on card 4 (fail -> stop, no test run): token selftest on all cvt dev items (initial = frozen method;
#      step-3000 checkpoint vs the training log's dev record), Talker export (LoRA merged in fp32 into a COPY of T1c, bf16,
#      T1c tensor names; logits vs the PEFT model on cvt dev sequences), si_infer smoke of the export on 6 cvt dev items
#      through the same talk wrapper (voicecond/idea2_e2e_talker.py --via vo).
#   2. token tables (idea2_e2e_tokens.py build, same arguments as M2's idea2_stream_tokens runs, V fallback ids of M2)
#      for the 6 OpenBroadcast-SI sets and the 3 OOD-3 subsets, spread over cards 4-7.
#   3. all 9 orchestrators in parallel (voicecond/idea2_e2e_orch.py = M2's voicecond_bench_orch / idea2_ood3_orch with
#      --model -> export on the talk jobs; --speech-from B's text, --talker t1c, --window 1, --cap135, --exclude 0 1 2 3,
#      SI_KV_CACHE_GIB=7.73 ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10): runs obsi_e2e_<c>_<d>__t1c and
#      idea2_ood3_e2e_<ss>__t1c (both TTS seeds each); then bench_eval, and for OOD-3 chunk LAAL, idea1_speech_metrics
#      measure + compare (B -> e2e and shrink+Vfb -> e2e per seed), TitaNet similarity (refs of M2).
#   4. idea2_e2e_test_summary bench / ood3 -> eval_outputs/mist_w3po_v1/voicecond_idea2/test_summary_e2e_{bench,ood3}.json
# Markers logs/mist_w3po_v1/voicecond/idea2/e2e_test/{E2E_<unit>_DONE,E2E_<unit>_FAILED,E2E_TEST_DONE,E2E_TEST_FAILED}.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_e2e_test_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/e2e_test_run.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
J=/opt/dlami/nvme/jasonleeeli
export PATH=$J/venvs/vllm_omni_030/bin:$PATH HF_HOME=$J/cache/huggingface TMPDIR=$J/work/mist_w3po_v1/tmp
export SI_KV_CACHE_GIB=7.73 ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10
export TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache TRITON_CACHE_DIR=$J/cache/triton VLLM_CACHE_ROOT=$J/cache/vllm_omni_030 \
  TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=$J/venvs/vllm_omni_030/bin/python
NPY=$J/venvs/nemo_diar/bin/python
EPY=$J/conda_envs/uniss-eval/bin/python
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
E2=eval_outputs/mist_w3po_v1/voicecond_idea2/e2e
RUN=$E2/idea2_e2e_main_v2
CK=$RUN/ckpt_step03000.pt
X=$E2/idea2_e2e_main_v2_export/talker_s03000/uniss
TOK=eval_outputs/mist_w3po_v1/voicecond_idea2/tokens
OUT=eval_outputs/mist_w3po_v1/voicecond_idea2/ood3
GL=logs/mist_w3po_v1/voicecond/idea2/e2e_test
SM=eval_outputs/mist_w3po_v1/voicecond_idea2/e2e_smoke/test_prep_s03000
CV=data/open_voicecond_v1/idea2/cvt
OD=data/open_voicecond_v1/idea2/ood3
DD=eval_outputs/mist_w3po_v1/data
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech
SL=logs/mist_w3po_v1/slots
RT=joint_v8h6k_20261113T073658Z_s06000__t1c
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
mkdir -p $GL
dry() { [ -n "${DRY_RUN:-}" ]; }
fail() { log "FAILED $1"; dry || date -u +%FT%TZ >> $GL/E2E_TEST_FAILED; exit 1; }

uuid_of() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v k="$1" '$1 == k {print $2}'; }
free_card() {   # stop the guard placeholder supervisor + children on card $1 (4-7 only), by PID, cmdline checked
  local k=$1 sup kids s a
  case "$k" in 4|5|6|7) ;; *) log "refuse to free card $k"; return;; esac
  sup=$(ps -eo pid,ppid,args | awk -v k="$k" '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ ("--devices " k " ") {print $1}')
  for s in $sup; do
    a=$(tr '\0' ' ' < /proc/$s/cmdline 2>/dev/null)
    case "$a" in *train_mist_workload.py*"--devices $k "*) ;; *) continue;; esac
    kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}' | tr '\n' ' ')
    log "card $k: SIGTERM placeholder supervisor $s children: $kids"
    dry || kill -TERM $kids $s 2>/dev/null
  done
}

# ---------------------------------------------------------------- 0. wait for main_v2
log "waiting for $RUN/DONE"
if ! dry; then
  until [ -e $RUN/DONE ] || [ -e $RUN/LAUNCHER_RC ]; do sleep 30; done
  [ -e $RUN/DONE ] || fail main_v2_not_done
  until [ -e $RUN/LAUNCHER_RC ]; do sleep 10; done
  until ! grep -qs "run idea2_e2e_main_v2" $SL/slotB.owner; do sleep 10; done
  [ -e $CK ] || fail no_ckpt_step03000
fi
log "main_v2 done; checkpoint $CK"
for k in 4 5 6 7; do free_card $k; done
dry || sleep 20

# ---------------------------------------------------------------- 1. dev-only checks (card 4)
step1() {
  mkdir -p $SM
  CUDA_VISIBLE_DEVICES=4 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_e2e_tokens.py selftest --ckpt $CK \
    --out $SM/token_selftest.json --device cuda:0 > $SM/token_selftest.log 2>&1 || return 1
  $VPY -I -c "
import json, sys
r = json.load(open('$SM/token_selftest.json'))
ok = r['init']['z_maxabs_vs_stored'] < 1e-3 and r['init']['u_maxabs_vs_numpy_shrink'] < 1e-3 \
     and r['ckpt'].get('max_abs_diff_vs_train_log', 1) < 0.02
print('token selftest', 'OK' if ok else 'BAD', r['init'].get('z_maxabs_vs_stored'), r['init'].get('u_maxabs_vs_numpy_shrink'),
      r['ckpt'].get('max_abs_diff_vs_train_log'))
sys.exit(0 if ok else 1)" || return 1
  if [ ! -e $X/model.safetensors ]; then
    CUDA_VISIBLE_DEVICES=4 $VPY -m experiments.mist_w3po_v1.voicecond.idea2_e2e_export --ckpt $CK --out $X --device cuda:0 \
      > $SM/export.log 2>&1 || return 1
  fi
  # si_infer smoke of the export on 6 cvt DEV items (never test items)
  head -6 $CV/teacher_in/items_zh2en_dev.jsonl > $SM/items_dev6.jsonl
  $VPY -I -c "
import json
ids = {json.loads(l)['id'] for l in open('$SM/items_dev6.jsonl')}
with open('$SM/blocks_dev6.jsonl', 'w') as f:
    for l in open('$CV/teacher_in/blocks_zh2en_dev.jsonl'):
        if json.loads(l)['id'] in ids: f.write(l)"
  CUDA_VISIBLE_DEVICES=4 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=.:third_party/Megatron-LM $VPY -m \
    experiments.mist_w3po_v1.voicecond.idea2_e2e_talker --talker-model $X --via vo --voice-override $CV/teacher_in/voice_gstar.jsonl \
    --model $T1C --direction zh2en --window 1 --output $SM/si_smoke --text-policy file --blocks-file $SM/blocks_dev6.jsonl \
    --items-file $SM/items_dev6.jsonl --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 --shard 0 \
    --num-shards 1 > $SM/si_smoke.log 2>&1 || return 1
  $VPY -I -c "
import json, sys
R = [json.loads(l) for l in open('$SM/si_smoke/part_0.jsonl')]
ok = len(R) == 6 and all(r['error'] is None and r['codes'] > 0 for r in R)
print('si smoke', 'OK' if ok else 'BAD', [(r['id'], r['codes'], r['error']) for r in R])
sys.exit(0 if ok else 1)" || return 1
  grep -q "idea2 e2e talker\] --model $X" $SM/si_smoke.log || return 1
}
if dry; then log "[dry] step 1 dev checks on card 4 -> $SM, export -> $X"
else
  log "step 1: dev-only checks"
  step1 > $GL/step1.log 2>&1 || fail step1_dev_checks
  log "step 1 OK: $(grep -h 'OK' $GL/step1.log | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- 2. token tables
tok_bench() {   # <cond> <dir> <card>
  local c=$1 d=$2
  [ -s $TOK/obsi_e2e_${c}_${d}.sidecar.json ] && return 0
  CUDA_VISIBLE_DEVICES=$3 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_e2e_tokens.py build --ckpt $CK --m2-tag obsi_${c}_${d} \
    --tag obsi_e2e_${c}_${d} --data data/open_voicecond_v1/bench/$c/$d --dir $d --run obsi_b_${c}_${d}__t1c \
    --windows $DD/windows_obsi_${c}_${d}_c2g06_v1/windows_test_online_${d}.jsonl --device cuda:0 > $GL/tokens_${c}_${d}.log 2>&1
}
tok_ood3() {    # <ss> <card>
  local ss=$1 a
  [ -s $TOK/ood3_e2e_$ss.jsonl ] && return 0
  case $ss in
    ami) a="--data data/ami_multispk_si_v1 --dir zh2en --session zh_mix_v8 --diar nemotron_lowlat_zh8 --lang zh --run jtest_bami_$RT --windows $DD/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl --meetings EN2002d IS1009c";;
    en)  a="--data data/ami_multispk_si_v1 --dir en2zh --session ihm_mix --diar nemotron_lowlat --lang en --run jtest_ben_$RT --windows $DD/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl --meetings EN2002d IS1009c";;
    ali) a="--data data/alimeeting_si_v1 --dir zh2en --session near_mix --diar nemotron_lowlat --lang zh --run jtest_bali_$RT --windows $DD/windows_test_c2g06_v1/windows_test_online_ali.jsonl --meetings R8009_M8021 R8009_M8024";;
  esac
  CUDA_VISIBLE_DEVICES=$2 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_e2e_tokens.py build --ckpt $CK --m2-tag ood3_$ss \
    --tag ood3_e2e_$ss $a --device cuda:0 > $GL/tokens_ood3_$ss.log 2>&1
}
if dry; then log "[dry] step 2 tokens: obsi_e2e_{id2,ood1,clean}_{zh2en,en2zh}, ood3_e2e_{ami,en,ali} on cards 4-7"
else
  log "step 2: token tables"
  pids=""; k=0
  for c in id2 ood1 clean; do for d in zh2en en2zh; do tok_bench $c $d $((4 + k % 4)) & pids="$pids $!"; k=$((k + 1)); done; done
  for ss in ami en ali; do tok_ood3 $ss $((4 + k % 4)) & pids="$pids $!"; k=$((k + 1)); done
  rc=0; for p in $pids; do wait $p || rc=1; done
  [ $rc = 0 ] || fail step2_tokens
  log "step 2 OK: $(grep -h token_agreement $GL/tokens_*.log | wc -l) tables"
fi

# ---------------------------------------------------------------- 3. orchestrators + metrics
bench_unit() {   # <cond> <dir>
  local c=$1 d=$2 N=obsi_e2e_$1_$2__t1c R
  R=$GL/$N
  [ -e $GL/E2E_${c}_${d}_DONE ] && return 0
  local CMD="$PY experiments/mist_w3po_v1/voicecond/idea2_e2e_orch.py bench --talker-model $X --cond $c --dir $d \
    --voice-override $TOK/obsi_e2e_${c}_${d}.sidecar.json --base $B2 --run $R --name $N --ckpt - --speech-from obsi_b_${c}_${d} \
    --talker t1c --window 1 --cap135 --exclude 0 1 2 3"
  if dry; then log "[dry] $CMD"; return 0; fi
  [ -e $R ] || { setsid nohup $CMD < /dev/null > $R.launch.log 2>&1 9>&- & log "$N orchestrator pid $!"; }
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$N failed"; date -u +%FT%TZ >> $GL/E2E_${c}_${d}_FAILED; return 1; }
  PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_eval --cond $c --dir $d --name $N > $GL/eval_${c}_${d}.log 2>&1 \
    || { log "bench_eval failed $N"; date -u +%FT%TZ >> $GL/E2E_${c}_${d}_FAILED; return 1; }
  date -u +%FT%TZ >> $GL/E2E_${c}_${d}_DONE; log "$N DONE"
}
ood3_args() {
  case $1 in
    ami) D=zh2en; S=test; DATA=data/ami_multispk_si_v1; SESS=zh_mix_v8; DIAR=nemotron_lowlat_zh8; MEET="EN2002d IS1009c"; DR=zh2en;;
    en)  D=en2zh; S=test; DATA=data/ami_multispk_si_v1; SESS=ihm_mix; DIAR=nemotron_lowlat; MEET="EN2002d IS1009c"; DR="";;
    ali) D=zh2en; S=ali;  DATA=data/alimeeting_si_v1; SESS=near_mix; DIAR=nemotron_lowlat; MEET="R8009_M8021 R8009_M8024"; DR=ali;;
  esac
}
ood3_unit() {   # <ss>
  local ss=$1 N R rc=0 n f sfx
  N=idea2_ood3_e2e_${ss}__t1c; R=$GL/$N
  [ -e $GL/E2E_ood3_${ss}_DONE ] && return 0
  ood3_args $ss
  local CMD="$PY experiments/mist_w3po_v1/voicecond/idea2_e2e_orch.py ood3 --talker-model $X --subset $ss \
    --voice-override $TOK/ood3_e2e_${ss}.jsonl --dir $D --set $S --base $B2 --run $R --name $N --ckpt - \
    --speech-from idea2_ood3_${ss}_txt --talker t1c --window 1 --cap135 --exclude 0 1 2 3"
  if dry; then log "[dry] $CMD"; return 0; fi
  [ -e $R ] || { setsid nohup $CMD < /dev/null > $R.launch.log 2>&1 9>&- & log "$N orchestrator pid $!"; }
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$N failed"; date -u +%FT%TZ >> $GL/E2E_ood3_${ss}_FAILED; return 1; }
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
      --out $OUT/$n.jsonl --procs 48 > $OUT/$n.measure.log 2>&1 || rc=1
  done
  for ref in b:idea2_ood3_b_${ss}__t1c shrinkvfb:idea2_ood3_sv_${ss}__t1c; do
    local tag=${ref%%:*} rn=${ref#*:}
    [ -e $OUT/compare_e2e_${tag}_${ss}.json ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics compare \
      --pair $OUT/$rn.jsonl:$OUT/$N.jsonl:${ss}_s1 $OUT/${rn}_tts2.jsonl:$OUT/${N}_tts2.jsonl:${ss}_s2 \
      --primary ${ss}_s1 ${ss}_s2 --out $OUT/compare_e2e_${tag}_${ss}.json > $OUT/compare_e2e_${tag}_${ss}.log 2>&1 || rc=1
  done
  CUDA_VISIBLE_DEVICES=5 $NPY experiments/mist_w3po_v1/voicecond/idea2_spksim_generic.py sim --tag ood3_$ss --device cuda:0 \
    --runs $N ${N}_tts2 > $OUT/spksim_e2e_${ss}.log 2>&1 || rc=1
  [ $rc = 0 ] && { date -u +%FT%TZ >> $GL/E2E_ood3_${ss}_DONE; log "$N DONE"; } || { date -u +%FT%TZ >> $GL/E2E_ood3_${ss}_FAILED; log "$N post failed"; }
  return $rc
}
log "step 3: 6 OpenBroadcast-SI + 3 OOD-3 orchestrators"
pids=""
for c in id2 ood1 clean; do for d in zh2en en2zh; do bench_unit $c $d & pids="$pids $!"; dry || sleep 5; done; done
for ss in ami en ali; do ood3_unit $ss & pids="$pids $!"; dry || sleep 5; done
rc=0; for p in $pids; do wait $p || rc=1; done
dry && { log "[dry] step 4 summaries: $EPY -m experiments.mist_w3po_v1.voicecond.idea2_e2e_test_summary bench|ood3"; exit 0; }

# ---------------------------------------------------------------- 4. summaries
log "step 4: summaries (units rc=$rc)"
for w in bench ood3; do
  PYTHONPATH=. $EPY -m experiments.mist_w3po_v1.voicecond.idea2_e2e_test_summary $w > $GL/summary_$w.log 2>&1 || rc=1
  log "summary $w: $(head -1 $GL/summary_$w.log)"
done
[ $rc = 0 ] && date -u +%FT%TZ >> $GL/E2E_TEST_DONE || date -u +%FT%TZ >> $GL/E2E_TEST_FAILED
log "E2E test runner exit rc=$rc"
