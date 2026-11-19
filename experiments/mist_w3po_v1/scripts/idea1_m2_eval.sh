#!/usr/bin/env bash
# Idea 1 M2 evaluation chain (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v3.4), after both trainings:
#  1. voice-token tables of both encoders (e2e: 2 phi_Z(y) - phi_Z(y + (sqrt2-1) bh); ctrl: phi_C(y)) for every demo stream
#  2. ID-1 test (idea1_m2_id1.py on the freest slot-A card) + speech metrics of its six arms
#  3. E1.2a: four talker arms swap_<dir>_{Ze2e,Zctrl} (idea1_swap_run.sh, freest card, engines 60 s apart) + metrics
#  4. E1.2b: four orchestrators game_b_<dir>__t1c_{e2e,ctrl} (idea1_game_orch.py, --cap135 --exclude 4 5 6 7), 60 s apart
#  5. after all (and the V reference arm game_b_*__t1c_vocals) finish: speech metrics of every new demo run, then
#     idea1_m2_summary.py (G2a) -> eval_outputs/mist_w3po_v1/voicecond_idea1/m2/g2a_summary.json; marker m2/EVAL_DONE
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea1_m2_eval.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_idea1/m2_eval.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
J=/opt/dlami/nvme/jasonleeeli
I=data/open_voicecond_v1/idea1
L=logs/mist_w3po_v1/voicecond_idea1
V1=eval_outputs/mist_w3po_v1/voicecond_idea1
M=$V1/m2; T=$V1/tokens; MET=$V1/metrics
PYE=$J/conda_envs/uniss-eval/bin/python
PYV=$J/venvs/vllm_omni_030/bin/python
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then log "[dry] ${*:0:400}"; else "$@"; fi; }
freest() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1<=3 {print $2, $1}' | sort -nr | head -1 | awk '{print $2}'; }
waitf() { [ -n "${DRY_RUN:-}" ] && { log "[dry] wait for $1"; return 0; }; until [ -e "$1" ]; do sleep 20; done; }
measure() {   # <talker dir> <metrics jsonl>
  [ -e $2 ] && return 0
  run env PYTHONPATH=. OMP_NUM_THREADS=1 $PYE -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics measure --talker $1 --out $2 --procs 32
}
waitf $M/train_zne.DONE; waitf $M/train_ctrl.DONE
if [ -z "${DRY_RUN:-}" ]; then grep -q rc=0 $M/train_zne.DONE && grep -q rc=0 $M/train_ctrl.DONE || { log "training failed"; exit 1; }; fi
export PYTHONPATH=. HF_HOME=$J/cache/huggingface OMP_NUM_THREADS=16
# 1. token tables
for mode in e2e ctrl; do
  ck=$M/$([ $mode = e2e ] && echo zne || echo ctrl)/encoder.pt
  for d in zh2en en2zh; do
    [ -e $T/game_${d}_$mode.jsonl ] || run $PYV -m experiments.mist_w3po_v1.voicecond.idea1_tokens --mode $mode --ckpt $ck \
      --items eval_outputs/mist_w3po_v1/speech/game_b_${d}__t1c/talker_in/items.jsonl \
      --vocals eval_outputs/mist_w3po_v1/game_muffle_v1/vocals_prompt/$d --out $T/game_${d}_$mode.jsonl --device cpu
  done
done
# 2. ID-1 test (background)
c=$(freest); log "ID-1 test on card $c"
( run env NEED_MIB=20000 PATH=$J/venvs/vllm_omni_030/bin:$PATH bash experiments/mist_w3po_v1/scripts/idea1_gpu_run.sh $c $M/id1_test.DONE \
    $PYV -m experiments.mist_w3po_v1.voicecond.idea1_m2_id1 --diag $I/diag_en $I/diag_zh --zne $M/zne/encoder.pt --ctrl $M/ctrl/encoder.pt \
    --out $M/id1_test --split test --device cuda:0 > $L/m2_id1_test.log 2>&1
  for arm in raw V Rt ctrl zne oracle; do measure $M/id1_test/$arm $MET/id1_test_$arm.jsonl; done
  [ -n "${DRY_RUN:-}" ] || date > $M/id1_test.MEASURED ) &
# 3. E1.2a
SP=$V1/speech
for d in zh2en en2zh; do for arm in Ze2e Zctrl; do
  D=$SP/swap_${d}_$arm/talker_in
  [ -e $D ] || { run mkdir -p $D; run cp -n eval_outputs/mist_w3po_v1/speech/game_swap_${d}_O/talker_in/items.jsonl eval_outputs/mist_w3po_v1/speech/game_swap_${d}_O/talker_in/blocks.jsonl $D/; }
done; done
c=$(freest); log "E1.2a on card $c"
run env SPECS="$c swap_zh2en_Ze2e zh2en $T/game_zh2en_e2e.jsonl;$c swap_zh2en_Zctrl zh2en $T/game_zh2en_ctrl.jsonl;$c swap_en2zh_Ze2e en2zh $T/game_en2zh_e2e.jsonl;$c swap_en2zh_Zctrl en2zh $T/game_en2zh_ctrl.jsonl" \
  TAG=M2 bash experiments/mist_w3po_v1/scripts/idea1_swap_run.sh
waitf $L/SWAP_M2_DONE
for d in zh2en en2zh; do for arm in Ze2e Zctrl; do measure $SP/swap_${d}_$arm/talker $MET/swap_${d}_$arm.jsonl; done; done
# 4. E1.2b
for mode in e2e ctrl; do for d in zh2en en2zh; do
  set_=$([ $d = zh2en ] && echo ali || echo test)
  R=$L/game_${d}_$mode
  if [ ! -e $R ]; then
    run $PY experiments/mist_w3po_v1/voicecond/idea1_game_orch.py --voice-override $T/game_${d}_$mode.jsonl --dir $d --set $set_ --base $B2 \
      --run $R --name game_b_${d}__t1c_$mode --ckpt - --speech-from game_b_$d --talker t1c --window 1 --cap135 --exclude 4 5 6 7 \
      > $R.launch.log 2>&1 &
    [ -n "${DRY_RUN:-}" ] || sleep 60
  fi
done; done
for x in e2e ctrl vocals; do for d in zh2en en2zh; do waitf $L/game_${d}_$x/events/DONE; done; done
# 5. metrics + summary
for x in e2e ctrl vocals; do for d in zh2en en2zh; do for s in "" _tts2; do
  measure eval_outputs/mist_w3po_v1/speech/game_b_${d}__t1c_$x$s/talker $MET/game_b_${d}__t1c_$x$s.jsonl
done; done; done
waitf $M/id1_test.MEASURED
run $PY -m experiments.mist_w3po_v1.voicecond.idea1_m2_summary --arms e2e ctrl vocals zne_Rt --out $M/g2a_summary.json
[ -n "${DRY_RUN:-}" ] || date > $M/EVAL_DONE
log "M2 eval done"
