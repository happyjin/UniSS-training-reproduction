#!/usr/bin/env bash
# v3 (2026-11-19 ~06:25 Prague, coordinator, user: "非训练任务尽量叠满显存跑，并行加快速度"): the three subsets run IN
# PARALLEL, and a subset's Z / C orchestrators start as soon as Idea 2's B arm of that subset has its talker inputs
# ($SPD/idea2_ood3_b_<ss>__t1c/talker_in/items.jsonl, unchanged for >= 120 s) and its text run decode/idea2_ood3_<ss>_txt
# exists, instead of after Idea 2's whole subset marker OOD3_<ss>_DONE. That marker is now awaited only before the
# speaker-similarity step (it needs Idea 2's slot references and B's finished speech). Orchestrator starts of all subsets
# go through one lock, 60 s apart (as v2). Inputs, flags, outputs, markers and the final analysis are identical to v2;
# everything that already exists is skipped, so the ami arms started by v1 are only waited for.
# v2 (2026-11-19, GPU stacking fix 548113295): every orchestrator launched here gets SI_KV_CACHE_GIB=7.73 (fixed KV cache,
# identical capacity, no vLLM memory profiling) ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10; otherwise identical to v1
# (an existing run dir is never relaunched, so the ami arms started by v1 keep running and are only waited for).
# Idea 1 M3 OOD-3 no-harm (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v4.5): per subset ss (ami = AMI zh->en,
# en = AMI en->zh, ali = AliMeeting zh->en; Idea 2's subset chain and B arm idea2_ood3_b_<ss>__t1c, read only), after
# Idea 2's marker logs/mist_w3po_v1/voicecond/idea2/ood3/OOD3_<ss>_DONE (FAILED -> skipped):
#   1. HDemucs vocals of B's prompts (idea1_bench_tokens vocals_gpu --items, freest slot-A card) -> data/open_voicecond_v1/
#      idea1/ood3_vocals/<ss>/; 2. sha1-checked token tables of Z / C (idea1_tokens --mode e2e|ctrl) -> m3/ood3/tokens_{z,c}_<ss>.jsonl
#   3. Z / C arms through Idea 2's wrapper voicecond/idea2_ood3_orch.py --voice-override (same subset chain, --speech-from
#      idea2_ood3_<ss>_txt, --talker t1c, --cap135, --exclude 4 5 6 7), run names idea1_ood3_{z,c}_<ss>__t1c, 60 s apart
#   4. chunk-protocol LAAL (as idea2_ood3_run.sh lat()), idea1_speech_metrics measure, TitaNet vs Idea 2's slot refs
#      (data/open_voicecond_v1/idea2/spkref/ood3_<ss>) for B / Z / C both seeds -> eval_outputs/.../voicecond_idea1/m3/ood3/
# Markers logs/mist_w3po_v1/voicecond_idea1/m3/OOD3_I1_<ss>_{DONE,FAILED,SKIPPED}; at the end idea1_m3_analysis ood3 ->
# m3/ood3_summary.json and logs/.../m3/OOD3_ANALYSIS_DONE.
#   [SUBSETS="ami en ali"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea1_ood3_run_v3.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond_idea1/ood3_run_v3.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
J=/opt/dlami/nvme/jasonleeeli
export PATH=$J/venvs/vllm_omni_030/bin:$PATH HF_HOME=$J/cache/huggingface
export SI_KV_CACHE_GIB=7.73 ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10   # v2: stacking fix (548113295)
VPY=$J/venvs/vllm_omni_030/bin/python; NPY=$J/venvs/nemo_diar/bin/python; EPY=$J/conda_envs/uniss-eval/bin/python
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
I2L=logs/mist_w3po_v1/voicecond/idea2/ood3
OD=data/open_voicecond_v1/idea2/ood3
L=logs/mist_w3po_v1/voicecond_idea1/m3
OUT=eval_outputs/mist_w3po_v1/voicecond_idea1/m3/ood3
M2=eval_outputs/mist_w3po_v1/voicecond_idea1/m2
DEC=eval_outputs/mist_w3po_v1/decode; SPD=eval_outputs/mist_w3po_v1/speech
SUBSETS=${SUBSETS:-"ami en ali"}
mkdir -p $L/ood3 $OUT
freest() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1<=3 {print $2, $1}' | sort -nr | head -1 | awk '{print $2}'; }
args() {
  case $1 in
    ami) D=zh2en; S=test; DR=zh2en;;
    en)  D=en2zh; S=test; DR="";;
    ali) D=zh2en; S=ali;  DR=ali;;
  esac
}
lat() {   # <ss> <run>
  local ss=$1 n=$2 f=$OUT/lat_chunk_$2.json
  args $ss
  [ -e $f ] && return 0
  if [ $D = zh2en ]; then
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $OD/${ss}_ref.jsonl --dr $DR \
      experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $OD/${ss}_chunkwin.jsonl \
      --out $f > $f.log 2>&1
  else
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $OD/${ss}_ref.jsonl --work $f.work \
      --run $DEC/$n --talker $SPD/$n/talker --win $OD/${ss}_chunkwin.jsonl --out $f > $f.log 2>&1
  fi
}
subset() {
  local ss=$1 rc=0 n m pids=""
  args $ss
  local ITEMS=$SPD/idea2_ood3_b_${ss}__t1c/talker_in/items.jsonl VOC=data/open_voicecond_v1/idea1/ood3_vocals/$ss
  if [ -n "${DRY_RUN:-}" ]; then
    log "[dry] $ss: wait $I2L/OOD3_${ss}_DONE; vocals $ITEMS -> $VOC; tokens z/c; orch idea1_ood3_{z,c}_${ss}__t1c --dir $D --set $S"
    return 0
  fi
  # v3: start on Idea 2's B talker inputs + text run (not on its whole subset marker)
  until { [ -s $ITEMS ] && [ $(( $(date +%s) - $(stat -c %Y $ITEMS) )) -ge 120 ] && [ -d $DEC/idea2_ood3_${ss}_txt ]; } \
        || [ -e $I2L/OOD3_${ss}_FAILED ]; do sleep 60; done
  [ -e $I2L/OOD3_${ss}_FAILED ] && { log "$ss: Idea 2 marked FAILED, skipped"; date > $L/OOD3_I1_${ss}_SKIPPED; return 0; }
  log "$ss: Idea 2 B talker inputs ready, starting"
  if [ ! -e $L/ood3/vocals_$ss.DONE ]; then
    c=$(freest)
    NEED_MIB=20000 PYTHONPATH=. bash experiments/mist_w3po_v1/scripts/idea1_gpu_run.sh $c $L/ood3/vocals_$ss.DONE \
      $EPY -m experiments.mist_w3po_v1.voicecond.idea1_bench_tokens vocals_gpu --items $ITEMS --out $VOC > $L/ood3/vocals_$ss.log 2>&1
  fi
  grep -q rc=0 $L/ood3/vocals_$ss.DONE || { log "$ss vocals failed"; date > $L/OOD3_I1_${ss}_FAILED; return 1; }
  for m in z c; do
    mode=$([ $m = z ] && echo e2e || echo ctrl); ck=$M2/$([ $m = z ] && echo zne || echo ctrl)/encoder.pt
    [ -e $OUT/tokens_${m}_$ss.jsonl ] || PYTHONPATH=. OMP_NUM_THREADS=16 $VPY -m experiments.mist_w3po_v1.voicecond.idea1_tokens \
      --mode $mode --ckpt $ck --items $ITEMS --vocals $VOC --out $OUT/tokens_${m}_$ss.jsonl --device cpu > $L/ood3/tokens_${m}_$ss.log 2>&1 || rc=1
  done
  [ $rc = 0 ] || { log "$ss tokens failed"; date > $L/OOD3_I1_${ss}_FAILED; return 1; }
  for m in z c; do
    n=idea1_ood3_${m}_${ss}__t1c; R=$L/ood3/$n
    if [ ! -e $R ]; then
      (   # v3: one orchestrator start per 60 s across all parallel subsets
        flock 8
        [ -e $R ] || { setsid nohup $PY experiments/mist_w3po_v1/voicecond/idea2_ood3_orch.py --subset $ss --voice-override $OUT/tokens_${m}_$ss.jsonl \
          --dir $D --set $S --base $B2 --run $R --name $n --ckpt - --speech-from idea2_ood3_${ss}_txt --talker t1c --window 1 \
          --cap135 --exclude 4 5 6 7 < /dev/null > $R.launch.log 2>&1 8>&- 9>&- &
          log "$n launched"; sleep 60; }
      ) 8> $L/ood3/launch_v3.lock
    fi
  done
  for m in z c; do
    R=$L/ood3/idea1_ood3_${m}_${ss}__t1c
    wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$R failed"; rc=1; }
  done
  [ $rc = 0 ] || { date > $L/OOD3_I1_${ss}_FAILED; return 1; }
  for m in z c; do for s in "" _tts2; do
    n=idea1_ood3_${m}_${ss}__t1c$s
    lat $ss $n || { log "LAAL failed $n"; rc=1; }
    [ -e $OUT/$n.jsonl ] || PYTHONPATH=. $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics measure --talker $SPD/$n/talker \
      --out $OUT/$n.jsonl --procs 32 > $OUT/$n.measure.log 2>&1 || rc=1
  done; done
  # v3: speaker similarity needs Idea 2's slot references and B's finished speech -> wait for its subset marker here
  until [ -e $I2L/OOD3_${ss}_DONE ] || [ -e $I2L/OOD3_${ss}_FAILED ]; do sleep 60; done
  [ -e $I2L/OOD3_${ss}_DONE ] || { log "$ss: Idea 2 marked FAILED after our runs; speaker similarity not run"; date > $L/OOD3_I1_${ss}_FAILED; return 1; }
  local RUNS=""
  for arm in idea2_ood3_b idea1_ood3_z idea1_ood3_c; do for s in "" _tts2; do RUNS="$RUNS ${arm}_${ss}__t1c$s=$SPD/${arm}_${ss}__t1c$s/talker"; done; done
  [ -e $OUT/spksim_$ss.json ] || PYTHONPATH=. OMP_NUM_THREADS=24 $NPY -m experiments.mist_w3po_v1.voicecond.idea1_spksim_runs \
    --ref-dir data/open_voicecond_v1/idea2/spkref/ood3_$ss --key-mode meeting_slot --runs $RUNS --out $OUT/spksim_$ss.json \
    --device cpu > $OUT/spksim_$ss.log 2>&1 || rc=1
  [ $rc = 0 ] && date > $L/OOD3_I1_${ss}_DONE || date > $L/OOD3_I1_${ss}_FAILED
  log "subset $ss rc=$rc"
}
pids=""   # v3: subsets in parallel
for ss in $SUBSETS; do subset $ss & pids="$pids $!"; sleep 5; done
for p in $pids; do wait $p || true; done
[ -n "${DRY_RUN:-}" ] && exit 0
PYTHONPATH=. $EPY -m experiments.mist_w3po_v1.voicecond.idea1_m3_analysis ood3 > $L/ood3_analysis.log 2>&1 && date > $L/OOD3_ANALYSIS_DONE
log "OOD3 done"
