#!/usr/bin/env bash
# Idea 2 prereg v3 §C (OOD-3 no-harm): per subset (ami = AMI zh->en Chinese-dubbed, en = AMI en->zh, ali = AliMeeting
# zh->en; 2 meetings each), the B arm (no override) and the shrink+Vfb arm (--voice-override tokens/ood3_<ss>.jsonl)
# through the identical subset chain (voicecond/idea2_ood3_orch.py: joint_orch with subset windows / refs, --speech-from
# the filtered baseline text idea2_ood3_<ss>_txt, --talker t1c, both TTS seeds, --exclude 0 1 2 3). Waits for the token
# table and for GATE (default: the esports full-fit Talker marker) so the GPUs are not oversubscribed. Afterwards (CPU):
# chunk-protocol speech LAAL (as joint_test.sh lat()), idea1_speech_metrics measure + compare (B -> sv per seed), TitaNet
# references + speaker similarity (idea2_spksim_generic.py). Markers logs/mist_w3po_v1/voicecond/idea2/ood3/OOD3_<ss>_{DONE,FAILED}.
#   [SUBSETS="ami en ali"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_ood3_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/ood3_run.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
VPY=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
NPY=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
EPY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
TOK=eval_outputs/mist_w3po_v1/voicecond_idea2/tokens
OD=data/open_voicecond_v1/idea2/ood3
GL=logs/mist_w3po_v1/voicecond/idea2/ood3
OUT=eval_outputs/mist_w3po_v1/voicecond_idea2/ood3
GATE=${GATE:-logs/mist_w3po_v1/voicecond/idea2/GAME_VFBF_TALKER_DONE}
SUBSETS=${SUBSETS:-"ami en ali"}
mkdir -p $GL $OUT
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech

args() {   # <ss> -> D S DATA SESS DIAR MEET DR
  case $1 in
    ami) D=zh2en; S=test; DATA=data/ami_multispk_si_v1; SESS=zh_mix_v8; DIAR=nemotron_lowlat_zh8; MEET="EN2002d IS1009c"; DR=zh2en;;
    en)  D=en2zh; S=test; DATA=data/ami_multispk_si_v1; SESS=ihm_mix; DIAR=nemotron_lowlat; MEET="EN2002d IS1009c"; DR="";;
    ali) D=zh2en; S=ali;  DATA=data/alimeeting_si_v1; SESS=near_mix; DIAR=nemotron_lowlat; MEET="R8009_M8021 R8009_M8024"; DR=ali;;
  esac
}
orch() {   # <ss> <arm b|sv> -> waits; returns 0 on success
  local ss=$1 arm=$2 N R vo=""
  args $ss
  N=idea2_ood3_${arm}_${ss}__t1c; R=$GL/$N
  [ $arm = sv ] && vo="--voice-override $TOK/ood3_${ss}.jsonl"
  local CMD="$PY experiments/mist_w3po_v1/voicecond/idea2_ood3_orch.py --subset $ss $vo --dir $D --set $S --base $B2 --run $R \
    --name $N --ckpt - --speech-from idea2_ood3_${ss}_txt --talker t1c --window 1 --cap135 --exclude 0 1 2 3"
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] $CMD"; return 0; fi
  if [ ! -e $R ]; then setsid nohup $CMD < /dev/null > $R.launch.log 2>&1 9>&- & log "$N orchestrator pid $!"; fi
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$N failed"; return 1; }
}
lat() {   # <ss> <run name>
  local ss=$1 n=$2 f
  args $ss
  f=$OUT/lat_chunk_$n.json
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
subset() {   # <ss>
  local ss=$1 rc=0 n
  args $ss
  if [ -z "${DRY_RUN:-}" ]; then
    until [ -s $TOK/ood3_${ss}.jsonl ]; do sleep 30; done
    until [ -e $GATE ]; do sleep 60; done
  fi
  orch $ss b & local p1=$!
  sleep 60
  orch $ss sv & local p2=$!
  wait $p1 || rc=1; wait $p2 || rc=1
  [ -n "${DRY_RUN:-}" ] && return 0
  [ $rc = 0 ] || { date -u +%FT%TZ >> $GL/OOD3_${ss}_FAILED; return 1; }
  for arm in b sv; do for sfx in "" _tts2; do
    n=idea2_ood3_${arm}_${ss}__t1c$sfx
    lat $ss $n || { log "chunk LAAL failed $n"; rc=1; }
    [ -e $OUT/$n.jsonl ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics measure --talker $SPD/$n/talker \
      --out $OUT/$n.jsonl --procs 48 > $OUT/$n.measure.log 2>&1 || rc=1
  done; done
  [ -e $OUT/compare_${ss}.json ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics compare \
    --pair $OUT/idea2_ood3_b_${ss}__t1c.jsonl:$OUT/idea2_ood3_sv_${ss}__t1c.jsonl:${ss}_s1 \
           $OUT/idea2_ood3_b_${ss}__t1c_tts2.jsonl:$OUT/idea2_ood3_sv_${ss}__t1c_tts2.jsonl:${ss}_s2 \
    --primary ${ss}_s1 ${ss}_s2 --out $OUT/compare_${ss}.json > $OUT/compare_${ss}.log 2>&1 || rc=1
  CUDA_VISIBLE_DEVICES=5 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_spksim_generic.py refs --data $DATA --session $SESS \
    --diar $DIAR --tag ood3_$ss --meetings $MEET --device cuda:0 > $OUT/spkref_${ss}.log 2>&1 || rc=1
  CUDA_VISIBLE_DEVICES=5 $NPY experiments/mist_w3po_v1/voicecond/idea2_spksim_generic.py sim --tag ood3_$ss --device cuda:0 \
    --runs idea2_ood3_b_${ss}__t1c idea2_ood3_b_${ss}__t1c_tts2 idea2_ood3_sv_${ss}__t1c idea2_ood3_sv_${ss}__t1c_tts2 \
    > $OUT/spksim_${ss}.log 2>&1 || rc=1
  [ $rc = 0 ] && date -u +%FT%TZ >> $GL/OOD3_${ss}_DONE || date -u +%FT%TZ >> $GL/OOD3_${ss}_FAILED
  log "subset $ss rc=$rc"
}
for ss in $SUBSETS; do subset $ss || true; done
log OOD3_RUN_EXIT
