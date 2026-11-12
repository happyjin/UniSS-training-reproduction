#!/usr/bin/env bash
# One dev5 checkpoint proxy of the joint bidirectional run (b) (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md
# §2.3 / §3 / §5), started detached by joint_stage1.sh / joint_stage2.sh for each saved checkpoint; never blocks training.
#   s1: joint/megatron_to_crf_ckpt.py converts the Megatron iteration onto O2 (CPU, = the merged model, tested), then
#   both directions' dev5 text decodes through joint_orch.py (--base O2): finalpost close_detect, wait policy, lag 2000,
#   ctx 0, batch 16, 1.0 s reads, 4 window shards, --cap135, loop guard -> text tcpBLEU all / ov / 1spk + text LAAL
#   (current protocol).  s2: the same on the stage-2 checkpoint (--base = the stage-1 merged model) plus both
#   directions' held-out loss (zh2en sched_dev_v1, en2zh sched_en2zh_dev_v1; tie-break only).
# Placement: the orchestrators exclude the (b) training cards 4-7 (EXCLUDE) and, through the shared exclude file
# (sync_train_exclude.sh), every owned training slot -- so they run on 0-3 only while 0-3 are not training cards, stacking
# within 135 GiB per card by real use (--cap135, shared ledger). An orchestrator that ends with failures is rerun once
# with the same arguments under <name>_r2; still failing -> marker <proxy dir>/s<step>.failed (the rule counts it as "no improvement").
# Then the chunk-protocol text LAAL (recorded only) and the marker <proxy dir>/s<step>.ok ("<utc> <zh2en name> <en2zh name>").
#   STAGE=s1|s2 RUN=<training run id> STEP=<n> BASE=<merged Thinker> SRC=<Megatron iter dir | CRF ckpt dir>
#     [EXCLUDE="4 5 6 7"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/joint_proxy.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
STAGE=${STAGE:?}; RUN=${RUN:?}; STEP=${STEP:?}; BASE=${BASE:?}; SRC=${SRC:?}; EXCLUDE=${EXCLUDE-4 5 6 7}
S5=$(printf %05d $STEP)
PD=$JL/$RUN/proxy
NZ=${STAGE}z_dev5_${RUN}_s$S5
NE=${STAGE}e_dev5_${RUN}_s$S5
case $STAGE in
  s1) CK=$JO/$RUN/crf_ckpt/step_$S5; HZ=""; HE="";;
  s2) CK=$SRC; HZ="--heldout $CRF/sched_dev_v1"; HE="--heldout $CRF/sched_en2zh_dev_v1";;
  *) echo "STAGE s1|s2"; exit 1;;
esac
EXA=""; [ -n "$EXCLUDE" ] && EXA="--exclude $EXCLUDE"
AZ="--dir zh2en --set dev5 --base $BASE $HZ --run $PD/z_s$S5 --name $NZ --ckpt $CK --talker none --shards 4 --shard-by window --window 1 $EXA --cap135"
AE="--dir en2zh --set dev5 --base $BASE $HE --run $PD/e_s$S5 --name $NE --ckpt $CK --talker none --shards 4 --shard-by window --window 1 $EXA --cap135"
if [ -n "${DRY_RUN:-}" ]; then
  [ $STAGE = s1 ] && echo "convert: $PY -m experiments.mist_w3po_v1.joint.megatron_to_crf_ckpt --checkpoint-dir $SRC --base $BASE --rank 32 --alpha 64 --out $CK"
  echo "zh2en: $JORCH $AZ"; $JORCH $AZ --dry | tail -1
  echo "en2zh: $JORCH $AE"; $JORCH $AE --dry | tail -1
  echo "chunk: speech_latency_lit --text-only --win $WZ_DEV5C ; en2zh_lat --text-only --win $WE_DEV5C ; markers $PD/s$S5.ok|failed"
  exit 0
fi
mkdir -p $PD
for m in ok failed; do [ -e $PD/s$S5.$m ] && { log "拒绝覆盖已有输出: $PD/s$S5.$m"; exit 1; }; done
fail() { log "proxy $STAGE step $STEP FAILED: $*"; ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $PD/s$S5.failed ); exit 1; }
if [ $STAGE = s1 ] && [ ! -e $CK/DONE ]; then
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.joint.megatron_to_crf_ckpt --checkpoint-dir $SRC --base $BASE \
    --rank 32 --alpha 64 --out $CK > $PD/convert_s$S5.log 2>&1 || fail "conversion ($(tail -1 $PD/convert_s$S5.log | cut -c1-200))"
  log "converted $SRC -> $CK"
fi
# user / coordinator 2026-11-12: (b)'s proxies use 0-3 only once (a)'s Thinker training there has exited
a_thinker_exited || log "step $STEP: waiting for (a)'s Thinker training to exit before using cards 0-3"
until a_thinker_exited; do sleep 60; done
for x in z e; do
  [ $x = z ] && A=$AZ || A=$AE
  setsid nohup $JORCH $A < /dev/null > $PD/${x}_s$S5.launch.log 2>&1 &
  log "step $STEP $x orchestrator pid $!"
  sleep 90
done
for x in z e; do
  [ $x = z ] && A=$AZ || A=$AE
  R=$PD/${x}_s$S5
  wait_done $R
  if [ -n "$(orch_failed $R)" ]; then
    # same configuration once more: a fresh orchestrator (a --resume keeps the exhausted jobs failed) under <name>_r2
    [ $x = z ] && { N0=$NZ; NZ=${NZ}_r2; } || { N0=$NE; NE=${NE}_r2; }
    A2=$(echo "$A" | sed "s# --run $R # --run ${R}_r2 #; s# --name $N0 # --name ${N0}_r2 #")
    log "step $STEP $x: orchestrator ended with failures ($(tail -1 $R/orch.log | cut -c1-200)); retry: $A2"
    setsid nohup $JORCH $A2 < /dev/null > $PD/${x}_s${S5}_r2.launch.log 2>&1 &
    log "step $STEP $x retry pid $!"
    sleep 60
    wait_done ${R}_r2
    [ -z "$(orch_failed ${R}_r2)" ] || fail "$x orchestrator failed again ($(tail -1 ${R}_r2/orch.log | cut -c1-200))"
  fi
done
mkdir -p $TE/$NZ $TE/$NE
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$NZ --text-only --win $WZ_DEV5C \
  --out $TE/$NZ/textlat_chunk_$NZ.json > $TE/$NZ/textlat_chunk.log 2>&1 || log "chunk text LAAL failed: $NZ (recorded only)"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_DEV --work $TE/$NE/textlat_chunk_$NE.json.work \
  --run $DEC/$NE --text-only --win $WE_DEV5C --out $TE/$NE/textlat_chunk_$NE.json > $TE/$NE/textlat_chunk.log 2>&1 \
  || log "chunk text LAAL failed: $NE (recorded only)"
( set -o noclobber; echo "$(date -u +%FT%TZ) $NZ $NE" > $PD/s$S5.ok )
log "proxy $STAGE step $STEP done: $NZ $NE"
