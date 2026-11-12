#!/usr/bin/env bash
# En->Zh, after the Thinker checkpoint is selected (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §3.3, §4,
# §5): the zh->en A' + PL-EM recipe with only the direction swapped, then the full dev evaluation.
#   1. wait for logs/mist_w3po_v1/en2zh/thinker_selection.json (en2zh_ckpt_proxy.sh); release the Thinker slot once
#      the training exited with rc 0
#   2. three detached orchestrators at once: the dev decode of the selected checkpoint (4996 windows, 8 shards by
#      meeting, --talker none: text score + text LAAL), the train16 decode + talker prep (en2zh_talker_orch traindec),
#      the candidate sampling (en2zh_talker_orch sample: waits for the prep; 4 Chinese CTC scorers)
#   3. dev: read-level merge (merge_read_segments --sep ''; pair_lost <= 2 with a complete output: go on, listed as a
#      known deviation; more: EN2ZH_STOP, ask the coordinator), dev5 cut of the merged decode (hyp + score_run)
#   4. PL-EM rows (talker_rows --mode plem --lang cmn --units char; lattice / lambda / budget as zh->en)
#   5. PL-EM training on TALKER_CARDS (4 cards; slot claim + takeover_train.sh; run_talker_crf.sh MODE=plem: T1c, full
#      talker, 1200 steps, saves every 400) and the exports of iterations 400 / 800 (1200 = the run's own export)
#   6. dev5 speech proxy of each export (one TTS seed, en2zh_orch --set dev5 --seeds 1, PL-EM c1b1s2 flags on the
#      read-level dev5 cut) -> the fixed rule (en2zh_select.py talker) -> talker_selection.json
#   7. full dev speech, two TTS seeds: (c) A'rl + PL-EM = main (the merged decode, the selected export) and (a) A' +
#      T1c c1+b1+s2 per segment = reference; then (CPU) gate_2seed (c) vs (a), the chunk-protocol latencies, talker CER,
#      block pauses (charified copies); (GPU, a card with >= 20 GiB free among CARDS_ALL) UTMOS / SIM, then the paired
#      UTMOS (c) vs (a)  ->  EN2ZH_DEV_DONE
#   RUN=<thinker run id> TALKER_CARDS=0,1,2,3|4,5,6,7 [PORT0=18900] [CARDS_ALL="0 1 2 3 4 5 6 7"] [DRY_RUN=1] \
#     bash experiments/mist_w3po_v1/scripts/en2zh_after_select.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/en2zh/after_select.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=${RUN:?}; TALKER_CARDS=${TALKER_CARDS:?}; PORT0=${PORT0:-18900}; CARDS_ALL=${CARDS_ALL:-0 1 2 3 4 5 6 7}
J=/opt/dlami/nvme/jasonleeeli
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface \
  MODELSCOPE_CACHE=$J/cache/modelscope OMP_NUM_THREADS=8
PY=$J/conda_envs/qwen3omni/bin/python
SC=experiments/mist_w3po_v1/scripts
TR=eval_outputs/mist_w3po_v1/crf/train
E=logs/mist_w3po_v1/en2zh
SL=logs/mist_w3po_v1/slots
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech
TE=eval_outputs/mist_w3po_v1/crf/t_eval
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
D=eval_outputs/mist_w3po_v1/data
WDEV=$D/windows_en2zh_c2g06_v1/windows_dev_online.jsonl
WDEVC=$D/windows_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl
W5=$D/windows_en2zh_c2g06_v1_dev5/windows_dev_online.jsonl
W5C=$D/windows_en2zh_c2g06_chunk_v1_dev5/windows_dev_online.jsonl
WTRAIN=$D/windows_en2zh_c2g06_train16_v1/windows_train_online.jsonl
REF=data/ami_multispk_si_v1/e2e_gstar_v1/windows_validation_ref.jsonl
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
SEL=$E/thinker_selection.json
TSEL=$E/talker_selection.json
L=logs/mist_w3po_v1/$RUN.log
case $TALKER_CARDS in 0,1,2,3) TSLOT=slotA;; 4,5,6,7) TSLOT=slotB;; *) echo "TALKER_CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
orch_failed() { tail -3 $1/orch.log 2>/dev/null | grep -q "DONE; failed/skipped \[\]" && echo "" || echo "failed"; }
ORCH="$PY $SC/en2zh_orch.py"
TORCH="$PY $SC/en2zh_talker_orch.py"
busy_on() {   # compute processes on the given cards that are not our placeholders
  local cards=$1 map pid bus idx cmd
  map=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null) || { echo "smi-failed"; return; }
  nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | while IFS=', ' read -r pid bus; do
    idx=$(echo "$map" | awk -F', ' -v b="$bus" '$2 == b {print $1}')
    case ",$cards," in *",$idx,"*) ;; *) continue;; esac
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || cmd="(unreadable)"
    case "$cmd" in *scripts/gpu_load/train_mist_workload.py*|*gpu_keeper.py*) ;; *) echo "$idx:$pid";; esac
  done
}
free_card() {   # a card of CARDS_ALL (not a training slot's) with >= 20 GiB free, the freest; empty if none
  local ex; ex=$(cat logs/mist_w3po_v1/talker_crf_v1/exclude_cards 2>/dev/null)
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
    awk -F', ' -v all=" $CARDS_ALL " -v ex=" $ex " 'index(all, " "$1" ") && !index(ex, " "$1" ") && $2 - $3 >= 20480 {print $2 - $3, $1}' | \
    sort -rn | head -1 | awk '{print $2}'
}
merge_ok() {   # merge_read_segments' json: complete output and pair_lost <= 2
  $PY - "$1" "$2" <<'EOF'
import json, sys
m = json.load(open(f"{sys.argv[1]}/merge_read_segments.json"))
n = sum(1 for _ in open(sys.argv[2]))
ok = m["windows"] == n and m["pair_lost"] <= 2
print(json.dumps({"windows": m["windows"], "expected": n, "pair_lost": m["pair_lost"], "go": ok}))
sys.exit(0 if ok else 1)
EOF
}
if [ -n "${DRY_RUN:-}" ]; then
  S5=01500; CK=$TR/$RUN/ckpt/step_$S5; F=dev_${RUN}_s$S5; TN=train16_${RUN}_s$S5; X=eval_outputs/mist_w3po_v1/talker_crf/talker_align_20261111T001542Z/export
  echo "(selection read from $SEL; DRY uses step $S5)"
  $ORCH --set dev --run /nonexistent/dev --name $F --ckpt $CK --talker none --window 1 --cap135 --dry | tail -1
  $TORCH --stage traindec --run /nonexistent/traindec --name $TN --ckpt $CK --dry | tail -1
  $TORCH --stage sample --run /nonexistent/sample --name $TN --ckpt $CK --traindec-run $E/traindec --scorer-port0 $PORT0 --dry | tail -1
  echo "rows: $PY -m experiments.mist_w3po_v1.talker.talker_rows --mode plem --lang cmn --units char --cands '$SPD/$TN/talker_cands/cands_*.jsonl' --win $WTRAIN --out eval_outputs/mist_w3po_v1/talker_crf/rows_en2zh/plem_$TN.pt"
  DRY_RUN=1 RUN_ID=talker_plem_en2zh_DRY MODE=plem ROWS=x.pt CARDS=$TALKER_CARDS MASTER_PORT=29823 bash experiments/mist_w3po_v1/scripts/run_talker_crf.sh | tail -1
  echo "(dev5 proxies / full dev chains below use an existing export only to check the commands: $X)"
  $ORCH --set dev5 --run /nonexistent/p5 --name dev5cut_${F}_rl__plem_i0400 --ckpt - --speech-from dev5cut_${F}_rl --talker plem --plem-export $X --window 1 --seeds 1 --cap135 --scorer-port0 $((PORT0 + 10)) --dry | tail -1
  $ORCH --set dev --run /nonexistent/c --name ${F}__rlplem --ckpt - --speech-from ${F}_rl --talker plem --plem-export $X --window 1 --cap135 --scorer-port0 $((PORT0 + 40)) --dry | tail -1
  $ORCH --set dev --run /nonexistent/a --name ${F}__seg --ckpt - --speech-from $F --talker c1b1s2 --window 1 --cap135 --scorer-port0 $((PORT0 + 50)) --dry | tail -1
  echo "free card now: $(free_card)"
  exit 0
fi
mkdir -p $E
[ -e $TSEL ] && { log "拒绝覆盖已有输出: $TSEL"; exit 1; }
# ---------------------------------------------------------------------------------------------- 1. selection, slot
log "waiting for $SEL"
until [ -e $SEL ]; do sleep 30; done
STEP=$($PY -c "import json; print(json.load(open('$SEL'))['chosen'])")
S5=$(printf %05d $STEP); CK=$TR/$RUN/ckpt/step_$S5
F=dev_${RUN}_s$S5; TN=train16_${RUN}_s$S5
log "selected Thinker checkpoint $CK"
until [ -e $L ] && grep -q CRF_RUN_EXIT $L; do sleep 30; done
if grep -q 'CRF_RUN_EXIT rc=0' $L; then
  for s in slotA slotB; do
    [ -e $SL/$s.owner ] && grep -q "$RUN" $SL/$s.owner && command mv -f $SL/$s.owner $SL/$s.released_$(date -u +%Y%m%dT%H%M%SZ) \
      && log "$s released (training $RUN exited rc 0)"
  done
else
  log "training $RUN exit: $(grep CRF_RUN_EXIT $L | tail -1); slot kept for inspection"
fi
# ---------------------------------------------------------------------------------------------- 2. three orchestrators
C1="$ORCH --set dev --run $E/dev --name $F --ckpt $CK --talker none --window 1 --cap135"
setsid nohup $C1 < /dev/null > $E/dev.launch.log 2>&1 &
log "dev decode orchestrator pid $!"
sleep 90
C2="$TORCH --stage traindec --run $E/traindec --name $TN --ckpt $CK"
setsid nohup $C2 < /dev/null > $E/traindec.launch.log 2>&1 &
log "train16 decode orchestrator pid $!"
C3="$TORCH --stage sample --run $E/sample --name $TN --ckpt $CK --traindec-run $E/traindec --scorer-port0 $PORT0"
setsid nohup $C3 < /dev/null > $E/sample.launch.log 2>&1 &
log "sampling orchestrator pid $!"
# ---------------------------------------------------------------------------------------------- 3. dev merge, dev5 cut
wait_done $E/dev
[ -z "$(orch_failed $E/dev)" ] || { log "EN2ZH_STOP: dev decode orchestrator ended with failures ($(tail -1 $E/dev/orch.log))"; exit 1; }
$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl --sep '' > $E/dev_merge.log 2>&1
merge_ok $DEC/${F}_rl $WDEV >> $E/dev_merge.log 2>&1 || { log "EN2ZH_STOP: dev merge ($(tail -1 $E/dev_merge.log)); ask the coordinator"; exit 1; }
log "dev merge: $(tail -1 $E/dev_merge.log)"
C5=$DEC/dev5cut_${F}_rl
mkdir -p $C5
$PY - $W5 $DEC/${F}_rl/hyp.jsonl $C5/hyp.jsonl <<'EOF'
import json, sys
ids = {json.loads(l)["id"] for l in open(sys.argv[1])}
rows = [l for l in open(sys.argv[2]) if json.loads(l)["id"] in ids]
assert len(rows) == len(ids), (len(rows), len(ids))
open(sys.argv[3], "x").writelines(rows)
EOF
$PY -m experiments.mist_w3po_v1.evaluation.score_run --run $C5 --windows $W5 --ref-windows $REF --set en2zh > $C5/score.log 2>&1 \
  || { log "EN2ZH_STOP: dev5 cut score failed"; exit 1; }
$PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $TE/$F/textlat_chunk_$F.json.work --run $DEC/$F \
  --text-only --win $WDEVC --out $TE/$F/textlat_chunk_$F.json > $E/dev_textlat_chunk.log 2>&1 || log "dev chunk text LAAL failed (recorded)"
# ---------------------------------------------------------------------------------------------- 4. rows
wait_done $E/sample
[ -z "$(orch_failed $E/sample)" ] || { log "EN2ZH_STOP: sampling ended with failures ($(tail -1 $E/sample/orch.log))"; exit 1; }
ROWS=eval_outputs/mist_w3po_v1/talker_crf/rows_en2zh/plem_$TN.pt
mkdir -p $(dirname $ROWS)
$PY -m experiments.mist_w3po_v1.talker.talker_rows --mode plem --lang cmn --units char --cands "$SPD/$TN/talker_cands/cands_*.jsonl" \
  --win $WTRAIN --out $ROWS > $E/rows.log 2>&1 || { log "EN2ZH_STOP: rows failed ($(tail -2 $E/rows.log | tr '\n' ' '))"; exit 1; }
log "PL-EM rows: $(tail -1 $E/rows.log | cut -c1-300)"
# ---------------------------------------------------------------------------------------------- 5. PL-EM training
RID=talker_plem_en2zh_$(date -u +%Y%m%dT%H%M%SZ)
log "waiting for $TSLOT ($TALKER_CARDS): no owner"
until [ ! -e $SL/$TSLOT.owner ]; do sleep 30; done
( set -o noclobber; printf 'who: En->Zh preparation agent (mist_w3po_v1 en2zh PL-EM)\ntask: En->Zh PL-EM talker training %s on cards %s, ~30 min; released by renaming to %s.released_<UTC>\nutc: %s\n' \
  "$RID" "$TALKER_CARDS" "$TSLOT" "$(date -u +%FT%TZ)" > $SL/$TSLOT.owner ) && log "$TSLOT.owner claimed" \
  || { log "EN2ZH_STOP: $TSLOT.owner claim failed"; exit 1; }
t0=$(date +%s)
until [ -z "$(busy_on $TALKER_CARDS)" ]; do
  [ $(( ($(date +%s) - t0) % 1800 )) -lt 30 ] && [ $(( $(date +%s) - t0 )) -gt 60 ] && log "still waiting: $TALKER_CARDS busy ($(busy_on $TALKER_CARDS | tr '\n' ' '))"
  sleep 30
done
TL=$E/train_talker_$RID.log
CARDS=$TALKER_CARDS LAUNCH="setsid nohup env RUN_ID=$RID MODE=plem ROWS=$ROWS CARDS=$TALKER_CARDS MASTER_PORT=29823 bash experiments/mist_w3po_v1/scripts/run_talker_crf.sh > $TL 2>&1 < /dev/null &" \
  bash experiments/mist_w3po_v1/scripts/takeover_train.sh
log "PL-EM training $RID launched on $TALKER_CARDS (log $TL)"
until [ -e $TL ] && grep -qE "TALKER_CRF_DONE|TALKER_CRF_FAILED|TALKER_CRF_EXPORT_FAILED" $TL; do sleep 30; done
command mv -f $SL/$TSLOT.owner $SL/$TSLOT.released_$(date -u +%Y%m%dT%H%M%SZ) && log "$TSLOT released"
grep -q TALKER_CRF_DONE $TL || { log "EN2ZH_STOP: PL-EM training: $(grep -E 'TALKER_CRF_' $TL | tail -1)"; exit 1; }
EXP=eval_outputs/mist_w3po_v1/talker_crf/$RID
for it in 400 800; do
  CUDA_VISIBLE_DEVICES="" $J/conda_envs/qwen3omni/bin/python experiments/uniss_qwen3_omni_v1/training/export_e2.py \
    --checkpoint-dir checkpoints/mist_w3po_v1/$RID/iter_$(printf %07d $it) --base $T1C --output $EXP/export_i$(printf %04d $it) \
    > $E/export_i$it.log 2>&1 || { log "EN2ZH_STOP: export of iteration $it failed"; exit 1; }
done
log "PL-EM exports: $EXP/export_i0400 $EXP/export_i0800 $EXP/export (iteration 1200)"
# ---------------------------------------------------------------------------------------------- 6. dev5 speech proxies
declare -A XP=([400]=$EXP/export_i0400 [800]=$EXP/export_i0800 [1200]=$EXP/export)
k=0
for it in 400 800 1200; do
  k=$((k + 1))
  n=dev5cut_${F}_rl__plem_i$(printf %04d $it)
  C="$ORCH --set dev5 --run $E/plem5_i$(printf %04d $it) --name $n --ckpt - --speech-from dev5cut_${F}_rl --talker plem \
--plem-export ${XP[$it]} --window 1 --seeds 1 --cap135 --scorer-port0 $((PORT0 + 10 * k))"
  setsid nohup $C < /dev/null > $E/plem5_i$(printf %04d $it).launch.log 2>&1 &
  log "dev5 speech proxy $n: orchestrator pid $!"
  sleep 120
done
CANDS=""
for it in 400 800 1200; do
  i4=$(printf %04d $it); n=dev5cut_${F}_rl__plem_i$i4
  wait_done $E/plem5_i$i4
  $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $TE/$n/lat_chunk_$n.json.work --run $DEC/$n \
    --talker $SPD/$n/talker --win $W5C --out $TE/$n/lat_chunk_$n.json > $TE/$n/lat_chunk.log 2>&1 || log "chunk LAAL failed: $n (recorded)"
  CANDS="$CANDS --cand $it $n"
done
$PY $SC/en2zh_select.py talker $CANDS --out $TSEL > $E/talker_selection.log 2>&1 || { log "EN2ZH_STOP: talker selection ($(tail -1 $E/talker_selection.log | cut -c1-300))"; exit 1; }
TIT=$($PY -c "import json; print(json.load(open('$TSEL'))['chosen'])")
TX=${XP[$TIT]}
log "EN2ZH_TALKER_SELECTED iteration $TIT ($TX)"
# ---------------------------------------------------------------------------------------------- 7. full dev speech
CC="$ORCH --set dev --run $E/devrlplem --name ${F}__rlplem --ckpt - --speech-from ${F}_rl --talker plem --plem-export $TX \
--window 1 --cap135 --scorer-port0 $((PORT0 + 40))"
setsid nohup $CC < /dev/null > $E/devrlplem.launch.log 2>&1 &
log "(c) A'rl + PL-EM orchestrator pid $!"
sleep 120
CA="$ORCH --set dev --run $E/devseg --name ${F}__seg --ckpt - --speech-from $F --talker c1b1s2 --window 1 --cap135 \
--scorer-port0 $((PORT0 + 50))"
setsid nohup $CA < /dev/null > $E/devseg.launch.log 2>&1 &
log "(a) A' + T1c c1b1s2 orchestrator pid $!"
wait_done $E/devrlplem
wait_done $E/devseg
O=$TE/${F}__rlplem
$PY -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand $DEC/${F}__rlplem $DEC/${F}__rlplem_tts2 \
  --ref $DEC/${F}__seg $DEC/${F}__seg_tts2 --out $O/gate_c_vs_a.json > $O/gate_c_vs_a.log 2>&1 || log "gate failed"
for nm in ${F}__rlplem ${F}__rlplem_tts2 ${F}__seg ${F}__seg_tts2; do
  base=${nm%_tts2}; Od=$TE/$base
  $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $Od/lat_chunk_$nm.json.work --run $DEC/$nm \
    --talker $SPD/$nm/talker --win $WDEVC --out $Od/lat_chunk_$nm.json > $Od/lat_chunk_$nm.log 2>&1 || log "chunk LAAL failed: $nm"
  $PY -m experiments.mist_w3po_v1.evaluation.talker_cer_zh --talker $SPD/$nm/talker --out $Od/cer_$nm.json > /dev/null 2>&1 \
    || log "talker CER failed: $nm"
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses --talker $Od/lat_$nm.json.work/talker \
    > $Od/pauses_$nm.jsonl 2> $Od/pauses_$nm.log || log "block pauses failed: $nm"
  c=$(free_card)
  if [ -n "$c" ]; then
    mkdir -p $Q/$nm
    CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$nm/talker \
      --items $SPD/$nm/talker_in/items.jsonl --out $Q/$nm/quality.json > $Od/quality_$nm.log 2>&1 || log "UTMOS / SIM failed: $nm"
  else
    log "no card with 20 GiB free for UTMOS / SIM of $nm: left for later"
  fi
done
$PY -m experiments.mist_w3po_v1.evaluation.listen_paired_meeting --cand $Q/${F}__rlplem $Q/${F}__rlplem_tts2 \
  --ref a $Q/${F}__seg $Q/${F}__seg_tts2 > $O/utmos_paired_c_vs_a.json 2> $O/utmos_paired.log || log "paired UTMOS failed"
( set -o noclobber; echo "thinker $CK talker $TX utc $(date -u +%FT%TZ)" > $E/EN2ZH_DEV_DONE ) || log "marker exists"
log "EN2ZH_DEV_DONE thinker step $STEP talker iteration $TIT"
