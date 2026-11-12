#!/usr/bin/env bash
# En->Zh TEST, run once (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §6): the AMI En->Zh test set (16
# meetings, original English ihm_mix + nemotron_lowlat, C2 gap-0.6 windows 5126, references = the v7 / v8 En->Zh test
# reference windows e2e_gstar_v1/windows_test_ref.jsonl), with the configuration fixed on dev: the selected Thinker
# checkpoint (thinker_selection.json) and the selected PL-EM export (talker_selection.json). Waits for
# logs/mist_w3po_v1/en2zh/EN2ZH_DEV_DONE (the full dev evaluation). Nothing is chosen on test.
#   1. test decode (8 shards by meeting, the dev decode flags) -> text tcpBLEU all / ov / 1spk + text LAAL
#   2. read-level merge (--sep ''; pair_lost <= 2 with a complete output: go on, a known deviation; more: EN2ZH_STOP)
#   3. two TTS seeds each: (c) A'rl + PL-EM (main) and (a) A' + T1c c1+b1+s2 per segment (reference)
#   4. (CPU) gate_2seed (c) vs (a), chunk-protocol latencies (speech both systems / seeds, text), talker CER, block
#      pauses; (GPU, a card with >= 20 GiB free among CARDS_ALL) UTMOS / SIM, paired UTMOS (c) vs (a) -> EN2ZH_TEST_DONE
#   RUN=<thinker run id> [PORT0=18900] [CARDS_ALL="0 1 2 3 4 5 6 7"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/en2zh_test.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/en2zh/test.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=${RUN:?}; PORT0=${PORT0:-18900}; CARDS_ALL=${CARDS_ALL:-0 1 2 3 4 5 6 7}
J=/opt/dlami/nvme/jasonleeeli
export PYTHONPATH=.:third_party/Megatron-LM TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface \
  MODELSCOPE_CACHE=$J/cache/modelscope OMP_NUM_THREADS=8
PY=$J/conda_envs/qwen3omni/bin/python
SC=experiments/mist_w3po_v1/scripts
TR=eval_outputs/mist_w3po_v1/crf/train
E=logs/mist_w3po_v1/en2zh
DEC=eval_outputs/mist_w3po_v1/decode
SPD=eval_outputs/mist_w3po_v1/speech
TE=eval_outputs/mist_w3po_v1/crf/t_eval
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
D=eval_outputs/mist_w3po_v1/data
WT=$D/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl
WTC=$D/windows_test_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl
REF=data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl
SEL=$E/thinker_selection.json
TSEL=$E/talker_selection.json
ORCH="$PY $SC/en2zh_orch.py"
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
orch_failed() { tail -3 $1/orch.log 2>/dev/null | grep -q "DONE; failed/skipped \[\]" && echo "" || echo "failed"; }
free_card() {
  local ex; ex=$(cat logs/mist_w3po_v1/talker_crf_v1/exclude_cards 2>/dev/null)
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
    awk -F', ' -v all=" $CARDS_ALL " -v ex=" $ex " 'index(all, " "$1" ") && !index(ex, " "$1" ") && $2 - $3 >= 20480 {print $2 - $3, $1}' | \
    sort -rn | head -1 | awk '{print $2}'
}
if [ -n "${DRY_RUN:-}" ]; then
  S5=01500; CK=$TR/$RUN/ckpt/step_$S5; FT=ft_en2zh_${RUN}_s$S5; X=eval_outputs/mist_w3po_v1/talker_crf/talker_align_20261111T001542Z/export
  echo "(selections read from $SEL / $TSEL; DRY uses step $S5 and an existing export only to check the commands)"
  $ORCH --set test --run /nonexistent/ft --name $FT --ckpt $CK --talker none --window 1 --cap135 --dry | tail -1
  $ORCH --set test --run /nonexistent/c --name ${FT}__rlplem --ckpt - --speech-from ${FT}_rl --talker plem --plem-export $X --window 1 --cap135 --scorer-port0 $((PORT0 + 60)) --dry | tail -1
  $ORCH --set test --run /nonexistent/a --name ${FT}__seg --ckpt - --speech-from $FT --talker c1b1s2 --window 1 --cap135 --scorer-port0 $((PORT0 + 70)) --dry | tail -1
  exit 0
fi
log "waiting for $E/EN2ZH_DEV_DONE"
until [ -e $E/EN2ZH_DEV_DONE ] && [ -e $SEL ] && [ -e $TSEL ]; do sleep 60; done
STEP=$($PY -c "import json; print(json.load(open('$SEL'))['chosen'])")
S5=$(printf %05d $STEP); CK=$TR/$RUN/ckpt/step_$S5
TX=$(sed -n 's/^thinker .* talker \(.*\) utc .*/\1/p' $E/EN2ZH_DEV_DONE)
[ -e "$TX/uniss/model.safetensors" ] || { log "EN2ZH_STOP: selected talker export not found ($TX)"; exit 1; }
FT=ft_en2zh_${RUN}_s$S5
for d in $DEC/$FT $DEC/${FT}_rl $E/ft $E/ftrlplem $E/ftseg; do [ -e $d ] && { log "拒绝覆盖已有输出: $d (the test runs once)"; exit 1; }; done
log "TEST once: Thinker $CK, talker $TX"
setsid nohup $ORCH --set test --run $E/ft --name $FT --ckpt $CK --talker none --window 1 --cap135 < /dev/null > $E/ft.launch.log 2>&1 &
log "test decode orchestrator pid $!"
wait_done $E/ft
[ -z "$(orch_failed $E/ft)" ] || { log "EN2ZH_STOP: test decode ended with failures ($(tail -1 $E/ft/orch.log))"; exit 1; }
$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $FT --dst ${FT}_rl --sep '' > $E/ft_merge.log 2>&1
$PY - $DEC/${FT}_rl $WT >> $E/ft_merge.log 2>&1 <<'EOF' || { log "EN2ZH_STOP: test merge ($(tail -1 $E/ft_merge.log)); ask the coordinator"; exit 1; }
import json, sys
m = json.load(open(f"{sys.argv[1]}/merge_read_segments.json"))
n = sum(1 for _ in open(sys.argv[2]))
ok = m["windows"] == n and m["pair_lost"] <= 2
print(json.dumps({"windows": m["windows"], "expected": n, "pair_lost": m["pair_lost"], "go": ok}))
sys.exit(0 if ok else 1)
EOF
log "test merge: $(tail -1 $E/ft_merge.log)"
setsid nohup $ORCH --set test --run $E/ftrlplem --name ${FT}__rlplem --ckpt - --speech-from ${FT}_rl --talker plem \
  --plem-export $TX --window 1 --cap135 --scorer-port0 $((PORT0 + 60)) < /dev/null > $E/ftrlplem.launch.log 2>&1 &
log "(c) test orchestrator pid $!"
sleep 120
setsid nohup $ORCH --set test --run $E/ftseg --name ${FT}__seg --ckpt - --speech-from $FT --talker c1b1s2 --window 1 \
  --cap135 --scorer-port0 $((PORT0 + 70)) < /dev/null > $E/ftseg.launch.log 2>&1 &
log "(a) test orchestrator pid $!"
wait_done $E/ftrlplem
wait_done $E/ftseg
O=$TE/${FT}__rlplem
mkdir -p $O $TE/$FT
$PY -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand $DEC/${FT}__rlplem $DEC/${FT}__rlplem_tts2 \
  --ref $DEC/${FT}__seg $DEC/${FT}__seg_tts2 --out $O/gate_c_vs_a.json > $O/gate_c_vs_a.log 2>&1 || log "gate failed"
$PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $TE/$FT/textlat_chunk_$FT.json.work --run $DEC/$FT \
  --text-only --win $WTC --out $TE/$FT/textlat_chunk_$FT.json > $TE/$FT/textlat_chunk.log 2>&1 || log "test chunk text LAAL failed"
for nm in ${FT}__rlplem ${FT}__rlplem_tts2 ${FT}__seg ${FT}__seg_tts2; do
  base=${nm%_tts2}; Od=$TE/$base
  $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $Od/lat_chunk_$nm.json.work --run $DEC/$nm \
    --talker $SPD/$nm/talker --win $WTC --out $Od/lat_chunk_$nm.json > $Od/lat_chunk_$nm.log 2>&1 || log "chunk LAAL failed: $nm"
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
$PY -m experiments.mist_w3po_v1.evaluation.listen_paired_meeting --cand $Q/${FT}__rlplem $Q/${FT}__rlplem_tts2 \
  --ref a $Q/${FT}__seg $Q/${FT}__seg_tts2 > $O/utmos_paired_c_vs_a.json 2> $O/utmos_paired.log || log "paired UTMOS failed"
log "EN2ZH_TEST_DONE"
