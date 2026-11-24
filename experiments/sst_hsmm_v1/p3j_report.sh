#!/usr/bin/env bash
# P3 for P7' (prereg change 15): copy of p3i_report.sh with P1 = p7p, P2 = p2j, output p3j; runs regardless of the dev verdict.
# P3 for P7 (prereg change 14): copy of p3h_report.sh + SST_GUARD=1, P2 = p2i, output p3i; runs regardless of the dev verdict.
# P3 for P6b (prereg change 13): copy of p3g_report.sh with P1 = p6b, P2 = p2h, output p3h; runs regardless of the dev verdict.
# REPORT-ONLY-IF-NO-GO test (prereg change 12, user request 2026-11-23 22:20 CET): p3g_orch.sh without the GO check (verdict only logged), Thinker --gpu-mem 0.6 after p2g's Thinker stage.
# P3 for P6 (prereg change 11): copy of p3f_orch.sh with P1 = p6, SST_CTRL = p6/control.json, gate p2g, output p3g.
# P3'''' (prereg change 9): copy of p3e_orch.sh with P1 = p5b (P5'), gate p2g, output p3g. Header below refers to the original.
# sst_hsmm_v1 P3''' test (prereg change 8): copy of p3c_orch.sh with the P5 Thinker (p5/merged + p5/train/gate_final.pt, threshold rule, SST_BLOCK_S=0.2), gate p2g GO, output p3g, p3_score --root p3j --rules pareto. Was P3'' (P4 Thinker, p2c, p3c); Was: P3 test (3.6 + change 2), cards 0-7. Built from voice_onset_v1/p3_orch.sh;
# only the Thinker blocks change (HSMM e2e Thinker instead of UE LA2). Refuses to start unless p2/summary.json says GO.
#  1. e2e_si_thinker on test_robust UE_abs items (5 main cells), 4 shards per direction, 8 cards -> p3/thinker_<d>
#  2. V1 tokens on NEW blocks (wave5_m_v1tokens, cards 0 / 1) || decide.py (CPU; j = 0 Kalman targets, onset = first write)
#  3. p3_search (posterior-MBR AbS), 8 shards on cards 0-7 -> p3/tokens/tokens_new_<d>.jsonl
#  4. Talker T1c (voice override, block, seed 20261021), 8 jobs  5. speech-only ASR, TitaNet, ECAPA2, PESQ
#  6. dip player on NEW, N (UE_mbr/mbr) and B_b (B/b), CPU  7. p3_score -> p3/summary.json, DONE
# Placeholders stopped by PID per card (lib.sh launch); watcher noplaceholder.sh. Refuses existing outputs.
#   setsid nohup bash experiments/sst_hsmm_v1/p3j_report.sh > logs/sst_hsmm_v1/p3j_report.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
P1=eval_outputs/sst_hsmm_v1/p7p
P2=eval_outputs/sst_hsmm_v1/p2j
R=eval_outputs/sst_hsmm_v1/p3j
T=eval_outputs/mist_w3po_v1/test_robust
MODEL=eval_outputs/voice_onset_v1/p1/train/model.pt
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
LOGD=logs/sst_hsmm_v1/p3j
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
export S1_DATA=/opt/dlami/nvme/jasonleeeli/projects/UniSS/$T/data
source experiments/sst_hsmm_v1/lib.sh
NENV=(env PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface S1_DATA=$S1_DATA)
SENV=(env PYTHONPATH=.:.tmp/b_abs_pyshim OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
declare -A TOKMS=([en2zh]=345.6 [zh2en]=332.5)
NS=8

if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R (refuse to overwrite)" >&2; exit 1; }
  # change 12: test runs regardless of the dev verdict (reported, not used to select)
  [ -s $MODEL ] && [ -s $P1/train/gate_final.pt ] && [ -e .tmp/b_abs_pyshim/einx/__init__.py ] || { echo "inputs missing" >&2; exit 1; }
  mkdir -p $LOGD $R/data $R/metrics $R/tokens $R/play
  git rev-parse HEAD > $R/commit.txt
  start_watcher
  for d in en2zh zh2en; do
    $PYQ -I - $T/UE_abs/data/items_$d.jsonl $R/data $d <<'PY' || die "split $d"
import sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows); b = [round(k * n / 4) for k in range(5)]
for k in range(4):
    with open(f"{sys.argv[2]}/items_{sys.argv[3]}_{k}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(rows[b[k]:b[k + 1]])
print(sys.argv[3], n, b)
PY
  done
fi

# change 12: wait until p2g has finished its Thinker stage (one vLLM Thinker per card at a time)
if [ -z "$DRY" ]; then
  log "waiting for p2g thinker stage"
  until grep -q "thinker done" logs/sst_hsmm_v1/p2j_orch.log 2>/dev/null; do [ -e $P2/FAILED ] || [ -e $P1/FAILED ] && die "p2j FAILED"; sleep 30; done
fi
# 1. HSMM Thinker, 8 shards
ps_=(); g=0
for d in en2zh zh2en; do for k in 0 1 2 3; do
  ps_+=("$(launch $g thinker_${d}_$k "${VENV[@]}" CUDA_VISIBLE_DEVICES=$g SST_GUARD=1 SST_RULE=threshold SST_BLOCK_S=0.2 SST_CTRL=$P1/control.json SST_GATE=$P1/train/gate_final.pt SST_TOKMS=${TOKMS[$d]} \
    $PYV -m experiments.sst_hsmm_v1.e2e_si_thinker --thinker $P1/merged --direction $d --gpu-mem 0.6 \
    --items-file $R/data/items_${d}_$k.jsonl --out $R/thinker_${d}_$k)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 40
done; done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do
    for k in 0 1 2 3; do [ -s $R/thinker_${d}_$k/blocks_${d}_w1.jsonl ] || die "thinker $d $k"; done
    mkdir -p $R/thinker_$d
    $PYQ -I - $T/UE_abs/data/items_$d.jsonl $R $d <<'PY' || die "merge blocks $d"
import json, sys
items, R, d = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [json.loads(l)["id"] for l in open(items, encoding="utf-8") if l.strip()]
rows = [l for k in range(4) for l in open(f"{R}/thinker_{d}_{k}/blocks_{d}_w1.jsonl", encoding="utf-8") if l.strip()]
assert [json.loads(l)["id"] for l in rows] == ids, (d, len(rows), len(ids))
with open(f"{R}/thinker_{d}/blocks_{d}_w1.jsonl", "x", encoding="utf-8") as fh:
    fh.writelines(rows)
print(d, len(rows))
PY
  done
fi
log "thinker done"

# 2. V1 tokens (GPU) and Kalman targets (CPU) on the NEW blocks
ps_=(); g=0
for d in en2zh zh2en; do
  ps_+=("$(launch $g v1tok_$d env PYTHONPATH=. S1_DATA="$S1_DATA" CUDA_VISIBLE_DEVICES=$g "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave5_m_v1tokens.py \
    --dir "$d" --blocks "$R/thinker_$d/blocks_${d}_w1.jsonl" --out-root "$R")")
  g=$((g + 1))
done
if [ -z "$DRY" ]; then
  env CUDA_VISIBLE_DEVICES= PYTHONPATH=. S1_DATA=$S1_DATA $PYN -m experiments.sst_hsmm_v1.decide --model $MODEL --blocks-root $R --out $R \
    > $LOGD/decide.log 2>&1 || die "decide"
fi
waitp "${ps_[@]}"
for d in en2zh zh2en; do [ -n "$DRY" ] || { [ -s $R/tokens/tokens_v1_$d.jsonl ] && [ -s $R/targets_$d.jsonl ]; } || die "tokens/targets $d"; done
log "v1 tokens + targets done"

# 3. posterior-MBR AbS search, 8 shards
ps_=()
for k in $(seq 0 $((NS - 1))); do
  ps_+=("$(launch $k search_$k env CUDA_VISIBLE_DEVICES=$k "${SENV[@]}" $PYN -m experiments.voice_onset_v1.p3_search \
    --tokens-root $R/tokens --targets $R --dirs en2zh zh2en --shard $k --num-shards $NS --out $R/search --device cuda:0)")
done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for d in en2zh zh2en; do for k in $(seq 0 $((NS - 1))); do [ -s $R/search/search_${d}_$k.jsonl ] || die "missing search_${d}_$k"; done; done
  $PYQ -I - "$R" "$NS" <<'PY' || die "merge search"
import json, sys
from pathlib import Path
root, ns = Path(sys.argv[1]), int(sys.argv[2])
for d in ("en2zh", "zh2en"):
    tg = [json.loads(l) for l in open(root / f"targets_{d}.jsonl") if l.strip()]
    rows = [json.loads(l) for k in range(ns) for l in open(root / "search" / f"search_{d}_{k}.jsonl") if l.strip()]
    assert {r["key"] for r in rows} == {t["key"] for t in tg} and len(rows) == len(tg), (d, len(rows), len(tg))
    with open(root / "tokens" / f"tokens_new_{d}.jsonl", "x") as fh:
        for r in rows:
            fh.write(json.dumps({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": r["ids"]}) + "\n")
    print(d, len(rows))
PY
fi
log "search done"

# 4. Talker T1c with the searched voice, 8 jobs
COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy block --gpu-mem 0.1 --seed 20261021)
ps_=(); g=0
for d in en2zh zh2en; do for s in 0 1 2 3; do
  ps_+=("$(launch $g talk_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
    -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $R/tokens/tokens_new_$d.jsonl \
    "${COMMON[@]}" --direction $d --output $R/new/talker_$d --shard $s --num-shards 4 \
    --blocks-file $R/thinker_$d/blocks_${d}_w1.jsonl --items-file $T/UE_abs/data/items_$d.jsonl)")
  g=$((g + 1)); [ -n "$DRY" ] || sleep 30
done; done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $R/new/talker_$d/part_$s.jsonl ] || die "talker $d $s"; done; done
log "talkers done"

# 5. ASR, then TitaNet / ECAPA2 / PESQ
ps_=(); g=0
for d in en2zh zh2en; do
  O=$R/new/talker_$d
  ps_+=("$(launch $g asr_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$O' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$O/results_speechonly.jsonl' --output '$O/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$O/asr_speech.jsonl'")")
  g=$((g + 1))
done
waitp "${ps_[@]}"
ps_=(); g=0
for d in en2zh zh2en; do
  O=$R/new/talker_$d; [ -n "$DRY" ] || [ -s $O/asr_speech.jsonl ] || die "asr $d"
  ps_+=("$(launch $g spk_$d env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. S1_DATA=$S1_DATA $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
    --dir $d --run $O --out $R/metrics/new_${d}_titanet.jsonl --device cuda:0)")
  ps_+=("$(launch $((g + 1)) ecapa2_$d env CUDA_VISIBLE_DEVICES=$((g + 1)) "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.test_abs_ecapa2 \
    --dir $d --run $O --out $R/metrics/ecapa2_new_${d}_spksim.jsonl --device cuda:0)")
  if [ -z "$DRY" ]; then env OMP_NUM_THREADS=1 PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $O \
    --out $R/metrics/new_${d}_squim.jsonl --procs 16 > $LOGD/pesq_$d.log 2>&1 & ps_+=($!); fi
  g=$((g + 2))
done
waitp "${ps_[@]}"
for d in en2zh zh2en; do for f in new_${d}_titanet new_${d}_squim ecapa2_new_${d}_spksim; do [ -n "$DRY" ] || [ -s $R/metrics/$f.jsonl ] || die "no $f"; done; done
log "metrics done"

# 6. dip player on NEW, N, B_b (CPU, 1 thread per worker)
[ -n "$DRY" ] && { echo "[dry] s0_apply dip x6 + p3_score" >&2; exit 0; }
touch $STOP
for d in en2zh zh2en; do
  for x in "NEW $R/new/talker_$d" "N $T/UE_mbr/mbr/talker_$d" "B_b $T/B/b/talker_$d"; do
    read -r name run <<< "$x"
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. $PYQ -m experiments.voice_onset_v1.s0_apply \
      --run $run --dir $d --policy dip --out $R/play/${name}_$d.jsonl --procs 32 > $LOGD/play_${name}_$d.log 2>&1 || die "play $name $d"
  done
done
log "play done"

# 7. score
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.p3_score --root p3j --rules pareto --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "p3 done"
