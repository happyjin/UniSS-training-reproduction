#!/usr/bin/env bash
# voice_onset_v1 P3 (reports/voice_onset_v1/P1_P3_prereg.zh-CN.md section 4), cards 4-7 ONLY. Built from test_abs_orch.sh.
#  0. p3_decide (CPU): onsets + posterior means -> UE_hmo/{data/blocks,targets}, UE_mbr/targets
#  1. V1 tokens on the UE_hmo prompts (wave5_m_v1tokens, cards 4 / 5)
#  2. p3_search (posterior-MBR AbS): arms hmo (tokens root UE_hmo/tokens) and mbr (UE/tokens), shard k on card 4+k
#  3. Talker (T1c, seed 20261021): UE_hmo with its delayed blocks, UE_mbr with the UE blocks; 4 jobs per card
#  4. speech-only ASR -> PESQ (CPU) + TitaNet + ECAPA2 ; 5. p3_score -> eval_outputs/voice_onset_v1/p3/summary.json
# Placeholders SIGTERMed by PID per card before each job; own watcher p3_noplaceholder.sh (POLL 1 s). Never touches guard
# 2726951, gpu_watch 52045, sync 1222453. Refuses existing run dirs.
#   setsid nohup bash experiments/voice_onset_v1/p3_orch.sh > logs/voice_onset_v1/p3_orch.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
ARMS=${ARMS:-hmo mbr}   # prereg change 18:45: all onsets j = 0 -> UE_hmo == UE_mbr, run ARMS=mbr
T=eval_outputs/mist_w3po_v1/test_robust
UE=$T/UE
H=$T/UE_hmo
M=$T/UE_mbr
R=eval_outputs/voice_onset_v1/p3
MODEL=eval_outputs/voice_onset_v1/p1/train/model.pt
LOGD=logs/voice_onset_v1/p3
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
export S1_DATA=/opt/dlami/nvme/jasonleeeli/projects/UniSS/$T/data
TALKER=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
PYQ=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
VENV=(env PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
      VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. OMP_NUM_THREADS=8
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
NENV=(env PYTHONPATH=. OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface S1_DATA=$S1_DATA)
SENV=(env PYTHONPATH=.:.tmp/b_abs_pyshim OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface)
PROTECT=" 2726951 52045 1222453 $$ "
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
die() { log "FAILED: $*"; [ -n "$DRY" ] || { mkdir -p $R; touch $R/FAILED; touch $STOP; }; exit 1; }
launch() {   # card name cmd... ; prints pid (DRY_RUN: prints the command to stderr and "DRY")
  local g=$1 name=$2; shift 2
  local lf=$LOGD/$name.log
  if [ -n "$DRY" ]; then printf '[dry] card %s %s: ' "$g" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  case " 4 5 6 7 " in *" $g "*) ;; *) die "card $g not allowed";; esac
  for pat in run_mist_workload.sh train_mist_workload.py; do
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/$pat*) ;; *) continue;; esac
      a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      case "$a" in *scripts/gpu_load/$pat*"--devices $g "*) kill -TERM "$pid" 2>/dev/null && log "SIGTERM placeholder $pid card $g" >&2;; esac
    done < <(ps -eo pid=,args=)
  done
  sleep 8
  local k left=""
  for k in $(seq 1 24); do
    left=""
    while read -r pid args; do
      case "$PROTECT" in *" $pid "*) continue;; esac
      case "$args" in *scripts/gpu_load/*"--devices $g "*) left="$left $pid"; kill -TERM "$pid" 2>/dev/null;; esac
    done < <(ps -eo pid=,args=)
    [ -z "$left" ] && break
    sleep 5
  done
  [ -n "$left" ] && log "WARNING: placeholders still on card $g after 120 s:$left" >&2
  setsid nohup "$@" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
  local p=$!
  echo "$g $p $name" >> "$PIDS"
  log "$name card $g pid $p" >&2
  echo $p
}
waitp() { for p in "$@"; do case "$p" in ''|*[!0-9]*) continue;; esac; while kill -0 "$p" 2>/dev/null; do sleep 15; done; done; }

[ -s $MODEL ] || die "no model $MODEL"
if [ -z "$DRY" ]; then
  for x in $M/mbr $R/summary.json; do [ -e $x ] && { echo "exists: $x (refuse to overwrite)" >&2; exit 1; }; done
  [ -e .tmp/b_abs_pyshim/einx/__init__.py ] || { echo "pyshim missing" >&2; exit 1; }
  mkdir -p "$LOGD" $R $H/metrics $M/metrics $H/tokens $M/tokens
  [ -e "$STOP" ] && { echo "STOP exists: $STOP" >&2; exit 1; }
  POLL=1 STOP=$STOP setsid nohup bash experiments/voice_onset_v1/p3_noplaceholder.sh > "$LOGD/noplaceholder.log" 2>&1 < /dev/null 8>&- 9>&- &
  log "orch $$; watcher $!"
  git rev-parse HEAD > $R/commit.txt
fi

# 0. decisions (CPU)
if [ -z "$DRY" ] && [ ! -s $H/targets_zh2en.jsonl ]; then
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PYN -m experiments.voice_onset_v1.p3_decide --model $MODEL --hmo $H --mbr $M > $LOGD/decide.log 2>&1 || die "decide"
fi
log "decide done"

# 1. V1 tokens on the delayed prompts (hmo arm only)
ps_=(); k=0
for d in en2zh zh2en; do
  case " $ARMS " in *" hmo "*) ;; *) continue;; esac
  g=$((4 + k)); k=$((k + 1))
  ps_+=("$(launch $g v1tok_$d env PYTHONPATH=. S1_DATA="$S1_DATA" CUDA_VISIBLE_DEVICES=$g "$PYQ" -I experiments/mist_w3po_v1/voicecond/wave5_m_v1tokens.py \
    --dir "$d" --blocks "$H/data/blocks_${d}_w1.jsonl" --out-root "$H")")
done
waitp "${ps_[@]}"
case " $ARMS " in *" hmo "*) for d in en2zh zh2en; do [ -n "$DRY" ] || [ -s $H/tokens/tokens_v1_$d.jsonl ] || die "no V1 tokens $d"; done;; esac
log "v1 tokens done"

# 2. search, both arms
ps_=()
for k in 0 1 2 3; do
  g=$((4 + k))
  case " $ARMS " in *" hmo "*) ps_+=("$(launch $g search_hmo_$k env CUDA_VISIBLE_DEVICES=$g "${SENV[@]}" $PYN -m experiments.voice_onset_v1.p3_search \
    --tokens-root $H/tokens --targets $H --dirs en2zh zh2en --shard $k --num-shards 4 --out $H/search --device cuda:0)");; esac
  ps_+=("$(launch $g search_mbr_$k env CUDA_VISIBLE_DEVICES=$g "${SENV[@]}" $PYN -m experiments.voice_onset_v1.p3_search \
    --tokens-root $UE/tokens --targets $M --dirs en2zh zh2en --shard $k --num-shards 4 --out $M/search --device cuda:0)")
done
waitp "${ps_[@]}"
if [ -z "$DRY" ]; then
  for arm in $ARMS; do X=$H; [ $arm = mbr ] && X=$M; for d in en2zh zh2en; do for k in 0 1 2 3; do [ -s $X/search/search_${d}_$k.jsonl ] || die "missing $X search_${d}_$k"; done; done; done
  $PYQ -I - "$H" "$M" "$ARMS" <<'PY' || die "merge"
import json, sys
from pathlib import Path
for root, arm in ((Path(sys.argv[1]), "hmo"), (Path(sys.argv[2]), "mbr")):
    if arm not in sys.argv[3].split():
        continue
    for d in ("en2zh", "zh2en"):
        tg = [json.loads(l) for l in open(root / f"targets_{d}.jsonl") if l.strip()]
        rows = [json.loads(l) for k in range(4) for l in open(root / "search" / f"search_{d}_{k}.jsonl") if l.strip()]
        assert {r["key"] for r in rows} == {t["key"] for t in tg} and len(rows) == len(tg), (arm, d, len(rows), len(tg))
        with open(root / "tokens" / f"tokens_{arm}_{d}.jsonl", "x") as fh:
            for r in rows:
                fh.write(json.dumps({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": r["ids"]}) + "\n")
        print(arm, d, len(rows))
PY
fi
log "search done"

# 3. Talker, both arms (4 jobs per card)
COMMON=(--model $TALKER --window 1 --text-policy file --speed-policy block --gpu-mem 0.1 --seed 20261021)
ps_=(); k=0
for arm in $ARMS; do
  X=$H; BL=$H/data; [ $arm = mbr ] && { X=$M; BL=$UE/thinker_; }
  for d in en2zh zh2en; do for s in 0 1 2 3; do
    g=$((4 + s))
    blk=$H/data/blocks_${d}_w1.jsonl; [ $arm = mbr ] && blk=$UE/thinker_$d/blocks_${d}_w1.jsonl
    ps_+=("$(launch $g talk_${arm}_${d}_$s "${VENV[@]}" SI_KV_CACHE_GIB=7.73 CUDA_VISIBLE_DEVICES=$g $PYV \
      -m experiments.mist_w3po_v1.voicecond.idea1_si_infer_vo --voice-override $X/tokens/tokens_${arm}_$d.jsonl \
      "${COMMON[@]}" --direction $d --output $X/$arm/talker_$d --shard $s --num-shards 4 \
      --blocks-file $blk --items-file $T/UE_abs/data/items_$d.jsonl)")
    [ -n "$DRY" ] || sleep 30
  done; done
done
waitp "${ps_[@]}"
for arm in $ARMS; do X=$H; [ $arm = mbr ] && X=$M
  for d in en2zh zh2en; do for s in 0 1 2 3; do [ -n "$DRY" ] || [ -s $X/$arm/talker_$d/part_$s.jsonl ] || die "talker $arm $d $s"; done; done; done
log "talkers done"

# 4. ASR + metrics
ps_=(); k=0
for arm in $ARMS; do X=$H; [ $arm = mbr ] && X=$M
  for d in en2zh zh2en; do
    g=$((4 + k % 4)); k=$((k + 1)); O=$X/$arm/talker_$d
    ps_+=("$(launch $g asr_${arm}_$d bash -c "set -e; $PYE reports/uniss_si_v4/speech_only.py '$O' && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. $PYE -m evaluation.asr_transcribe --input '$O/results_speechonly.jsonl' --output '$O/asr_speech.jsonl' --device cuda:0 --batch-size 4 && test -f '$O/asr_speech.jsonl'")")
  done
done
waitp "${ps_[@]}"
ps_=(); k=0
for arm in $ARMS; do X=$H; [ $arm = mbr ] && X=$M
  for d in en2zh zh2en; do
    O=$X/$arm/talker_$d; [ -n "$DRY" ] || [ -s $O/asr_speech.jsonl ] || die "asr $arm $d"
    g=$((4 + k % 4)); k=$((k + 1))
    ps_+=("$(launch $g spk_${arm}_$d env CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. S1_DATA=$S1_DATA $PYN -m experiments.mist_w3po_v1.voicecond.wave3_v_spksim \
      --dir $d --run $O --out $X/metrics/${arm}_${d}_spksim.jsonl --device cuda:0)")
    ps_+=("$(launch $g ecapa2_${arm}_$d env CUDA_VISIBLE_DEVICES=$g "${NENV[@]}" $PYN -m experiments.mist_w3po_v1.voicecond.test_abs_ecapa2 \
      --dir $d --run $O --out $X/metrics/ecapa2_${arm}_${d}_spksim.jsonl --device cuda:0)")
    if [ -z "$DRY" ]; then PYTHONPATH=. $PYE experiments/mist_w3po_v1/voicecond/wave2_s1_score.py pesq --run $O \
      --out $X/metrics/${arm}_${d}_squim.jsonl --procs 16 > $LOGD/pesq_${arm}_$d.log 2>&1 & ps_+=($!); fi
  done
done
waitp "${ps_[@]}"
for arm in $ARMS; do X=$H; [ $arm = mbr ] && X=$M
  for d in en2zh zh2en; do for m in spksim squim; do [ -n "$DRY" ] || [ -s $X/metrics/${arm}_${d}_$m.jsonl ] || die "no $m $arm $d"; done
  [ -n "$DRY" ] || [ -s $X/metrics/ecapa2_${arm}_${d}_spksim.jsonl ] || die "no ecapa2 $arm $d"; done; done
log "metrics done"

# 5. score
if [ -n "$DRY" ]; then echo "[dry] score" >&2; exit 0; fi
touch $STOP
PYTHONPATH=. $PYQ -m experiments.voice_onset_v1.p3_score --out $R/summary.json > $LOGD/score.log 2>&1 || die "score"
touch $R/DONE
log "orch done"
