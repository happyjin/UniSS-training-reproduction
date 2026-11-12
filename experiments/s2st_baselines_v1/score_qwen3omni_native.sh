#!/bin/bash
# Score baseline (c), Qwen3-Omni native on CVSS-T test, with the paper's Table 1 protocol (paper/baselines_run_plan_v1.md
# §3): speech_only.py -> evaluation.asr_transcribe (Whisper-large-v3 / Paraformer, batch 16) -> corpus BLEU as
# paper/bayes_simul_s2st_v11/scripts/cvss_table.py; cvss_table1_metrics.py (AutoPCP, SIM-O wavlm-base-plus-sv).
# Needs all NUM_SHARDS part files complete (4,897 rows a direction); refuses otherwise. Never overwrites: each step
# skips an output that exists (the scorers refuse or skip themselves), the summary is written with noclobber.
#
#   GPU=4 bash experiments/s2st_baselines_v1/score_qwen3omni_native.sh        DRY_RUN=1 prints the steps only
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
OUT=${OUT:-eval_outputs/s2st_baselines_v1/qwen3omni_native_cvss}
NUM_SHARDS=${NUM_SHARDS:-6}
GPU=${GPU:?GPU=<card>}
DRY_RUN=${DRY_RUN:-0}
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope
ts() { TZ=Europe/Prague date +%H:%M:%S; }
for d in en2zh zh2en; do
  n=$(cat "$OUT/cvss_$d"/part_*of${NUM_SHARDS}.jsonl 2>/dev/null | wc -l)
  u=$(cat "$OUT/cvss_$d"/part_*of${NUM_SHARDS}.jsonl 2>/dev/null | "$PYE" -c "import sys,json; print(len({json.loads(l)['id'] for l in sys.stdin if l.strip()}))")
  echo "[$(ts)] $d: $n rows, $u unique ids"
  if [ "$u" != 4897 ] || [ "$n" != 4897 ]; then echo "refusing: $d is not complete (need 4897 unique rows)"; exit 1; fi
done
if [ "$DRY_RUN" = 1 ]; then echo "would run speech_only, asr_transcribe, cvss_table1_metrics on card $GPU, then the summary"; exit 0; fi
for d in en2zh zh2en; do
  R="$OUT/cvss_$d"
  "$PYE" reports/uniss_si_v4/speech_only.py "$R" || exit 1
  if [ ! -s "$R/asr_speech.jsonl" ]; then
    echo "[$(ts)] ASR $d"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. "$PYE" -m evaluation.asr_transcribe --input "$R/results_speechonly.jsonl" \
      --output "$R/asr_speech.jsonl" --device cuda:0 --batch-size 16 || exit 1
  fi
  echo "[$(ts)] A.PCP / SIM-O $d"
  PYTHONPATH=. "$PYE" reports/uniss_si_v4/cvss_table1_metrics.py "$R" "$OUT/metrics_$d" "$GPU" || exit 1
done
PYTHONPATH=. "$PYE" - "$OUT" <<'PY' || exit 1
import glob, json, statistics as st, sys
from pathlib import Path
from evaluation.text_metrics import corpus_bleu
out = Path(sys.argv[1]); summ = {}
for d, lang in (("en2zh", "cmn"), ("zh2en", "eng")):
    R = out / f"cvss_{d}"
    res = [json.loads(l) for l in open(R / "results.jsonl", encoding="utf-8")]
    heard = {x["id"]: x.get("asr_text", "") for x in map(json.loads, open(R / "asr_speech.jsonl", encoding="utf-8"))}
    asr = corpus_bleu([heard.get(r["id"], "") for r in res], [r["translation_ref"] for r in res], language=lang)["score"]
    text = corpus_bleu([r["translated_text"] for r in res], [r["translation_ref"] for r in res], language=lang)["score"]
    m = out / f"metrics_{d}"
    ap = [json.loads(l)["autopcp_score"] for f in glob.glob(str(m / "apcp/*.jsonl")) for l in open(f) if "autopcp_score" in json.loads(l)]
    sim = float(next(iter(json.load(open(m / "sim_o_baseplus.json"))["groups"].values()))["mean"])
    summ[d] = {"n": len(res), "n_no_audio": sum(1 for r in res if r.get("error")), "n_asr": len(heard),
               "asr_bleu": round(asr, 2), "text_bleu": round(text, 2), "apcp": round(st.mean(ap), 3) if ap else None,
               "n_apcp": len(ap), "sim_o_baseplus": round(sim, 3)}
with open(out / "summary.json", "x", encoding="utf-8") as fh:
    json.dump(summ, fh, indent=1)
print(json.dumps(summ, indent=1))
PY
echo "[$(ts)] QWEN3OMNI_NATIVE_SCORED -> $OUT/summary.json"
