#!/bin/bash
# Finish the remaining handbook schemes once their training ends. Every scheme waits for its own final checkpoint,
# then renders test (render_backend.py behind the S6-future fixed:1 plans, or run_stream_eval for S1), then audio
# metrics. Outputs that exist are skipped (run_jobs done files, audio_metrics.json).
#   all_schemes_driver.sh <cards>
set -euo pipefail
CARDS="${1:?cards}"
ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1
P=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-train/bin/python
V=/opt/dlami/nvme/jasonleeeli/venvs/tts_schemes/bin/python
A=$ROOT/runs/all_schemes
BB=$ROOT/runs/e0_backbone_v2/G_60000.pth
CFG=$ROOT/configs/ljs_char_base.json
cd "$ROOT"
log() { echo "$(date -u +%H:%M:%S) $*"; }
IFS=, read -ra C <<< "$CARDS"
PLANNER=$($P -c "import json;print('$ROOT/runs/final/s6_future/stream_%d.pth'%json.load(open('$ROOT/runs/final/s6_future/best.json'))['best_step'])")

render_jobs() {  # backend ckpt_json out_json
  $P - "$1" "$2" "$3" <<'EOF'
import json, sys
be, ck, out = sys.argv[1:]
root = "/opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/llm_vits_route_v1"
planner = open(f"{root}/runs/all_schemes/.planner").read().strip()
jobs = []
for s in range(4):
  d = f"{root}/runs/all_schemes/render_test/{be}"
  cmd = ["/opt/dlami/nvme/jasonleeeli/venvs/tts_schemes/bin/python", "scripts/render_backend.py", "--backend", be,
         "--model", planner, "--ckpt", ck, "--backbone-config", f"{root}/configs/ljs_char_base.json",
         "--backbone-ckpt", f"{root}/runs/e0_backbone_v2/G_60000.pth", "--split", "test", "--shard", str(s),
         "--num-shards", "4", "--out", d]
  jobs.append({"id": f"render-{be}-{s}", "cmd": cmd, "done": f"{d}/commits.shard{s}.jsonl"})
json.dump(jobs, open(out, "w"), indent=1)
EOF
}

audio() {  # run_dir card
  [ -f "$1/audio_metrics.json" ] || CUDA_VISIBLE_DEVICES=$2 $P scripts/eval_audio.py "$1" --align-dir runs/final/align > "$1/eval_audio.log" 2>&1
}

scheme() {  # name wait_file ckpt_json card_a card_b
  local name=$1 wait=$2 ck=$3 ca=$4 cb=$5
  until [ -f "$wait" ]; do sleep 60; done
  log "$name: training done, rendering test"
  [ -f "$A/jobs_render_$name.json" ] || render_jobs "$name" "$ck" "$A/jobs_render_$name.json"
  $P scripts/run_jobs.py "$A/jobs_render_$name.json" --cards "$ca,$cb" --per-card 2 --log-dir "$A/joblogs" >> "$A/render_$name.log" 2>&1
  audio "$A/render_test/$name" "$ca"
  log "$name: audio metrics done"
}

echo "$PLANNER" > "$A/.planner"
scheme s5 "$A/s5/S5_20000.pth" "{\"s5\": \"$A/s5/S5_20000.pth\"}" "${C[0]}" "${C[1]}" &
scheme t4 "$A/t4_dec/T4dec_30000.pth" "{\"t4flow\": \"$A/t4/T4flow_30000.pth\", \"t4dec\": \"$A/t4_dec/T4dec_30000.pth\"}" "${C[2]}" "${C[3]}" &
scheme s3 "$A/s3/s3_40000.pth" "{\"s3\": \"$A/s3/s3_40000.pth\"}" "${C[4]}" "${C[5]}" &
scheme s4 "$A/s4/s4_40000.pth" "{\"s4\": \"$A/s4/s4_40000.pth\"}" "${C[6]}" "${C[7]}" &
scheme s2 "$A/s2/S2_60000.pth" "{\"s2\": \"$A/s2/S2_60000.pth\"}" "${C[0]}" "${C[1]}" &
(
  until [ -f "$A/s1/best.json" ]; do sleep 60; done
  log "s1: training done, fixed lookahead val + test"
  for split in val test; do
    [ -f "$A/jobs_s1_$split.json" ] || $P scripts/make_jobs.py --stage fixed --exp "$A" --backbone-ckpt "$BB" --noise-w 0.8 \
      --models s1 --ks 0,1 --split $split --shards 4 --out "$A/jobs_s1_$split.json"
    $P scripts/run_jobs.py "$A/jobs_s1_$split.json" --cards "${C[2]},${C[3]}" --per-card 3 --log-dir "$A/joblogs" >> "$A/s1_eval.log" 2>&1
  done
  [ -f "$A/jobs_s1_render.json" ] || $P scripts/make_jobs.py --stage fixed --exp "$A" --backbone-ckpt "$BB" --noise-w 0.8 \
    --models s1 --ks 1 --split test --shards 4 --render --tag _render --out "$A/jobs_s1_render.json"
  $P scripts/run_jobs.py "$A/jobs_s1_render.json" --cards "${C[2]},${C[3]}" --per-card 2 --log-dir "$A/joblogs" >> "$A/s1_eval.log" 2>&1
  audio "$A/eval_render_test/s1_fixed1" "${C[2]}"
  log "s1: done"
) &
wait
for d in s5 t4 s3 s4 s2; do [ -f "$A/render_test/$d/audio_metrics.json" ] || { log "missing audio metrics for $d"; exit 1; }; done
[ -f "$A/eval_render_test/s1_fixed1/audio_metrics.json" ] || { log "missing audio metrics for s1"; exit 1; }
log "ALL_SCHEMES_DONE"
