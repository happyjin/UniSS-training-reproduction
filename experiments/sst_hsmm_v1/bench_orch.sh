#!/usr/bin/env bash
# prereg change 38-0: bench_cons on edev, one card per direction ($CARDS, default "0 1"), K / max-tokens from $K / $MT (default 6 / 24)
# -> eval_outputs/sst_hsmm_v1/$OUTN/{reads,summary}_<d>.*; decision per prereg 38-0 -> $OUTN/decision.json. Refuses outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
: "${OUTN:?}"
K=${K:-6}; MT=${MT:-24}
R=eval_outputs/sst_hsmm_v1/$OUTN
LOGD=logs/sst_hsmm_v1/$OUTN
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
CA=(${CARDS:-0 1}); ci=0; ps_=()
for d in en2zh zh2en; do
  ps_+=("$(launch ${CA[$ci]} bench_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=${CA[$ci]} $PYV -m experiments.sst_hsmm_v1.bench_cons --direction $d \
    --n-items 40 --k $K --max-tokens $MT --out $R/$d)"); ci=$((ci + 1)); [ -n "$DRY" ] || sleep 20
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for d in en2zh zh2en; do [ -s $R/$d/summary_$d.json ] || die "bench $d"; done
touch $STOP
$PYQ - $R <<'PY' || die "decision"
import json, sys
R = sys.argv[1]
s = {d: json.load(open(f"{R}/{d}/summary_{d}.json")) for d in ("en2zh", "zh2en")}
ok = all(s[d]["extra"]["p95_ms"] <= 150 and s[d]["B"]["p95_ms"] <= 1000 for d in s)
json.dump({"go": ok, "rule": "per direction: p95 extra <= 150 ms and pass-B p95 <= 1000 ms (prereg 38-0)",
           **{d: {"A_p95": s[d]["A"]["p95_ms"], "B_p95": s[d]["B"]["p95_ms"], "A_med": s[d]["A"]["median_ms"], "B_med": s[d]["B"]["median_ms"],
                  "extra_med": s[d]["extra"]["median_ms"], "extra_p95": s[d]["extra"]["p95_ms"]} for d in s}}, open(f"{R}/decision.json", "x"), indent=1)
print(open(f"{R}/decision.json").read())
PY
touch $R/DONE; log "$OUTN done"
