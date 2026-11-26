#!/usr/bin/env bash
# prereg change 42-1b: = bi_test_orch.sh with --enforce-eager on the BI jobs A, B (C unchanged), output bi1b.
# Was: prereg change 42-1: vLLM batch-invariant determinism test. hsc13 (p3h8 policy) on p0rl/data (500 / direction), members from
# es1/members (sigma 0.01). Per direction: A = VLLM_BATCH_INVARIANT=1 {base, p0_pos}; B = BI=1 {base, p0_pos, p0_neg};
# C = BI=0 {base, p0_pos}. 6 jobs on cards 0-5 -> eval_outputs/sst_hsmm_v1/bi1/{A,B,C}; then bi_compare -> bi1/result.json.
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
E=$D0/bi1b
R=$E
M=$D0/es1/members
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
LOGD=logs/sst_hsmm_v1/bi1b
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $E ] && { echo "exists: $E" >&2; exit 1; }
  mkdir -p $LOGD $E; git rev-parse HEAD > $E/commit.txt
  start_watcher
fi
job() {   # card name out bi drafters names direction
  launch $1 $2 "${VENV[@]}" VLLM_BATCH_INVARIANT=$4 CUDA_VISIBLE_DEVICES=$1 $PYV -m experiments.sst_hsmm_v1.hsc13_si_thinker --mode hsc --variant n2 \
    --samples 0 --params $D0/h14/params.json --theta 0.75 --verifier $D0/v/lora/a1.0000 --second-view enh --views $D0/e1/views/$7 \
    --thinker $NTH --direction $7 --gpu-mem 0.6 --drafters "$5" --names "$6" --items-file $D0/p0rl/data/items_$7.jsonl --out $3 ${8:-}
}
ps_=(); ci=0
for d in en2zh zh2en; do
  ps_+=("$(job $ci A_$d $E/A 1 "none,$M/p0_pos" "base,p0_pos" $d --enforce-eager)"); ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
  ps_+=("$(job $ci B_$d $E/B 1 "none,$M/p0_pos,$M/p0_neg" "base,p0_pos,p0_neg" $d --enforce-eager)"); ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
  ps_+=("$(job $ci C_$d $E/C 0 "none,$M/p0_pos" "base,p0_pos" $d)"); ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
touch $STOP
$PYE -I - $E > $E/result.json 2> $LOGD/compare.err <<'PY' || die "compare"
import json, sys, glob
E = sys.argv[1]
def rows(p):
    try:
        return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
    except FileNotFoundError:
        return None
res = {}
for d in ("en2zh", "zh2en"):
    r = {}
    for nm in ("base", "p0_pos"):
        a, b, c = (rows(f"{E}/{x}/blocks_{d}_{nm}.jsonl") for x in "ABC")
        if a is None or b is None:
            r[nm] = {"error": "missing output"}
            continue
        ch = lambda x, y: sum(x[i]["translation"] != y[i]["translation"] or x[i]["blocks"] != y[i]["blocks"] for i in x) / len(x)
        r[nm] = {"changed_A_vs_B": ch(a, b), "changed_A_vs_C": ch(a, c) if c else None}
    sec = {}
    for x in "ABC":
        f = glob.glob(f"{E}/{x}/job_{d}_base.combo.json")
        sec[x] = json.load(open(f[0]))["seconds"] if f else None
    r["seconds"] = sec
    r["ratio_A_over_C"] = sec["A"] / sec["C"] if sec["A"] and sec["C"] else None
    res[d] = r
ok = all(isinstance(res[d][nm].get("changed_A_vs_B"), float) and res[d][nm]["changed_A_vs_B"] <= 0.01 for d in res for nm in ("base", "p0_pos"))
ok = ok and all(res[d]["ratio_A_over_C"] is not None and res[d]["ratio_A_over_C"] <= 3.0 for d in res)
res["pass"] = bool(ok)
print(json.dumps(res, indent=1))
PY
touch $E/DONE; log "bi1b done"
