#!/usr/bin/env bash
# prereg change 37-2: SAP support-lag precheck (sap_lag.py) on the Emilia dev (UniST held-out, edev) and 600 spontaneous SAP
# training items per direction (e1 = p4 samples), cards from $CARDS (default 0 1 2 3) -> eval_outputs/sst_hsmm_v1/sap_lag/{ho,tr}_<d>.jsonl,
# then the CPU summary -> sap_lag/summary.txt. Refuses existing outputs.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D=eval_outputs/sst_hsmm_v1
R=$D/sap_lag
LOGD=logs/sst_hsmm_v1/sap_lag
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $R ] && { echo "exists: $R" >&2; exit 1; }
  mkdir -p $LOGD $R; git rev-parse HEAD > $R/commit.txt
  start_watcher
fi
CA=(${CARDS:-0 1 2 3}); ci=0; ps_=()
for d in en2zh zh2en; do
  ps_+=("$(launch ${CA[$ci]} ho_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=${CA[$ci]} $PYV -m experiments.sst_hsmm_v1.sap_lag --direction $d \
    --items $D/edev/data/items_$d.jsonl --rows $D/data/heldout_$d.jsonl --out $R/ho_$d.jsonl)"); ci=$((ci + 1)); [ -n "$DRY" ] || sleep 40
  ps_+=("$(launch ${CA[$ci]} tr_$d "${VENV[@]}" CUDA_VISIBLE_DEVICES=${CA[$ci]} $PYV -m experiments.sst_hsmm_v1.sap_lag --direction $d \
    --items $D/e1/data/items_$d.jsonl --rows $D/data/train_$d.jsonl --spont --limit 600 --out $R/tr_$d.jsonl)"); ci=$((ci + 1)); [ -n "$DRY" ] || sleep 40
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for x in ho tr; do for d in en2zh zh2en; do [ -s $R/${x}_$d.jsonl ] || die "sap_lag $x $d"; done; done
touch $STOP
PYTHONPATH=. $PYQ -m experiments.sst_hsmm_v1.sap_lag --summary $R > $R/summary.txt 2>&1 || die "summary"
$PYQ - $R <<'PY' || die "decision"
import json, sys, glob
sys.path.insert(0, "."); from experiments.sst_hsmm_v1.sap_lag import lag
R = sys.argv[1]
def L(f, cond=None):
    rs = [json.loads(l) for l in open(f)]
    rs = [r for r in rs if r["C"] >= 2 and (cond is None or r["id"].startswith(cond + "__"))]
    return sum(lag(r, "n_sap")[0] for r in rs) / len(rs)
Lho = sum(L(f"{R}/ho_{d}.jsonl", c) for d in ("en2zh", "zh2en") for c in ("clean", "noisy")) / 4
Ltr = sum(L(f"{R}/tr_{d}.jsonl") for d in ("en2zh", "zh2en")) / 2
go = Lho >= 0.25 and Lho - Ltr >= 0.10
json.dump({"L_ho": Lho, "L_tr": Ltr, "go_retrain": go, "rule": "L_ho >= 0.25 and L_ho - L_tr >= 0.10 (prereg 37-2)"}, open(f"{R}/decision.json", "x"), indent=1)
print(json.dumps({"L_ho": round(Lho, 3), "L_tr": round(Ltr, 3), "go_retrain": go}))
PY
touch $R/DONE; log "sap_lag done"
