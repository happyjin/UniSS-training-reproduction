#!/usr/bin/env bash
# prereg change 42-2: P0-ES rerun with VLLM_BATCH_INVARIANT=1 = es_orch.sh (change 41) with every hsc13 job under BI=1 and outputs es2 / es2b.
# Was: prereg change 41 (P0-ES): parameter-space exploration precheck of the drafter LoRA.
#  1. es_make members (8 antithetic pairs, sigma $SIG) -> $E/members
#  2. phase 1: 8 hsc13 jobs (2 directions x 4 groups; group g = base + p{2g}_pos/neg + p{2g+1}_pos/neg) on cards 0-7, p3h8 policy,
#     greedy, items = p0rl/data (500 / direction) -> $E/run_g<g>
#  3. es_analyze phase1 -> $E/phase1.json (+ .grad.json); G0-ES fails at sigma 0.01 -> ONE rerun of 1-3 at sigma 0.03 ($E -> ${E}b)
#  4. es_make step -> $E/step (step_pos, step_neg); phase 2: 2 hsc13 jobs (base, step_pos, step_neg) -> $E/run_step
#  5. es_analyze phase2 -> $E/phase2.json; $E/DONE. Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
DRY=${DRY_RUN:-}
D0=eval_outputs/sst_hsmm_v1
DATA=$D0/p0rl/data
DREF=$D0/p0rl/data_ref
NTH=eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05
VER=$D0/v/lora/a1.0000
PRM=$D0/h14/params.json
LOGD=logs/sst_hsmm_v1/es2
export PIDS=$LOGD/jobs.pids
STOP=$LOGD/watcher.stop
R=$D0/es2
source experiments/sst_hsmm_v1/lib.sh
if [ -z "$DRY" ]; then
  [ -e $D0/es2 ] && { echo "exists: $D0/es2" >&2; exit 1; }
  mkdir -p $LOGD $D0/es2; git rev-parse HEAD > $D0/es2/commit.txt
  start_watcher
fi
job() {   # card name out drafters names direction
  launch $1 $2 "${VENV[@]}" VLLM_BATCH_INVARIANT=1 CUDA_VISIBLE_DEVICES=$1 $PYV -m experiments.sst_hsmm_v1.hsc13_si_thinker --mode hsc --variant n2 --samples 0 \
    --params $PRM --theta 0.75 --verifier $VER --second-view enh --views $D0/e1/views/$6 --thinker $NTH --direction $6 --gpu-mem 0.6 \
    --drafters "$4" --names "$5" --items-file $DATA/items_$6.jsonl --out $3
}
phase1() {   # E sigma
  local E=$1 SIG=$2 ps_=() ci=0 g d dr nm
  [ -n "$DRY" ] || { $PYV -m experiments.sst_hsmm_v1.es_make members --out $E/members --pairs 8 --sigma $SIG > $LOGD/make_$(basename $E).log 2>&1 || die "make $E"; }
  for d in en2zh zh2en; do for g in 0 1 2 3; do
    dr="none"; nm="base"
    for k in $((2 * g)) $((2 * g + 1)); do dr="$dr,$E/members/p${k}_pos,$E/members/p${k}_neg"; nm="$nm,p${k}_pos,p${k}_neg"; done
    ps_+=("$(job $ci $(basename $E)_${d}_g$g $E/run_g$g "$dr" "$nm" $d)")
    ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
  done; done
  waitp "${ps_[@]}"
  [ -n "$DRY" ] && return 0
  for d in en2zh zh2en; do for g in 0 1 2 3; do [ -s $E/run_g$g/blocks_${d}_base.jsonl ] || die "phase1 $E $d $g"; done; done
  PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.es_analyze phase1 --jobs "$E/run_g*" --data $DREF --out $E/phase1.json > $LOGD/phase1_$(basename $E).log 2>&1 || die "analyze phase1 $E"
}
phase1 $D0/es2 0.01
E=$D0/es2; SIG=0.01
if [ -z "$DRY" ] && ! $PYE -c "import json,sys; sys.exit(0 if json.load(open('$E/phase1.json'))['G0_pass'] else 1)"; then
  log "G0-ES failed at sigma 0.01 -> one rerun at 0.03 (prereg change 41)"
  [ -e $D0/es2b ] && die "es1b exists"
  mkdir -p $D0/es2b; E=$D0/es2b; SIG=0.03
  phase1 $E $SIG
  $PYE -c "import json,sys; sys.exit(0 if json.load(open('$E/phase1.json'))['G0_pass'] else 1)" || { log "G0-ES failed at 0.03 too"; touch $R/G0_FAILED; touch $STOP; touch $R/DONE; exit 0; }
fi
[ -n "$DRY" ] || { $PYV -m experiments.sst_hsmm_v1.es_make step --out $E/step --sigma $SIG --grad $E/phase1.json.grad.json > $LOGD/make_step.log 2>&1 || die "make step"; }
ps_=(); ci=0
for d in en2zh zh2en; do
  ps_+=("$(job $ci es2step_$d $E/run_step "none,$E/step/step_pos,$E/step/step_neg" "base,step_pos,step_neg" $d)")
  ci=$((ci + 1)); [ -n "$DRY" ] || sleep 45
done
waitp "${ps_[@]}"
[ -n "$DRY" ] && exit 0
for d in en2zh zh2en; do [ -s $E/run_step/blocks_${d}_step_pos.jsonl ] || die "phase2 $d"; done
touch $STOP
PYTHONPATH=. $PYE -m experiments.sst_hsmm_v1.es_analyze phase2 --jobs "$E/run_step" --data $DREF --out $E/phase2.json > $LOGD/phase2.log 2>&1 || die "analyze phase2"
echo "$E" > $R/final_dir.txt
touch $R/DONE; log "es2 done ($E)"
