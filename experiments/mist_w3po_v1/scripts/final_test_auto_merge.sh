#!/usr/bin/env bash
# Final test, VE1 / D1v read-level merges (coordinator pre-authorisation 2026-11-12 ~22:20 Prague): when a v2 chain driver
# stops a merge (FT_MERGE_STOP <decode name> in chain_v2_<set>.log), continue its PL-EM chain with
# final_test_after_merge.sh only if ALL hold:
#   1. the merge stopped on "pairing lost" and every lost window is the known mechanism (merge_pairlost_check.py:
#      all_known_mechanism, segment / block write_ms of the same entry differ)
#   2. pair_lost <= 1 % of the set's windows
#   3. the merged copy is complete (hyp lines = windows, score_text.json)
# otherwise log FT_AUTO_MERGE_REFUSED and do nothing (the coordinator is told). One waiter per (set, Thinker).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/final_test_auto_merge.sh
# Run detached: setsid nohup bash .../final_test_auto_merge.sh > logs/mist_w3po_v1/final_test/auto_merge.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/final_test
DEC=eval_outputs/mist_w3po_v1/decode
log() { echo "$(date -u +%FT%TZ) $*"; }
one() {   # <set> <thinker tag>
  local S=$1 T=$2 NW F O R PL OKM SH HL
  case $S in ami) NW=3973;; ali) NW=4065;; esac
  F=ft_${S}_${T}_s3000; O=eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/$S
  until grep -qE "FT_MERGE_(STOP|DEVIATION) $F:|merged $F:|FT_DECODE_FAILED ${S}_$T" $E/chain_v2_$S.log; do sleep 30; done
  grep -q "FT_MERGE_STOP $F:" $E/chain_v2_$S.log || { log "$F: no merge stop (driver continued or decode failed)"; return 0; }
  grep -q 'pairing lost' $E/${S}_merge_$F.log || { log "FT_AUTO_MERGE_REFUSED $F: merge did not stop on pairing lost"; return 1; }
  R=$O/merge_pairlost_$F.json
  [ -e $R ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.merge_pairlost_check --src $F > $R
  read PL OKM SH <<< "$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['pair_lost'], int(d['all_known_mechanism']), d['share'])" $R)"
  HL=$(wc -l < $DEC/${F}_rl/hyp.jsonl 2>/dev/null || echo 0)
  if [ "$OKM" = 1 ] && [ $((PL * 100)) -le $NW ] && [ "$HL" = $NW ] && [ -s $DEC/${F}_rl/score_text.json ]; then
    log "FT_AUTO_MERGE_CONTINUE $F: pair_lost $PL ($SH of $NW), all known mechanism, copy complete (coordinator pre-authorisation)"
    SET=$S TAG=${T}rlplem PL=$PL bash experiments/mist_w3po_v1/scripts/final_test_after_merge.sh > $E/after_merge_${S}_${T}rlplem.log 2>&1
    log "after-merge ${S}_${T}rlplem rc=$?"
  else
    log "FT_AUTO_MERGE_REFUSED $F: pair_lost $PL ($SH), known mechanism $OKM, hyp lines $HL / $NW: not continued, report"
  fi
}
if [ -n "${DRY_RUN:-}" ]; then
  for S in ali ami; do for T in ve1 d1v; do echo "$S $T: driver line: $(grep -E "ft_${S}_${T}_s3000" $E/chain_v2_$S.log | grep -E 'MERGE|merged' | cut -c1-120)"; done; done
  exit 0
fi
for S in ali ami; do for T in ve1 d1v; do one $S $T & done; done
wait
log "FT_AUTO_MERGE_WAITERS_DONE"
