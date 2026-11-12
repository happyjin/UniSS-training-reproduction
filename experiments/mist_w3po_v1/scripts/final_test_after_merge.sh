#!/usr/bin/env bash
# Final test: continue ONE read-level PL-EM speech chain after the registered merge rule stopped it (FT_MERGE_STOP,
# pair_lost > 2), only on the coordinator's decision (logged in LOG): the merged copy <src>_rl must be complete (hyp lines
# = windows of the set, score_text.json, merge_read_segments.json) and its pair_lost must equal PL (the value the
# decision was made on). Then the same orchestrator command / post-hoc measures as final_test_chain_v2.sh.
#   SET=ami|ali TAG=aprlplem|ve1rlplem|d1vrlplem PL=<pair_lost> [DRY_RUN=1] bash final_test_after_merge.sh
# Run detached: setsid nohup bash .../final_test_after_merge.sh > logs/mist_w3po_v1/final_test/after_merge_<set>_<tag>.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
SET=${SET:?}; TAG=${TAG:?}; PL=${PL:?}
case $SET in ami) NW=3973; P0=20100; WS=zh2en;; ali) NW=4065; P0=20200; WS=ali;; *) exit 1;; esac
case $TAG in aprlplem) T=aprime; PO=10;; ve1rlplem) T=ve1; PO=60;; d1vrlplem) T=d1v; PO=70;; *) exit 1;; esac
export PYTHONPATH=. TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/final_test
DEC=eval_outputs/mist_w3po_v1/decode
O=eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/$SET
WIN=eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_$WS.jsonl
F=ft_${SET}_${T}_s3000; N=${F}__rlplem
CMD="$PY experiments/mist_w3po_v1/scripts/final_test_orch.py --set $SET --win $WIN --window 1 --cap135 --run $E/${SET}_$TAG --name $N --ckpt - --speech-from ${F}_rl --talker plem --scorer-port0 $((P0 + PO))"
log() { echo "$(date -u +%FT%TZ) $*"; }
jget() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], 'missing'))" "$1" "$2" 2>/dev/null || echo missing; }
MJ=$DEC/${F}_rl/merge_read_segments.json
GOT=$(jget $MJ pair_lost); HL=$(wc -l < $DEC/${F}_rl/hyp.jsonl 2>/dev/null || echo 0)
OK=1
[ "$GOT" = "$PL" ] || { OK=0; echo "pair_lost $GOT != $PL"; }
[ "$HL" = $NW ] && [ -s $DEC/${F}_rl/score_text.json ] || { OK=0; echo "merged copy incomplete (hyp lines $HL)"; }
for d in $E/${SET}_$TAG $DEC/$N $DEC/${N}_tts2; do [ -e $d ] && { OK=0; echo "exists: $d"; }; done
if [ -n "${DRY_RUN:-}" ]; then echo "ok $OK; $CMD"; $CMD --dry | tail -n 1; exit 0; fi
[ $OK = 1 ] || { log "not started"; exit 1; }
setsid nohup $CMD < /dev/null > $E/${SET}_$TAG.launch.log 2>&1 &
log "launched ${SET}_$TAG pid $! (coordinator decision, pair_lost $PL known deviation): $CMD"
sleep 20
[ -e $E/${SET}_$TAG/orch.log ] || { log "FT_LAUNCH_FAILED ${SET}_$TAG"; exit 1; }
until [ -e $E/${SET}_$TAG/events/DONE ]; do sleep 30; done
if grep -q 'DONE; failed/skipped \[\]' $E/${SET}_$TAG/orch.log; then log "FT_CHAIN_DONE ${SET}_$TAG"; else log "FT_CHAIN_FAILED ${SET}_$TAG"; fi
SET=$SET NAME=$N bash experiments/mist_w3po_v1/scripts/final_test_post.sh > $O/post_$N.log 2>&1
log "post-hoc $N rc=$?"
log "FT_AFTER_MERGE_DONE ${SET}_$TAG"
