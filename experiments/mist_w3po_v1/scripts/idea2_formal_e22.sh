#!/usr/bin/env bash
# Idea 2 M1b formal E2.2 (prereg v2 §C, §F) on the full 100 h: waits for data/open_voicecond_v1/idea2/FULL_CHAIN_DONE
# (idea2_full_chain_v2.sh: all streams preprocessed, v~ features, E2.0 re-check e20_all), then fits the contraction model
# on the FULL train split (prior / b_M from e20_all, kinds fixed from the pilot) and evaluates on the full test split
# (80 streams) on CPU. Outputs eval_outputs/mist_w3po_v1/voicecond_idea2/{shrink_fit_all, shrink_eval_all_test}.
# Marker data/open_voicecond_v1/idea2/FORMAL_E22_DONE (or FORMAL_E22_FAILED).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_formal_e22.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/formal_e22.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
I2=data/open_voicecond_v1/idea2
E=eval_outputs/mist_w3po_v1/voicecond_idea2
PY=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface CUDA_VISIBLE_DEVICES=
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then log "[dry] $*"; else log "run: $*"; eval "$@"; fi; }
if [ -z "${DRY_RUN:-}" ]; then
  until [ -e $I2/FULL_CHAIN_DONE ]; do
    ls $I2/FULL_CHAIN_FAILED_* > /dev/null 2>&1 && { log "full chain failed"; date -u +%FT%TZ >> $I2/FORMAL_E22_FAILED; exit 1; }
    sleep 30
  done
fi
run "$PY -I experiments/mist_w3po_v1/voicecond/idea2_shrink.py fit --which all --e20 $E/e20_all --out $E/shrink_fit_all" \
  && run "$PY -I experiments/mist_w3po_v1/voicecond/idea2_shrink.py eval --which all --split test --e20 $E/e20_all --fit $E/shrink_fit_all --out $E/shrink_eval_all_test" \
  || { log FAILED; [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ >> $I2/FORMAL_E22_FAILED; exit 1; }
[ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ >> $I2/FORMAL_E22_DONE
log FORMAL_E22_DONE
