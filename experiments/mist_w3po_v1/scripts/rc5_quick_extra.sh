#!/usr/bin/env bash
# RC5's extra quick-check report (thinker_rc5_prereg_v1 §4, coordinator 2026-11-12 ~03:05): once t_after_train_rl.sh
# has written the dev5 GO verdict (go4.json), the CPU diagnosis (evaluation/early_commit_diag.py --light) of RC5's and
# T5's dev5 decodes against A': swap-in gains of the >= 10 s windows and of the >= 2-speaker-output windows, BLEU
# components, unmatched tokens, words per segment, within / cross-segment n-gram precision. Not a GO line. CPU only.
#   TAG=rc5 [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/rc5_quick_extra.sh   (run detached: setsid nohup ... &)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
TAG=${TAG:-rc5}
export PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES=
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
DEC=eval_outputs/mist_w3po_v1/decode
QO=eval_outputs/mist_w3po_v1/crf/t_eval/dev5_${TAG}_s3000
CMD="$PY -m experiments.mist_w3po_v1.evaluation.early_commit_diag --light --run $DEC/dev5_${TAG}_s3000 --run $DEC/dev5_t5_s3000 --out $QO/early_commit_diag_light.json"
if [ -n "${DRY_RUN:-}" ]; then echo "$CMD"; exit 0; fi
until [ -e $QO/go4.json ]; do sleep 60; done
$CMD > $QO/early_commit_diag_light.log 2>&1
echo "$(date -u +%FT%TZ) RC5_EXTRA_EXIT rc=$?" >> $QO/early_commit_diag_light.log
