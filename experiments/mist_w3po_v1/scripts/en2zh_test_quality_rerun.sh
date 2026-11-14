#!/usr/bin/env bash
# Rerun of en2zh_test.sh's UTMOS / SIM step (2026-11-14): the four talker_quality runs failed at 09:17 Prague because the
# night relaunch's environ had no TORCH_HOME -> torch.hub looked in /root/.cache, tried to download the UTMOS weights
# and hit an SSL error. The weights are in /opt/dlami/nvme/jasonleeeli/cache/torch/hub/checkpoints. Same commands as
# en2zh_test.sh lines 98-109 with TORCH_HOME set; each run waits for a card with >= 20 GiB free (as free_card there).
# A run is skipped when its quality.json exists; the paired UTMOS file is written only if absent.
#   setsid nohup bash experiments/mist_w3po_v1/scripts/en2zh_test_quality_rerun.sh > logs/mist_w3po_v1/en2zh/quality_rerun.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
SPD=eval_outputs/mist_w3po_v1/speech; TE=eval_outputs/mist_w3po_v1/crf/t_eval; Q=eval_outputs/mist_w3po_v1/talker_crf/quality
FT=ft_en2zh_en2zh_v2p_20261112T204539Z_s02000
O=$TE/${FT}__rlplem
log() { echo "$(date -u +%FT%TZ) $*"; }
free_card() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
  awk -F', ' '$2 - $3 >= 20480 {print $2 - $3, $1}' | sort -rn | head -1 | awk '{print $2}'; }
[ -e $TORCH_HOME/hub/checkpoints/utmos22_strong_step7459_v1.pt ] || { log "UTMOS weights missing under $TORCH_HOME"; exit 1; }
for nm in ${FT}__rlplem ${FT}__rlplem_tts2 ${FT}__seg ${FT}__seg_tts2; do
  base=${nm%_tts2}; Od=$TE/$base
  [ -s $Q/$nm/quality.json ] && { log "$nm: quality.json exists, skipped"; continue; }
  if [ -n "${DRY_RUN:-}" ]; then echo "would run talker_quality $nm (free card now: '$(free_card)')"; continue; fi
  until c=$(free_card); [ -n "$c" ]; do sleep 60; done
  mkdir -p $Q/$nm
  log "$nm on card $c"
  CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$nm/talker \
    --items $SPD/$nm/talker_in/items.jsonl --out $Q/$nm/quality.json > $Od/quality_${nm}.rerun.log 2>&1 \
    && log "$nm done" || log "UTMOS / SIM failed again: $nm (see $Od/quality_${nm}.rerun.log)"
done
[ -n "${DRY_RUN:-}" ] && { echo "would write $O/utmos_paired_c_vs_a.json (exists: $([ -s $O/utmos_paired_c_vs_a.json ] && echo y || echo n))"; exit 0; }
if [ -s $O/utmos_paired_c_vs_a.json ]; then log "paired file exists, kept"; else
  $PY -m experiments.mist_w3po_v1.evaluation.listen_paired_meeting --cand $Q/${FT}__rlplem $Q/${FT}__rlplem_tts2 \
    --ref a $Q/${FT}__seg $Q/${FT}__seg_tts2 > $O/utmos_paired_c_vs_a.json.tmp 2> $O/utmos_paired.rerun.log \
    && command mv -f $O/utmos_paired_c_vs_a.json.tmp $O/utmos_paired_c_vs_a.json && log "paired UTMOS done" || log "paired UTMOS failed again"
fi
log "QUALITY_RERUN_DONE"
