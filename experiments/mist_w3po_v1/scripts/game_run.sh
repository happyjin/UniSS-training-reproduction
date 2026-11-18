#!/usr/bin/env bash
# Game (HOK esports commentary) listening demo (reports/mist_w3po_v1/game_multispk_demo_plan.zh-CN.md §3): baseline B
# (joint_v8h6k merged s06000 + identity, the v11 / Table 1 B row = alieval_run.sh row_b) on the 10 game excerpts through
# game_orch.py. Per direction, in parallel: text decode (window 1, --cap135) -> T1c speech (frozen T1c, no merge, no
# PL-EM; two TTS seeds). Run dirs logs/mist_w3po_v1/game/<dir>{,_t1c}; marker logs/mist_w3po_v1/game/GAME_<dir>_DONE
# (or GAME_<dir>_FAILED). Refuses to overwrite a run dir.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/game_run.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/game/run.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
GL=logs/mist_w3po_v1/game
GO="$PY $SC/game_orch.py"
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
IDN=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/identity
mkdir -p $GL
run_orch() {   # <run dir> <command...>
  local R=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "$(basename $R): $*"; "$@" --dry | tail -1; return 0; fi
  [ -e $R ] && { log "$R exists (not overwritten)"; return 1; }
  setsid nohup "$@" < /dev/null > $R.launch.log 2>&1 9>&- &
  log "$(basename $R) orchestrator pid $!"
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$(basename $R) failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
}
chain() {   # <dir> <set>
  local d=$1 s=$2 n=game_b_$1
  if run_orch $GL/$d $GO --dir $d --set $s --base $B2 --run $GL/$d --name $n --ckpt $IDN --talker none --window 1 --cap135 \
     && run_orch $GL/${d}_t1c $GO --dir $d --set $s --base $B2 --run $GL/${d}_t1c --name ${n}__t1c --ckpt - \
          --speech-from $n --talker t1c --window 1 --cap135; then
    [ -n "${DRY_RUN:-}" ] || { date -u +%FT%TZ > $GL/GAME_${d}_DONE; log "GAME_${d}_DONE"; }
  else
    [ -n "${DRY_RUN:-}" ] || { date -u +%FT%TZ > $GL/GAME_${d}_FAILED; log "GAME_${d}_FAILED"; }
  fi
}
if [ -n "${DRY_RUN:-}" ]; then chain zh2en ali; chain en2zh test; exit 0; fi
chain zh2en ali &
sleep 60
chain en2zh test &
wait
log "GAME_RUN_EXIT"
