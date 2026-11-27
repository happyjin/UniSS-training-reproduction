# portfolio 2026-11-27: card pool, sourced AFTER lib.sh (overrides launch). launch <requested cards> <name> <cmd...> ignores the
# requested card numbers: it takes as many free cards from POOL_CARDS (default 0-7) as were requested, under per-card flock locks
# (logs/sst_hsmm_v1/cardlocks/card<g>.lock; held by the job's flock processes and inherited by the job and its children, so a
# card stays locked until every process of the job, orphan engines included, has exited), and rewrites every
# "CUDA_VISIBLE_DEVICES=<requested>" (whole arg or inside a bash -c string) to the assigned cards. Placeholders of the assigned
# cards are stopped by PID inside the job (pool_run.sh). Waits (polling 20 s) while not enough cards are free. Prints the pid.
POOL_CARDS=${POOL_CARDS:-"0 1 2 3 4 5 6 7"}
LOCKD=/opt/dlami/nvme/jasonleeeli/projects/UniSS/logs/sst_hsmm_v1/cardlocks
[ -n "${DRY:-}" ] || mkdir -p "$LOCKD"
launch() {
  local gs=$1 name=$2; shift 2
  local lf=$LOGD/$name.log n g p rc asg waited=0 a
  if [ -n "${DRY:-}" ]; then printf '[dry] pool(%s) %s: ' "$gs" "$name" >&2; printf '%q ' "$@" >&2; echo >&2; echo DRY; return 0; fi
  [ -e "$lf" ] && die "log exists: $lf"
  n=$(echo ${gs//,/ } | wc -w)
  while true; do
    local got=()
    for g in $POOL_CARDS; do
      flock -n "$LOCKD/card$g.lock" true 2>/dev/null && got+=("$g")
      [ ${#got[@]} -ge "$n" ] && break
    done
    if [ ${#got[@]} -ge "$n" ]; then
      asg=$(IFS=,; echo "${got[*]}")
      local args=() fl=()
      for a in "$@"; do
        a="${a//CUDA_VISIBLE_DEVICES=$gs /CUDA_VISIBLE_DEVICES=$asg }"
        [ "$a" = "CUDA_VISIBLE_DEVICES=$gs" ] && a="CUDA_VISIBLE_DEVICES=$asg"
        args+=("$a")
      done
      for g in "${got[@]}"; do fl+=(flock -n -E 75 "$LOCKD/card$g.lock"); done
      setsid nohup "${fl[@]}" bash experiments/sst_hsmm_v1/pool_run.sh "$asg" "${args[@]}" > "$lf" 2>&1 < /dev/null 8>&- 9>&- &
      p=$!
      sleep 3
      if ! kill -0 "$p" 2>/dev/null; then
        wait "$p"; rc=$?
        if [ "$rc" = 75 ]; then command mv -f "$lf" "$lf.race$(date +%s)"; continue; fi
      fi
      echo "$asg $p $name" >> "$PIDS"
      log "$name cards $asg pid $p (pool)" >&2
      echo "$p"
      return 0
    fi
    [ $waited = 0 ] && log "$name: waiting for $n free card(s)" >&2
    waited=1
    sleep 20
  done
}
