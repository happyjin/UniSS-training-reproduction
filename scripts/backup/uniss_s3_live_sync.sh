#!/usr/bin/env bash
# Near-real-time backup of projects/UniSS to S3, beside the full-tree loop (/usr/local/bin/jasonleeeli-s3-backup.sh,
# ~3 h per round). Every INTERVAL s: upload the regular files modified since the previous scan (find -newer a stamp,
# one s5cmd batch). Never deletes anything on S3. Failed uploads are retried next round if the file still exists.
# Every OUTSIDE_EVERY rounds, files changed elsewhere under /opt/dlami/nvme/jasonleeeli are summarised (per top-two
# directories) into outside_changes.log; they are not uploaded here, the full-tree loop covers them.
#
#   setsid nohup bash scripts/backup/uniss_s3_live_sync.sh > /dev/null 2>&1 &
#   first round starts from START (default: the last full-tree round's start), state in /opt/dlami/nvme/jasonleeeli/s3_live_sync
set -u

ROOT=/opt/dlami/nvme/jasonleeeli
SRC=$ROOT/projects/UniSS
DST=s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli/projects/UniSS
STATE=$ROOT/s3_live_sync
INTERVAL=${INTERVAL:-60}
OUTSIDE_EVERY=${OUTSIDE_EVERY:-30}
START=${START:-"2026-10-30 04:20:40"}
export AWS_REGION=ap-southeast-3 PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$STATE"
exec 9>"$STATE/lock"
flock -n 9 || { echo "another live sync is running"; exit 1; }
exec >> "$STATE/sync.log" 2>&1
log() { echo "[$(date -u '+%F %T')] $*"; }

[ -e "$STATE/stamp" ] || touch -d "$START UTC" "$STATE/stamp"
[ -e "$STATE/outside_stamp" ] || touch -d "$START UTC" "$STATE/outside_stamp"
touch "$STATE/retry.txt"
log "live sync started: $SRC -> $DST every ${INTERVAL}s, from $(date -u -r "$STATE/stamp" '+%F %T')"

round=0
while true; do
    round=$((round + 1))
    touch "$STATE/stamp.next"                       # taken before the scan: files written during it go to the next round
    cmds="$STATE/cmds.txt"; : > "$cmds"
    n=0; skipped=0
    while IFS= read -r -d '' f; do
        case "$f" in *'"'*|*$'\n'*) skipped=$((skipped + 1)); echo "$f" >> "$STATE/unsupported_names.txt"; continue ;; esac
        echo "cp --raw \"$f\" \"$DST/${f#"$SRC"/}\"" >> "$cmds"; n=$((n + 1))
    done < <(nice -n 10 find "$SRC" -xdev -type f -newer "$STATE/stamp" -print0 2>/dev/null)
    r=0
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        echo "cp --raw \"$f\" \"$DST/${f#"$SRC"/}\"" >> "$cmds"; r=$((r + 1))
    done < "$STATE/retry.txt"
    : > "$STATE/retry.txt"
    if [ $((n + r)) -gt 0 ]; then
        t0=$(date +%s)
        nice -n 10 ionice -c3 s5cmd --log error --numworkers 64 run "$cmds" > "$STATE/s5cmd_out.txt" 2>&1
        rc=$?
        fails=$(grep -c '^ERROR' "$STATE/s5cmd_out.txt")
        grep '^ERROR "cp ' "$STATE/s5cmd_out.txt" | sed -E 's#^ERROR "cp --raw=true (.*) s3://stevehguo-ap-southeast-3/.*$#\1#' \
            | sort -u > "$STATE/retry.txt"
        [ "$fails" -gt 0 ] && grep '^ERROR' "$STATE/s5cmd_out.txt" | tail -3 | cut -c1-400
        log "round $round: $n changed + $r retried, $fails failed (rc $rc, $(( $(date +%s) - t0 ))s), $skipped unsupported names"
    fi
    command mv -f "$STATE/stamp.next" "$STATE/stamp"

    if [ $((round % OUTSIDE_EVERY)) -eq 1 ]; then
        touch "$STATE/outside_stamp.next"
        out=$(nice -n 10 find "$ROOT" -xdev \( -path "$SRC" -o -path "$STATE" \) -prune -o -type f -newer "$STATE/outside_stamp" \
              -printf '%P\n' 2>/dev/null | cut -d/ -f1-2 | sort | uniq -c | sort -rn)
        if [ -n "$out" ]; then
            { echo "[$(date -u '+%F %T')] changed outside UniSS since $(date -u -r "$STATE/outside_stamp" '+%F %T') (count, top dirs):"
              echo "$out"; } >> "$STATE/outside_changes.log"
        fi
        command mv -f "$STATE/outside_stamp.next" "$STATE/outside_stamp"
    fi
    sleep "$INTERVAL"
done
