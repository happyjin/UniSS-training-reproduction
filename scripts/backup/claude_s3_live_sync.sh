#!/usr/bin/env bash
# Every INTERVAL s (default 180), mirror Claude Code's history and memory under /root/.claude (projects/, file-history/,
# skills/, plugins/, settings.json; never .credentials.json) to S3 claude_backup/root_dot_claude with s5cmd sync:
# uploads new or changed files only, never deletes on S3. Log and lock in /opt/dlami/nvme/jasonleeeli/s3_live_sync.
#
#   setsid nohup bash scripts/backup/claude_s3_live_sync.sh > /dev/null 2>&1 &
set -u

SRC=/root/.claude
DST=s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli/claude_backup/root_dot_claude
STATE=/opt/dlami/nvme/jasonleeeli/s3_live_sync
INTERVAL=${INTERVAL:-180}
export AWS_REGION=ap-southeast-3 PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$STATE"
exec 8>"$STATE/claude.lock"
flock -n 8 || { echo "another claude sync is running"; exit 1; }
exec >> "$STATE/claude_sync.log" 2>&1
log() { echo "[$(date -u '+%F %T')] $*"; }
log "claude sync started: $SRC -> $DST every ${INTERVAL}s"

while true; do
    t0=$(date +%s); n=0; bad=0
    for d in projects file-history skills plugins; do
        [ -d "$SRC/$d" ] || continue
        out=$(nice -n 10 s5cmd --log error sync "$SRC/$d/" "$DST/$d/" 2>&1); rc=$?
        [ $rc -eq 0 ] || { bad=$((bad + 1)); echo "$out" | tail -3 | cut -c1-400; }
    done
    out=$(nice -n 10 s5cmd --log error cp "$SRC/settings.json" "$DST/settings.json" 2>&1) || { bad=$((bad + 1)); echo "$out" | tail -2; }
    log "round done in $(( $(date +%s) - t0 ))s, $bad failed parts"
    sleep "$INTERVAL"
done
