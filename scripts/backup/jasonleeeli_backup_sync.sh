#!/usr/bin/env bash
# Whole-tree backup of /opt/dlami/nvme/jasonleeeli to S3 under jasonleeeli_backup/. Never deletes anything on S3.
#   1. full pass: s5cmd sync (no --delete, size/mtime check, 256 workers) of the whole tree;
#   2. once after the first full pass: a store-mode (-0) zip of the tree streamed to $ZIP (zip64, no local copy);
#   3. then every INTERVAL s: upload the regular files modified since the previous scan (find -newer a stamp,
#      one s5cmd batch); failed uploads are retried next round if the file still exists.
#
#   setsid nohup bash scripts/backup/jasonleeeli_backup_sync.sh > /dev/null 2>&1 &
#   state in /opt/dlami/nvme/jasonleeeli/s3_backup_sync; SKIP_ZIP=1 skips step 2
set -u

SRC=/opt/dlami/nvme/jasonleeeli
DST=s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli/jasonleeeli_backup
ZIP=s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli/jasonleeeli_backup.zip
STATE=$SRC/s3_backup_sync
INTERVAL=${INTERVAL:-300}
WORKERS=${WORKERS:-256}
S5=$SRC/tools/s5cmd_dl/bin/s5cmd
export AWS_REGION=ap-southeast-3 PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

mkdir -p "$STATE"
exec 9>"$STATE/lock"
flock -n 9 || { echo "another backup sync is running"; exit 1; }
exec >> "$STATE/sync.log" 2>&1
log() { echo "[$(date -u '+%F %T')] $*"; }

# 1. full pass (repeated until it finishes without errors, so a restart resumes it)
if [ ! -e "$STATE/full_ok" ]; then
    touch "$STATE/stamp.full"                       # files written during the full pass go to the first incremental round
    for attempt in 1 2 3; do
        t0=$(date +%s)
        log "full pass attempt $attempt: $SRC -> $DST ($WORKERS workers)"
        nice -n 10 "$S5" --log error --numworkers "$WORKERS" sync --no-follow-symlinks --exclude 's3_backup_sync/*' \
            "$SRC/" "$DST/" > "$STATE/full_out.txt" 2>&1
        rc=$?
        fails=$(grep -c '^ERROR' "$STATE/full_out.txt")
        log "full pass attempt $attempt: rc $rc, $fails errors, $(( $(date +%s) - t0 ))s"
        [ "$fails" -gt 0 ] && grep '^ERROR' "$STATE/full_out.txt" | tail -3 | cut -c1-400
        [ "$rc" -eq 0 ] && [ "$fails" -eq 0 ] && break
    done
    command mv -f "$STATE/stamp.full" "$STATE/stamp"
    touch "$STATE/full_ok"
fi

# 2. zip of the tree (same content as the S3 copy), streamed and cut into 50 GiB volumes $ZIP.001, .002, ...
#    (no local copy); restore: cat jasonleeeli_backup.zip.* > jasonleeeli_backup.zip, or open .zip.001 in 7-Zip
if [ "${SKIP_ZIP:-0}" != 1 ] && [ ! -e "$STATE/zip_ok" ]; then
    t0=$(date +%s)
    log "zip: $SRC -> $ZIP.NNN (50 GiB volumes)"
    export S5 ZIP
    ( cd "$SRC" && nice -n 10 zip -0 -r -q -y - . -x 's3_backup_sync/*' ) \
        | split -b 50G -d -a 3 --numeric-suffixes=1 \
            --filter='"$S5" --log error pipe -c 16 -p 256 "$ZIP.$(echo "$FILE" | sed s/^x//)"' - x \
        > "$STATE/zip_out.txt" 2>&1
    rc=("${PIPESTATUS[@]}")
    log "zip: zip rc ${rc[0]}, split/upload rc ${rc[1]}, $(( $(date +%s) - t0 ))s"
    { [ "${rc[0]}" -eq 0 ] || [ "${rc[0]}" -eq 18 ]; } && [ "${rc[1]}" -eq 0 ] && touch "$STATE/zip_ok"
fi

# 3. incremental rounds
touch "$STATE/retry.txt"
log "incremental sync every ${INTERVAL}s from $(date -u -r "$STATE/stamp" '+%F %T')"
round=0
while true; do
    round=$((round + 1))
    touch "$STATE/stamp.next"
    cmds="$STATE/cmds.txt"; : > "$cmds"
    n=0; skipped=0
    while IFS= read -r -d '' f; do
        case "$f" in *'"'*|*$'\n'*) skipped=$((skipped + 1)); echo "$f" >> "$STATE/unsupported_names.txt"; continue ;; esac
        echo "cp --raw \"$f\" \"$DST/${f#"$SRC"/}\"" >> "$cmds"; n=$((n + 1))
    done < <(nice -n 10 find "$SRC" -xdev -path "$STATE" -prune -o -type f -newer "$STATE/stamp" -print0 2>/dev/null)
    r=0
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        echo "cp --raw \"$f\" \"$DST/${f#"$SRC"/}\"" >> "$cmds"; r=$((r + 1))
    done < "$STATE/retry.txt"
    : > "$STATE/retry.txt"
    if [ $((n + r)) -gt 0 ]; then
        t0=$(date +%s)
        nice -n 10 "$S5" --log error --numworkers 64 run "$cmds" > "$STATE/s5cmd_out.txt" 2>&1
        rc=$?
        fails=$(grep -c '^ERROR' "$STATE/s5cmd_out.txt")
        grep '^ERROR "cp ' "$STATE/s5cmd_out.txt" | sed -E 's#^ERROR "cp --raw=true (.*) s3://stevehguo-ap-southeast-3/.*$#\1#' \
            | sort -u > "$STATE/retry.txt"
        [ "$fails" -gt 0 ] && grep '^ERROR' "$STATE/s5cmd_out.txt" | tail -3 | cut -c1-400
        log "round $round: $n changed + $r retried, $fails failed (rc $rc, $(( $(date +%s) - t0 ))s), $skipped unsupported names"
    fi
    command mv -f "$STATE/stamp.next" "$STATE/stamp"
    sleep "$INTERVAL"
done
