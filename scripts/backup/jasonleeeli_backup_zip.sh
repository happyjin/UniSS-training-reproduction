#!/usr/bin/env bash
# Parallel zip of /opt/dlami/nvme/jasonleeeli into standalone ~50 GiB volumes on S3 (store mode, zip64, streamed,
# no local copy). Files are listed in path order and packed greedily into volumes; a single file larger than the
# limit gets a volume of its own. Each volume is a complete zip: unzip all of them into one folder to restore the
# tree. Symlinks are stored as links (-y). Never deletes anything on S3; a finished volume is not redone on restart.
#
#   setsid nohup bash scripts/backup/jasonleeeli_backup_zip.sh > /dev/null 2>&1 &
#   state in /opt/dlami/nvme/jasonleeeli/s3_backup_zip; JOBS parallel streams (default 12)
set -u

SRC=/opt/dlami/nvme/jasonleeeli
DST=s3://stevehguo-ap-southeast-3/speech_model_backup/jasonleeeli/jasonleeeli_backup_zip
STATE=$SRC/s3_backup_zip
JOBS=${JOBS:-12}
LIMIT=$((50 * 1024 * 1024 * 1024))
S5=$SRC/tools/s5cmd_dl/bin/s5cmd
export AWS_REGION=ap-southeast-3 PATH="/usr/local/bin:/usr/bin:/bin:$PATH" SRC DST STATE S5

mkdir -p "$STATE/lists" "$STATE/done"
exec 9>"$STATE/lock"
flock -n 9 || { echo "another backup zip is running"; exit 1; }
exec >> "$STATE/zip.log" 2>&1
log() { echo "[$(date -u '+%F %T')] $*"; }

# 1. volume lists (made once; a restart reuses them so volume contents stay fixed)
if [ ! -e "$STATE/lists_ok" ]; then
    log "listing $SRC"
    ( cd "$SRC" && nice -n 10 find . -xdev \( -path ./s3_backup_sync -o -path ./s3_backup_zip \) -prune \
        -o \( -type f -o -type l \) -printf '%s\t%P\n' 2>/dev/null ) | LC_ALL=C sort -t $'\t' -k2,2 \
    | awk -F'\t' -v lim="$LIMIT" -v dir="$STATE/lists" '
        index($2, "\n") { next }
        { if (n == 0 || (sz > 0 && sz + $1 > lim)) { n++; sz = 0; f = sprintf("%s/part%04d.txt", dir, n) }
          print $2 > f; sz += $1; tot += $1 }
        END { printf "%d volumes, %.1f GiB\n", n, tot / 2^30 }' > "$STATE/lists_summary.txt"
    touch "$STATE/lists_ok"
fi
log "volumes: $(cat "$STATE/lists_summary.txt"), $JOBS parallel streams -> $DST/"

# 2. zip + upload each volume
one() {
    p=$(basename "$1" .txt)
    [ -e "$STATE/done/$p" ] && return 0
    t0=$(date +%s)
    ( cd "$SRC" && nice -n 10 zip -0 -q -y -@ - < "$1" ) \
        | "$S5" --log error pipe -c 16 -p 256 "$DST/jasonleeeli_backup.$p.zip" > "$STATE/lists/$p.out" 2>&1
    rc=("${PIPESTATUS[@]}")
    echo "[$(date -u '+%F %T')] $p: zip rc ${rc[0]}, upload rc ${rc[1]}, $(( $(date +%s) - t0 ))s"
    { [ "${rc[0]}" -eq 0 ] || [ "${rc[0]}" -eq 18 ]; } && [ "${rc[1]}" -eq 0 ] && touch "$STATE/done/$p"
}
export -f one
ls "$STATE"/lists/part*.txt | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {}
log "finished: $(ls "$STATE/done" | wc -l) of $(ls "$STATE"/lists/part*.txt | wc -l) volumes uploaded"
