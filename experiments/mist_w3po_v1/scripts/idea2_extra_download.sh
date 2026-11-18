#!/usr/bin/env bash
# Idea 2 (CA-Kalman) extra open-data downloads, same in-place range technique as voicecond_download.sh (that running
# script is not modified). Files land in data/open_voicecond_v1/idea2/raw/<dir>/<file>:
#   fsd50k/FSD50K.eval_audio.z01   first volume of the split FSD50K eval archive (the .zip in raw/fsd50k is volume 2 of 2,
#                                  which is why its zip integrity check failed); the .zip is linked next to it
#   cnceleb/cn-celeb_v2.tar.gz     CN-Celeb v1 (openslr 82), real noisy multi-genre speech for the unsupervised term
# Each file: Content-Length first, NPAR parallel HTTP range requests written in place (curl | dd conv=notrunc), size
# check, then <file>.DONE. A file whose .DONE exists is skipped; an existing file without .DONE is NOT overwritten.
# Sequential (FSD50K first). Marker data/open_voicecond_v1/idea2/raw/EXTRA_DONE (or EXTRA_FAILED).
#   [NPAR=8] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_extra_download.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2_extra_download.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=data/open_voicecond_v1/idea2/raw
NPAR=${NPAR:-8}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
FILES=(
  "fsd50k https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01?download=1 FSD50K.eval_audio.z01"
  "cnceleb https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz"
)
fetch() {   # <dir> <url> [<name>]
  local d=$R/$1 url=$2 name=${3:-$(basename "${2%%\?*}")} f size chunk i s e pids rc
  f=$d/$name
  mkdir -p $d
  [ -e $f.DONE ] && { log "skip (done): $f"; return 0; }
  [ -e $f ] && { log "REFUSE: $f exists without .DONE (not overwritten)"; return 1; }
  size=$(curl -sIL -m 60 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  [ -z "$size" ] || [ "$size" -lt 1 ] && { log "FAIL no content-length: $url"; return 1; }
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] $f <- $url ($size B, $NPAR ranges)"; return 0; fi
  log "start $f ($size B)"
  truncate -s $size $f
  chunk=$(( (size + NPAR - 1) / NPAR )); pids=""
  for i in $(seq 0 $((NPAR - 1))); do
    s=$((i * chunk)); e=$((s + chunk - 1)); [ $e -ge $size ] && e=$((size - 1)); [ $s -gt $e ] && continue
    ( for a in 1 2 3 4 5; do
        curl -sfL --retry 5 --retry-delay 5 -r $s-$e "$url" | dd of=$f bs=4M seek=$s oflag=seek_bytes conv=notrunc status=none \
          && exit 0
        sleep 10
      done; exit 1 ) &
    pids="$pids $!"
  done
  rc=0; for p in $pids; do wait $p || rc=1; done
  [ $rc = 0 ] && [ "$(stat -c %s $f)" = "$size" ] || { log "FAIL $f (rc $rc, size $(stat -c %s $f) / $size)"; return 1; }
  case $f in
    *.tar.gz|*.tgz) gzip -t $f 2>/dev/null || { log "FAIL gzip integrity $f"; return 1; };;
  esac
  date -u +%FT%TZ > $f.DONE
  log "done $f"
}
mkdir -p $R/fsd50k
# volume 2 of the split archive (downloaded by voicecond_download.sh) next to volume 1, as a symlink (never a copy)
[ -e $R/fsd50k/FSD50K.eval_audio.zip ] || [ -n "${DRY_RUN:-}" ] || \
  ln -s ../../../raw/fsd50k/FSD50K.eval_audio.zip $R/fsd50k/FSD50K.eval_audio.zip
fail=0
for spec in "${FILES[@]}"; do
  set -- $spec
  fetch "$@" || fail=1
done
[ -n "${DRY_RUN:-}" ] && exit 0
if [ $fail = 0 ]; then date -u +%FT%TZ > $R/EXTRA_DONE; log "EXTRA_DONE"; else date -u +%FT%TZ > $R/EXTRA_FAILED; log "EXTRA_FAILED"; fi
