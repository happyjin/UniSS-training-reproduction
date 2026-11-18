#!/usr/bin/env bash
# FSD50K eval audio is a SPLIT zip (FSD50K.eval_audio.z01 + .zip); voicecond_download.sh fetched only the .zip, whose
# integrity test then failed ("spans multiple disks"). This fetches the missing .z01 part (8 range requests written in
# place; NO curl --retry, a failed range is re-fetched from its start with dd seek), checks the size, then merges both
# parts with `zip -s 0` into FSD50K.eval_audio.merged.zip, tests it (unzip -tq) and writes <file>.DONE markers.
# Never overwrites: refuses if a target exists without .DONE.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea1_fetch_fsd50k_z01.sh   (checked with bash -n / DRY_RUN only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=data/open_voicecond_v1/raw/fsd50k
URL='https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01?download=1'
F=$D/FSD50K.eval_audio.z01; M=$D/FSD50K.eval_audio.merged.zip; NPAR=8
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
if [ ! -e $F.DONE ]; then
  [ -e $F ] && { log "REFUSE: $F exists without .DONE"; exit 1; }
  size=$(curl -sIL -m 60 "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  [ -z "$size" ] && { log "FAIL no content-length"; exit 1; }
  [ -n "${DRY_RUN:-}" ] && { log "[dry] $F ($size B) then merge -> $M"; exit 0; }
  log "start $F ($size B)"; truncate -s $size $F
  chunk=$(( (size + NPAR - 1) / NPAR )); pids=""
  for i in $(seq 0 $((NPAR - 1))); do
    s=$((i * chunk)); e=$((s + chunk - 1)); [ $e -ge $size ] && e=$((size - 1))
    ( for a in 1 2 3 4 5 6 7 8; do
        n=$(curl -sfL -m 3600 -r $s-$e "$URL" | dd of=$F bs=4M seek=$s oflag=seek_bytes conv=notrunc status=none iflag=fullblock 2>&1; echo ${PIPESTATUS[0]})
        [ "$n" = 0 ] && exit 0
        log "range $s-$e attempt $a failed (curl rc $n), retry"; sleep 15
      done; exit 1 ) &
    pids="$pids $!"
  done
  rc=0; for p in $pids; do wait $p || rc=1; done
  [ $rc = 0 ] && [ "$(stat -c %s $F)" = "$size" ] || { log "FAIL $F rc=$rc"; exit 1; }
  date -u +%FT%TZ > $F.DONE; log "done $F"
fi
[ -n "${DRY_RUN:-}" ] && { log "[dry] merge -> $M"; exit 0; }
[ -e $M.DONE ] && { log "skip merge (done)"; exit 0; }
[ -e $M ] && { log "REFUSE: $M exists without .DONE"; exit 1; }
zip -q -s 0 $D/FSD50K.eval_audio.zip --out $M || { log "FAIL zip -s 0"; exit 1; }
unzip -tq $M > /dev/null || { log "FAIL unzip -t $M"; exit 1; }
date -u +%FT%TZ > $M.DONE; log "done $M"
