#!/usr/bin/env bash
# Idea 1 M2 (prereg v3): AISHELL-1 (OpenSLR 33, Apache 2.0, 400 speakers) as the Chinese TRAINING speech -- AISHELL-3's
# test/wav folder holds 170 train speakers too, so no AISHELL-3 speaker is free of the ID-1 dev / test split.
# NPAR range requests written in place (curl | dd at the right offset, NO curl --retry: a failed range is re-fetched
# from its start), alternating between the openslr.org and the trmal.net mirror; size + gzip -t checked, then .DONE.
# Refuses an existing file without .DONE.
#   [DRY_RUN=1] [NPAR=12] bash experiments/mist_w3po_v1/scripts/idea1_fetch_aishell1.sh   (checked with bash -n / DRY_RUN only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
D=data/open_voicecond_v1/raw/aishell1
U1=https://www.openslr.org/resources/33/data_aishell.tgz
U2=https://openslr.trmal.net/resources/33/data_aishell.tgz
F=$D/data_aishell.tgz; NPAR=${NPAR:-12}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
[ -e $F.DONE ] && { log "skip (done)"; exit 0; }
[ -e $F ] && { log "REFUSE: $F exists without .DONE"; exit 1; }
size=$(curl -sIL -m 60 "$U1" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
[ -z "$size" ] && { log "FAIL no content-length"; exit 1; }
[ -n "${DRY_RUN:-}" ] && { log "[dry] $F ($size B, $NPAR ranges over 2 mirrors)"; exit 0; }
mkdir -p $D; log "start $F ($size B)"; truncate -s $size $F
chunk=$(( (size + NPAR - 1) / NPAR )); pids=""
for i in $(seq 0 $((NPAR - 1))); do
  s=$((i * chunk)); e=$((s + chunk - 1)); [ $e -ge $size ] && e=$((size - 1))
  u=$U1; [ $((i % 2)) = 1 ] && u=$U2
  ( for a in 1 2 3 4 5 6 7 8; do
      n=$(curl -sfL -m 7200 -r $s-$e "$u" | dd of=$F bs=4M seek=$s oflag=seek_bytes conv=notrunc status=none iflag=fullblock 2>&1; echo ${PIPESTATUS[0]})
      [ "$n" = 0 ] && exit 0
      log "range $s-$e attempt $a failed (curl rc $n), retry"; sleep 15
      [ $a = 4 ] && { [ "$u" = "$U1" ] && u=$U2 || u=$U1; }
    done; exit 1 ) &
  pids="$pids $!"
done
rc=0; for p in $pids; do wait $p || rc=1; done
[ $rc = 0 ] && [ "$(stat -c %s $F)" = "$size" ] || { log "FAIL $F rc=$rc"; exit 1; }
gzip -t $F 2>/dev/null || { log "FAIL gzip integrity $F"; exit 1; }
date -u +%FT%TZ > $F.DONE; log "done $F"
