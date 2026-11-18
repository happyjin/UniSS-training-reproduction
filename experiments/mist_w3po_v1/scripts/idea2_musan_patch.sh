#!/usr/bin/env bash
# Idea 2: a private complete copy of MUSAN without waiting for one throttled range of the shared download.
# voicecond_download.sh's range 2 of musan.tar.gz runs at ~40 KB/s (all other ranges finished). This script does not
# touch the shared file: it copies data/open_voicecond_v1/raw/musan/musan.tar.gz to
# data/open_voicecond_v1/idea2/raw/musan/musan.tar.gz, downloads bytes [FROM, TO] (the unwritten tail of that range, FROM
# taken from the shared dd's progress minus a 64 MiB safety margin) with NPAR parallel range requests written in place
# into the private copy, then gzip -t; success -> <copy>.DONE. Refuses to overwrite an existing copy.
#   FROM=<byte> TO=<byte> [NPAR=8] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_musan_patch.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
SRCF=data/open_voicecond_v1/raw/musan/musan.tar.gz
D=data/open_voicecond_v1/idea2/raw/musan
F=$D/musan.tar.gz
URL=https://www.openslr.org/resources/17/musan.tar.gz
NPAR=${NPAR:-8}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
: "${FROM:?}" "${TO:?}"
[ -e $F ] && { log "REFUSE: $F exists"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then log "[dry] cp $SRCF $F; ranges $FROM-$TO in $NPAR parts; gzip -t"; exit 0; fi
mkdir -p $D
cp --sparse=never $SRCF $F || { log "FAIL copy"; exit 1; }
size=$((TO - FROM + 1)); chunk=$(( (size + NPAR - 1) / NPAR )); pids=""
for i in $(seq 0 $((NPAR - 1))); do
  s=$((FROM + i * chunk)); e=$((s + chunk - 1)); [ $e -gt $TO ] && e=$TO; [ $s -gt $e ] && continue
  ( for a in 1 2 3 4 5; do
      curl -sfL --retry 5 --retry-delay 5 -r $s-$e "$URL" | dd of=$F bs=4M seek=$s oflag=seek_bytes conv=notrunc status=none && exit 0
      sleep 5
    done; exit 1 ) &
  pids="$pids $!"
done
rc=0; for p in $pids; do wait $p || rc=1; done
[ $rc = 0 ] || { log "FAIL ranges"; exit 1; }
gzip -t $F 2>/dev/null || { log "FAIL gzip integrity $F"; exit 1; }
date -u +%FT%TZ > $F.DONE
log "done $F"
