#!/usr/bin/env bash
# OpenBroadcast-SI v1, OOD-1 background corpora (plan §7.2e; corpora never used by Idea 1 / Idea 2):
#   FMA-small (music bed; per-track CC licences in fma_metadata tracks.csv)   os.unil.cloud.switch.ch/fma
#   DEMAND 16 kHz, 5 crowded public environments (ambience bursts; CC BY 4.0)  zenodo 1227121
#   ESC-50 (sparse SFX; CC BY-NC 3.0; clips whose Freesound id is in FSD50K are excluded at build time)  GitHub
# -> data/open_voicecond_v1/raw_ood/{fma,demand,esc50}/<file> + <file>.DONE (size checked; zip integrity tested).
# Large files: 8 parallel range requests written in place (dd seek), a failed range is re-fetched from its start.
# Never overwrites: a target without .DONE is refused.
#   [DRY_RUN=1] bash scripts/voicecond_bench_download_ood.sh     (checked with bash -n / DRY_RUN only)
# Run detached: setsid nohup bash scripts/voicecond_bench_download_ood.sh > logs/mist_w3po_v1/voicecond_bench/download_ood.log 2>&1 < /dev/null 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=data/open_voicecond_v1/raw_ood
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
fetch() {   # <url> <dest> <nparallel>
  local url=$1 F=$2 np=$3 size chunk pids i s e rc
  [ -e $F.DONE ] && { log "skip (done) $F"; return 0; }
  [ -e $F ] && { log "REFUSE: $F exists without .DONE"; return 1; }
  size=$(curl -sIL -m 60 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  [ "$size" = 0 ] && size=
  [ -z "$size" ] && [ "$np" -gt 1 ] && { log "FAIL no content-length $url"; return 1; }
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] $F ($size B) <- $url"; return 0; fi
  mkdir -p $(dirname $F)
  log "start $F ($size B)"
  if [ "$np" -le 1 ]; then
    curl -sfL -m 7200 --retry 5 -o $F "$url" || { log "FAIL curl $F"; return 1; }
  else
    truncate -s $size $F
    chunk=$(( (size + np - 1) / np )); pids=""
    for i in $(seq 0 $((np - 1))); do
      s=$((i * chunk)); e=$((s + chunk - 1)); [ $e -ge $size ] && e=$((size - 1))
      ( for a in 1 2 3 4 5 6 7 8; do
          n=$(curl -sfL -m 7200 -r $s-$e "$url" | dd of=$F bs=4M seek=$s oflag=seek_bytes conv=notrunc status=none iflag=fullblock 2>&1; echo ${PIPESTATUS[0]})
          [ "$n" = 0 ] && exit 0
          log "range $s-$e attempt $a failed (curl rc $n), retry"; sleep 15
        done; exit 1 ) &
      pids="$pids $!"
    done
    rc=0; for p in $pids; do wait $p || rc=1; done
    [ $rc = 0 ] || { log "FAIL ranges $F"; return 1; }
  fi
  [ -z "$size" ] || [ "$(stat -c %s $F)" = "$size" ] || { log "FAIL size $F"; return 1; }
  case $F in *.zip) unzip -tq $F > /dev/null 2>&1 || { log "FAIL zip integrity $F"; return 1; };; esac
  date -u +%FT%TZ > $F.DONE; log "done $F"
}
fail=0
fetch 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip' $R/fma/fma_metadata.zip 4 || fail=1
for e in PCAFETER PRESTO PSTATION SPSQUARE TMETRO; do
  fetch "https://zenodo.org/records/1227121/files/${e}_16k.zip?download=1" $R/demand/${e}_16k.zip 1 || fail=1
done
fetch 'https://github.com/karoldvl/ESC-50/archive/master.zip' $R/esc50/ESC-50-master.zip 1 || fail=1
fetch 'https://os.unil.cloud.switch.ch/fma/fma_small.zip' $R/fma/fma_small.zip 8 || fail=1
[ -n "${DRY_RUN:-}" ] && exit 0
if [ $fail = 0 ]; then date -u +%FT%TZ > $R/ALL_DONE; log ALL_DONE; else date -u +%FT%TZ > $R/SOME_FAILED; log SOME_FAILED; fi
