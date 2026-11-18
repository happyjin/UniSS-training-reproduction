#!/usr/bin/env bash
# Open-data downloads for the voice-condition ideas (reports/mist_w3po_v1/game_voice_condition_ideas_plan.zh-CN.md §7):
# MUSAN, FSD50K eval (+ ground truth / metadata), AISHELL-3, LibriSpeech test-clean / dev-clean / train-clean-100 /
# train-clean-360 -> data/open_voicecond_v1/raw/<name>/<file>. Each file: Content-Length read first, N parallel HTTP
# range requests written in place (curl | dd conv=notrunc at the right offset, no part files), size checked, then
# <file>.DONE. A file whose .DONE exists is skipped; an existing incomplete file is NOT overwritten (logged, skipped).
# Two files at a time. Marker data/open_voicecond_v1/raw/DOWNLOAD_DONE (or DOWNLOAD_FAILED).
#   [NPAR=8] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/voicecond_download.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/download.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=data/open_voicecond_v1/raw
NPAR=${NPAR:-8}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
FILES=(
  "musan https://www.openslr.org/resources/17/musan.tar.gz"
  "fsd50k https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip?download=1 FSD50K.eval_audio.zip"
  "fsd50k https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1 FSD50K.ground_truth.zip"
  "fsd50k https://zenodo.org/records/4060432/files/FSD50K.metadata.zip?download=1 FSD50K.metadata.zip"
  "librispeech https://www.openslr.org/resources/12/test-clean.tar.gz"
  "librispeech https://www.openslr.org/resources/12/dev-clean.tar.gz"
  "aishell3 https://www.openslr.org/resources/93/data_aishell3.tgz"
  "librispeech https://www.openslr.org/resources/12/train-clean-100.tar.gz"
  "librispeech https://www.openslr.org/resources/12/train-clean-360.tar.gz"
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
          && [ "$(stat -c %s $f)" -ge $((e + 1)) ] && exit 0
        sleep 10
      done; exit 1 ) &
    pids="$pids $!"
  done
  rc=0; for p in $pids; do wait $p || rc=1; done
  [ $rc = 0 ] && [ "$(stat -c %s $f)" = "$size" ] || { log "FAIL $f (rc $rc, size $(stat -c %s $f) / $size)"; return 1; }
  case $f in
    *.tar.gz|*.tgz) gzip -t $f 2>/dev/null || { log "FAIL gzip integrity $f"; return 1; };;
    *.zip) python3 -c "import zipfile,sys; sys.exit(0 if zipfile.ZipFile('$f').testzip() is None else 1)" 2>/dev/null || { log "FAIL zip integrity $f"; return 1; };;
  esac
  date -u +%FT%TZ > $f.DONE
  log "done $f"
}
mkdir -p $R
fail=0; running=0
for spec in "${FILES[@]}"; do
  set -- $spec
  ( fetch "$@" ) &
  running=$((running + 1))
  if [ $running -ge 2 ]; then wait -n || fail=1; running=$((running - 1)); fi
done
while [ $running -gt 0 ]; do wait -n || fail=1; running=$((running - 1)); done
[ -n "${DRY_RUN:-}" ] && exit 0
if [ $fail = 0 ]; then date -u +%FT%TZ > $R/DOWNLOAD_DONE; log "DOWNLOAD_DONE"; else date -u +%FT%TZ > $R/DOWNLOAD_FAILED; log "DOWNLOAD_FAILED"; fi
