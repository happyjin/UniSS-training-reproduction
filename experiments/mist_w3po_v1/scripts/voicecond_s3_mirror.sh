#!/usr/bin/env bash
# Local mirror of the open-corpus archives on team S3 used by the voice-condition work (plan §7.2b): CoVoST 2 (s2tt),
# Common Voice / LibriSpeech (asr), WenetSpeech cuts, manifests -> data/open_voicecond_v1/s3_mirror/zeyangsong_data/.
# s5cmd with --no-clobber (never overwrites); size check against S3; marker COPY_DONE (or COPY_FAILED).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/voicecond_s3_mirror.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/s3_mirror.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
S=s3://stevehguo-ap-southeast-3/speech_model_backup/zeyangsong/data
D=data/open_voicecond_v1/s3_mirror/zeyangsong_data
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
mkdir -p $D
fail=0
for f in manifests.tar.zst wenetspeech.tar.zst s2tt.tar.zst asr.tar.zst; do
  want=$(aws s3 ls $S/$f | awk '{print $3}')
  if [ -n "${DRY_RUN:-}" ]; then log "[dry] $S/$f ($want B) -> $D/$f"; continue; fi
  if [ -e $D/$f ] && [ "$(stat -c %s $D/$f)" = "$want" ]; then log "skip (complete): $f"; continue; fi
  [ -e $D/$f ] && { log "REFUSE: $D/$f exists with a different size (not overwritten)"; fail=1; continue; }
  log "copy $f ($want B)"
  s5cmd --numworkers 64 cp --no-clobber --concurrency 32 --part-size 64 $S/$f $D/$f || { log "FAIL s5cmd $f"; fail=1; continue; }
  [ "$(stat -c %s $D/$f)" = "$want" ] && log "done $f" || { log "FAIL size $f"; fail=1; }
done
[ -n "${DRY_RUN:-}" ] && exit 0
if [ $fail = 0 ]; then date -u +%FT%TZ > $D/COPY_DONE; log COPY_DONE; else date -u +%FT%TZ > $D/COPY_FAILED; log COPY_FAILED; fi
