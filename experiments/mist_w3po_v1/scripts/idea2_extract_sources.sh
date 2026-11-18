#!/usr/bin/env bash
# Idea 2: unpack the open source corpora into data/open_voicecond_v1/idea2/src/<name>/ as soon as each download is
# complete (its .DONE marker exists). Never overwrites (tar -k / unzip -n); each unpack writes src/<tag>.UNPACKED.
#   librispeech test-clean, dev-clean, train-clean-100, train-clean-360   (raw/librispeech/*.tar.gz)
#   aishell3                                                               (raw/aishell3/data_aishell3.tgz)
#   musan                                                                  (raw/musan/musan.tar.gz)
#   fsd50k eval  (split zip: idea2/raw/fsd50k/FSD50K.eval_audio.z01 + .zip -> zip -s 0 -> joined zip -> unzip)
#   cv_zh        inner shards idea2/cv_zh/asr/common_voice/audio/zh-CN/*/*.tar (after idea2/cv_zh/EXTRACT_DONE)
# Polls every 30 s until all are unpacked (or a source failed), then writes src/ALL_UNPACKED.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_extract_sources.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2_extract_sources.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RAW=data/open_voicecond_v1/raw
I2=data/open_voicecond_v1/idea2
SRC=$I2/src
mkdir -p $SRC
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
run() { if [ -n "${DRY_RUN:-}" ]; then log "[dry] $*"; else eval "$@"; fi; }

untar() {   # <tag> <archive> <dest>
  local tag=$1 a=$2 d=$3
  [ -e $SRC/$tag.UNPACKED ] && return 0
  [ -e $a.DONE ] || return 1
  log "unpack $tag <- $a"
  mkdir -p $d
  if run "tar -x -k -f $a -C $d"; then [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ > $SRC/$tag.UNPACKED; log "done $tag"
  else log "FAIL $tag"; [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ > $SRC/$tag.FAILED; fi
  return 0
}

fsd() {
  local tag=fsd50k_eval z=$I2/raw/fsd50k/FSD50K.eval_audio.z01 j=$I2/raw/fsd50k/FSD50K.eval_audio.joined.zip
  [ -e $SRC/$tag.UNPACKED ] && return 0
  [ -e $z.DONE ] || return 1
  [ -e $RAW/fsd50k/FSD50K.eval_audio.zip ] || return 1
  log "join + unpack $tag"
  if [ ! -e $j ]; then run "zip -q -s 0 $I2/raw/fsd50k/FSD50K.eval_audio.zip --out $j" || { log "FAIL join"; date -u +%FT%TZ > $SRC/$tag.FAILED; return 0; }; fi
  mkdir -p $SRC/fsd50k
  if run "unzip -q -n $j -d $SRC/fsd50k"; then [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ > $SRC/$tag.UNPACKED; log "done $tag"
  else log "FAIL $tag"; [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ > $SRC/$tag.FAILED; fi
  return 0
}

cvzh() {
  local tag=cv_zh t ok=1
  [ -e $SRC/$tag.UNPACKED ] && return 0
  [ -e $I2/cv_zh/EXTRACT_DONE ] || return 1
  mkdir -p $SRC/cv_zh
  for t in $I2/cv_zh/asr/common_voice/audio/zh-CN/*/*.tar; do
    log "unpack $tag shard $t"
    run "tar -x -k -f $t -C $SRC/cv_zh" || ok=0
  done
  if [ $ok = 1 ]; then [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ > $SRC/$tag.UNPACKED; log "done $tag"
  else log "FAIL $tag"; [ -n "${DRY_RUN:-}" ] || date -u +%FT%TZ > $SRC/$tag.FAILED; fi
  return 0
}

while true; do
  pending=0
  for s in test-clean dev-clean train-clean-100 train-clean-360; do
    untar ls_$s $RAW/librispeech/$s.tar.gz $SRC/librispeech || pending=1
  done
  untar aishell3 $RAW/aishell3/data_aishell3.tgz $SRC/aishell3 || pending=1
  untar musan $RAW/musan/musan.tar.gz $SRC/musan || pending=1
  fsd || pending=1
  cvzh || pending=1
  [ -n "${DRY_RUN:-}" ] && exit 0
  if [ $pending = 0 ]; then date -u +%FT%TZ > $SRC/ALL_UNPACKED; log ALL_UNPACKED; exit 0; fi
  # give up on a source whose download failed for good
  [ -e $RAW/DOWNLOAD_FAILED ] && [ -e $I2/raw/EXTRA_FAILED ] && { log "downloads failed; stopping"; exit 1; }
  sleep 30
done
