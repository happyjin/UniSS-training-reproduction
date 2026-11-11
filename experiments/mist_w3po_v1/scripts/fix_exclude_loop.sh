#!/usr/bin/env bash
# Corrects the talker_crf exclude files (2026-11-11 00:45 UTC bug in start_talker_crf_scheme.sh: release_cards renames
# train_active_<s> to train_active_<s>.ended_<t>, which its own glob train_active_* still matched, so a finished
# training's cards stayed excluded and the eval could not start). Every 20 s: such files are renamed to
# ended_train_active_<s>.<t> (out of the glob), and the shared + per-run exclude_cards are rewritten from the exact
# train_active_plem / train_active_align files. Stops after MAX_S or when STOP exists. Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
L=logs/mist_w3po_v1/talker_crf_v1
MAX_S=${MAX_S:-21600}; t0=$(date +%s)
echo $$ > $L/fix_exclude.pid
while [ $(( $(date +%s) - t0 )) -lt "$MAX_S" ] && [ ! -e $L/fix_exclude.STOP ]; do
  for f in $L/train_active_*.ended_*; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); command mv -f "$f" "$L/ended_${b/.ended_/.}"
    echo "$(date -u +%FT%TZ) renamed $b" >> $L/fix_exclude.log
  done
  ex=""
  for s in plem align; do [ -e $L/train_active_$s ] && ex="$ex $(tr ',' ' ' < $L/train_active_$s)"; done
  for d in $L $L/sample $L/precheck $L/eval_plem $L/eval_align; do
    [ -d "$d" ] || continue
    if [ "$(cat $d/exclude_cards 2>/dev/null)" != "$ex" ]; then
      echo "$ex" > "$d/exclude_cards.tmp" && command mv -f "$d/exclude_cards.tmp" "$d/exclude_cards"
      echo "$(date -u +%FT%TZ) $d/exclude_cards <- [$ex]" >> $L/fix_exclude.log
    fi
  done
  sleep 20
done
