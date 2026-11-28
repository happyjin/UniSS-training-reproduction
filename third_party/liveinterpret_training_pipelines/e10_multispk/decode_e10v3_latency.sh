#!/usr/bin/env bash
# E10 v3 latency fixes, decoding only (reports/mist_w3po_v1/e10v3_latency_analysis.zh-CN.md; user 2026-11-28 "尽量加快速度"):
#   arm T  (cards 0-3): --write_threshold 0.3                     -> output/demo_game_v3t/raw_{zh2en,en2zh}
#   arm TC (cards 4-7): --write_threshold 0.3 --max_wait_steps 4  -> output/demo_game_v3tc/raw_{zh2en,en2zh}
# Same checkpoint (e10v3_joint/checkpoint-6000), sessions, windows, TTS seed and loop guard as the v3 demo decode.
# A keeper stops the guard's no-checkpoint placeholders on cards 0-7 by PID (the guard itself is never touched);
# slots A and B are held during the run and released at the end; DONE / FAILED marker in output/markers.
# Run detached:  setsid nohup bash e10_multispk/decode_e10v3_latency.sh > <log> 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
U=/opt/dlami/nvme/jasonleeeli/projects/UniSS
E=$U/third_party/liveinterpret_training_pipelines/e10_multispk
SL=$U/logs/mist_w3po_v1/slots
PY=/opt/dlami/nvme/jasonleeeli/venvs/simuls2s/bin
OUT=$E/output
MK=$OUT/markers
J=$OUT/e10v3_joint/v0-20261121-100836/checkpoint-6000
DRY=${DRY_RUN:-}
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }

stop_placeholders() {
  local c sup kids
  for c in 0 1 2 3 4 5 6 7; do
    for sup in $(ps -eo pid=,ppid=,args= | awk -v c="--devices $c " '$2==1 && index($0,"train_mist_workload.py") && index($0,c) {print $1}'); do
      kids=$(ps -eo pid=,ppid= | awk -v p="$sup" '$2==p {print $1}')
      log "card $c: stop placeholder supervisor $sup (children: ${kids:-none})"
      [ -n "$DRY" ] || kill -TERM $sup $kids 2>/dev/null
    done
  done
  return 0
}
KEEPER=""
stop_keeper() { [ -n "$KEEPER" ] && kill -TERM "$KEEPER" 2>/dev/null; KEEPER=""; }
release() {
  local s
  for s in A B; do
    [ -e "$SL/slot$s.owner" ] && command mv -f "$SL/slot$s.owner" "$SL/slot$s.released_$(date -u +%Y%m%dT%H%M%SZ)" && log "slot $s released"
  done
  return 0
}
fail() { log "FAILED: $*"; stop_keeper; [ -n "$DRY" ] || { mkdir -p "$MK"; echo "$*" > "$MK/E10V3_LAT_FAILED"; release; }; exit 1; }

[ -d "$J" ] || { log "missing checkpoint $J"; exit 1; }
for s in A B; do [ -e "$SL/slot$s.owner" ] && { log "slot $s is held: $(cat "$SL/slot$s.owner")"; exit 1; }; done
for d in "$OUT/demo_game_v3t" "$OUT/demo_game_v3tc"; do [ -e "$d" ] && { log "$d exists; refusing"; exit 1; }; done
if [ -z "$DRY" ]; then
  mkdir -p "$MK"
  for s in A B; do echo "E10 v3 latency decode (threshold / max-wait), $(date -u +%FT%TZ), pid $$" > "$SL/slot$s.owner"; done
  ( while true; do stop_placeholders; sleep 10; done ) &
  KEEPER=$!
  log "card keeper $KEEPER"
fi
trap 'stop_keeper' EXIT
stop_placeholders
[ -n "$DRY" ] || sleep 20

cd "$U"
pids=""
for arm in T TC; do
  case $arm in
    T)  DEMO=$OUT/demo_game_v3t;  G0=0; EXTRA="--write_threshold 0.3";;
    TC) DEMO=$OUT/demo_game_v3tc; G0=4; EXTRA="--write_threshold 0.3 --max_wait_steps 4";;
  esac
  [ -n "$DRY" ] || mkdir -p "$DEMO"
  k=0
  for d in zh2en en2zh; do
    case $d in zh2en) M=gz01,gz02,gz03; S=mix; T=en;; en2zh) M=ge01,ge02,ge04; S=ihm_mix; T=zh;; esac
    for s in 0 1 2 3 4 5; do
      g=$(( G0 + k % 4 )); k=$((k+1))
      if [ -n "$DRY" ]; then
        log "[dry] arm $arm card $g: run_e10 --mode e10v3 $EXTRA $d shard $s/6 -> $DEMO/raw_$d"
      else
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 E10_THREADS=4 PATH=$PY:$PATH "$PY/python" "$E/run_e10.py" --mode e10v3 \
          --set custom --data "data/game_esports_si_v1/$d" --sess "$S" --tgt "$T" \
          --windows "eval_outputs/mist_w3po_v1/data/windows_game_${d}_c2g06_v1/windows_test_online_${d}.jsonl" \
          --meetings "$M" --adapter "$J" --out "$DEMO/raw_$d" --shard "$s" --num-shards 6 $EXTRA \
          > "$DEMO/decode_${d}_$s.log" 2>&1 < /dev/null &
        pids="$pids $!"
        sleep 10
      fi
    done
  done
done
nf=0
for p in $pids; do wait "$p" || nf=$((nf+1)); done
[ "$nf" = 0 ] || fail "$nf decoders failed (see output/demo_game_v3t*/decode_*.log)"
stop_keeper
[ -n "$DRY" ] || { echo "T: $OUT/demo_game_v3t TC: $OUT/demo_game_v3tc" > "$MK/E10V3_LAT_DONE"; release; }
log "E10 v3 latency decodes done"
