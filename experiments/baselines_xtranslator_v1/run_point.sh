#!/usr/bin/env bash
# Launch the driver shards of one point x direction (one shard per ASR / X-Voice server pair, shared LMT server).
#   POINT=xt-default DIR=en2zh CARD=3 NSH=4 ASR_BASE=18001 TTS_BASE=18201 MT_PORT=18100 [ASR_CHUNK=0.6] \
#   [ITEMS=<items jsonl>] [OUTROOT=<dir>] [LIMIT=n] [IDS=a,b] bash run_point.sh
# Resumable: a shard skips ids already in its part file; an existing shard config is reused only if it is identical.
set -uo pipefail
: "${POINT:?}" "${DIR:?}" "${CARD:?}" "${NSH:?}" "${ASR_BASE:?}" "${TTS_BASE:?}" "${MT_PORT:?}"
R=/opt/dlami/nvme/jasonleeeli/projects/UniSS
V=/opt/dlami/nvme/jasonleeeli/venvs/xtranslator
EXP=$R/experiments/baselines_xtranslator_v1
ITEMS=${ITEMS:-$R/eval_outputs/baselines_xtranslator_v1/items_$DIR.jsonl}
OUT=${OUTROOT:-$R/eval_outputs/baselines_xtranslator_v1}/$POINT/realsi_$DIR
case "$POINT" in xt-a1.6) [ "${ASR_CHUNK:-0.6}" = 1.6 ] || { echo "xt-a1.6 needs ASR_CHUNK=1.6 servers"; exit 1; } ;;
  *) [ "${ASR_CHUNK:-0.6}" = 0.6 ] || { echo "$POINT needs ASR_CHUNK=0.6 servers"; exit 1; } ;; esac
mkdir -p "$OUT/configs" "$OUT/logs"
export PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$CARD
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope
export SSL_CERT_FILE=$($V/bin/python -m certifi)
export XT_ASR_CHUNK_SIZE_SEC=${ASR_CHUNK:-0.6}
for k in $(seq 0 $((NSH - 1))); do
  cfg=$OUT/configs/shard$k.json
  tmp=$OUT/configs/.shard$k.$$.json
  "$V/bin/python" "$EXP/make_config.py" --point "$POINT" --direction "$DIR" --asr-port $((ASR_BASE + k)) \
    --mt-port "$MT_PORT" --tts-port $((TTS_BASE + k)) --run-dir "$OUT" --out "$tmp" > /dev/null || exit 1
  if [ -e "$cfg" ]; then
    cmp -s "$cfg" "$tmp" || { echo "refusing: $cfg differs from this launch"; exit 1; }
  else
    command mv -f "$tmp" "$cfg"
  fi
  extra=()
  [ -n "${LIMIT:-}" ] && extra+=(--limit "$LIMIT")
  [ -n "${IDS:-}" ] && extra+=(--ids "$IDS")
  setsid nohup "$V/bin/python" "$EXP/xt_realsi.py" --direction "$DIR" --items "$ITEMS" --config "$cfg" \
    --point "$POINT" --out "$OUT" --shard "$k" --num-shards "$NSH" "${extra[@]}" \
    >> "$OUT/logs/driver_shard$k.log" 2>&1 < /dev/null &
  echo $! > "$OUT/logs/driver_shard$k.pid"
  echo "$(date -u +%FT%TZ) shard $k pid $! cfg $cfg" >> "$OUT/logs/launch.log"
done
