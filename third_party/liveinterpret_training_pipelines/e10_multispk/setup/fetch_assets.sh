#!/usr/bin/env bash
# E10: fetch the LiveInterpret S5 runtime assets into liveinterpret_training_pipelines/{checkpoints,external}.
# Skips anything already present (never overwrites, never deletes).
#
#   setsid nohup bash e10_multispk/setup/fetch_assets.sh > e10_multispk/runs/fetch_assets.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # liveinterpret_training_pipelines
B=s3://stevehguo-ap-southeast-3/speech_model_backup
CK="${ROOT}/checkpoints"

fetch_dir() {   # fetch_dir <s3 prefix> <local dir>
    if [[ -e "$2/_FETCH_DONE" ]]; then echo "[fetch] skip $2 (done)"; return; fi
    mkdir -p "$2"
    aws s3 cp --recursive --only-show-errors "$1" "$2"
    date -u +%FT%TZ > "$2/_FETCH_DONE"
    echo "[fetch] $1 -> $2"
}

# S4 base (Voxtral + Qwen3.5-4B Thinker + 0.6B Talker), the adapter's base_model_name_or_path.
fetch_dir "${B}/stevehguo/model_checkpoint/streaming_si_swift/trained_streaming_si_4b_enzhjako/S4_1031v2_ckpt93767/" \
          "${CK}/S4_1031v2_ckpt93767"
# CosyVoice3 flow / HiFT / CAM++ (the base dir has no speech_decoder/).
fetch_dir "${B}/stevehguo/model_checkpoint/streaming_si_swift/base_streaming_si_4b/speech_decoder/" \
          "${CK}/speech_decoder"
# Speech tokenizer for the source-voice flow prompt. Confirmed by the user on 2026-11-19 (S3 multipart ETag 840fd747...-116).
TOK="${CK}/cosyvoice3/speech_tokenizer_v3.batch.onnx"
if [[ ! -e "${TOK}" ]]; then
    mkdir -p "$(dirname "${TOK}")"
    aws s3 cp --only-show-errors "${B}/stevehguo/model_checkpoint/cosyvoice3/speech_tokenizer_v3.batch.onnx" "${TOK}"
    echo "[fetch] speech_tokenizer_v3.batch.onnx"
fi

# CosyVoice official code (flow/HiFT modules + Matcha-TTS submodule).
CV="${ROOT}/external/CosyVoice"
if [[ ! -e "${CV}/.git" ]]; then
    mkdir -p "${ROOT}/external"
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "${CV}"
fi
git -C "${CV}" rev-parse HEAD > "${ROOT}/external/CosyVoice.commit"
echo "[fetch] CosyVoice @ $(cat "${ROOT}/external/CosyVoice.commit")"
echo "[fetch] done"
