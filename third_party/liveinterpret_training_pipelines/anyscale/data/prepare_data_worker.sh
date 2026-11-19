#!/usr/bin/env bash
# =============================================================================
# Prepare-data worker script. Dispatched to each non-head Ray node by
# anyscale/data/dispatch_prepare_data.py.
# =============================================================================
# 每个 worker 节点独立执行（tokyo-h100:18 P5 节点，CPU-only Ray task）：
#   1. TTS:  从 /mnt/fsx/.../task=tts/...           读取 → text_to_speech.jsonl
#   2. S2S:  从 /mnt/fsx/.../task=speech2speech/...   读取 → speech_to_speech_translation.jsonl
#   3. Merge: TTS:S2S = 5:1                          → bridge_alignment_joint.jsonl
#
# 产物目录：${BASE_PATH}/datasets/jsonl/（本地 NVMe via /local hostPath）
#
# SMALL_DATA=1 时仅处理少量行用于流程验证（--limit）。
# 确认无误后将 job_prepare_data.yaml 中 SMALL_DATA 改为 "0" 即可全量处理。
#
# 幂等性：
#   - .worker_prepare_data_done marker 存在 → 完整跳过
#   - 无 marker 但个别步骤产物已存在且非空 → 该步骤跳过（断点续传）
#   - 前次不完整运行产生的空文件 → 自动清理
#
# Layout:
#   <repo>/anyscale/data/prepare_data_worker.sh   ← this file
#   <repo>/dataset_maker/                         ← maker 脚本位置
#   REPO_ROOT = dirname(this) / ../..
# =============================================================================
set -euo pipefail

: "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BASE_PATH="${BASE_PATH:-/local/speech-language-model/liveinterpret_training_pipelines}"

DATA_MAKER_DIR="${BASE_PATH}/datasets/jsonl"
MODEL_DIR="${BASE_PATH}/models"

# ---- SMALL_DATA 参数 ----
SMALL_DATA="${SMALL_DATA:-0}"
SMALL_DATA_LIMIT="${SMALL_DATA_LIMIT:-1000}"

# ---- 输出路径 ----
TTS_OUTPUT="${DATA_MAKER_DIR}/text_to_speech.jsonl"
S2S_OUTPUT="${DATA_MAKER_DIR}/speech_to_speech_translation.jsonl"
FINAL_OUTPUT="${DATA_MAKER_DIR}/bridge_alignment_joint.jsonl"
MARKER="${DATA_MAKER_DIR}/.worker_prepare_data_done"

# ---- 本 job 产出物清单 ----
# FORCE_RERUN=1 时仅删除以下这些本 job 自己的产物，目录下其它（不属于本 job 的）
# jsonl 保留原样。后续新增训练任务的 *_OUTPUT 时，请一并加入此清单。
JOB_OUTPUTS=(
    "${TTS_OUTPUT}"
    "${S2S_OUTPUT}"
    "${FINAL_OUTPUT}"
)

# ---- FSx 数据源路径（只读） ----
FSX_TTS="/mnt/fsx/data/curated/task=tts/curation_version=v1"
FSX_S2S="/mnt/fsx/data/curated/task=speech2speech/curation_version=v1"

# ---- FSx 挂载根 ----
# S3 bucket 镜像挂载在此处，TTS maker 把样本里的 s3://<bucket>/<key> 音频路径
# 自动改写为 ${FSX_MOUNT}/<key> 本地路径并校验文件存在，训练时才能读到原始音频。
FSX_MOUNT="${FSX_MOUNT:-/mnt/fsx}"

# =============================================================================
# 幂等性：已完成则跳过（FORCE_RERUN=1 时清掉产物重新生成）
# =============================================================================
FORCE_RERUN="${FORCE_RERUN:-0}"
if [[ "${FORCE_RERUN}" == "1" ]]; then
    echo "[prepare-data-worker] $(hostname) FORCE_RERUN=1 — removing this job's existing outputs (other jsonl in ${DATA_MAKER_DIR} are kept)."
    for f in "${JOB_OUTPUTS[@]}"; do
        if [[ -f "$f" ]]; then
            echo "[prepare-data-worker]   removing existing output: $f"
            rm -f "$f"
        fi
    done
    rm -f "${MARKER}"
fi
if [[ -f "${MARKER}" ]]; then
    echo "[prepare-data-worker] $(hostname) ${MARKER} exists — all steps already done."
    exit 0
fi

# ---- Helpers ----
is_nonempty() { [[ -f "$1" && -s "$1" ]]; }

clean_empty() {
    # 清理前次不完整运行产生的空文件（非空文件保留用于断点续传）
    if [[ -f "$1" ]] && [[ ! -s "$1" ]]; then
        echo "[prepare-data-worker] removing stale empty file: $1"
        rm -f "$1"
    fi
}

mkdir -p "${DATA_MAKER_DIR}" "${MODEL_DIR}"

# ---- AWS CLI tuning ----
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB

echo "================================================================"
echo "[prepare-data-worker] $(hostname) starting at $(date -Iseconds)"
echo "[prepare-data-worker] REPO_ROOT=${REPO_ROOT}"
echo "[prepare-data-worker] BASE_PATH=${BASE_PATH}"
echo "[prepare-data-worker] SMALL_DATA=${SMALL_DATA}"
if [[ "${SMALL_DATA}" == "1" ]]; then
    echo "[prepare-data-worker]   --limit=${SMALL_DATA_LIMIT}"
fi
echo "================================================================"

aws sts get-caller-identity --output text || {
    echo "[prepare-data-worker][ERROR] AWS credentials missing on worker." >&2
    exit 1
}

df -h /local 2>/dev/null | tail -1 || df -h / | tail -1

# ---- 校验 /mnt/fsx ----
if [[ ! -d "/mnt/fsx/data/curated" ]]; then
    echo "[prepare-data-worker][ERROR] /mnt/fsx/data/curated not accessible." >&2
    ls /mnt/fsx/ 2>/dev/null || echo "  /mnt/fsx/ empty or unreadable" >&2
    exit 1
fi
echo "[prepare-data-worker] /mnt/fsx accessible: $(ls /mnt/fsx/data/curated/ 2>/dev/null | head -5)"

# =============================================================================
# Step 1: TTS dataset → text_to_speech.jsonl
#   Input  : /mnt/fsx/.../task=tts/.../dataset=*/examples.parquet
#   Output : ${DATA_MAKER_DIR}/text_to_speech.jsonl
# =============================================================================
echo "[prepare-data-worker] (1/3) TTS: ${FSX_TTS} → ${TTS_OUTPUT}"

TTS_ARGS=(--input-dir "${FSX_TTS}" --output "${TTS_OUTPUT}" --sampling-rate 16000 --fsx-mount "${FSX_MOUNT}")
if [[ "${SMALL_DATA}" == "1" ]]; then
    TTS_ARGS+=(--limit "${SMALL_DATA_LIMIT}")
fi

if is_nonempty "${TTS_OUTPUT}"; then
    echo "[prepare-data-worker] (1/3) ${TTS_OUTPUT} exists ($(wc -l < "${TTS_OUTPUT}") lines) — skip."
else
    clean_empty "${TTS_OUTPUT}"
    if [[ ! -d "${FSX_TTS}" ]]; then
        echo "[prepare-data-worker][ERROR] TTS FSx dir not found: ${FSX_TTS}" >&2
        exit 1
    fi
    echo "[prepare-data-worker] (1/3) running make_text_to_speech.py ..."
    python3 "${REPO_ROOT}/dataset_maker/make_text_to_speech.py" "${TTS_ARGS[@]}"
    echo "[prepare-data-worker] (1/3) TTS done: $(wc -l < "${TTS_OUTPUT}") lines"
fi

# =============================================================================
# Step 2: S2S dataset → speech_to_speech_translation.jsonl
#   Input  : /mnt/fsx/.../task=speech2speech/.../*.parquet + audio dirs
#   Output : ${DATA_MAKER_DIR}/speech_to_speech_translation.jsonl
#
#   maker 构建 in-memory audio index (os.listdir) → O(1) 查找，
#   避免逐行 os.path.isfile() 的 FSx 网络往返开销。
# =============================================================================
echo "[prepare-data-worker] (2/3) S2S: ${FSX_S2S} → ${S2S_OUTPUT}"

S2S_ARGS=(--input-dir "${FSX_S2S}" --output "${S2S_OUTPUT}" --latency-budget-ms 2000)
if [[ "${SMALL_DATA}" == "1" ]]; then
    S2S_ARGS+=(--limit "${SMALL_DATA_LIMIT}")
fi

if is_nonempty "${S2S_OUTPUT}"; then
    echo "[prepare-data-worker] (2/3) ${S2S_OUTPUT} exists ($(wc -l < "${S2S_OUTPUT}") lines) — skip."
else
    clean_empty "${S2S_OUTPUT}"
    if [[ ! -d "${FSX_S2S}" ]]; then
        echo "[prepare-data-worker][ERROR] S2S FSx dir not found: ${FSX_S2S}" >&2
        exit 1
    fi
    # ---- 预热 FSx 缓存：并发读取所有 WAV 文件头，触发 FSx 从 S3 拉取数据块 ----
    # 若不预热，S2S maker 逐行 wave.open() 每次冷读 ~2.7s，57 个 parquet 需 ~3 天。
    # 预热后 wave.open() 命中 FSx SSD 缓存，单次 ~7ms，全量 S2S 几分钟完成。
    echo "[prepare-data-worker] (2/3) warming FSx cache (source_audio + target_audio, parallel=32) ..."
    WARM_T0=$SECONDS
    for AUDIO_SUBDIR in source_audio target_audio; do
        AUDIO_DIR="${FSX_S2S}/${AUDIO_SUBDIR}"
        if [[ -d "${AUDIO_DIR}" ]]; then
            FILE_COUNT=$(find "${AUDIO_DIR}" -type f 2>/dev/null | wc -l)
            echo "[prepare-data-worker]   warming ${AUDIO_SUBDIR} (${FILE_COUNT} files) ..."
            find "${AUDIO_DIR}" -type f -print0 2>/dev/null | \
                xargs -0 -P 32 -I {} dd if="{}" bs=44 count=1 of=/dev/null 2>/dev/null
            echo "[prepare-data-worker]   ${AUDIO_SUBDIR} done (${FILE_COUNT} files)"
        else
            echo "[prepare-data-worker]   ${AUDIO_SUBDIR} not found, skip"
        fi
    done
    WARM_ELAPSED=$((SECONDS - WARM_T0))
    echo "[prepare-data-worker] (2/3) FSx warm done in ${WARM_ELAPSED}s"

    echo "[prepare-data-worker] (2/3) launching S2S maker (FSx, in-memory index) ..."
    python3 "${REPO_ROOT}/dataset_maker/make_speech_to_speech_translation.py" "${S2S_ARGS[@]}"
    echo "[prepare-data-worker] (2/3) S2S done: $(wc -l < "${S2S_OUTPUT}") lines"
fi

# =============================================================================
# Step 3: Merge TTS + S2S → bridge_alignment_joint.jsonl (TTS:S2S = 5:1)
#
#   merge_datasets.py 参数说明：
#     --inputs "${S2S_OUTPUT}" "${TTS_OUTPUT}"
#     --ratios 1 5
#   表示 S2S:TTS 权重 = 1:5，即最终 TTS 样本数 ≈ 5 × S2S 样本数。
#   scale = min(S2S_lines/1, TTS_lines/5)
# =============================================================================
echo "[prepare-data-worker] (3/3) Merge TTS:S2S = 5:1 → ${FINAL_OUTPUT}"

if is_nonempty "${FINAL_OUTPUT}"; then
    echo "[prepare-data-worker] (3/3) ${FINAL_OUTPUT} exists ($(wc -l < "${FINAL_OUTPUT}") lines) — skip."
else
    clean_empty "${FINAL_OUTPUT}"

    if ! is_nonempty "${S2S_OUTPUT}"; then
        echo "[prepare-data-worker][ERROR] S2S output empty/missing: ${S2S_OUTPUT}" >&2
        exit 1
    fi
    if ! is_nonempty "${TTS_OUTPUT}"; then
        echo "[prepare-data-worker][ERROR] TTS output empty/missing: ${TTS_OUTPUT}" >&2
        exit 1
    fi

    python3 "${REPO_ROOT}/dataset_maker/merge_datasets.py" \
        --inputs "${S2S_OUTPUT}" "${TTS_OUTPUT}" \
        --ratios 1 5 \
        --output-name bridge_alignment_joint \
        --output-dir "${DATA_MAKER_DIR}" \
        --shuffle-output

    echo "[prepare-data-worker] (3/3) Merge done: $(wc -l < "${FINAL_OUTPUT}") lines"
fi

# =============================================================================
# Step 4: 产物音频路径抽检校验
#   从最终 JSONL 抽样若干条，校验其 audios / target_audio 均为 ${FSX_MOUNT}
#   下的本地文件且真实存在，确认训练时能正确读取原始音频。
# =============================================================================
echo "[prepare-data-worker] (verify) verifying audio paths in ${FINAL_OUTPUT} (sample) ..."
FSX_MOUNT="${FSX_MOUNT}" python3 - "${FINAL_OUTPUT}" <<'PYEOF'
import json, os, sys

path = sys.argv[1]
fsx_mount = os.environ.get("FSX_MOUNT", "/mnt/fsx")
sample_n = 50
checked = missing = not_fsx = 0
shown = 0

def audio_paths(rec):
    for p in rec.get("audios") or []:
        yield p
    if rec.get("target_audio"):
        yield rec["target_audio"]

with open(path, encoding="utf-8") as f:
    i = -1
    for i, line in enumerate(f):
        if i >= sample_n:
            break
        rec = json.loads(line)
        for ap in audio_paths(rec):
            checked += 1
            if not ap.startswith(fsx_mount + "/"):
                not_fsx += 1
                if shown < 5:
                    print(f"  [WARN] not under {fsx_mount}: {ap}", file=sys.stderr)
                    shown += 1
            elif not os.path.isfile(ap):
                missing += 1
                if shown < 5:
                    print(f"  [WARN] missing on FSx: {ap}", file=sys.stderr)
                    shown += 1

print(f"  audio path check: sampled={min(sample_n, i + 1)} records, "
      f"checked={checked}, not_under_fsx={not_fsx}, missing={missing}")
if not_fsx or missing:
    print("  [ERROR] some sampled audio paths are not readable on FSx.", file=sys.stderr)
    sys.exit(1)
print("  audio path check: OK")
PYEOF
echo "[prepare-data-worker] (verify) audio path verification done."

# =============================================================================
# Done
# =============================================================================
touch "${MARKER}"
df -h /local 2>/dev/null | tail -1 || true
echo "================================================================"
echo "[prepare-data-worker] $(hostname) ALL DONE at $(date -Iseconds)"
echo "[prepare-data-worker] Output summary:"
echo "  JSONL:  $(du -sh \"${DATA_MAKER_DIR}\" 2>/dev/null || echo '?')"
ls -lh "${DATA_MAKER_DIR}"/*.jsonl 2>/dev/null || true
echo "================================================================"