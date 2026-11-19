#!/usr/bin/env bash
set -euo pipefail

# Back up training artifacts and the text-to-speech translation dataset.
#
# Usage:
#   bash tools/backup_training_to_s3.sh
#
# Optional overrides:
#   AWS_PROFILE=backup S5_WORKERS=16 S5_PART_CONCURRENCY=4 \
#     bash tools/backup_training_to_s3.sh
#   SNAPSHOT_NAME=2026-07-22 bash tools/backup_training_to_s3.sh
#   DRY_RUN=1 bash tools/backup_training_to_s3.sh

REPO_ROOT="${REPO_ROOT:-/opt/dlami/nvme/zeyangsong/projects/liveinterpret_training_pipelines}"
DATA_ROOT="${DATA_ROOT:-/opt/dlami/nvme/zeyangsong/data/sft_t2a_translate_converted}"
BACKUP_BASE="${BACKUP_BASE:-s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup}"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-$(date -u +%F)}"
DEST_ROOT="${DEST_ROOT:-${BACKUP_BASE}/${SNAPSHOT_NAME}}"
AWS_PROFILE="${AWS_PROFILE:-backup}"
AWS_REGION="${AWS_REGION:-ap-northeast-1}"
S5_WORKERS="${S5_WORKERS:-16}"
S5_PART_CONCURRENCY="${S5_PART_CONCURRENCY:-4}"
S5_PART_SIZE_MIB="${S5_PART_SIZE_MIB:-128}"
DRY_RUN="${DRY_RUN:-0}"

export AWS_REGION
export AWS_DEFAULT_REGION="${AWS_REGION}"

if ! command -v s5cmd >/dev/null 2>&1; then
    echo "ERROR: s5cmd is not installed or is not on PATH." >&2
    exit 1
fi

for required_path in "${REPO_ROOT}/logs" "${REPO_ROOT}/output" "${DATA_ROOT}"; do
    if [[ ! -d "${required_path}" ]]; then
        echo "ERROR: required source directory does not exist: ${required_path}" >&2
        exit 1
    fi
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
status_dir="${REPO_ROOT}/logs/s3_backup/${run_id}"
mkdir -p "${status_dir}"

s5_global=(
    s5cmd
    --profile "${AWS_PROFILE}"
    --numworkers "${S5_WORKERS}"
    --retry-count 20
    --stat
)
if [[ "${DRY_RUN}" == "1" ]]; then
    s5_global+=(--dry-run)
fi

s5_cp=(
    cp
    --no-follow-symlinks
    --concurrency "${S5_PART_CONCURRENCY}"
    --part-size "${S5_PART_SIZE_MIB}"
)

upload_dir() {
    local source_dir="$1"
    local destination="$2"
    echo "[$(date -u +%FT%TZ)] START ${source_dir} -> ${destination}"
    "${s5_global[@]}" "${s5_cp[@]}" "${source_dir}/" "${destination}/"
    echo "[$(date -u +%FT%TZ)] DONE  ${source_dir} -> ${destination}"
}

upload_file() {
    local source_file="$1"
    local destination="$2"
    [[ -f "${source_file}" ]] || return 0
    "${s5_global[@]}" "${s5_cp[@]}" "${source_file}" "${destination}/"
}

upload_training_logs() {
    local source_dir="$1"
    local destination="$2"
    echo "[$(date -u +%FT%TZ)] START ${source_dir} -> ${destination}"
    "${s5_global[@]}" "${s5_cp[@]}" \
        --exclude 's3_backup/*' \
        "${source_dir}/" "${destination}/"
    echo "[$(date -u +%FT%TZ)] DONE  ${source_dir} -> ${destination}"
}

# Keep only a compact per-root size summary; a per-file inventory for the raw
# audio tree is hundreds of MB and is unnecessary for the backup itself.
printf 'bytes\tpath\n' >"${status_dir}/local_inventory.tsv"
for source_path in \
    "${REPO_ROOT}/logs" \
    "${REPO_ROOT}/output" \
    "${DATA_ROOT}"; do
    du -sb "${source_path}" 2>>"${status_dir}/inventory_warnings.log" \
        >>"${status_dir}/local_inventory.tsv" || true
done

echo "run_id=${run_id}" >"${status_dir}/backup.conf"
echo "destination=${DEST_ROOT}" >>"${status_dir}/backup.conf"
echo "aws_profile=${AWS_PROFILE}" >>"${status_dir}/backup.conf"
echo "aws_region=${AWS_REGION}" >>"${status_dir}/backup.conf"
echo "s5_workers_per_lane=${S5_WORKERS}" >>"${status_dir}/backup.conf"
echo "multipart_concurrency_per_file=${S5_PART_CONCURRENCY}" >>"${status_dir}/backup.conf"
echo "part_size_mib=${S5_PART_SIZE_MIB}" >>"${status_dir}/backup.conf"

# Two bulk lanes run concurrently. s5cmd also parallelizes objects and multipart
# chunks inside each lane.
upload_dir \
    "${REPO_ROOT}/output" \
    "${DEST_ROOT}/liveinterpret_training_pipelines/output" \
    >"${status_dir}/output.log" 2>&1 &
output_pid=$!

upload_dir \
    "${DATA_ROOT}" \
    "${DEST_ROOT}/data/sft_t2a_translate_converted" \
    >"${status_dir}/dataset.log" 2>&1 &
dataset_pid=$!

# Small artifacts and the exact code/config used for the runs are a third lane.
(
    upload_training_logs \
        "${REPO_ROOT}/logs" \
        "${DEST_ROOT}/liveinterpret_training_pipelines/logs"

    for name in exp_results benchmark_results inference_outputs; do
        [[ -d "${REPO_ROOT}/${name}" ]] || continue
        upload_dir \
            "${REPO_ROOT}/${name}" \
            "${DEST_ROOT}/liveinterpret_training_pipelines/${name}"
    done

    for name in custom docs pipelines scripts tools dataset_maker inference; do
        [[ -d "${REPO_ROOT}/${name}" ]] || continue
        upload_dir \
            "${REPO_ROOT}/${name}" \
            "${DEST_ROOT}/liveinterpret_training_pipelines/source/${name}"
    done

    for name in README.md requirements.txt pyproject.toml setup.py; do
        upload_file \
            "${REPO_ROOT}/${name}" \
            "${DEST_ROOT}/liveinterpret_training_pipelines/source"
    done
) >"${status_dir}/metadata.log" 2>&1 &
metadata_pid=$!

status=0
for lane in \
    "output:${output_pid}" \
    "dataset:${dataset_pid}" \
    "metadata:${metadata_pid}"; do
    lane_name="${lane%%:*}"
    lane_pid="${lane##*:}"
    if wait "${lane_pid}"; then
        echo "[$(date -u +%FT%TZ)] ${lane_name}: complete" | tee -a "${status_dir}/summary.log"
    else
        lane_status=$?
        echo "[$(date -u +%FT%TZ)] ${lane_name}: failed (exit ${lane_status})" | tee -a "${status_dir}/summary.log"
        status=1
    fi
done

echo "Detailed logs: ${status_dir}"
exit "${status}"
