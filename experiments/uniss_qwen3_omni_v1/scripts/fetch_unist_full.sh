#!/usr/bin/env bash
# Fetch the rest of the UniST corpus: the 188 manifests and the 629 audio
# shards that are not on this disk yet.
#
# Incremental by construction. aws s3 sync skips a key whose local copy
# matches on size and mtime, so an interrupted run resumes by being run
# again, and a completed run is a no-op rather than two terabytes of
# re-download.
#
# Everything lands under data/, which is where this project keeps corpora.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"

SRC="${SRC:-s3://aigc-anyscale-hyperpod-124355679795-ap-northeast-1-an/zeyangsong_backup/opt/dlami/nvme/zeyangsong/data/unist/converted/unist_qwen3_s2s_v2}"
MAN_DST="${MAN_DST:-${REPO_ROOT}/data/unist_omni_v1/manifests}"
AUD_DST="${AUD_DST:-${REPO_ROOT}/data/unist_omni_v1/speech}"
JOBS="${JOBS:-64}"
mkdir -p "${MAN_DST}" "${AUD_DST}"

# Refuse to start a 2 TiB download that cannot finish: a disk that fills
# mid-sync leaves truncated tars, and a truncated tar reads as a short
# utterance rather than as an error.
NEED_GIB="${NEED_GIB:-2100}"
FREE_GIB=$(df -BG --output=avail "${REPO_ROOT}" | tail -1 | tr -dc '0-9')
echo "可用 ${FREE_GIB} GiB，需要约 ${NEED_GIB} GiB"
if [ "${FREE_GIB}" -lt "${NEED_GIB}" ]; then
  echo "空间不足，拒绝开始"; exit 1
fi

# 64, measured. 16 gives 274 MiB/s and 128 gives 96 -- past the knee the
# connections thrash rather than share. s5cmd was tried too and came back
# at 101 MiB/s on the same tar, so the faster tool is the slower one here.
aws configure set default.s3.max_concurrent_requests "${JOBS}"
aws configure set default.s3.multipart_chunksize "${CHUNK:-32MB}"
aws configure set default.s3.multipart_threshold "${CHUNK:-32MB}"

echo "[$(date -u +%H:%M)] manifests"
aws s3 sync "${SRC}/manifests/" "${MAN_DST}/" \
  --exclude "*" --include "manifest-train-*.jsonl" --only-show-errors
echo "  本地 manifest: $(ls "${MAN_DST}"/manifest-train-*.jsonl 2>/dev/null | wc -l) / 198"

echo "[$(date -u +%H:%M)] audio shards"
# One sync per rank directory rather than one over the whole prefix.
# Measured on this link: a single process tops out at 367 MiB/s and gets
# *worse* if its own concurrency is raised past 64 -- 128 gives 96 -- but
# eight independent processes at 64 each reach 1,869. The ceiling is
# inside one CLI process's scheduling, not the network, which is also why
# s5cmd's 256 workers came back at 101: they are all in one process too.
ranks=$(aws s3 ls "${SRC}/speech/unist_bicodec_qwen3_v2/" | awk '{print $NF}' | tr -d '/')
for r in ${ranks}; do
  mkdir -p "${AUD_DST}/unist_bicodec_qwen3_v2/${r}"
  aws s3 sync "${SRC}/speech/unist_bicodec_qwen3_v2/${r}/" \
              "${AUD_DST}/unist_bicodec_qwen3_v2/${r}/" \
    --exclude "*" --include "*.tar" --only-show-errors &
done
wait
echo "  本地 tar: $(find "${AUD_DST}" -name '*.tar' | wc -l) / 662"

# Size is the only integrity signal sync leaves behind; a tar that is
# short is the failure mode that would otherwise surface as missing rows
# three hours into training.
echo "[$(date -u +%H:%M)] 校验大小"
aws s3 ls "${SRC}/speech/unist_bicodec_qwen3_v2/" --recursive \
  | awk '$NF ~ /\.tar$/ {n=split($NF,p,"/"); print p[n], $3}' | sort > /tmp/_s3_tars.txt
find "${AUD_DST}" -name '*.tar' -printf '%f %s\n' | sort > /tmp/_local_tars.txt
join /tmp/_s3_tars.txt /tmp/_local_tars.txt 2>/dev/null \
  | awk '$2 != $3 {print "  大小不符:", $1, "S3", $2, "本地", $3; bad++} END {print "  大小不符的 tar:", bad+0}'
comm -23 <(cut -d' ' -f1 /tmp/_s3_tars.txt) <(cut -d' ' -f1 /tmp/_local_tars.txt) \
  | wc -l | sed 's/^/  仍缺少的 tar: /'
echo "FETCH_DONE"
