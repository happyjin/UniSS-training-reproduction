#!/usr/bin/env bash
# SimulS2ST-Omni：swift 版本与源项目的三阶段对拍（CPU，tiny 夹具）。
#
#   SRC_PY=<源环境 python>   SRC_ROOT=<SimulS2ST-Omni-stage123>   SW_PY=<本仓库环境 python> \
#   FIXTURE=/tmp/simuls2st_smoke OUT=/tmp/simuls2st_parity  bash run_parity.sh [single|ddp|all]
#
# single：单进程，每个阶段跑三份——src（源代码 + 源环境）、nat（迁入代码 + HF Trainer + 本仓库环境）、
#         sw（swift sft）。nat vs sw 只差接入层，要求逐位一致（rtol 1e-6）；src vs sw 还含 torch 2.6→2.8、
#         transformers 4.52→4.57、peft 0.18→0.15 的数值差异（rtol 1e-3）。阶段交接各用自己上一阶段的产物。
# ddp：   2 进程（gloo）。swift 经 DDP forward 同步梯度；参照为 nat + --ddp_forward（源的多卡语义修正版），
#         同时核对每个 rank 每一步的 batch 与源采样器一致（src 2 进程的 batch 序列）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"
: "${SRC_PY:?需要 SRC_PY（源环境 python）}" "${SRC_ROOT:?需要 SRC_ROOT（SimulS2ST-Omni-stage123 目录）}"
: "${SW_PY:?需要 SW_PY（本仓库环境 python，含 ms-swift 4.4.0）}"
: "${FIXTURE:=/tmp/simuls2st_smoke}" "${OUT:=/tmp/simuls2st_parity}"
MODE="${1:-all}"
SW_BIN="$(dirname "${SW_PY}")"
# PYTHONHASHSEED：源 Stage 3 的延迟倍数 m 由 hash((seed, index, "m")) 抽取，字符串哈希随进程而变。
# BLAS 单线程：多线程 BLAS（macOS Accelerate / MKL）的归约顺序随进程而变，会带来 fp32 舍入级差异。
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
[[ -f "${FIXTURE}/model/config.json" ]] || { echo "缺夹具 ${FIXTURE}（先跑 make_smoke_fixture.py）" >&2; exit 1; }

last_ckpt() { ls -d "$1"/checkpoint-* | sort -t- -k2 -n | tail -1; }

run_src()   { "${SRC_PY}" "${HERE}/native_trace.py" --impl source --src_root "${SRC_ROOT}" "$@"; }
run_nat()   { "${SW_PY}" "${HERE}/native_trace.py" --impl ported "$@"; }
run_swift() {   # run_swift <cfg> <out_dir> <trace> [extra...]
    local cfg="$1" out="$2" trace="$3"; shift 3
    local a=(); while IFS= read -r x; do a+=("$x"); done < <(
        "${SW_PY}" -c "import sys; sys.path.insert(0, '${REPO}/custom'); from simuls2st_omni.integration.swift_args import main; main(['${cfg}'])")
    SIMULS2ST_CONFIG="${cfg}" TRACE_OUT="${trace}" "${SW_BIN}/swift" sft \
        --external_plugins "${REPO}/custom/plugin.py" "${HERE}/swift_trace_plugin.py" \
        --model "${FIXTURE}/model" "${a[@]}" --output_dir "${out}" --use_cpu true --device_map cpu "$@"
}
cmp() { "${SW_PY}" "${HERE}/compare.py" "$@"; }

single() {
    rm -rf "${OUT}/src" "${OUT}/nat" "${OUT}/sw" "${OUT}/trace"; mkdir -p "${OUT}/trace" "${OUT}/logs"
    "${SW_PY}" "${HERE}/make_parity_configs.py" --fixture "${FIXTURE}" --out "${OUT}" --sides src nat sw
    for n in 1 2 3; do
        local st="stage${n}"
        run_src --out "${OUT}/trace/src_${st}.json" -- --config "${OUT}/cfg/src/${st}.yaml" > "${OUT}/logs/src_${st}.log" 2>&1
        run_nat --out "${OUT}/trace/nat_${st}.json" -- --config "${OUT}/cfg/nat/${st}.yaml" > "${OUT}/logs/nat_${st}.log" 2>&1
        [[ $n == 2 ]] && export SIMULS2ST_INIT_TALKER_FROM="$(last_ckpt "${OUT}/sw/stage1")"
        [[ $n == 3 ]] && export SIMULS2ST_MERGE_ADAPTERS_FROM="$(last_ckpt "${OUT}/sw/stage2")"
        run_swift "${OUT}/cfg/sw/${st}.yaml" "${OUT}/sw/${st}" "${OUT}/trace/sw_${st}.json" > "${OUT}/logs/sw_${st}.log" 2>&1
        unset SIMULS2ST_INIT_TALKER_FROM SIMULS2ST_MERGE_ADAPTERS_FROM
        cmp "${OUT}/trace/nat_${st}.json" "${OUT}/trace/sw_${st}.json" --rtol 1e-6 --label "${st} nat vs swift"
        cmp "${OUT}/trace/src_${st}.json" "${OUT}/trace/sw_${st}.json" --rtol 1e-3 --label "${st} src vs swift"
    done
}

ddp() {
    local D="${OUT}/ddp"; rm -rf "${D}"; mkdir -p "${D}/trace" "${D}/logs"
    "${SW_PY}" "${HERE}/make_parity_configs.py" --fixture "${FIXTURE}" --out "${D}" --sides src nat sw
    for n in 1 2 3; do
        local st="stage${n}" unused=()
        [[ $n != 1 ]] && unused=(--find_unused)
        "$(dirname "${SRC_PY}")/torchrun" --nproc_per_node 2 --master_port $((29700 + n)) "${HERE}/native_trace.py" \
            --impl source --src_root "${SRC_ROOT}" --out "${D}/trace/src_${st}.json" -- --config "${D}/cfg/src/${st}.yaml" \
            > "${D}/logs/src_${st}.log" 2>&1
        "${SW_BIN}/torchrun" --nproc_per_node 2 --master_port $((29710 + n)) "${HERE}/native_trace.py" \
            --impl ported --ddp_forward ${unused[@]+"${unused[@]}"} --out "${D}/trace/nat_${st}.json" -- --config "${D}/cfg/nat/${st}.yaml" \
            > "${D}/logs/nat_${st}.log" 2>&1
        [[ $n == 2 ]] && export SIMULS2ST_INIT_TALKER_FROM="$(last_ckpt "${D}/sw/stage1")"
        [[ $n == 3 ]] && export SIMULS2ST_MERGE_ADAPTERS_FROM="$(last_ckpt "${D}/sw/stage2")"
        NPROC_PER_NODE=2 MASTER_PORT=$((29720 + n)) \
            run_swift "${D}/cfg/sw/${st}.yaml" "${D}/sw/${st}" "${D}/trace/sw_${st}.json" > "${D}/logs/sw_${st}.log" 2>&1
        unset SIMULS2ST_INIT_TALKER_FROM SIMULS2ST_MERGE_ADAPTERS_FROM
        cmp "${D}/trace/nat_${st}.json" "${D}/trace/sw_${st}.json" --ranks 2 --rtol 1e-6 --label "${st} ddp nat(ddp_forward) vs swift"
        "${SW_PY}" - "${D}/trace/src_${st}.json" "${D}/trace/sw_${st}.json" "${st}" <<'EOF'
import json, sys
ref, got, st = sys.argv[1], sys.argv[2], sys.argv[3]
for r in range(2):
    a = json.load(open(ref if r == 0 else f'{ref}.rank{r}'))['steps']
    b = json.load(open(got if r == 0 else f'{got}.rank{r}'))['steps']
    assert [x['ids'] for x in a] == [x['ids'] for x in b], f'{st} rank{r}: batch 序列与源采样器不同'
sw0 = json.load(open(got))['checksum']; sw1 = json.load(open(f'{got}.rank1'))['checksum']
assert sw0 == sw1, f'{st}: swift 两个 rank 训练后的参数不同（梯度未同步）'
print(f'[{st} ddp src vs swift] OK (两个 rank 的 batch 序列与源采样器逐步一致；swift 两 rank 参数完全相同)')
EOF
    done
}

case "${MODE}" in
    single) single ;;
    ddp) ddp ;;
    all) single; ddp ;;
    *) echo "用法: run_parity.sh [single|ddp|all]" >&2; exit 1 ;;
esac
