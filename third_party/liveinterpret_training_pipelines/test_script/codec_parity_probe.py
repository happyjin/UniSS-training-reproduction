# codec 编码确定性 / 跨缓存一致性探针（2026-08-25，D′ vs B 分叉排查）
# 问题：D′（merge 训练逻辑 + zeyang 数据管线）未设 BRIDGE_ALIGNMENT_CODEC_CACHE，
#   codec 全部重编码（容器层缓存）；B 用 NVMe 磁盘缓存（B 自己编码的）。
# 本探针：对 B 缓存中已有编码的音频现场重编码，对比 codes 是否逐位一致；
#   同音频连续两次重编码互相对比（判定 GPU 编码本身的确定性）。
# 用法：python3 codec_parity_probe.py（容器内需 1 GPU、/nvme 挂载、/s3all 代码桶）
import glob
import os
import sys

import torch

sys.path.insert(0, '/s3all/deferzhou/liveinterpret_zeyang_dev_parity/custom')
os.environ.setdefault(
    'QWEN3_TTS_TOKENIZER',
    '/nvme/iegg-speech-model-models/origin_models/Qwen3-TTS-Tokenizer-12Hz')

from codec_tokenizer import _normalize_codes, get_codec_tokenizer  # noqa: E402

B_CACHE = '/nvme/.cache/bridge_alignment_codec'
NEW_CACHE = '/tmp/codec_probe_cache'
EOS_ID = 2150
SAMPLE_N = int(os.environ.get('PROBE_N', '20'))


def main() -> None:
    pts = sorted(glob.glob(os.path.join(B_CACHE, '*.pt')))[:SAMPLE_N]
    print(f'[probe] B cache dir={B_CACHE}, sampled {len(pts)} .pt files', flush=True)
    tk = get_codec_tokenizer(cache_dir=NEW_CACHE)
    target_sr = tk._input_sample_rate()
    n_same = n_diff = n_skip = 0
    for p in pts:
        obj = torch.load(p, map_location='cpu')
        audio_path = obj.get('audio_path') if isinstance(obj, dict) else None
        codes_b = obj['codes'] if isinstance(obj, dict) and 'codes' in obj else obj
        if not audio_path or not os.path.exists(audio_path):
            print(f'[skip] {os.path.basename(p)}: audio missing -> {audio_path}', flush=True)
            n_skip += 1
            continue
        codes_new1 = _normalize_codes(tk._encode_with_resample(audio_path, target_sr))
        codes_new2 = _normalize_codes(tk._encode_with_resample(audio_path, target_sr))
        codes_b = _normalize_codes(codes_b)
        det = torch.equal(codes_new1, codes_new2)
        name = os.path.basename(audio_path)
        if codes_b.shape != codes_new1.shape:
            print(f'[{name}] SHAPE-DIFF B{tuple(codes_b.shape)} vs new{tuple(codes_new1.shape)} det={det}', flush=True)
            n_diff += 1
            continue
        neq = int((codes_b != codes_new1).sum())
        total = codes_b.numel()
        eos_flip = int(((codes_b == EOS_ID) != (codes_new1 == EOS_ID)).sum())
        print(f'[{name}] det={det} mismatch={neq}/{total} ({neq / max(total, 1):.4%}) eos_pos_flip={eos_flip}', flush=True)
        if neq == 0:
            n_same += 1
        else:
            n_diff += 1
    print(f'[summary] identical={n_same} diff={n_diff} skip={n_skip}', flush=True)


if __name__ == '__main__':
    main()
