"""S3-w（桥接预热）验收：同一批 Emilia 样本上比较 cond_mode=full 与 text_only 的 talker speech CE。

    python custom/streaming_si/scripts/s3w_bridge_gate.py --model <S3-w checkpoint> \
        --data <Emilia parquet 目录> [--baseline <S3-w 起点目录>]

判据（旧版源项目 docs/model_architecture.md §5，S3-w 的门）：full 的 speech CE ≤ text_only。
即接入 thinker 语义后至少不比只用文本 embedding 差，桥接已「对上」，才进入 S3 放开 talker 主干。

`--baseline` 额外报告起点模型的两项 CE（合并后未预热时实测 text_only 3.51 / full 4.95），用来看预热的改善幅度。
样本按 stride 从各 parquet 抽取（中英都覆盖），编码口径与 S3-w / S3 训练一致（strip_prompt、无音色 prompt）。
S3-w 只用约 1/10 的数据训练约 1000 万参数的桥接，抽到训练样本的影响可以忽略。
需要 1 张 GPU（4B thinker + talker 前向，bf16）。
"""
import argparse
import glob
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.data.si_encode import collate_si, encode_s1  # noqa: E402


def load_samples(tokenizer, data_dir, n_files, per_file, seed):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        raise SystemExit(f'没有 parquet: {data_dir}')
    rng = random.Random(seed)
    pick = files[::max(1, len(files) // n_files)][:n_files]
    out = []
    for f in pick:
        rows = pq.read_table(f, columns=['conversations', 'answer_speech_tokens']).to_pylist()
        for r in rng.sample(rows, min(per_file, len(rows))):
            conv = json.loads(r['conversations']) if isinstance(r['conversations'], str) else r['conversations']
            try:
                out.append(encode_s1(tokenizer, conv, r['answer_speech_tokens'], None, max_length=512,
                                     max_speech_len=750, max_prompt_speech_len=125, ref_drop_ratio=1.0,
                                     strip_prompt=True, text_scheduled_sampling=0.0,
                                     text_vocab_size=int(tokenizer.vocab_size), rng=rng))
            except Exception:
                continue
    return out, len(pick)


@torch.no_grad()
def speech_ce(model, samples, cond_mode, batch, device):
    model.config.cond_mode = cond_mode
    tot = n = 0.0
    for i in range(0, len(samples), batch):
        b = collate_si(samples[i:i + batch], pad_token_id=0)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        res = model(**b)
        w = float((b['speech_tokens'] != -100).sum()) if 'speech_tokens' in b else 1.0
        tot += res.loss.float().item() * w
        n += w
    return tot / n


def evaluate(path, samples, batch, device):
    model = StreamingSIForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    model.config.text_loss_weight, model.config.speech_loss_weight = 0.0, 1.0
    out = {m: speech_ce(model, samples, m, batch, device) for m in ('text_only', 'full')}
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='S3-w 产出的 checkpoint 目录')
    ap.add_argument('--data', required=True, help='Emilia parquet 目录')
    ap.add_argument('--baseline', help='可选：S3-w 起点（merge_stage_checkpoints.py 的产出）')
    ap.add_argument('--n_files', type=int, default=16)
    ap.add_argument('--per_file', type=int, default=32)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model)
    samples, nf = load_samples(tok, args.data, args.n_files, args.per_file, args.seed)
    print(f'[gate] 样本 {len(samples)} 条（{nf} 个 parquet，按 stride 抽取）')
    report = {}
    if args.baseline:
        report['baseline'] = evaluate(args.baseline, samples, args.batch, device)
        print(f"[gate] 起点  text_only={report['baseline']['text_only']:.4f}  full={report['baseline']['full']:.4f}")
    report['model'] = r = evaluate(args.model, samples, args.batch, device)
    ok = r['full'] <= r['text_only']
    print(f"[gate] 预热后 text_only={r['text_only']:.4f}  full={r['full']:.4f}  "
          f"full - text_only = {r['full'] - r['text_only']:+.4f}  -> {'PASS' if ok else 'FAIL'}")
    report['pass'] = ok
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
