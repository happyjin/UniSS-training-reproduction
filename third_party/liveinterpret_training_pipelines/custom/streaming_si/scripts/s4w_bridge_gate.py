"""S4-w（跨池桥接预热）验收 = 门 A：同一批 UNISS 样本上，按语言方向比较 cond_mode=full 与 text_only 的 talker speech CE。

    python custom/streaming_si/scripts/s4w_bridge_gate.py --model <S4-w checkpoint> \
        --data '<P2 parquet glob>' --wav_dir <p2_source_wav 目录> [--baseline <S3 checkpoint>]

判据（旧版源项目 scripts/diag_s3_w.py 的门 A）：每个方向 full - text_only < -margin（默认 0.005）。
即 thinker 路在跨语数据上带来了 text 路之外的信息，桥接已对上，才进入 S4 正式解冻全部。
差值在 ±margin 内 = thinker 路没起作用（被 fusion 门控关掉）；> margin = thinker 路是负资产。
**不看训练 loss 曲线**：桥接预热期曲线本就近乎平，看曲线会误判成「没在学」。

门 B（会不会收尾：生成/真实长度比 ≈ 1）只有 talker 主干解冻后才可能改善，属于 S4 正式的验收，不在这里。

`--baseline` 额外报告起点（S3）的两项 CE，用来看预热的改善幅度。
分方向报告：综合指标会掩盖弱方向（eng->cmn 约占 2/3）。
样本按 stride 从各 parquet 抽取、按与训练相同的口径编码（SI_MAX_AUDIO_SEC 等取 train_s2st.sh 的默认值，
超长样本丢弃）。训练没有留出验证集；S4-w 只看约 64 万 / 1928 万行，抽到训练样本的影响可以忽略。
需要训练镜像（read_p2_wav 依赖 swift 插件模块）和 1 张 GPU。
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.model.base_model import VoxtralAudioProcessor  # noqa: E402
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.integration.register_streaming_si import read_p2_wav  # noqa: E402
from streaming_si.data.si_encode import IGNORE_INDEX, SIEncodeError, collate_si, encode_st, normalize_fsq  # noqa: E402
from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor  # noqa: E402

# 与 train_s2st.sh 的默认口径一致
LIMITS = dict(max_length=768, max_target_len=256, max_speech_len=750, max_audio_sec=30.0, min_audio_sec=0.5)


def load_samples(tokenizer, mel_fn, data_glob, wav_dir, n_files, per_dir, seed):
    """按方向各取 per_dir 条（从 n_files 个等间隔 parquet 里抽）。返回 {direction: [encoded, ...]}。"""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(data_glob))
    if not files:
        raise SystemExit(f'没有 parquet: {data_glob}')
    rng = random.Random(seed)
    pick = files[::max(1, len(files) // n_files)][:n_files]
    pool = defaultdict(list)
    cols = ['source_wav', 'translation', 'src_lang', 'tgt_lang', 'target_fsq']
    for f in pick:
        for r in pq.read_table(f, columns=cols).to_pylist():
            if r['source_wav'] and '::' in r['source_wav'] and r['target_fsq'] is not None:
                pool[f"{r['src_lang']}->{r['tgt_lang']}"].append(r)
    out, skipped = {}, 0
    for d, rows in sorted(pool.items()):
        rng.shuffle(rows)
        enc = []
        for r in rows:
            if len(enc) >= per_dir:
                break
            tar, loc = r['source_wav'].split('::', 1)
            try:
                wav = read_p2_wav(f'{os.path.join(wav_dir, tar)}::{loc}', LIMITS['min_audio_sec'],
                                  LIMITS['max_audio_sec'], drop_long=True)
                if wav is None:
                    raise SIEncodeError('读不到源音频')
                codes = normalize_fsq(r['target_fsq'])
                enc.append(encode_st(tokenizer, mel_fn(wav, sr=16000), r['translation'] or '', r['tgt_lang'],
                                     max_length=LIMITS['max_length'], max_target_len=LIMITS['max_target_len'],
                                     speech_tokens=codes, max_speech_len=LIMITS['max_speech_len'],
                                     with_answer_mask=True, drop_overlong=True))
            except SIEncodeError:
                skipped += 1
        out[d] = enc
    return out, len(pick), skipped


@torch.no_grad()
def speech_ce(model, samples, cond_mode, batch, device):
    """token 加权的 speech CE（各 batch 有效语音 token 数不同，不能对 batch 均值再平均）。"""
    model.config.cond_mode = cond_mode
    tot, n = 0.0, 0
    for i in range(0, len(samples), batch):
        b = collate_si(samples[i:i + batch], pad_token_id=0)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        res = model(**b)
        logits, target = res.speech_logits.reshape(-1, res.speech_logits.size(-1)), res.speech_target.reshape(-1)
        valid = target != IGNORE_INDEX
        tot += F.cross_entropy(logits[valid].float(), target[valid], reduction='sum').item()
        n += int(valid.sum())
    return tot / max(n, 1), n


def evaluate(path, by_dir, batch, device):
    model = StreamingSIForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    if model.audio_encoder is None:
        raise SystemExit(f'{path} 未含 Voxtral 音频塔')
    model.config.text_loss_weight, model.config.speech_loss_weight = 0.0, 1.0
    out = {}
    for d, samples in by_dir.items():
        out[d] = {}
        for m in ('text_only', 'full'):
            out[d][m], out[d]['n_tokens'] = speech_ce(model, samples, m, batch, device)
    model.config.cond_mode = 'full'
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='S4-w 产出的 checkpoint 目录')
    ap.add_argument('--data', required=True, help="P2 parquet glob，如 '/nvme/_swift_prep/p2_merged_*/p2_fsq_v3/*.parquet'")
    ap.add_argument('--wav_dir', required=True, help='对应的 p2_source_wav 目录（tar）')
    ap.add_argument('--baseline', help='可选：S4-w 起点（S3 的 checkpoint）')
    ap.add_argument('--n_files', type=int, default=16)
    ap.add_argument('--per_dir', type=int, default=256, help='每个方向的样本数')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--margin', type=float, default=0.005)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model)
    mel = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.model))
    by_dir, nf, skipped = load_samples(tok, mel, args.data, args.wav_dir, args.n_files, args.per_dir, args.seed)
    print(f'[gate A] {nf} 个 parquet，' + '，'.join(f'{d} {len(s)} 条' for d, s in by_dir.items())
          + f'（跳过超长/读不到 {skipped} 条）')
    report = {}
    for name, path in (('baseline', args.baseline), ('model', args.model)):
        if not path:
            continue
        report[name] = evaluate(path, by_dir, args.batch, device)
        for d, r in report[name].items():
            diff = r['full'] - r['text_only']
            print(f"[gate A] {'起点' if name == 'baseline' else '预热后'} {d:9s} text_only={r['text_only']:.4f}  "
                  f"full={r['full']:.4f}  full-text_only={diff:+.4f}  ({r['n_tokens']} speech tokens)")
    diffs = {d: r['full'] - r['text_only'] for d, r in report['model'].items()}
    ok = all(v < -args.margin for v in diffs.values())
    for d, v in diffs.items():
        verdict = ('PASS' if v < -args.margin else
                   'FAIL：thinker 路没起作用（差值在 margin 内）' if v < args.margin else
                   'FAIL：thinker 路是负资产（full 更差）')
        print(f'[gate A] {d}: {verdict}')
    report['pass'] = ok
    print(json.dumps(report, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
