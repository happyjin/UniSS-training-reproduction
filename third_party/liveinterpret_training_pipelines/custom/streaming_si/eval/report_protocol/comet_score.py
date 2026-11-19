"""ASR-COMET / Text-COMET（口径同 liveinterpret_training_pipelines/custom/streaming_si/eval/score_eval.py）。

* 模型 `Unbabel/wmt22-comet-da`，`load_from_checkpoint` + `predict(batch_size=32, gpus=1)`，取 `system_score`
  （= 逐句分数均值），保留 4 位小数。
* 输入三元组 src / mt / ref 都用**未归一化**原文：src = 清单 `src_text`，ref = 清单 `reference`；
  `text_comet` 的 mt 是模型输出的译文 `text`；
  `asr_comet` 的 mt 是 asr_lat.jsonl 的 `asr` 转写（我方口径：zh 目标 Qwen3-ASR，en 目标 Whisper）；
  `asr_whisper_comet` 的 mt 一律用 Whisper-large-v3 转写（zh 目标取 `asr_ts`，en 目标即 `asr`），
  与对方 score_eval.py 的 ASR 口径完全一致。
* 句集合与 make_report.py 一致：清单里有推理结果的句子；失败 / 空输出按空串计入（不剔除）。

输出 `<root>/runs/<set>/<out_name>`（默认 comet.json）：{<key>_comet, <key>_seg, n, model}。
用法：python comet_score.py --root <out_root> --sets a,b,c --model <model.ckpt> [--batch 32]
                           [--keys asr,text,asr_whisper] [--out_name comet.json]
"""
import argparse
import json
import os


def read_jsonl(p):
    out = {}
    if p and os.path.isfile(p):
        for l in open(p, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                if r['id'] not in out or r.get('status', 'ok') == 'ok':
                    out[r['id']] = r
    return out


def whisper_text(r):
    """Whisper 转写：en 目标的 `asr` 本身就是 Whisper；zh 目标（asr 为 Qwen3-ASR）取时间戳那一遍的 `asr_ts`。"""
    if not r:
        return ''
    return (r.get('asr') if r.get('asr_model', 'whisper') == 'whisper' else r.get('asr_ts')) or ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--model', required=True, help='wmt22-comet-da 的 checkpoints/model.ckpt')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--keys', default='asr,text')
    ap.add_argument('--out_name', default='comet.json')
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(',') if k.strip()]

    from comet import load_from_checkpoint
    model = load_from_checkpoint(args.model)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        d = os.path.join(args.root, 'runs', name)
        if not os.path.isfile(os.path.join(d, 'results.jsonl')):
            print(f'[comet] skip {name}: 无结果', flush=True)
            continue
        man = [json.loads(l) for l in open(os.path.join(args.root, 'manifests', f'{name}.jsonl'), encoding='utf-8')
               if l.strip()]
        res = read_jsonl(os.path.join(d, 'results.jsonl'))
        lat = read_jsonl(os.path.join(d, 'asr_lat.jsonl'))
        rows = [m for m in man if m['id'] in res]
        srcs = [m['src_text'] for m in rows]
        refs = [m['reference'] for m in rows]
        mts = dict(
            text=[((res[m['id']].get('text') or '') if res[m['id']].get('status') == 'ok' else '').strip()
                  for m in rows],
            asr=[(lat.get(m['id'], {}).get('asr') or '').strip() for m in rows],
            asr_whisper=[whisper_text(lat.get(m['id'])).strip() for m in rows])
        out = dict(n=len(rows), model=args.model)
        for key in keys:
            data = [dict(src=s, mt=t, ref=r) for s, t, r in zip(srcs, mts[key], refs)]
            pred = model.predict(data, batch_size=args.batch, gpus=1, progress_bar=False)
            out[f'{key}_comet'] = round(float(pred.system_score), 4)
            out[f'{key}_seg'] = {m['id']: round(float(x), 4) for m, x in zip(rows, pred.scores)}
            print(f'[comet] {name} {key}_comet = {out[f"{key}_comet"]:.4f} (n={len(rows)})', flush=True)
        json.dump(out, open(os.path.join(d, args.out_name), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
