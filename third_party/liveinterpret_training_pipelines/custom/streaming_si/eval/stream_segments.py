"""流式系统的统一后处理：把 S5 与 SimulS2ST-Omni（SimulEval）的输出整理成同一种「分段」表示，
导出评测用的结果 jsonl（与 score_eval 的输入格式一致），以及做 LAAL 所需的分段时间线。

每句得到若干语音段 seg_i = (emit_nca, emit_ca, dur)，单位秒，时间轴以源音频开始为 0：
* `emit_nca`：该段发出时已读入的源音频时长（SimulEval 的 `delays`，computation-unaware）。
* `emit_ca` ：SimulEval 的 `elapsed` 口径 = 已读源音频 + 自本句开始的墙钟（computation-aware）。
* `dur`    ：该段语音时长。

播放时间线按 SimulEval `SpeechOutputInstance.summarize` 的规则：start_i = max(prev_end, emit_i)，
前一段没播完就排队。NCA / CA 各排一条。

来源：
* **S5**：`steps[*]` 里 WRITE 步的 `source_end_ms`（实为秒）、`speech_tokens`（25Hz，每 token 40ms）、
  `elapsed_sec`；尾部 flush 的 token 记在源读完时刻 / `tail_elapsed_sec`。S5 是整句末尾一次合成波形，
  声码器耗时 `vocoder_sec` 按累计 token 比例摊到各段的 CA 时刻上（近似「逐段合成」）。
  波形是各段 token 首尾相接合成的，故分段在波形上的位置 = 累计 token × 40ms（按实际波形长度等比校正）。
* **Omni**：SimulEval `instances.log` 的 `delays` / `elapsed`（ms，由 omni_simuleval_wrap 补回）/
  `durations`；它写出的 wav 是按 NCA 时间线插了等待静音的，这里按 `intervals` 把各段切出来首尾拼接，
  得到「只有语音」的波形（ASR / UTMOS 用它，与离线系统可比）。
  译文取 agent 的 `text_predictions.jsonl`（按实例顺序对应，并用源音频时长交叉校验）。

用法：
  python stream_segments.py s5   --results <s5 results.jsonl,...> --out <dir>
  python stream_segments.py omni --runs <omni_L*/shard*_dir,...>  --out <dir>
输出 `<dir>/results.jsonl`（id, text, wav, output_sec, wall_sec, source_sec, segments）。
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf

TOK_SEC = 0.04            # CosyVoice speech token 25Hz


def s5_rows(paths):
    for p in paths:
        for l in open(p, encoding='utf-8'):
            if not l.strip():
                continue
            r = json.loads(l)
            out = dict(id=r['id'], tgt_lang=r.get('tgt_lang'), source_sec=r.get('source_sec'),
                       text=r.get('text', ''), status=r.get('status'), wall_sec=r.get('wall_sec'))
            if r.get('status') != 'ok' or not r.get('wav'):
                out.update(wav=None, output_sec=0.0, segments=[])
                yield out
                continue
            src = float(r['source_sec'])
            segs = []                                     # (emit_nca, elapsed_wall, n_tok)
            for s in r.get('steps', []):
                n = int(s.get('speech_tokens') or 0)
                if s.get('decision') == 'write' and n > 0:
                    segs.append([min(float(s['source_end_ms']), src), float(s.get('elapsed_sec', np.nan)), n])
            nt = int(r.get('tail_speech_tokens') or 0)
            if nt > 0:
                segs.append([src, float(r.get('tail_elapsed_sec', np.nan)), nt])
            total = sum(x[2] for x in segs)
            wav_sec = float(r.get('output_sec') or 0.0)
            scale = wav_sec / (total * TOK_SEC) if total else 1.0      # 波形长度与 token 数的微小偏差
            voc = float(r.get('vocoder_sec') or 0.0)
            seg_out, cum = [], 0
            for emit, wall, n in segs:
                cum += n
                # SimulEval elapsed = 已读源 + 墙钟；声码器按累计 token 比例摊入
                ca = emit + wall + voc * cum / total
                seg_out.append(dict(emit_nca=round(emit, 4), emit_ca=round(ca, 4),
                                    dur=round(n * TOK_SEC * scale, 4)))
            out.update(wav=r['wav'], output_sec=wav_sec, segments=seg_out)
            yield out


def omni_rows(run_dirs, out_dir):
    wav_dir = os.path.join(out_dir, 'wav')
    os.makedirs(wav_dir, exist_ok=True)
    man = {}
    for d in run_dirs:
        ids = [l.strip() for l in open(os.path.join(d, 'ids.txt')) if l.strip()]
        logs = [json.loads(l) for l in open(os.path.join(d, 'instances.log')) if l.strip()] \
            if os.path.isfile(os.path.join(d, 'instances.log')) else []
        logs = {int(x['index']): x for x in logs}
        side = [json.loads(l) for l in open(os.path.join(d, 'text_predictions.jsonl'), encoding='utf-8')] \
            if os.path.isfile(os.path.join(d, 'text_predictions.jsonl')) else []
        side.sort(key=lambda x: x['record_id'])
        for k, uid in enumerate(ids):
            info = logs.get(k)
            row = dict(id=uid, run=os.path.basename(d.rstrip('/')))
            if info is None:
                row.update(status='missing', text='', wav=None, output_sec=0.0, segments=[], wall_sec=None)
                yield row
                continue
            src = float(info['source_length']) / 1000.0
            text = ''
            if k < len(side):
                sd = side[k]
                if abs(float(sd.get('source_length_ms', 0)) / 1000.0 - src) < 0.06:   # 交叉校验对应关系
                    text = sd.get('pred_text', '') or ''
                else:
                    row['text_note'] = f'sidecar 第 {k} 条时长 {sd.get("source_length_ms")}ms 与实例不符'
            delays = [float(x) / 1000 for x in info.get('delays', [])]
            elapsed = [float(x) / 1000 for x in info.get('elapsed', [])]
            durs = [float(x) / 1000 for x in info.get('durations', [])]
            intervals = info.get('intervals', [])
            segs, pieces = [], []
            if delays and os.path.isfile(info['prediction']):
                tl, sr = sf.read(info['prediction'], dtype='float32')
                base = float(delays[0])                          # SimulEval 的 wav 从第一段发出时刻开始
                for i, (st, du) in enumerate(intervals):
                    a = int(round((float(st) / 1000 - base) * sr))
                    b = a + int(round(float(du) / 1000 * sr))
                    pieces.append(tl[a:b])
                    segs.append(dict(emit_nca=round(delays[i], 4),
                                     emit_ca=round(elapsed[i], 4) if i < len(elapsed) else None,
                                     dur=round(durs[i], 4)))
            if pieces:
                wav = np.concatenate(pieces)
                path = os.path.join(wav_dir, f'{uid}.wav')
                sf.write(path, wav, sr)
                row.update(wav=os.path.abspath(path), output_sec=round(wav.size / sr, 3))
            else:
                row.update(wav=None, output_sec=0.0)
            row.update(status='ok' if pieces else 'empty_speech', text=text, segments=segs,
                       source_sec=round(src, 3), wall_sec=info.get('wall_sec'))
            yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=['s5', 'omni'])
    ap.add_argument('--results', default='')
    ap.add_argument('--runs', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.kind == 's5':
        paths = sorted(p for x in args.results.split(',') for p in glob.glob(x))
        rows = list(s5_rows(paths))
    else:
        dirs = sorted(p for x in args.runs.split(',') for p in glob.glob(x) if os.path.isdir(p))
        rows = list(omni_rows(dirs, args.out))
    with open(os.path.join(args.out, 'results.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ok = sum(1 for r in rows if r.get('status') == 'ok')
    notes = sum(1 for r in rows if r.get('text_note'))
    print(f'{args.kind}: {len(rows)} 条，ok {ok}，旁路译文对不上 {notes} -> {args.out}/results.jsonl')


if __name__ == '__main__':
    main()
