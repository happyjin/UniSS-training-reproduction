"""核验 P2（UniST）训练数据里「源语音 → 目标语音」的音色与副语言一致性。

背景与动机
----------
`docs/paralinguistic_preservation.md` 曾下结论「数据里韵律有、情感很少，目标 TTS 把情感压平了」，
依据是 emotion2vec 的配对余弦只比随机配对高 0.02~0.03、eng→cmn 目标 90% 判为中立。

但该推理有两处不牢：
1. **指标失效**：我们在 RealSI/ACL 评测里独立观察到 emotion2vec 的 ESim delta 对**所有**系统
   都只有 0.01~0.03（见 `inference_outputs/.../README.md` 观察第 8 条），该指标在本任务上几乎
   不携带信息，不能用它的小 delta 反推「数据里没有情感」；
2. **「中立」判定**：emotion2vec 的 9 类分类器对 TTS 干净音频本身偏向中立，类别分布不可直接
   与真实录音比。

而数据生成侧的事实是：`p2_unist_to_fsq.py` 头注释写明**源/目标共用同一个 `bicodec_global`**
（同说话人），即目标音频是以源说话人的 BiCodec global 条件解码的。所以音色应当是对齐的。

本脚本直接测量，不依赖单一指标：
* SIM-o（WavLM x-vector 余弦）：源↔目标 配对 vs 随机配对 —— 判定**音色**是否对齐；
* AutoPCP（官方 v2 口径，layer 9 + symmetrize）：判定**韵律一致性**；
* ESim（emotion2vec）：一并报出，但按其已知的低区分度解读；
* 逐句声学量（F0 均值/标准差、能量标准差、语速、停顿占比）的源-目标相关，作为不依赖
  神经指标的交叉验证。

并按「源语音情感强度」排序导出 wav 对，供人工评审（情感夸张的样本最能看出是否真的被压平）。

目标 tar 不在训练节点上，但在 S3 同前缀下；parquet 的 `source_wav` / `target_wav` 列都是
`'<tar>::offset:length'`，故用 S3 **字节范围 GET** 精确取单条，无需同步整个 7GB tar。

用法
----
    python check_p2_pair_para.py --bucket <b> --prefix <p> --parquet_key <k> \
        --n 400 --out_dir <dir> --wavlm <dir> --emotion2vec <dir> \
        --autopcp <model.config> --w2v <dir> [--export 12]
"""
import argparse
import io
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SR = 16000


def s3_range(bucket, key, off, ln):
    """字节范围取对象。容器里没有 boto3，走 aws CLI（输出到临时文件再读）。"""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as t:
        tp = t.name
    try:
        p = subprocess.run(['aws', 's3api', 'get-object', '--bucket', bucket, '--key', key,
                            '--range', f'bytes={off}-{off + ln - 1}', tp],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip()[:200])
        with open(tp, 'rb') as f:
            return f.read()
    finally:
        os.path.isfile(tp) and os.unlink(tp)


def s3_download(bucket, key, dest):
    import subprocess
    p = subprocess.run(['aws', 's3', 'cp', f's3://{bucket}/{key}', dest, '--quiet'],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:300])


def decode(buf):
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(buf), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def acoustics(wav):
    """不依赖神经模型的副语言量，用于交叉验证。"""
    import librosa
    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=400, sr=SR,
                                 frame_length=1024, hop_length=256)
    v = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256)[0]
    sil = float((rms < max(rms.max() * 0.05, 1e-4)).mean()) if rms.size else 0.0
    return dict(
        f0_mean=float(np.log(v).mean()) if v.size else None,
        f0_std=float(np.log(v).std()) if v.size > 1 else None,
        energy_std=float(np.log(rms + 1e-8).std()) if rms.size > 1 else None,
        voiced_ratio=float(np.mean(voiced)) if voiced is not None and len(voiced) else None,
        pause_ratio=sil,
        dur=float(wav.size / SR),
    )


def pearson(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 10:
        return None, 0
    import math
    n = len(pairs)
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs))
    sy = math.sqrt(sum((b - my) ** 2 for _, b in pairs))
    if sx == 0 or sy == 0:
        return None, n
    return sum((a - mx) * (b - my) for a, b in pairs) / (sx * sy), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bucket', required=True)
    ap.add_argument('--prefix', required=True, help='到 machine_X_shard 的前缀（不含末尾斜杠）')
    ap.add_argument('--parquet_key', required=True, help='相对 prefix 的 parquet，如 p2_fsq_v3/fsq-train-00000.parquet')
    ap.add_argument('--src_dir', default='p2_source_wav')
    ap.add_argument('--tgt_dir', default='p2_target_wav')
    ap.add_argument('--n', type=int, default=400, help='每个方向采样条数')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--wavlm', required=True)
    ap.add_argument('--emotion2vec', required=True)
    ap.add_argument('--autopcp', required=True)
    ap.add_argument('--w2v', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--export', type=int, default=12, help='导出多少对 wav 供人工评审')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import soundfile as sf
    from streaming_si.eval.run_para_metrics import AutoPCP, ESim, SimO

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 读 parquet（整文件下载，仅 ~30MB）
    pkey = f'{args.prefix}/{args.parquet_key}'
    local_pq = os.path.join(args.out_dir, os.path.basename(args.parquet_key))
    if not os.path.isfile(local_pq):
        s3_download(args.bucket, pkey, local_pq)
    cols = ['id', 'src_lang', 'tgt_lang', 'source_wav', 'target_wav',
            'source_transcript', 'translation', 'duration_ratio']
    rows = pq.read_table(local_pq, columns=cols).to_pylist()
    rows = [r for r in rows if r['source_wav'] and r['target_wav']
            and '::' in r['source_wav'] and '::' in r['target_wav']]
    by_dir = {}
    for r in rows:
        by_dir.setdefault(f"{r['src_lang']}->{r['tgt_lang']}", []).append(r)
    rng = random.Random(args.seed)
    samp = []
    for d, rs in sorted(by_dir.items()):
        rng.shuffle(rs)
        samp += rs[:args.n]
    print(f'[p2] parquet 共 {len(rows)} 行，方向 {sorted(by_dir)}，采样 {len(samp)} 条', flush=True)

    # ---- 取波形（S3 字节范围）
    wav_dir = os.path.join(args.out_dir, 'wav')
    os.makedirs(wav_dir, exist_ok=True)
    got = []
    for i, r in enumerate(samp, 1):
        try:
            st, sl = r['source_wav'].split('::', 1)
            tt, tl = r['target_wav'].split('::', 1)
            so, sn = (int(x) for x in sl.split(':'))
            to, tn = (int(x) for x in tl.split(':'))
            sw = decode(s3_range(args.bucket, f'{args.prefix}/{args.src_dir}/{st}', so, sn))
            tw = decode(s3_range(args.bucket, f'{args.prefix}/{args.tgt_dir}/{tt}', to, tn))
            if sw.size < SR * 0.5 or tw.size < SR * 0.5:
                continue
            ps = os.path.join(wav_dir, f"{r['id']}_src.wav")
            pt = os.path.join(wav_dir, f"{r['id']}_tgt.wav")
            sf.write(ps, sw, SR)
            sf.write(pt, tw, SR)
            got.append(dict(row=r, src=ps, tgt=pt, src_wav=sw, tgt_wav=tw))
        except Exception as e:
            print(f'  [skip] {r["id"]}: {type(e).__name__}: {e}', flush=True)
        if i % 50 == 0:
            print(f'  取样 {i}/{len(samp)}  成功 {len(got)}', flush=True)
    if not got:
        raise SystemExit('没取到任何样本对')
    print(f'[p2] 取到 {len(got)} 对', flush=True)

    # ---- 随机配对基线：换一条**别的样本**的源语音
    n = len(got)
    rand_idx = [(k + n // 2) % n for k in range(n)]

    simo = SimO(args.wavlm, args.device)
    pcp = AutoPCP(args.autopcp, args.w2v, args.device)
    esim = ESim(args.emotion2vec, args.device)

    recs = []
    B = 8
    for s in range(0, n, B):
        chunk = list(range(s, min(s + B, n)))
        try:
            srcs = [got[k]['src'] for k in chunk]
            tgts = [got[k]['tgt'] for k in chunk]
            rnds = [got[rand_idx[k]]['src'] for k in chunk]
            import torch
            tt_ = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
            es = [simo.emb(tt_(got[k]['src_wav'])) for k in chunk]
            et = [simo.emb(tt_(got[k]['tgt_wav'])) for k in chunk]
            er = [simo.emb(tt_(got[rand_idx[k]]['src_wav'])) for k in chunk]
            pt_ = pcp.emb(tgts, B)
            ps_ = pcp.emb(srcs, B)
            pr_ = pcp.emb(rnds, B)
            sc = pcp.scores(ps_, pt_, B)
            scr = pcp.scores(pr_, pt_, B)
            vs = [esim.emb(p) for p in srcs]
            vt = [esim.emb(p) for p in tgts]
            vr = [esim.emb(p) for p in rnds]
            for j, k in enumerate(chunk):
                g = got[k]
                recs.append(dict(
                    id=g['row']['id'], direction=f"{g['row']['src_lang']}->{g['row']['tgt_lang']}",
                    sim_o=round(float(es[j] @ et[j]), 4), sim_o_rand=round(float(er[j] @ et[j]), 4),
                    autopcp=round(float(sc[j]), 4), autopcp_rand=round(float(scr[j]), 4),
                    esim=round(float(vs[j] @ vt[j]), 4), esim_rand=round(float(vr[j] @ vt[j]), 4),
                    src=g['src'], tgt=g['tgt'],
                    src_text=(g['row']['source_transcript'] or '')[:160],
                    tgt_text=(g['row']['translation'] or '')[:160],
                    duration_ratio=g['row'].get('duration_ratio'),
                    acou_src=acoustics(g['src_wav']), acou_tgt=acoustics(g['tgt_wav']),
                ))
        except Exception as e:
            print(f'[FAIL] batch@{s}: {type(e).__name__}: {e}', flush=True)
        print(f'[p2 score] {min(s + B, n)}/{n}', flush=True)

    out_p = os.path.join(args.out_dir, 'pairs.jsonl')
    with open(out_p, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # ---- 汇总
    def m(key):
        v = [r[key] for r in recs if r.get(key) is not None]
        return sum(v) / len(v) if v else float('nan')

    print('\n==== 源 → 目标（训练数据本身）====')
    print(f'样本 {len(recs)} 对')
    for k in ('sim_o', 'autopcp', 'esim'):
        print(f'  {k:8s} 配对 {m(k):.4f}   随机 {m(k + "_rand"):.4f}   delta {m(k) - m(k + "_rand"):+.4f}')
    print('\n  不依赖神经指标的交叉验证（源 vs 目标 的逐句相关）：')
    for key in ('f0_mean', 'f0_std', 'energy_std', 'pause_ratio', 'voiced_ratio', 'dur'):
        c, cnt = pearson([r['acou_src'].get(key) for r in recs],
                         [r['acou_tgt'].get(key) for r in recs])
        print(f'    {key:13s} r = {"n/a" if c is None else f"{c:+.3f}"}  (n={cnt})')

    # ---- 按「源语音情感强度」挑夸张样本供人工评审
    # 强度代理：源的 F0 标准差 + 能量标准差（都做 z-score 后相加），不依赖情感分类器
    def zs(vals):
        v = [x for x in vals if x is not None]
        if len(v) < 2:
            return [0.0] * len(vals)
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
        return [0.0 if x is None else (x - mu) / sd for x in vals]

    zf = zs([r['acou_src'].get('f0_std') for r in recs])
    ze = zs([r['acou_src'].get('energy_std') for r in recs])
    for r, a, b in zip(recs, zf, ze):
        r['expressiveness'] = round(a + b, 3)
    recs.sort(key=lambda r: -r['expressiveness'])

    rev_dir = os.path.join(args.out_dir, 'review')
    os.makedirs(rev_dir, exist_ok=True)
    import shutil
    lines = ['# 人工评审：P2 训练数据的源语音 vs 目标语音', '',
             '按「源语音表现力」（F0 标准差 + 能量标准差的 z-score 之和）从高到低取样，',
             '即**情感/语调最夸张**的样本。请听每组的 `*_src.wav`（源）与 `*_tgt.wav`（目标），',
             '判断目标是否保留了源的情感、语气、强调与语速。', '',
             '| # | id | 方向 | 表现力 | SIM-o | AutoPCP | 源文 | 译文 |',
             '|---|---|---|---|---|---|---|---|']
    picked = []
    seen_dir = {}
    for r in recs:                                   # 两个方向各取一半，避免只看一个方向
        d = r['direction']
        if seen_dir.get(d, 0) >= args.export // max(1, len(by_dir)):
            continue
        seen_dir[d] = seen_dir.get(d, 0) + 1
        picked.append(r)
        if len(picked) >= args.export:
            break
    for i, r in enumerate(picked, 1):
        sd = os.path.join(rev_dir, f"{i:02d}_{r['id']}")
        os.makedirs(sd, exist_ok=True)
        shutil.copy(r['src'], os.path.join(sd, 'src.wav'))
        shutil.copy(r['tgt'], os.path.join(sd, 'tgt.wav'))
        lines.append(f"| {i} | `{r['id']}` | {r['direction']} | {r['expressiveness']:+.2f} | "
                     f"{r['sim_o']:.3f} | {r['autopcp']:.2f} | {r['src_text'][:48]} | {r['tgt_text'][:48]} |")
    lines += ['', '## 汇总（全部样本）', '',
              '| 指标 | 配对 | 随机 | delta |', '|---|---|---|---|']
    for k in ('sim_o', 'autopcp', 'esim'):
        lines.append(f'| {k} | {m(k):.4f} | {m(k + "_rand"):.4f} | {m(k) - m(k + "_rand"):+.4f} |')
    with open(os.path.join(rev_dir, 'REVIEW.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n人工评审样本 {len(picked)} 组 -> {rev_dir}/REVIEW.md')
    print(f'逐句指标 -> {out_p}')


if __name__ == '__main__':
    main()
