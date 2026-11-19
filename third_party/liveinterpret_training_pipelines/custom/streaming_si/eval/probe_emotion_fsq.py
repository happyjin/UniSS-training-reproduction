"""输出端情感保留率（行动 4）：FSQ 单码本 + flow/HiFT 能不能承载情感？

假设（COLM 2026 论文的推论）：论文证明「语音生成目标」让 Omni 保住风格，但前提是
speech tokenizer 表示得了风格。我们的 CV3 FSQ 是**单码本 ~320bps、ASR 目标训练、设计上
去掉说话人与风格**（CosyVoice 的风格靠 flow prompt 补）。若 FSQ+flow 这条输出链路本身
压平情感，那么 b/c 等输入侧实验都救不了情感——信息进得来，出不去。

测法（与实验 1「韵律保留率」同构，指标换成情感）：

    源音频 --speech_tokenizer_v3--> FSQ token --flow(+prompt/零向量)--> HiFT --> 重建波形

    情感保留率 = cos(e2v(重建), e2v(原始))          —— 逐条
    塌缩度     = 重建向量两两余弦 vs 原始两两余弦     —— 重建若挤成一团而原始分散，
               说明这条链路抹掉了情感方差（「全成中立」的向量版）
    随机基线   = cos(e2v(重建_i), e2v(原始_j))，i≠j —— emotion2vec 随机配对本身有 0.82~0.90，
               保留率须显著高于基线才算「保留了这条音频的情感」

判读：
* 保留率 ≈ 1 且不塌缩 → 输出端不设限，情感弱是输入/数据问题（维持原结论）
* 保留率 ≈ 随机基线或重建塌缩 → **FSQ 单码本表示不了情感**，b/c 都救不了，
  要救情感得先动输出端（多码本 / flow 情感 prompt / 改数据）

用法（0a3ebd，1 卡）：
  /opt/venv/bin/python probe_emotion_fsq.py --n 200 --out_dir /nvme/emo_fsq
"""
import argparse
import glob
import io
import json
import os

import numpy as np

OUT_SR = 24000
P2 = '/nvme/speech_model_backup/stevehguo/data_fortrain/open_data_0803_fortrain'


def tar_read(shard_dir, ref):
    """'src-train-XXXXX.tar::offset:size' -> 音频字节（偏移直接指向 tar 内数据，不解 tar）。"""
    name, loc = ref.split('::', 1)
    off, ln = (int(x) for x in loc.split(':'))
    with open(os.path.join(shard_dir, name), 'rb') as f:
        f.seek(off)
        return f.read(ln)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--out_dir', default='/nvme/emo_fsq')
    ap.add_argument('--decoder', default='/fsx/stevehguo/eval/models/s4_ck/speech_decoder')
    ap.add_argument('--emotion2vec', default='/fsx/stevehguo/tts_bench/models/emotion2vec_plus_large')
    ap.add_argument('--cosyvoice_repo', default='/fsx/stevehguo/eval/code/CosyVoice')
    ap.add_argument('--prompt_sec', type=float, default=10.0)
    ap.add_argument('--ode_steps', type=int, default=5)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--score_only', action='store_true',
                    help='跳过 FSQ 往返，直接对 out_dir/{orig,zero,clone} 现有 wav 打分')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import soundfile as sf
    import torch
    if args.score_only:
        ids_only = sorted(f[:-4] for f in os.listdir(os.path.join(args.out_dir, 'orig'))
                          if f.endswith('.wav'))
        rows = [(i, None) for i in ids_only]
        print(f'score_only: {len(rows)} 条', flush=True)
    else:
        rows = _roundtrip(args, sf, torch)
    _score(args, rows, torch)


def _roundtrip(args, sf, torch):
    from streaming_si.inference.source_prompt import SourcePrompt, find_tokenizer, load_wav24
    from streaming_si.model.model_si import SI
    from streaming_si.model.speech_decoder import find_config_yaml

    # 1) 解码器（flow + HiFT），与 probe_vocoder.py 同一加载方式
    yaml_name = find_config_yaml(args.decoder)
    model = SI.__new__(SI)
    object.__setattr__(model, 'config', type('C', (), {'talker_config_yaml': yaml_name})())
    SI.attach_speech_decoder(model, args.decoder, device=args.device,
                             cosyvoice_repo=args.cosyvoice_repo)
    tok_path = find_tokenizer(args.decoder, None)
    assert tok_path, '找不到 speech_tokenizer_v3.onnx'
    sp = SourcePrompt(tok_path)
    print(f'tokenizer: {tok_path}', flush=True)

    # 2) 取 N 条源音频（machine_0 第一个 parquet，按行序）
    import pyarrow.parquet as pq
    f = sorted(glob.glob(f'{P2}/machine_0_shard/p2_fsq_v3/*.parquet'))[0]
    shard_wav = f'{P2}/machine_0_shard/p2_source_wav'
    t = pq.read_table(f, columns=['id', 'source_wav']).to_pydict()
    rows = list(zip(t['id'], t['source_wav']))[:args.n]

    # 3) FSQ 往返：tokenize -> decode（zero / clone 两种 speaker 条件）
    for n, (rid, ref) in enumerate(rows, 1):
        wav16, sr = sf.read(io.BytesIO(tar_read(shard_wav, ref)), dtype='float32', always_2d=False)
        if wav16.ndim > 1:
            wav16 = wav16.mean(1)
        tok = sp.tokens(wav16)
        for mode in ('zero', 'clone'):
            try:
                if mode == 'clone':
                    ptok, pfeat = sp(wav16, max_sec=args.prompt_sec,
                                     wav24=load_wav24(io.BytesIO(tar_read(shard_wav, ref))))
                    wav = model.decode_speech(tok, spk_emb=None, prompt_speech_tokens=ptok,
                                              prompt_feat=pfeat, ode_steps=args.ode_steps)
                else:
                    wav = model.decode_speech(tok, spk_emb=None, ode_steps=args.ode_steps)
                if hasattr(wav, 'detach'):
                    wav = wav.detach().float().cpu().numpy()
                d = os.path.join(args.out_dir, mode)
                os.makedirs(d, exist_ok=True)
                sf.write(os.path.join(d, f'{rid}.wav'), np.asarray(wav).reshape(-1), OUT_SR)
            except Exception as exc:
                print(f'[FAIL] {rid}/{mode}: {type(exc).__name__}: {exc}', flush=True)
        if n % 20 == 0 or n == len(rows):
            print(f'[fsq] {n}/{len(rows)}', flush=True)

    # 原始音频落盘（16k），供 emotion2vec 统一读取
    orig = os.path.join(args.out_dir, 'orig')
    os.makedirs(orig, exist_ok=True)
    for rid, ref in rows:
        w, _ = sf.read(io.BytesIO(tar_read(shard_wav, ref)), dtype='float32', always_2d=False)
        sf.write(os.path.join(orig, f'{rid}.wav'), w.reshape(-1), 16000)
    return rows


def _score(args, rows, torch):
    # 4) emotion2vec 打分
    from funasr import AutoModel
    e2v = AutoModel(model=args.emotion2vec, device=args.device, disable_update=True,
                    disable_pbar=True, disable_log=True)

    def emb(p):
        r = e2v.generate(p, granularity='utterance', extract_embedding=True)
        return torch.nn.functional.normalize(
            torch.from_numpy(np.asarray(r[0]['feats'], dtype=np.float32)).reshape(-1), dim=-1).numpy()

    ids = [r for r, _ in rows]
    E = {k: {i: emb(os.path.join(args.out_dir, k, f'{i}.wav')) for i in ids
             if os.path.isfile(os.path.join(args.out_dir, k, f'{i}.wav'))}
         for k in ('orig', 'zero', 'clone')}

    def pairwise(V):
        ks = list(V)
        cs = [float(V[a] @ V[b]) for x, a in enumerate(ks) for b in ks[x + 1:]]
        return float(np.mean(cs)), float(np.std(cs))

    rep = {}
    print(f'\n===== FSQ 往返情感保留（n={len(E["orig"])}）=====')
    print(f'{"模式":<8}{"保留率":>9}{"随机基线":>10}{"差":>8}{"重建两两余弦":>14}{"原始两两余弦":>14}')
    o_mean, o_std = pairwise(E['orig'])
    for mode in ('zero', 'clone'):
        pair = [(E[mode][i], E['orig'][i]) for i in E[mode] if i in E['orig']]
        ret = float(np.mean([float(a @ b) for a, b in pair]))
        # 随机配对：循环错位（与 run_para_metrics 同思路）
        ks = [i for i in E[mode] if i in E['orig']]
        rnd = float(np.mean([float(E[mode][a] @ E['orig'][b])
                             for x, a in enumerate(ks) for b in [ks[(x + 7) % len(ks)]]]))
        m_mean, m_std = pairwise(E[mode])
        rep[mode] = dict(retention=round(ret, 4), random=round(rnd, 4),
                         delta=round(ret - rnd, 4), collapse_mean=round(m_mean, 4),
                         orig_mean=round(o_mean, 4))
        print(f'{mode:<8}{ret:>9.4f}{rnd:>10.4f}{ret - rnd:>+8.4f}'
              f'{m_mean:>10.4f}±{m_std:.3f}{o_mean:>10.4f}±{o_std:.3f}')
    print('\n判读：')
    print(f'  原始两两余弦 {o_mean:.3f} —— 高于它的塌缩才叫塌缩（emotion2vec 随机配对本身不低）')
    for mode in ('zero', 'clone'):
        r = rep[mode]
        flat = r['collapse_mean'] > o_mean + 0.02
        dead = r['delta'] < 0.02
        print(f'  {mode}: delta={r["delta"]:+.4f}' + ('（≈随机基线，**情感没保留**）' if dead else '')
              + (f'，重建两两 {r["collapse_mean"]:.3f} vs 原始 {o_mean:.3f}'
                 + ('（**塌缩**）' if flat else '')))
        if dead or flat:
            print('    -> FSQ+flow 表示不了情感：b/c 都救不了情感，得先动输出端或数据')
    dst = os.path.join(args.out_dir, 'report.json')
    json.dump(rep, open(dst, 'w'), indent=2)
    print(f'-> {dst}')


if __name__ == '__main__':
    main()
