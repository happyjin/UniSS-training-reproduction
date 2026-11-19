"""Bridge 四点连续韵律探针（A/B/C/D/E）—— 判断连续韵律在桥接与融合环节的丢失位置。

背景：风格分类（修复版）显示情绪类别能过 Bridge，但连续韵律（F0 起伏/能量起伏/语速/停顿）
是否过桥未知。此前漏斗只测到 thinker 末层（tgt_L32，S4@6000 为 0.092）。

测点（全部取自真实部署路径，bridge_layer 保持 ckpt 的 -1，不改任何配置）：
  A  实际传入 Bridge 的 thinker hidden = forward_thinker(...).last_hidden_state
     （pick_bridge_hidden(-1) 的返回值，已过 final norm；在调用处捕获，不按层号猜）
  B  GatedBridge(thinker_proj) 输出        —— 检查投影/门控
  C  Fusion 相加后、final RMSNorm 前       —— 检查融合加和（fusion.norm 的输入）
  D  Fusion 最终输出（fusion.norm 的输出） —— 检查归一化
  E  Talker 第一层 @cond 段                —— 与修复版风格探针衔接

目标（4 项）：lf0_std(F0 起伏) / en_std(能量起伏) / rate(语速) / pause(停顿占比)。
Y = 目标韵律对文本特征回归后的残差（与 paraling_probe / probe_mlp 完全同一口径），
表示 = 各测点 mean⊕std 池化，回归 = RidgeCV，折 = KFold(5, shuffle, seed=0)。
说话人分组：P2/CommonVoice 无 speaker 标签（parquet 与 prosody.pkl 均已核实），
主口径沿用漏斗同款折以保持逐位可比；配对差异（B−A、D−B）用 bootstrap 置信区间。

用法（GPU，可分片）：
  python probe_bridge_prosody.py extract --ck ... --shard 0 --nshards 7 \
      --directions eng->cmn,cmn->eng --out /nvme/bridge_prosody
  python probe_bridge_prosody.py analyze --out /nvme/bridge_prosody
"""
import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SI_CODE', '/workspace/custom'))

PKL = '/fsx/si_s3_work/paraling/out/prosody.pkl'
STAGE = '/fsx/si_s3_work/paraling/stage'      # src tar 实际位置（pkl 里记的是当年 Pod 内路径）
KEYS = ['lf0_mean', 'lf0_std', 'final_rise', 'en_mean', 'en_std', 'rate', 'pause', 'voiced']
TARGETS = ['lf0_std', 'en_std', 'rate', 'pause']
POINTS = ['A', 'B', 'C', 'D', 'E', 'S', 'DS']
# 消融测点（探针级，不训练）：
#   S  = 逐 token log-RMS(C) —— Fusion 末尾 RMSNorm 抹掉的唯一自由度（每 token 1 维幅值）
#   DS = D ⊕ log-RMS(C)       —— "保留幅值的 scale 通道"：若能补回 C→D 的损失，说明丢的正是幅值
PAIRS = (('A', 'B'), ('B', 'D'), ('D', 'E'), ('D', 'DS'), ('C', 'DS'))


def read_wav(tar_dir, ref):
    import soundfile as sf
    name, loc = ref.split('::', 1)
    off, ln = map(int, loc.split(':'))
    with open(os.path.join(tar_dir, name), 'rb') as fp:
        fp.seek(off)
        wav, sr = sf.read(io.BytesIO(fp.read(ln)), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(1)
    return wav, sr


def pool(x):
    """(T, D) -> mean⊕std，float16。与 paraling_probe._pool 逐位相同。"""
    x = x.float()
    return np.concatenate([x.mean(0).cpu().numpy(), x.std(0, unbiased=False).cpu().numpy()]).astype(np.float16)


def extract(args):
    import pickle
    import torch
    from transformers import AutoTokenizer
    from streaming_si.base_model import VoxtralAudioProcessor
    from streaming_si.si_encode import audio_token_id, encode_st, normalize_fsq
    from streaming_si.modeling_streaming_si import StreamingSIForConditionalGeneration
    from streaming_si.voxtral_compat import VoxtralFeatureExtractor

    with open(PKL, 'rb') as f:
        data = pickle.load(f)

    # 注意：不设 talker_bridge_layer！部署口径 -1，A 直接取 last_hidden_state。
    model = StreamingSIForConditionalGeneration.from_pretrained(
        args.ck, dtype=torch.bfloat16).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.ck)
    mel_fn = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.ck))
    a_id = audio_token_id(tok)

    cap = {}

    def _hook_B(m, a, o):
        cap.setdefault('B', o)      # 注意：hook 返回非 None 会替换模块输出，必须不写 return

    def _hook_CD(m, a, o):
        cap.setdefault('C', a[0])   # Fusion 相加后、RMSNorm 前
        cap.setdefault('D', o)     # Fusion 最终输出

    hooks = [
        model.talker.thinker_proj.register_forward_hook(_hook_B),
        model.talker.fusion.norm.register_forward_hook(_hook_CD),
    ]

    for direction in args.directions.split(','):
        d = data[direction]
        rows = d['rows'][args.shard::args.nshards]
        res = {p: [] for p in POINTS}
        ok = []
        for r in rows:
            try:
                sw, ssr = read_wav(STAGE, r['source_wav'])
                if ssr != 16000:
                    import librosa
                    sw = librosa.resample(sw, orig_sr=ssr, target_sr=16000)
                sw = sw[:30 * 16000]
                mel = mel_fn(torch.from_numpy(sw), sr=16000)
                e = encode_st(tok, mel, r['translation'] or '', r['tgt_lang'], max_length=768,
                              max_target_len=256, speech_tokens=normalize_fsq(r['target_fsq']),
                              max_speech_len=750, with_answer_mask=True)
                ids = torch.tensor([e['input_ids']], device=args.device)
                am = torch.tensor([e['answer_mask']], device=args.device)   # (1,L)：gather_answer_hidden 按 dim=1 求和
                ans_pos = am[0].bool()      # 与 cond_mask（S4 mode='answer'）完全一致：进 Bridge 的 token 集
                if int(ans_pos.sum()) < 1:
                    raise ValueError('空译文位置')
                with torch.no_grad():
                    cap.clear()
                    out = model.forward_thinker(
                        ids, audio_inputs=e['audio_inputs'][None].to(args.device),
                        audio_lens=torch.tensor([e['audio_lens']], device=args.device))
                    A = out.last_hidden_state          # 部署路径：pick_bridge_hidden(-1) 的返回值
                    cond, cond_lens = model.build_talker_condition(out, ids, am)  # 触发 B/C/D 钩子
                    inputs, attn, _ = model.talker.build_inputs(cond, cond_lens)
                    t_out = model.talker.backbone(inputs_embeds=inputs, attention_mask=attn,
                                                   output_hidden_states=True)
                B, C, D = cap['B'], cap['C'], cap['D']
                c_len = int(cond_lens[0])
                cmask = torch.zeros(inputs.size(1), dtype=torch.bool, device=args.device)
                cmask[1:1 + c_len] = True
                res['A'].append(pool(A[0, ans_pos]))    # 末层，进 Bridge 的那批 token（与 B/C/D 同一 token 集）
                res['B'].append(pool(B[0]))             # GatedBridge 后，已是收集后的译文位置
                res['C'].append(pool(C[0]))             # 融合加和后、归一化前
                res['D'].append(pool(D[0]))             # 融合输出
                res['E'].append(pool(t_out.hidden_states[1][0, cmask]))   # talker L1 @cond 段
                c32 = C[0].float()
                logrms = torch.log(c32.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)   # (T,1)
                res['S'].append(pool(logrms))
                res['DS'].append(pool(torch.cat([D[0].float(), logrms], -1)))
                ok.append(True)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                print(f'  [FAIL] {r.get("id")}: {type(ex).__name__}: {ex}', flush=True)
                ok.append(False)
        os.makedirs(args.out, exist_ok=True)
        np.savez(f'{args.out}/shard{args.nshards}_{args.shard}_{direction.replace("->", "2")}.npz',
                 ok=np.array(ok), **{p: np.stack(v) if v else np.zeros((0, 2), np.float16) for p, v in res.items()})
        print(f'[extract] {direction} shard{args.shard}/{args.nshards}: '
              f'{sum(ok)}/{len(rows)} 条写出', flush=True)
    for h in hooks:
        h.remove()


def analyze(args):
    import glob
    import pickle
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    with open(PKL, 'rb') as f:
        data = pickle.load(f)
    report = {}
    for direction in sorted({os.path.basename(p).split('_', 2)[2][:-4]
                             for p in glob.glob(f'{args.out}/shard*_*.npz')}):
        d = data[direction.replace('2', '->', 1)]
        shards = sorted(glob.glob(f'{args.out}/shard*_{direction}.npz'))
        zs = [np.load(p, allow_pickle=True) for p in shards]
        # 提取时分片按 rows[shard::nshards] **交错**取样；必须按原始行号还原顺序再与 Y 对齐
        # （旧版直接顺序拼接 → X/Y 错位，全部 R²≈0，2026-10-22 发现并修复）。
        n_rows = len(d['rows'])
        gidx, okc = [], []
        for p, z in zip(shards, zs):
            ns, s = map(int, os.path.basename(p)[len('shard'):].split('_')[:2])
            gi = np.arange(s, n_rows, ns)
            assert len(gi) == len(z['ok']), f'{p}: 行数 {len(z["ok"])} != 期望 {len(gi)}'
            gidx.append(gi); okc.append(z['ok'])
        gidx, okc = np.concatenate(gidx), np.concatenate(okc).astype(bool)
        valid = gidx[okc]                       # X 拼接后各行对应的原始行号
        order = np.argsort(valid)
        rows_ok = valid[order]
        n_tot = n_rows
        ok = np.zeros(n_rows, dtype=bool); ok[rows_ok] = True
        tgt = np.asarray(d['tgt'], dtype=np.float64)[rows_ok]
        txt = np.asarray(d['txt'], dtype=np.float64)[rows_ok]
        Y = tgt[:, [KEYS.index(k) for k in TARGETS]].copy()
        # 残差化：目标韵律去掉文本可解释部分（与 paraling_probe 完全一致）
        Z = np.column_stack([np.ones(len(txt)), txt])
        Y = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]

        Xs, preds = {}, {}
        points = [p for p in POINTS if all(p in z.files for z in zs)]   # 兼容无消融测点的旧 npz
        pairs = [(lo, hi) for lo, hi in PAIRS if lo in points and hi in points]
        for p in points:
            X = np.concatenate([z[p] for z in zs], 0).astype(np.float64)[order]
            Xs[p] = X
            oof = np.zeros_like(Y)
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
                m = RidgeCV(alphas=np.logspace(-1, 6, 15), alpha_per_target=True)
                m.fit(sx.transform(X[tr]), sy.transform(Y[tr]))
                oof[te] = sy.inverse_transform(m.predict(sx.transform(X[te])))
            preds[p] = oof

        def r2(pred, idx):
            e = ((Y[idx] - pred[idx]) ** 2).sum(0)
            t = ((Y[idx] - Y[idx].mean(0)) ** 2).sum(0)
            return 1 - e / np.maximum(t, 1e-12)

        rng = np.random.default_rng(0)
        all_idx = np.arange(len(Y))
        n_boot = 2000
        rows = []
        for ti, tk in enumerate(TARGETS):
            r = {'target': tk}
            for p in points:
                r[p] = round(float(r2_single(Y[:, ti], preds[p][:, ti])), 4)
            for lo, hi in pairs:
                deltas = []
                for _ in range(n_boot):
                    b = rng.choice(all_idx, len(all_idx), replace=True)
                    deltas.append(r2_single(Y[b, ti], preds[hi][b, ti]) -
                                   r2_single(Y[b, ti], preds[lo][b, ti]))
                deltas = np.sort(np.array(deltas))
                r[f'{hi}-{lo}_ci'] = [round(float(deltas[int(.025 * n_boot)]), 4),
                                      round(float(deltas[int(.975 * n_boot)]), 4)]
            rows.append(r)
        report[direction] = {'n': int(ok.sum()), 'n_tot': int(n_tot), 'rows': rows}
        print(f'\n=== {direction}  n={int(ok.sum())}/{n_tot}  Y=目标韵律去文本残差  '
              f'折=KFold(5,shuffle,seed=0) ===')
        print(f"{'目标':<8}" + ''.join(f'{p:>8}' for p in points) +
              ''.join(f"{hi + '-' + lo + ' 95%CI':>18}" for lo, hi in pairs))
        for r in rows:
            print(f"{r['target']:<8}" + ''.join(f"{r[p]:>8.3f}" for p in points) +
                  ''.join(f"  [{r[f'{hi}-{lo}_ci'][0]:+.3f},{r[f'{hi}-{lo}_ci'][1]:+.3f}]"
                          for lo, hi in pairs))
    import json
    json.dump(report, open(f'{args.out}/report.json', 'w'), ensure_ascii=False, indent=2)
    print(f"-> {args.out}/report.json")
    print('注：P2/CommonVoice 无 speaker 标签（parquet 与 prosody.pkl 均已核实），'
          '无法做真实说话人分组；主折式与既有漏斗逐位一致，配对差异用 bootstrap CI。')


def r2_single(y, pred):
    e = ((y - pred) ** 2).sum()
    t = ((y - y.mean()) ** 2).sum()
    return 1 - e / max(t, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--ck', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--shard', type=int, default=0)
    e.add_argument('--nshards', type=int, default=1)
    e.add_argument('--directions', default='eng->cmn,cmn->eng')
    e.add_argument('--device', default='cuda:0')
    a = sub.add_parser('analyze')
    a.add_argument('--out', required=True)
    args = ap.parse_args()
    (extract if args.cmd == 'extract' else analyze)(args)


if __name__ == '__main__':
    main()
