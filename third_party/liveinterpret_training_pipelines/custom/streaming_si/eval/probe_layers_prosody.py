"""跨层连续韵律探针：S4 的 audio encoder(32L) → adapter → audio_proj → thinker(32L) → Bridge → talker(28L)。

目的：把 SUMMARY §10 的 A–E 四点扩展成**逐层**曲线，定位连续韵律（F0 起伏 / 能量起伏 /
语速 / 停顿占比）在哪一层丢失、哪一层恢复。同一批 P2 对、同一译文位置 mask、
同一 Y（目标韵律对文本特征回归的残差）、同一 RidgeCV + KFold(5, shuffle, seed=0)。

测点（全部取自部署路径，bridge_layer 最终保持 −1）：
  enc_Lk   (k=1..32)  Voxtral 编码器第 k 层输出（50Hz，仅取有效帧）
  adapter            编码器 adapter 输出（12.5Hz，3072d）
  aproj              audio_proj 输出（→ thinker 宽度）
  thk_aud_Lk         thinker 第 k 层 @ 音频 token 位置（k=0 为输入 embedding）
  thk_ans_Lk         thinker 第 k 层 @ 译文 token 位置（进 Bridge 的那一路）
  A / B / C / D / E   Bridge 四点（同 probe_bridge_prosody 口径）
  tkr_Lk             talker 第 k 层 @ cond 段（k=1..28）

用法（GPU，可分片跑）：
  python probe_layers_prosody.py extract --ck <S4 ckpt> --shard 0 --nshards 3 --out /nvme/layer_prosody
  python probe_layers_prosody.py analyze --out /nvme/layer_prosody
"""
import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SI_CODE', '/workspace/custom'))
from streaming_si.eval.probe_bridge_prosody import (PKL, STAGE, KEYS, TARGETS,  # noqa: E402
                                                    pool, read_wav)

N_ENC = 32
N_TKR = 28


def extract(args):
    import pickle
    import torch
    from transformers import AutoTokenizer
    from streaming_si.base_model import VoxtralAudioProcessor
    from streaming_si.si_encode import audio_token_id, encode_st, normalize_fsq
    from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor

    with open(PKL, 'rb') as f:
        data = pickle.load(f)
    model = StreamingSIForConditionalGeneration.from_pretrained(
        args.ck, dtype=torch.bfloat16).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.ck)
    mel_fn = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.ck))
    a_id = audio_token_id(tok)

    enc = model.audio_encoder.encoder            # VoxtralEncoder：embedder/layers/norm
    n_enc = len(enc.layers)
    n_thk = model.config.num_hidden_layers
    n_tkr = len(model.talker.backbone.layers)

    cap = {}

    def _mk(key):
        def _f(m, a, o):                          # 返回 None，绝不替换模块输出
            cap.setdefault(key, o)
        return _f

    hooks = []
    for i in range(n_enc):
        hooks.append(enc.layers[i].register_forward_hook(_mk(f'enc_L{i + 1}')))
    hooks.append(enc.norm.register_forward_hook(_mk('enc_norm')))
    for i in range(n_tkr):
        hooks.append(model.talker.backbone.layers[i].register_forward_hook(_mk(f'tkr_L{i + 1}')))
    def _mk_CD(m, a, o):
        cap.setdefault('C', a[0])        # 相加后、RMSNorm 前
        cap.setdefault('D', o)           # 归一化后（不写 return，否则会替换模块输出）

    hooks.append(model.talker.thinker_proj.register_forward_hook(_mk('B')))
    hooks.append(model.talker.fusion.norm.register_forward_hook(_mk_CD))
    hooks.append(model.audio_proj.register_forward_hook(_mk('aproj')))   # 投影后（12.5Hz）

    # audio_proj 只在 encode_audio_inputs 路径上跑（forward_thinker 传 mel 会绕过它），故走完整 forward。
    orig_bridge = model.config.talker_bridge_layer
    model.config.talker_bridge_layer = 0         # 让 thinker 返回全部层；算 cond 前恢复
    print(f'enc {n_enc}L  thinker {n_thk}L  talker {n_tkr}L', flush=True)

    for direction in args.directions.split(','):
        d = data[direction]
        rows = d['rows'][args.shard::args.nshards]
        res, ok = {}, []
        for n, r in enumerate(rows, 1):
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
                am = torch.tensor([e['answer_mask']], device=args.device)
                aud = e['audio_inputs'][None].to(args.device)
                alen = int(e['audio_lens'])
                ans_pos = am[0].bool()
                aud_pos = ids[0] == a_id
                if int(ans_pos.sum()) < 1 or int(aud_pos.sum()) < 1:
                    raise ValueError('空位置')
                with torch.no_grad():
                    cap.clear()
                    model.config.talker_bridge_layer = 0
                    # 走部署路径：encode_audio_inputs 内部是 encoder → _apply_adapter → audio_proj，
                    # 与训练/推理完全一致（不能手动拼 adapter，输入是 4 帧拼接后的 5120 维）。
                    emb = model.encode_audio_inputs(
                        aud, torch.tensor([alen], device=args.device))
                    out = model.forward_thinker(
                        ids, inputs_embeds=None, audio_inputs=aud,
                        audio_lens=torch.tensor([alen], device=args.device))
                    hs = out.hidden_states
                    model.config.talker_bridge_layer = orig_bridge
                    cond, cond_lens = model.build_talker_condition(out, ids, am)
                    inputs, attn, _ = model.talker.build_inputs(cond, cond_lens)
                    _ = model.talker.backbone(inputs_embeds=inputs, attention_mask=attn,
                                              output_hidden_states=True)
                # 编码器有效帧数：adapter 之后 = 音频 token 数（12.5Hz），编码器侧为其 4 倍
                n_tok = int(aud_pos.sum())
                v_enc = min(int(cap['enc_L1'].shape[1]), max(1, alen))
                sample = {
                    'A': pool(out.last_hidden_state[0, ans_pos]),
                    'B': pool(cap['B'][0]),
                    'C': pool(cap['C'][0]),
                    'D': pool(cap['D'][0]),
                }
                for i in range(1, n_enc + 1):
                    sample[f'enc_L{i}'] = pool(cap[f'enc_L{i}'][0, :v_enc])
                sample['enc_norm'] = pool(cap['enc_norm'][0, :v_enc])
                if 'aproj' in cap:
                    apr = cap['aproj']
                    apr = (apr[0] if isinstance(apr, (list, tuple)) else apr)
                    sample['aproj'] = pool(apr.float()[0, :n_tok] if apr.dim() == 3 else apr[0, :n_tok])
                if emb is not None and len(emb):
                    sample['adapter'] = pool(emb[0][:n_tok].float())
                for k in range(n_thk + 1):
                    sample[f'thk_aud_L{k}'] = pool(hs[k][0, aud_pos])
                    sample[f'thk_ans_L{k}'] = pool(hs[k][0, ans_pos])
                c_len = int(cond_lens[0])
                cmask = torch.zeros(inputs.size(1), dtype=torch.bool, device=args.device)
                cmask[1:1 + c_len] = True
                for i in range(1, n_tkr + 1):
                    sample[f'tkr_L{i}'] = pool(cap[f'tkr_L{i}'][0, cmask])
                if not res:
                    res = {k: [] for k in sample}
                for k, v in sample.items():
                    res[k].append(v)
                ok.append(True)
            except Exception as ex:
                import traceback
                if os.environ.get('PROBE_DEBUG'):
                    traceback.print_exc()
                print(f'  [FAIL] {r.get("id")}: {type(ex).__name__}: {ex}', flush=True)
                ok.append(False)
            if n % 100 == 0:
                print(f'[layers {args.shard}] {n}/{len(rows)} ok={sum(ok)}', flush=True)
        os.makedirs(args.out, exist_ok=True)
        np.savez(f'{args.out}/shard{args.nshards}_{args.shard}_{direction.replace("->", "2")}.npz',
                 ok=np.array(ok),
                 **{k: np.stack(v) if v else np.zeros((0, 2), np.float16) for k, v in res.items()})
        print(f'[extract] {direction} shard{args.shard}/{args.nshards}: {sum(ok)}/{len(rows)} 写出',
              flush=True)
    for h in hooks:
        h.remove()


def analyze(args):
    import glob
    import json
    import pickle
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    with open(PKL, 'rb') as f:
        data = pickle.load(f)
    out = {}
    # 文件名形如 shard{nshards}_{shard}_{eng2cmn|cmn2eng}；nshards 位数可变，故用后缀精确匹配
    dirs = sorted({os.path.basename(p).rsplit('_', 1)[1][:-4]
                   for p in glob.glob(f'{args.out}/shard*_*.npz')})
    for direction in dirs:
        d = data[direction.replace('2', '->', 1)]
        shards = []
        for p in sorted(glob.glob(f'{args.out}/shard*_{direction}.npz')):
            z = np.load(p, allow_pickle=True)
            if int(np.asarray(z['ok']).sum()) == 0:
                continue                 # 跳过空分片（调试残留 / 全失败）
            shards.append(p)
        if not shards:
            print(f'  {direction}: 无有效分片，跳过')
            continue
        zs = [np.load(p, allow_pickle=True) for p in shards]
        n_rows = len(d['rows'])
        gidx, okc = [], []
        for p, z in zip(shards, zs):
            ns, s = map(int, os.path.basename(p)[len('shard'):].split('_')[:2])
            gi = np.arange(s, n_rows, ns)
            assert len(gi) == len(z['ok']), f'{p} 行数不符'
            gidx.append(gi); okc.append(z['ok'])
        gidx, okc = np.concatenate(gidx), np.concatenate(okc).astype(bool)
        order = np.argsort(gidx[okc]); rows_ok = gidx[okc][order]
        tgt = np.asarray(d['tgt'], np.float64)[rows_ok]
        txt = np.asarray(d['txt'], np.float64)[rows_ok]
        Y = tgt[:, [KEYS.index(k) for k in TARGETS]].copy()
        Z = np.column_stack([np.ones(len(txt)), txt])
        Y = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]

        keys = [k for k in zs[0].files if k != 'ok']
        fold = list(KFold(5, shuffle=True, random_state=0).split(np.zeros((len(Y), 1))))
        table = {}
        for k in keys:
            X = np.concatenate([z[k] for z in zs], 0).astype(np.float64)[order]
            if X.shape[1] == 0:
                continue
            oof = np.zeros_like(Y)
            for tr, te in fold:
                sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
                m = RidgeCV(alphas=np.logspace(-1, 6, 15), alpha_per_target=True)
                m.fit(sx.transform(X[tr]), sy.transform(Y[tr]))
                oof[te] = sy.inverse_transform(m.predict(sx.transform(X[te])))
            for ti, tk in enumerate(TARGETS):
                e = ((Y[:, ti] - oof[:, ti]) ** 2).sum()
                t = ((Y[:, ti] - Y[:, ti].mean()) ** 2).sum()
                table.setdefault(k, {})[tk] = round(float(1 - e / max(t, 1e-12)), 4)
            print(f'  {k:<12} ' + '  '.join(f'{tk}={table[k][tk]:+.3f}' for tk in TARGETS), flush=True)
        out[direction] = {'n': int(len(Y)), 'table': table}
        json.dump(out, open(f'{args.out}/layers_report.json', 'w'), ensure_ascii=False, indent=2)
    print(f'-> {args.out}/layers_report.json')


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
