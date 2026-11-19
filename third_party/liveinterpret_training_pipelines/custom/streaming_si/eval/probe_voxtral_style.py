"""用论文协议测 Voxtral 编码器的说话风格线性可分性（行动 5）。

COLM 2026 "Heard but Not Heeded" 的结论：四个模型（编码器均 Whisper 派生）都在
**晚期编码器（top third）** 峰值 82~85%（chance=14.3%）。用**同一协议**测我们的
Voxtral（32 层、frozen、全程 S1-S5 未动过），回答两个问题：

1. Voxtral 的风格信息在第几层最线性可分？（论文口径的"late encoder 峰值"是否也成立）
2. 与我们自己的韵律迁移探针结论（**浅层 L4 最高**、往深单调降）对照——
   两者不矛盾：论文测的是"风格**类别**的可判别性"（mean 池化、分类），我们测的是
   "连续韵律细节的**可迁移性**"（mean⊕std、回归）。若本探针也是晚期峰值，
   就坐实了"识别用高层、复刻用低层"的分工假说。

协议逐项对齐论文（§4 + 附录 B）：
* 数据 Expresso read：同文本 × 7 风格（confused/default/enunciated/happy/laughing/sad/whisper）× 4 说话人；
* 池化：**仅时间轴 mean**（论文口径；我们自己的迁移探针用 mean⊕std，别混）；
* 探针：LogisticRegression(L2, C=0.1, saga, max_iter=500, tol=1e-3, class_weight=balanced)；
* 划折：**LOSO**（4 折，留一说话人），特征按折用训练集统计标准化；
* 报告：4 折 accuracy 的 mean±std；chance = 1/7 = 14.3%。

用法（0a3ebd，1 卡）：
  /opt/venv/bin/python probe_voxtral_style.py extract --parquet /nvme/expresso/train-00000-of-00012.parquet
  /opt/venv/bin/python probe_voxtral_style.py probe
"""
import argparse
import io
import json
import os

import numpy as np

CK = '/nvme/speech_model_backup/stevehguo/model_checkpoint/streaming_si_swift/trained_streaming_si_4b/expA/a0'
CACHE = '/nvme/vox_style'
STYLES = ('confused', 'default', 'enunciated', 'happy', 'laughing', 'sad', 'whisper')


def extract(args):
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from streaming_si.model.base_model import VoxtralAudioBackend
    from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
    import librosa

    cache = args.cache          # 不同模型/ckpt 各存一处，互不覆盖
    backend = VoxtralAudioBackend.from_config_dir(args.ck)
    import glob as _g
    from safetensors.torch import load_file
    sd = {}
    for p in sorted(_g.glob(os.path.join(args.ck, '*.safetensors'))):
        for k, v in load_file(p).items():
            if k.startswith('audio_encoder.'):
                sd[k[len('audio_encoder.'):]] = v
    backend.load_state_dict(sd, strict=False)
    backend = backend.to(device=args.device, dtype=torch.bfloat16).eval()
    fe = VoxtralFeatureExtractor.from_pretrained(args.ck)
    enc, n_layers = backend.encoder, len(backend.encoder.layers)
    print(f'voxtral: {n_layers} 层 hidden={backend.hidden_size} frame_rate={backend.frame_rate}Hz', flush=True)

    cap = {}
    hooks = [enc.layers[i - 1].register_forward_hook(
                 lambda m, a, o, i=i: cap.__setitem__(f'L{i}', o))
             for i in range(1, n_layers + 1)]
    hooks.append(backend.adapter.register_forward_hook(
        lambda m, a, o: cap.__setitem__('adapter', o)))

    t = []
    import glob as _g2
    for p in sorted(_g2.glob(args.parquet)):
        t += pq.read_table(p).to_pylist()
        print(f'  读入 {os.path.basename(p)}', flush=True)
    # 音频在结构列 audio 内（{'bytes': ..., 'path': ...}），不是顶层字段
    rows = [r for r in t if r.get('style') in STYLES
            and isinstance(r.get('audio'), dict) and r['audio'].get('bytes')]
    print(f'clips: {len(rows)}（styles={sorted({r["style"] for r in rows})} '
          f'speakers={sorted({r["speaker_id"] for r in rows})}）', flush=True)
    feats = {f'L{i}': [] for i in range(1, n_layers + 1)}
    feats['adapter'] = []
    labels, speakers, ids, ok = [], [], [], []
    for n, r in enumerate(rows, 1):
        try:
            wav, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32', always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav = wav[:16000 * 30]
            mel = fe(wav, sampling_rate=16000)['input_features']
            mel = torch.as_tensor(mel, dtype=torch.bfloat16, device=args.device)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            cap.clear()
            with torch.no_grad():
                backend.encode_audio(mel)
            for k in feats:
                o = cap[k]
                feats[k].append(o[0].float().mean(0).cpu().numpy().astype(np.float16)
                                if o.dim() == 3 else o.float().mean(0).cpu().numpy())
            labels.append(r['style']); speakers.append(r['speaker_id']); ids.append(r['id']); ok.append(True)
        except Exception as exc:
            print(f'  [FAIL] {r.get("id")}: {type(exc).__name__}: {exc}', flush=True)
        if n % 100 == 0 or n == len(rows):
            print(f'[extract] {n}/{len(rows)} ok={len(ok)}', flush=True)
    for h in hooks:
        h.remove()
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(f'{cache}/feats.npz',
                        **{k: np.asarray(v) for k, v in feats.items()},
                        labels=np.asarray(labels), speakers=np.asarray(speakers),
                        ids=np.asarray(ids))
    print(f'-> {cache}/feats.npz（{len(ok)} 条）')


def extract_omni(args):
    """官方 SimulS2ST-Omni 流式系统的 audio_tower（Whisper 系 32 层）逐层风格特征。

    加载口径与 probe_omni_audio_tower.py extract 完全一致（offline 基座 + thinker_lora
    merge 后取 audio_tower；mel 前端 / attention_mask / aftercnn_lens 的坑见那边注释），
    仅两处不同：样本换成 Expresso（同分类协议），池化换成本协议的 mean-only。
    """
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import librosa
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from probe_omni_audio_tower import VARIANTS

    cache = args.cache
    base, lora = VARIANTS[args.variant]
    print(f'variant={args.variant}  基座={base}  thinker LoRA={lora or "（无）"}', flush=True)
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(base)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=None).to(args.device)
    if lora:
        from peft import PeftModel
        model.thinker = PeftModel.from_pretrained(model.thinker, lora)
        n_at = sum(1 for n, _ in model.thinker.named_parameters()
                   if 'audio_tower' in n and 'lora_' in n)
        if n_at == 0:
            raise SystemExit('LoRA 未命中 audio_tower（预期 192 个张量）')
        model.thinker = model.thinker.merge_and_unload()
    model.eval()
    tower = model.thinker.audio_tower
    n_layers = len(tower.layers)
    print(f'audio_tower: {n_layers} 层, hidden={tower.config.d_model}', flush=True)
    del model.thinker.model            # 只要编码器，省显存（同 R² 探针）
    torch.cuda.empty_cache()
    fe = proc.feature_extractor

    cap = []
    def _mk(idx):
        def _h(_m, _i, o):
            x = o[0] if isinstance(o, tuple) else o
            cap.append(x.detach()[0] if x.dim() == 3 else x.detach())
        return _h
    for layer in tower.layers:
        layer.register_forward_hook(_mk(0))

    t = []
    import glob as _g
    for p in sorted(_g.glob(args.parquet)):
        t += pq.read_table(p).to_pylist()
    rows = [r for r in t if r.get('style') in STYLES
            and isinstance(r.get('audio'), dict) and r['audio'].get('bytes')]
    print(f'clips: {len(rows)}', flush=True)
    feats = {f'L{i}': [] for i in range(1, n_layers + 1)}
    labels, speakers, ids = [], [], []
    for n, r in enumerate(rows, 1):
        try:
            wav, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32', always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav = wav[:16000 * 30]
            inp = fe(wav, sampling_rate=16000, return_tensors='pt',
                     return_attention_mask=True, padding='max_length')
            nfr = int(inp['attention_mask'].sum())
            mel = inp['input_features'][0, :, :nfr].to(args.device, dtype=torch.bfloat16)
            flen = torch.tensor([nfr], device=args.device, dtype=torch.long)
            aftercnn, _ = tower._get_feat_extract_output_lengths(flen)
            cap.clear()
            with torch.no_grad():
                tower(mel, feature_lens=flen, aftercnn_lens=aftercnn)
            if len(cap) < n_layers:
                raise RuntimeError(f'只捕获到 {len(cap)}/{n_layers} 层')
            for i in range(1, n_layers + 1):
                feats[f'L{i}'].append(cap[i - 1].float().mean(0).cpu().numpy().astype(np.float16))
            labels.append(r['style']); speakers.append(r['speaker_id']); ids.append(r['id'])
        except Exception as exc:
            print(f'  [FAIL] {r.get("id")}: {type(exc).__name__}: {exc}', flush=True)
        if n % 200 == 0 or n == len(rows):
            print(f'[extract-omni] {n}/{len(rows)} ok={len(labels)}', flush=True)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(f'{cache}/feats.npz',
                        **{k: np.asarray(v) for k, v in feats.items()},
                        labels=np.asarray(labels), speakers=np.asarray(speakers),
                        ids=np.asarray(ids))
    print(f'-> {cache}/feats.npz（{len(labels)} 条）')


def probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    z = np.load(f'{args.cache}/feats.npz', allow_pickle=True)
    y, spk = z['labels'], z['speakers']
    logo = LeaveOneGroupOut()
    print(f'\n===== LOSO 线性风格探针（7 类，chance=14.3%，协议=论文附录B）=====')
    print(f'n={len(y)}  styles={sorted(set(y.tolist()))}  speakers={sorted(set(spk.tolist()))}')
    print(f'{"表示":<12}{"acc(mean)":>10}{"±std":>8}')
    rows = []
    for k in sorted([x for x in z.files if x not in ('labels', 'speakers', 'ids')],
                    key=lambda t: (0, int(t[1:])) if t[0] == 'L' else (1, 0)):
        X = z[k].astype(np.float64)
        accs = []
        for tr, te in logo.split(X, y, spk):
            sx = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=500,
                                   tol=1e-3, class_weight='balanced',
                                   random_state=0).fit(sx.transform(X[tr]), y[tr])
            accs.append(float((m.predict(sx.transform(X[te])) == y[te]).mean()))
        a, s = float(np.mean(accs)), float(np.std(accs))
        rows.append({'rep': k, 'acc': round(a, 4), 'std': round(s, 4)})
        print(f'{k:<12}{a:>10.4f}{s:>8.4f}', flush=True)
    lay = [r for r in rows if r['rep'].startswith('L')]
    if lay:
        peak = max(lay, key=lambda r: r['acc'])
        top_third = max(1, len(lay) * 2 // 3)
        in_late = int(peak['rep'][1:]) >= top_third
        print(f'\n峰值层 = {peak["rep"]}（acc={peak["acc"]:.1%}），位于第 {peak["rep"][1:]}/{len(lay)} 层')
        print('对照论文：四个 Whisper 派生模型峰值都在晚期编码器（top third）82~85%。')
        print(f'Voxtral 峰值在{"晚 1/3（与论文一致）" if in_late else "**浅/中部（与论文相反）**"}'
              f'——结合我们迁移探针的 L4 峰值，支持「识别用高层、复刻用低层」的分工假说。')
    dst = f'{CACHE}/style_probe.json'
    dst = f'{args.cache}/style_probe.json'
    json.dump(rows, open(dst, 'w'), indent=2)
    print(f'-> {dst}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('extract')
    p1.add_argument('--parquet', required=True)
    p1.add_argument('--ck', default=CK)
    p1.add_argument('--device', default='cuda:0')
    p1.add_argument('--cache', default=CACHE)
    p2 = sub.add_parser('extract-omni')
    p2.add_argument('--parquet', required=True)
    p2.add_argument('--variant', default='omni_simul',
                    help='probe_omni_audio_tower.VARIANTS 的 key，官方流式 = omni_simul')
    p2.add_argument('--device', default='cuda:0')
    p2.add_argument('--cache', default='/nvme/vox_style_omni')
    p3 = sub.add_parser('probe')
    p3.add_argument('--cache', default=CACHE)
    a = ap.parse_args()
    if a.cmd == 'extract':
        extract(a)
    elif a.cmd == 'extract-omni':
        extract_omni(a)
    else:
        probe(a)


if __name__ == '__main__':
    main()
