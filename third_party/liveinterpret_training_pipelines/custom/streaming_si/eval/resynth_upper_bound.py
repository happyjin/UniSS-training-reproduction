"""复制合成上界：源音频 → 自己的 speech token → flow/HiFT 重建，测 SIM-o。

目的：把 talker 完全排除在外，量出「flow + CAM++（+ 可选声学 prompt）」这条声码器通路
重建源说话人音色的**上限**。解释路线：

* 上界 ≈ 0.95：声码器通路没问题，S2ST 的 SIM-o 差距来自 talker 产出的 token 与
  跨语种内容（换语言、换时长），换说话人编码器收益有限；
* 上界 ≈ 0.80：声码器通路本身就是天花板（CAM++ 的 192 维向量信息不足 /
  `spk_embed_affine_layer` 拟合有限），此时才值得动编码器——但必须连带重训该层。

与 S2ST 推理共用同一套 CAM++、tokenizer、flow、HiFT 与 ODE 步数，只换掉 token 来源，
因此可与 `run_timbre_ablation.sh` 的各臂直接比。

用法：
    python resynth_upper_bound.py --model_dir <s4_ck> --input_jsonl <manifest> \
        --out_dir <dir> --cosyvoice_repo <CosyVoice> [--source_prompt] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streaming_si.inference.infer_s2st import OUT_SR, SR, CampplusSpeaker, load_wav16  # noqa: E402
from streaming_si.inference.source_prompt import SourcePrompt, find_tokenizer  # noqa: E402
from streaming_si.model.speech_decoder import resolve_speech_decoder  # noqa: E402
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402


def load_wav24(path):
    import soundfile as sf
    import librosa
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav if sr == OUT_SR else librosa.resample(wav, orig_sr=sr, target_sr=OUT_SR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--cosyvoice_repo', default=os.environ.get('COSYVOICE_REPO'))
    ap.add_argument('--prompt_tokenizer', default=None)
    ap.add_argument('--source_prompt', action='store_true',
                    help='除 spk_emb 外再给 flow 声学 prompt（取源音频前 prompt_sec 秒）。'
                         '注意 prompt 与被重建内容同源，上界会偏乐观')
    ap.add_argument('--prompt_sec', type=float, default=10.0)
    ap.add_argument('--no_speaker', dest='speaker', action='store_false',
                    help='不给 CAM++ 向量（配合 --source_prompt 可单独量 prompt 通路的贡献）')
    ap.add_argument('--ode_steps', type=int, default=5)
    ap.add_argument('--max_audio_sec', type=float, default=30.0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args(argv)

    if not args.cosyvoice_repo:
        raise SystemExit('需要 --cosyvoice_repo 或环境变量 COSYVOICE_REPO')
    tok_path = find_tokenizer(None, args.prompt_tokenizer)
    if tok_path is None:
        raise SystemExit('找不到 speech_tokenizer_v3.onnx：复制合成需要它把源音频编成 token')

    os.makedirs(args.out_dir, exist_ok=True)
    model = StreamingSIForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    dec_dir, yaml = resolve_speech_decoder(args.model_dir, None, None)
    model.config.talker_config_yaml = yaml
    model.attach_speech_decoder(dec_dir, device=args.device,
                                cosyvoice_repo=args.cosyvoice_repo, strict=True)
    spk = None
    if args.speaker:
        onnx = os.path.join(dec_dir, 'campplus.onnx')
        if not os.path.isfile(onnx):
            raise SystemExit(f'{dec_dir} 缺 campplus.onnx')
        spk = CampplusSpeaker(onnx)
    sp = SourcePrompt(tok_path)
    print(f'[resynth] tokenizer={tok_path} spk={"campplus" if spk else "无"} '
          f'prompt={"开" if args.source_prompt else "关"} ode={args.ode_steps}', file=sys.stderr)

    with open(args.input_jsonl, encoding='utf-8') as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        items = items[:args.limit]

    import soundfile as sf
    out_p = os.path.join(args.out_dir, 'results.jsonl')
    n_ok = 0
    with open(out_p, 'w', encoding='utf-8') as out:
        for k, it in enumerate(items, 1):
            t0 = time.time()
            row = dict(id=it['id'], tgt_lang=it.get('tgt_lang'))
            try:
                wav16 = load_wav16(it['audio'], max_sec=args.max_audio_sec)
                tokens = sp.tokens(wav16).reshape(-1).tolist()
                p_tok, p_feat = (None, None)
                if args.source_prompt:
                    p_tok, p_feat = sp(wav16, max_sec=args.prompt_sec, position='head',
                                       wav24=load_wav24(it['audio']))
                emb = spk(wav16) if spk else None
                wav = model.decode_speech(tokens, spk_emb=emb, ode_steps=args.ode_steps,
                                          prompt_speech_tokens=p_tok, prompt_feat=p_feat)
                pcm = wav.detach().float().cpu().numpy().reshape(-1)
                path = os.path.join(args.out_dir, f"{it['id']}.wav")
                sf.write(path, pcm, OUT_SR)
                row.update(status='ok', wav=path, source_sec=round(wav16.size / SR, 2),
                           output_sec=round(pcm.size / OUT_SR, 2), n_tokens=len(tokens),
                           source_prompt_tokens=0 if p_tok is None else int(len(p_tok)),
                           wall_sec=round(time.time() - t0, 3))
                n_ok += 1
            except Exception as e:                                   # 单条失败不中断整轮
                row.update(status=f'error: {type(e).__name__}: {e}')
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            if k % 20 == 0 or k == len(items):
                print(f'[resynth] {k}/{len(items)} ok={n_ok}', file=sys.stderr)
    print(f'[resynth] 完成 {n_ok}/{len(items)} -> {out_p}', file=sys.stderr)
    return 0 if n_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
