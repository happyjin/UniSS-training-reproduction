"""验证一：CosyVoice3 声码器链路（speech_tokenizer_v3 -> flow -> HiFT）对各语言的重建质量。

这条链路是 S4/S5 的「speech token -> 波形」后端，推理时冻结、不参与训练。它只吃 token 不看文本，
架构上与语言无关；真正决定能否支持某语言的是 **token 空间对该语言的覆盖**。本脚本用「重建」
把这一层单独隔出来测：

    源音频(16k) --tokenizer--> speech token(25Hz) --flow(+可选声学 prompt)--> mel --HiFT--> 24k 波形

重建音频与源音频的差距 = 这条链路在该语言上的信息损失**上界**（上游 thinker / talker 不参与）。
它回答的是「声码器能不能发出这门语言的音」，不是「我们的模型会不会说这门语言」。

两种 prompt 条件，用于区分损失来源：
* `zero` （主口径）：不给 prompt、说话人向量取零（中性音色），纯粹测 token -> 波形的内容承载能力，
  没有任何来自源音频的声学泄漏。
* `clone`：flow 带源语音声学 prompt（prompt_token + prompt_feat，取开头 10s），与 S4/S5 推理一致。
  **注意**：这里 prompt 取自同一条音频，与被重建的内容有重叠，对重建是偏乐观的
  （S4/S5 实际用法里 prompt 来自源语音、生成的是译文 token，没有内容重叠）。
  因此 `clone` 只用来看「带声学 prompt 时音质/音色能到什么程度」，内容准确率看 `zero`。

用法：python probe_vocoder.py --manifest m.jsonl --out_dir out --decoder <speech_decoder 目录> \
         [--tokenizer <speech_tokenizer_v3.onnx>] [--cosyvoice_repo <CosyVoice>] [--device cuda:0]
输出 results.jsonl：{id, lang, mode, wav, source_sec, output_sec, n_tokens, tokens_per_sec, ...}
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf

OUT_SR = 24000
SR16 = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--decoder', required=True, help='含 flow.pt / hift.pt / cosyvoice3.yaml 的目录')
    ap.add_argument('--tokenizer', default=None, help='speech_tokenizer_v3.onnx；缺省按 decoder 目录与环境变量找')
    ap.add_argument('--cosyvoice_repo', default=os.environ.get('COSYVOICE_REPO'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--modes', default='clone,zero')
    ap.add_argument('--prompt_sec', type=float, default=10.0)
    ap.add_argument('--ode_steps', type=int, default=5)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    import torchaudio
    from streaming_si.inference.source_prompt import SourcePrompt, find_tokenizer, load_wav24
    from streaming_si.model.model_si import SI
    from streaming_si.model.speech_decoder import find_config_yaml

    yaml_name = find_config_yaml(args.decoder)                        # manifest.json > cosyvoice3.yaml
    model = SI.__new__(SI)                      # 只借 attach_speech_decoder / decode_speech
    object.__setattr__(model, 'config', type('C', (), {'talker_config_yaml': yaml_name})())
    SI.attach_speech_decoder(model, args.decoder, device=args.device,
                             cosyvoice_repo=args.cosyvoice_repo)
    tok_path = find_tokenizer(args.decoder, args.tokenizer)
    assert tok_path, f'找不到 speech_tokenizer_v3.onnx（decoder={args.decoder}）'
    print(f'tokenizer: {tok_path}', flush=True)
    sp = SourcePrompt(tok_path)

    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    modes = args.modes.split(',')
    os.makedirs(args.out_dir, exist_ok=True)
    out_f = open(os.path.join(args.out_dir, 'results.jsonl'), 'w', encoding='utf-8')
    for n, r in enumerate(rows, 1):
        wav16, sr = sf.read(r['audio'], dtype='float32', always_2d=False)
        assert sr == SR16, f'{r["id"]} 不是 16k'
        wav24 = load_wav24(r['audio'])
        all_tok = sp.tokens(wav16)                                    # 整段 token，用于重建
        for mode in modes:
            rec = dict(id=r['id'], lang=r['lang'], mode=mode, source_sec=r['source_sec'],
                       text=r['text'], n_tokens=int(all_tok.size),
                       tokens_per_sec=round(all_tok.size / r['source_sec'], 2))
            try:
                t0 = time.perf_counter()
                if mode == 'clone':
                    ptok, pfeat = sp(wav16, max_sec=args.prompt_sec, wav24=wav24)
                    # 官方 flow 内部自己拼 prompt 并只返回 prompt 之后的 mel，这里直接喂整段 token
                    wav = model.decode_speech(all_tok, spk_emb=None, prompt_speech_tokens=ptok,
                                              prompt_feat=pfeat, ode_steps=args.ode_steps)
                    rec['prompt_tokens'] = int(ptok.size)
                else:
                    wav = model.decode_speech(all_tok, spk_emb=None, ode_steps=args.ode_steps)
                wall = time.perf_counter() - t0
                if hasattr(wav, 'detach'):                            # decode_speech 返回 CUDA tensor
                    wav = wav.detach().float().cpu().numpy()
                arr = np.asarray(wav, dtype=np.float32).reshape(-1)
                d = os.path.join(args.out_dir, mode)
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f'{r["id"]}.wav')
                sf.write(path, arr, OUT_SR)
                rec.update(wav=os.path.abspath(path), status='ok', wall_sec=round(wall, 3),
                           output_sec=round(arr.size / OUT_SR, 3))
            except Exception as exc:
                rec.update(status=f'{type(exc).__name__}: {exc}')
                print(f'[FAIL] {r["id"]}/{mode}: {exc}', flush=True)
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out_f.flush()
        if n % 20 == 0 or n == len(rows):
            print(f'[vocoder] {n}/{len(rows)}', flush=True)
    out_f.close()
    _ = torchaudio                                                    # 仅确保依赖可用


if __name__ == '__main__':
    main()
