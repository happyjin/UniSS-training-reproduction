"""验证一（Omni 版）：SimulS2ST-Omni 声码器链路对各语言的重建质量。

与 `probe_vocoder.py`（CosyVoice3 链路）**同一口径**，用于横向比较两套声码器在
印尼语 / 马来语上的覆盖情况：

    源音频(16k) --DualCodec(w2v-bert 语义量化)--> semantic code(12.5Hz, 16384)
                --VoiceBox(flow matching, 可选 prompt)--> mel --Vocos--> 24k 波形

两档与 CosyVoice3 版对齐：
* `zero` （主口径）：不给 prompt，纯测 token -> 波形的内容承载能力；
* `clone`：给源音频开头 10s 的 (prompt_codes, prompt_mel)，与官方 `synthesize` 用法一致。
  注意 prompt 取自同一条音频、与被重建内容重叠，对重建偏乐观，只用于看音色能力。

VoiceBox 的 `codes_to_mel` 默认 `n_timesteps=16 / cfg_scale=2.0 / rescale_cfg=0.75`（官方值），
这里原样沿用，不调参。

需要官方仓库在 PYTHONPATH 上（`<repo>`、`<repo>/external/SimulEval`、`<repo>/src/train`）。

用法：python probe_vocoder_omni.py --manifest m.jsonl --out_dir out --model_root <SimulS2ST-Omni 包> \
         [--device cuda:0] [--prompt_sec 10]
输出 results.jsonl：与 probe_vocoder.py 同字段，便于同一套打分脚本处理。
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

OUT_SR = 24000
SR16 = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model_root', required=True, help='含 offline/ voicebox/ dualcodec/ w2v/ 的目录')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--modes', default='clone,zero')
    ap.add_argument('--prompt_sec', type=float, default=10.0)
    ap.add_argument('--n_timesteps', type=int, default=16)        # 官方默认
    ap.add_argument('--cfg_scale', type=float, default=2.0)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    root = args.model_root
    os.environ.setdefault('SIMULS2ST_MODEL_ROOT', root)
    os.environ.setdefault('SIMULS2ST_W2V_BERT_PATH', os.path.join(root, 'w2v'))

    import torchaudio
    from src.inference.generate_tts_offline import SemanticCodeTTSInference
    from src.voicebox.tokenizer import VoiceBoxAudioTokenizer
    from src.voicebox.util import load_config

    cfg = load_config(os.path.join(root, 'voicebox', 'voicebox_config.json'))
    cfg.model.dual_codec.pretrained_path = os.path.join(root, 'dualcodec', 'dualcodec.safetensors')
    cfg.model.kmeans.stat_mean_var_path = os.path.join(root, 'dualcodec', 'w2v_bert_stats.pt')

    # VoiceBox + Vocos + mel 前端：复用官方 from_pretrained（thinker/talker 不参与重建，但该入口一并建好）
    tts = SemanticCodeTTSInference.from_pretrained(
        checkpoint_path=os.path.join(root, 'offline'),
        omni_model_path=os.path.join(root, 'offline'),
        voicebox_path=os.path.join(root, 'voicebox', 'voicebox.safetensors'),
        vocos_path=os.path.join(root, 'voicebox', 'vocos.safetensors'),
        voicebox_config_path=os.path.join(root, 'voicebox', 'voicebox_config.json'),
        device=args.device,
    )
    tokenizer = VoiceBoxAudioTokenizer(cfg, args.device)
    print('omni 声码器链路就绪', flush=True)

    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    modes = args.modes.split(',')
    os.makedirs(args.out_dir, exist_ok=True)
    out_f = open(os.path.join(args.out_dir, 'results.jsonl'), 'w', encoding='utf-8')
    for n, r in enumerate(rows, 1):
        wav16, sr = sf.read(r['audio'], dtype='float32', always_2d=False)
        assert sr == SR16, f'{r["id"]} 不是 16k'
        wav24 = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav16)), SR16, OUT_SR).numpy()
        with torch.inference_mode():
            codes = tokenizer.wav2token(wav16)                   # (1, T) semantic code
        ntok = int(codes.shape[-1])
        for mode in modes:
            rec = dict(id=r['id'], lang=r['lang'], mode=mode, source_sec=r['source_sec'],
                       text=r['text'], n_tokens=ntok,
                       tokens_per_sec=round(ntok / r['source_sec'], 2))
            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    if mode == 'clone':
                        n16 = int(min(args.prompt_sec, wav16.size / SR16) * SR16)
                        pw16, pw24 = wav16[:n16], wav24[:int(n16 / SR16 * OUT_SR)]
                        pcodes = tokenizer.wav2token(pw16)
                        pmel = tts.extract_mel_feature(
                            torch.from_numpy(np.ascontiguousarray(pw24)).unsqueeze(0).to(args.device))
                        # VoiceBox 内部拼 prompt_codes + generated_codes，只输出 prompt 之后的 mel
                        mel = tts.codes_to_mel(codes.to(args.device), prompt_codes=pcodes.to(args.device),
                                               prompt_mel=pmel, n_timesteps=args.n_timesteps,
                                               cfg_scale=args.cfg_scale)
                        rec['prompt_tokens'] = int(pcodes.shape[-1])
                    else:
                        mel = tts.codes_to_mel(codes.to(args.device), n_timesteps=args.n_timesteps,
                                               cfg_scale=args.cfg_scale)
                    arr = np.asarray(tts.mel_to_audio(mel), dtype=np.float32).reshape(-1)
                wall = time.perf_counter() - t0
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
            print(f'[omni-vocoder] {n}/{len(rows)}', flush=True)
    out_f.close()


if __name__ == '__main__':
    main()
