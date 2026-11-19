"""SimulS2ST-Omni **离线** S2ST 批量推理（官方代码路径，模型只加载一次）。

官方 `src/inference/run_offline.py` 每次只跑一条、每条都重新加载 19GB 模型，不适合跑上千条。
这里直接调它用的同一个官方类与同一组默认参数：

    OmniTalkerSemanticTTSInference.from_pretrained(...).e2e(
        audio_16k, audio_24k, source_lang, target_lang,
        max_s2tt_new_tokens=256, max_tts_new_tokens=500, do_sample=False)

音频读取也复用官方 `run_offline.load_audio`（单声道 + 重采样）。除了「只加载一次 + 批量循环 + 计时」，
没有改动任何推理逻辑。

需要官方仓库在 PYTHONPATH 上（`<repo>`、`<repo>/external/SimulEval`、`<repo>/src/train`）。

用法：python run_omni_offline.py --manifest m.jsonl --out_dir out --model_root models/SimulS2ST-Omni \
          [--shard i --num_shards n] [--device cuda:0] [--limit N]
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf

LANG = {'en': 'English', 'zh': 'Chinese'}
OUT_SR = 24000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model_root', required=True)
    ap.add_argument('--checkpoint_path', default=None,
                    help='官方 from_pretrained 的 checkpoint_path（talker_components.pt + 可选 thinker_lora/）。'
                         '缺省 <model_root>/offline（官方发布模型）；评测复现的 Stage 2 时指向其输出目录，'
                         '基座仍是 <model_root>/offline，与 Stage 2 训练起点一致')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--max_text_tokens', type=int, default=256)   # run_offline.py 默认
    ap.add_argument('--max_audio_tokens', type=int, default=500)  # run_offline.py 默认
    args = ap.parse_args()

    from src.inference.infer_omni_talker_tts import OmniTalkerSemanticTTSInference
    from src.inference.run_offline import load_audio

    root = args.model_root
    model = OmniTalkerSemanticTTSInference.from_pretrained(
        checkpoint_path=args.checkpoint_path or os.path.join(root, 'offline'),
        omni_model_path=os.path.join(root, 'offline'),
        voicebox_path=os.path.join(root, 'voicebox/voicebox.safetensors'),
        vocos_path=os.path.join(root, 'voicebox/vocos.safetensors'),
        voicebox_config_path=os.path.join(root, 'voicebox/voicebox_config.json'),
        device=args.device,
    )

    rows = [json.loads(l) for l in open(args.manifest, encoding='utf-8') if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    if args.num_shards > 1:                           # 直接喂分好的清单时传 --num_shards 1，--shard 只用作输出后缀
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    wav_dir = os.path.join(args.out_dir, 'wav')
    os.makedirs(wav_dir, exist_ok=True)
    out_f = open(os.path.join(args.out_dir, f'results.{args.shard:02d}.jsonl'), 'w', encoding='utf-8')
    for n, r in enumerate(rows, 1):
        rec = dict(id=r['id'], audio=r['audio'], tgt_lang=r['tgt_lang'], source_sec=r['source_sec'])
        try:
            a16 = load_audio(r['audio'], 16000)
            a24 = load_audio(r['audio'], 24000)
            t0 = time.perf_counter()
            gen, thinker_text, codes = model.e2e(
                audio_16k=a16, audio_24k=a24,
                source_lang=LANG[r['src_lang']], target_lang=LANG[r['tgt_lang']],
                max_s2tt_new_tokens=args.max_text_tokens,
                max_tts_new_tokens=args.max_audio_tokens,
                do_sample=False,
            )
            wall = time.perf_counter() - t0
            wav = np.asarray(gen, dtype=np.float32).reshape(-1)
            path = os.path.join(wav_dir, f'{r["id"]}.wav')
            sf.write(path, wav, OUT_SR)
            rec.update(text=thinker_text, wav=os.path.abspath(path), wall_sec=round(wall, 3),
                       output_sec=round(wav.size / OUT_SR, 3), n_codes=int(codes.numel()), status='ok')
        except Exception as exc:                      # 单条失败不中断，记录原因
            rec.update(status=f'{type(exc).__name__}: {exc}', text='', wall_sec=None, output_sec=0.0)
            print(f'[FAIL] {r["id"]}: {exc}', flush=True)
        out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        out_f.flush()
        if n % 20 == 0 or n == len(rows):
            print(f'[shard {args.shard}] {n}/{len(rows)}', flush=True)
    out_f.close()


if __name__ == '__main__':
    main()
