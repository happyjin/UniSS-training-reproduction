"""S2 文本评测：整段 vs chunk 送入 Voxtral encoder。

复用 infer_s2st 的 text_only 路径；--chunk_ms>0 时把整段 mel 按 chunk 逐块
encode_audio_chunk 增量编码（与流式同传的 encoder 路径一致），特征拼接后注入。
"""
import argparse, json, os, sys, time
import numpy as np
import soundfile as sf
import torch

import streaming_si.inference.infer_s2st as is2
from streaming_si.inference.infer_s2st import build_prompt
from streaming_si.model.base_model import SR, VoxtralAudioProcessor, mel_frames_for_chunk_ms
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration
from streaming_si.model.voxtral_compat import VoxtralFeatureExtractor
from transformers import AutoTokenizer

CHUNK_MS = 0


def load_mono16k(path):
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def _chunk_embeds(model, ids, mel):
    """替换 is2._prompt_embeds：chunk>0 时逐块增量编码。mel: (1,128,T)。"""
    if CHUNK_MS == 0:
        return is2._orig_prompt_embeds(model, ids, mel)
    cm = mel_frames_for_chunk_ms(CHUNK_MS)
    fs, cache = [], None
    for s in range(0, mel.size(-1), cm):
        f, cache = model.encode_audio_chunk(mel[..., s:s + cm], cache)
        if f is not None and f.size(1) > 0:
            fs.append(f)
    feats = torch.cat(fs, 1)[0] if fs else None          # (n, hidden)
    h = model.thinker.embed_tokens(ids)
    return model.inject_audio_features(ids, h, [feats], seqlen=ids.size(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--chunk_ms', type=int, default=0, help='0=整段；480/640/960/2400')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max_length', type=int, default=768)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--max_audio_sec', type=float, default=30.0)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    global CHUNK_MS
    CHUNK_MS = args.chunk_ms
    is2._orig_prompt_embeds = is2._prompt_embeds
    is2._prompt_embeds = _chunk_embeds

    dev = args.device
    model = StreamingSIForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    mel_proc = VoxtralAudioProcessor(VoxtralFeatureExtractor.from_pretrained(args.model_dir))
    eos_id = tok.eos_token_id

    os.makedirs(args.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.input_jsonl, encoding='utf-8')]
    if args.limit:
        rows = rows[:args.limit]
    out_path = os.path.join(args.out_dir, 'results.jsonl')
    t0 = time.time()
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, r in enumerate(rows):
            wav = load_mono16k(r['audio'])
            if args.max_audio_sec:
                wav = wav[:int(args.max_audio_sec * SR)]
            mel = mel_proc(torch.from_numpy(wav), sr=SR)
            prompt, mel_c, _ = build_prompt(tok, mel, r['tgt_lang'], max_length=args.max_length)
            gen, stop = is2.generate_text(
                model, prompt, mel_c, eos_id=eos_id,
                max_new_tokens=args.max_new_tokens, temperature=1.0, top_p=0.0)
            text = tok.decode(gen, skip_special_tokens=True).strip()
            f.write(json.dumps(dict(id=r['id'], text=text, reference=r.get('reference'),
                                    src_text=r.get('src_text'), tgt_lang=r['tgt_lang'],
                                    stop=stop, n_prompt=len(prompt)), ensure_ascii=False) + '\n')
            f.flush()
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f'  {i+1}/{len(rows)}  {el:.0f}s  avg {el/(i+1):.2f}s/条', flush=True)
    print(f'DONE {len(rows)} 条 in {(time.time()-t0)/60:.1f} min -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
