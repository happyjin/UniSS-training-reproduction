"""S1 / S3-w / S3 talker 的 TTS 推理：文本 -> speech token -> 24kHz WAV（人工试听用）。

    python custom/streaming_si/inference/infer_tts.py --model_dir <S1 或 S3 checkpoint> --input_jsonl texts.jsonl \
        --out_dir out/tts [--speaker_wav ref.wav] [--device cpu]

jsonl 每行 {"id", "text", "lang": "zh" | "en"}。

输入布局与 Emilia-FSQ 训练数据（S1 / S3-w / S3）逐项一致（si_encode.encode_s1 + 训练 shift）：
  * 指令 = 训练数据里的模板：en "Speak: <text>"、zh "把这段文本读出来。<text>"；
    S3-w 起训练去掉了指令里重复的原文（SI_STRIP_PROMPT_TEXT），用 --strip_prompt 对齐；
  * 译文 = assistant 内容（en 前面带一个空格，与数据一致），不含 eos；
  * talker 条件按 answer_mask 取译文位置（S1 的 cond_mode=text_only 时 thinker 路置零，只用文本 embedding）；
  * talker：`[sos] + cond + [task]` 后自回归到 eos（RAS：top_k 25 / top_p 0.8，与 infer_s2st.py 相同）；
  * 声码：<model_dir>/speech_decoder 的 Fun-CosyVoice3 flow + HiFT；说话人向量取自 --speaker_wav（CAM++），
    缺省为零向量（中性音色）。音色只由 flow 决定，与 talker 无关。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # custom/

from streaming_si.inference.infer_s2st import OUT_SR, CampplusSpeaker, generate_speech, load_wav16  # noqa: E402
from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration  # noqa: E402
from streaming_si.model.speech_decoder import resolve_speech_decoder  # noqa: E402

# S1 训练数据（tts-stage1-fsq）按语种用各自的指令模板，推理必须对齐；
# 这里沿用该管线 PROMPTS 里的短式模板（与 en 的 "Speak:" 同一风格）。
INSTR = {
    'en': 'Speak: ',
    'zh': '把这段文本读出来。',
    'ja': '読み上げ：',
    'ko': '읽기:',
}
_ALIAS = {'zh': 'zh', 'cmn': 'zh', 'zho': 'zh',
          'en': 'en', 'eng': 'en',
          'ja': 'ja', 'jpn': 'ja',
          'ko': 'ko', 'kor': 'ko'}


def build_ids(tokenizer, text: str, lang: str, strip_prompt: bool):
    lang = _ALIAS.get(lang.lower(), 'en')
    answer = (' ' + text.strip()) if lang == 'en' else text.strip()
    instr = INSTR[lang] if strip_prompt else INSTR[lang] + text.strip()
    return (tokenizer.encode(instr, add_special_tokens=False),
            tokenizer.encode(answer, add_special_tokens=False))


@torch.inference_mode()
def talker_condition(model, prompt_ids, text_ids):
    device = model.thinker.embed_tokens.weight.device
    x = torch.tensor([list(prompt_ids) + list(text_ids)], dtype=torch.long, device=device)
    am = torch.tensor([[0] * len(prompt_ids) + [1] * len(text_ids)], dtype=torch.long, device=device)
    out = model.forward_thinker(x)
    cond, lens = model.build_talker_condition(out, x, am)
    return cond[:, :int(lens[0])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--speaker_wav', default=None, help='音色参考（CAM++ 提取说话人向量）；缺省零向量')
    ap.add_argument('--strip_prompt', action='store_true', help='指令里不重复原文（S3-w 起的训练口径）')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', default=None, help='缺省：cuda 用 bfloat16，cpu 用 float32')
    ap.add_argument('--cosyvoice_repo', default=None, help='缺省读 COSYVOICE_REPO')
    ap.add_argument('--max_speech_tokens', type=int, default=750)
    ap.add_argument('--top_k', type=int, default=25)
    ap.add_argument('--top_p', type=float, default=0.8)
    ap.add_argument('--ode_steps', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import soundfile as sf
    dtype = getattr(torch, args.dtype or ('bfloat16' if args.device.startswith('cuda') else 'float32'))
    model = StreamingSIForConditionalGeneration.from_pretrained(args.model_dir, dtype=dtype).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    dec_dir, yaml = resolve_speech_decoder(args.model_dir, None, None)
    repo = args.cosyvoice_repo or os.environ.get('COSYVOICE_REPO')
    if not repo:
        raise SystemExit('需要 --cosyvoice_repo 或 COSYVOICE_REPO')
    model.config.talker_config_yaml = yaml
    model.attach_speech_decoder(dec_dir, device=args.device, cosyvoice_repo=repo, strict=True)
    emb = None
    if args.speaker_wav:
        emb = CampplusSpeaker(os.path.join(dec_dir, 'campplus.onnx'))(load_wav16(args.speaker_wav))
    print(f'[tts] model={args.model_dir} cond_mode={model.config.cond_mode} device={args.device} dtype={dtype} '
          f'speaker={"zero" if emb is None else os.path.basename(args.speaker_wav)}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.input_jsonl, encoding='utf-8') as f:
        items = [json.loads(line) for line in f if line.strip()]
    rows = []
    for i, it in enumerate(items):
        torch.manual_seed(args.seed + i)
        t0 = time.time()
        prompt_ids, text_ids = build_ids(tok, it['text'], it['lang'], args.strip_prompt)
        cond = talker_condition(model, prompt_ids, text_ids)
        tokens, stop = generate_speech(model, cond, max_speech_tokens=args.max_speech_tokens,
                                       top_k=args.top_k, top_p=args.top_p)
        row = dict(id=it['id'], lang=it['lang'], text=it['text'], n_text_tokens=len(text_ids),
                   n_speech_tokens=len(tokens), speech_stop=stop)
        if tokens:
            wav = model.decode_speech(tokens, spk_emb=emb, ode_steps=args.ode_steps)
            wav = wav.detach().float().cpu().numpy().reshape(-1)
            row['wav'] = os.path.join(args.out_dir, f"{it['id']}.wav")
            sf.write(row['wav'], wav, OUT_SR)
            row['output_sec'] = round(wav.size / OUT_SR, 2)
            row['speech_tokens_per_text_token'] = round(len(tokens) / max(len(text_ids), 1), 2)
        row['wall_sec'] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[tts] {it['id']}: {len(tokens)} speech tokens ({stop}), {row.get('output_sec')}s audio, "
              f"{row['wall_sec']}s", flush=True)
    with open(os.path.join(args.out_dir, 'results.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
