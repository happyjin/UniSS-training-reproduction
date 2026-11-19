"""S4 离线推理耗时分解：给 S2STPipeline.translate 的各阶段套计时，不改推理逻辑。

分段：mel 前端、thinker 生成译文（Qwen3.5 每 token 整段重算）、thinker 条件重跑、
talker 生成 speech token、CAM++ 说话人向量（CPU onnx）、源 prompt tokenizer（CPU onnx）、
flow + vocos 合成。

用法：profile_s4.py <model_dir> <audio.wav> <tgt_lang> <cosyvoice_repo> [out.json]
"""
import collections
import json
import os
import sys
import time

import numpy as np
import torch

T = collections.defaultdict(float)
N = collections.defaultdict(int)


def timed(name, fn):
    def wrap(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        T[name] += time.perf_counter() - t0
        N[name] += 1
        return r
    return wrap


def main():
    model_dir, audio, tgt_lang, cosy_repo = sys.argv[1:5]
    out_p = sys.argv[5] if len(sys.argv) > 5 else None
    os.environ.setdefault('COSYVOICE_REPO', cosy_repo)
    from streaming_si.inference import infer_s2st as M

    M.generate_text = timed('2_thinker生成译文', M.generate_text)
    M.talker_condition = timed('3_thinker条件重跑', M.talker_condition)
    M.generate_speech = timed('4_talker生成speech', M.generate_speech)

    pipe = M.S2STPipeline(model_dir, device='cuda:0', cosyvoice_repo=cosy_repo)
    pipe.mel = timed('1_mel前端', pipe.mel)
    if pipe.speaker is not None:
        pipe.speaker = timed('5_说话人向量onnx', pipe.speaker)
    if pipe.prompt is not None:
        pipe.prompt = timed('6_源prompt_onnx', pipe.prompt)
    pipe.model.decode_speech = timed('7_flow_vocos', pipe.model.decode_speech)

    wav16 = M.load_wav16(audio)
    for it in range(2):                       # 第 1 遍预热，取第 2 遍
        if it == 1:
            T.clear(); N.clear()
        t0 = time.perf_counter()
        res = pipe.translate(wav16, tgt_lang)
        total = time.perf_counter() - t0

    src = res['source_sec']
    rec = {k: round(v, 3) for k, v in T.items()}
    rec.update(total=round(total, 3), src_sec=round(src, 2), rtf=round(total / src, 3),
               n_text=len(res['text_token_ids']), n_speech=len(res['speech_tokens']),
               audio=os.path.basename(audio))
    print(f"\n{os.path.basename(audio)}  源 {src:.1f}s | 译文 {rec['n_text']} tok | "
          f"speech {rec['n_speech']} tok | 总 {total:.2f}s  RTF {total / src:.2f}")
    print('阶段'.ljust(22), '耗时'.rjust(9), '占比'.rjust(7), '调用'.rjust(6))
    for k in sorted(T):
        print(k.ljust(22), f'{T[k]:.2f}s'.rjust(9), f'{100 * T[k] / total:.0f}%'.rjust(7), str(N[k]).rjust(6))
    rest = total - sum(T.values())
    print('其它'.ljust(21), f'{rest:.2f}s'.rjust(9), f'{100 * rest / total:.0f}%'.rjust(7))
    if out_p:
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        json.dump(rec, open(out_p, 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
