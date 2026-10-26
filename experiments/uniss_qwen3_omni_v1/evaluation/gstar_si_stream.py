"""G-STAR-SI inference with HF (docs/multi_speaker_si_plan_v7.zh-CN.md): the merged Thinker reads a window's audio
with the Nemotron-3 speaker stream interleaved K:1 and writes <|spk k|><|t a|>translation<|t b|> ... (offline: the
whole window at once). vLLM cannot take the interleaved embeddings, so this is a batched greedy decoder over the HF
model with a KV cache; positions are arange (left padding only shifts a row's positions by a constant, which RoPE
does not see).

  --inst gstar   the multi-speaker model, posteriors from the window's --post-key (offline | oracle_offline);
                 --zero-sd feeds zeros as speaker stream (O-nosd)
  --inst sd      a plain translation Thinker (the original release for B-off): no speaker stream; the whole window
                 is one segment of slot 1

Output per window: {"id", "raw", "translation", "segments": [{"spk", "start_s", "end_s", "text"}]}.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream --thinker <merged> \
        --windows data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl --out <dir>/hyp.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
    AUDIO_PAD, EOS_ID, N_SLOT, SDProjector, interleave, n_speaker_tokens, post_to_tokens)

REPO = Path(__file__).resolve().parents[3]
RELEASE = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
SEG = re.compile(r"<\|spk(\d+)\|><\|t(\d+)\|>(.*?)<\|t(\d+)\|>", re.S)


def parse(raw: str, dur_s: float, inst: str) -> list[dict]:
    if inst == "sd":
        t = raw.replace("<|im_end|>", "").strip()
        return [{"spk": 1, "start_s": 0.0, "end_s": round(dur_s, 2), "text": t}] if t else []
    return [{"spk": int(k), "start_s": round(int(a) * 0.08, 2), "end_s": round(int(b) * 0.08, 2), "text": x.strip()}
            for k, a, x, b in SEG.findall(raw) if x.strip()]


def load_thinker(path: str):
    """The Thinker of a merged checkpoint, loaded straight onto the card with a bounded CPU thread count: eight shards
    each converting the MoE experts on the CPU with every core at once took 20-30 min to load."""
    import gc
    import os

    from transformers import Qwen3OmniMoeForConditionalGeneration

    torch.set_num_threads(int(os.environ.get("GSTAR_THREADS", max(4, (os.cpu_count() or 32) // 8))))
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": 0})
    thinker = omni.thinker.eval()
    del omni
    gc.collect()
    torch.cuda.empty_cache()
    return thinker


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inst", choices=["gstar", "sd"], default="gstar")
    ap.add_argument("--post-key", default="offline")
    ap.add_argument("--zero-sd", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

    from experiments.uniss_qwen3_omni_v1.training import pretrain_thinker_gstar as pg

    windows = [json.loads(l) for l in open(args.windows, encoding="utf-8")][args.shard:: args.num_shards]
    if args.limit:
        windows = windows[: args.limit]
    proc = pg.processor(RELEASE)
    tok = proc.tokenizer
    dev = torch.device("cuda")
    thinker = load_thinker(args.thinker)
    sd_proj = None
    if args.inst == "gstar":
        sd_proj = SDProjector(N_SLOT, thinker.config.text_config.hidden_size)
        sd_proj.load_state_dict(torch.load(Path(args.thinker) / "sd_proj.pt"))
        sd_proj = sd_proj.to(dev, torch.bfloat16).eval()
    emb_layer = thinker.get_input_embeddings()
    t0 = time.time()
    with open(out, "x", encoding="utf-8") as fh:
        for b0 in range(0, len(windows), args.batch):
            ws = windows[b0: b0 + args.batch]
            wavs = []
            for w in ws:
                x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
                assert sr == 16000
                wavs.append(x.mean(1))
            f = proc.feature_extractor(wavs, sampling_rate=16000, padding=True, truncation=False,
                                       return_attention_mask=True, return_tensors="pt")
            n_audio = _get_feat_extract_output_lengths(f["attention_mask"].sum(1)).tolist()
            prompts = []
            for w, n in zip(ws, n_audio):
                m = n_speaker_tokens(n) if args.inst == "gstar" else 0
                prompts.append(tok(pg.head(args.inst).replace("<|audio_pad|>", "<|audio_pad|>" * (n + m)),
                                   add_special_tokens=False)["input_ids"])
            L = max(len(p) for p in prompts)
            ids = torch.full((len(ws), L), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((len(ws), L), dtype=torch.long)
            for i, p in enumerate(prompts):   # left padding
                ids[i, L - len(p):] = torch.as_tensor(p)
                mask[i, L - len(p):] = 1
            ids, mask = ids.to(dev), mask.to(dev)
            emb = emb_layer(ids)
            a = thinker.get_audio_features(f["input_features"].to(dev, torch.bfloat16),
                                           f["attention_mask"].to(dev)).last_hidden_state.to(emb.dtype)
            parts, pos = [], 0
            for w, n, x in zip(ws, n_audio, wavs):
                ai = a[pos: pos + n]
                pos += n
                if args.inst == "gstar":   # the token grid as in training: the wav's own length
                    post = np.load(w["post"][args.post_key]).astype(np.float32)
                    pt = torch.as_tensor(post_to_tokens(post, n, len(x) / 16000)).to(dev, torch.bfloat16)
                    s = sd_proj(pt[None])[0]
                    if args.zero_sd:
                        s = torch.zeros_like(s)
                    ai = interleave(ai, s.to(emb.dtype))
                parts.append(ai)
            emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat(parts, 0))
            o = thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=True, return_dict=True)
            pkv = o.past_key_values
            nxt = thinker.lm_head(o.last_hidden_state[:, -1]).float().argmax(-1)
            gen = [[] for _ in ws]
            done = torch.zeros(len(ws), dtype=torch.bool, device=dev)
            for _ in range(args.max_new):
                for i in range(len(ws)):
                    if not done[i]:
                        gen[i].append(int(nxt[i]))
                done |= nxt == EOS_ID
                if bool(done.all()):
                    break
                mask = torch.cat([mask, torch.ones((len(ws), 1), dtype=mask.dtype, device=dev)], 1)
                o = thinker.model(inputs_embeds=emb_layer(nxt[:, None]), attention_mask=mask, past_key_values=pkv,
                                  use_cache=True, return_dict=True)
                pkv = o.past_key_values
                nxt = thinker.lm_head(o.last_hidden_state[:, -1]).float().argmax(-1)
            for w, g in zip(ws, gen):
                raw = tok.decode(g, skip_special_tokens=False)
                segs = parse(raw, w["dur_ms"] / 1000, args.inst)
                fh.write(json.dumps({"id": w["id"], "raw": raw, "translation": "".join(s["text"] for s in segs),
                                     "segments": segs}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{b0 + len(ws)}/{len(windows)} windows, {time.time() - t0:.0f}s", flush=True)
    print(f"GSTAR_STREAM_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
