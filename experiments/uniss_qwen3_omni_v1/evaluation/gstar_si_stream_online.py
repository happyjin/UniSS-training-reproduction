"""G-STAR-SI streaming inference (docs/multi_speaker_si_plan_v8_streaming.zh-CN.md §4), HF, batched over windows.

Every --window s (and at the window's end) the Thinker reads the audio heard so far with the speaker stream of the
posteriors the diarizer has finalised by then (frames later than t - --lookahead-ms are zero, as in training), and
its reply so far as the start of its answer, then:
  --policy wait  (a model trained on streaming rows) generates until <|box_end|>; all of it is committed;
  --policy la2   (any model, e.g. the offline O-G1) generates to the end; the longest common prefix, in tokens,
                 of this continuation and the previous one is committed (LocalAgreement-2).
At the window's end the rest is committed (<|box_end|> barred). Nothing committed is revised.

Committed tokens are parsed as they arrive: a speaker token opens a segment, the first time token is its start, the
text follows, the second time token closes it. Output per window: {"id", "raw", "translation",
"segments": [{"spk", "start_s", "end_s", "text", "write_ms"}], "blocks": [{"write_ms", "spk", "start_s", "text"}]}
-- a block is the text a read committed inside one segment (what the talker can speak at that moment).

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online --thinker <merged> --policy la2 \
        --windows <windows_test_ref.jsonl> --post-key lowlat --out <dir>/hyp.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream import RELEASE, load_thinker
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
    AUDIO_PAD, EOS_ID, NEW0, NEW_END, N_SLOT, SPK0, SDProjector, interleave, n_speaker_tokens, post_to_tokens)

WAIT_ID = 151649   # <|box_end|>


class Parser:
    """Committed token ids -> segments and per-read blocks."""

    def __init__(self, tok):
        self.tok = tok
        self.spk = None
        self.start = None
        self.text_ids = []
        self.segments = []
        self.open = False

    def feed(self, ids: list[int], write_ms: int) -> list[dict]:
        blocks, buf = [], []

        def flush():
            if buf and self.spk is not None:
                t = self.tok.decode(buf, skip_special_tokens=False)
                if t.strip():
                    blocks.append({"write_ms": write_ms, "spk": self.spk, "start_s": self.start, "text": t})
                self.text_ids += buf
            buf.clear()

        for i in ids:
            if SPK0 <= i < NEW_END:
                flush()
                if self.open:   # a segment the model left without its end time: close it by its length
                    text = self.tok.decode(self.text_ids, skip_special_tokens=False).strip()
                    if text and self.spk is not None:
                        self.segments.append({"spk": self.spk, "start_s": self.start,
                                              "end_s": round(self.start + max(0.5, 0.25 * len(text)), 2),
                                              "text": text, "write_ms": write_ms})
                self.spk, self.start, self.open, self.text_ids = i - SPK0 + 1, None, False, []
            elif NEW0 <= i < SPK0:
                t = round((i - NEW0) * 0.08, 2)
                if self.start is None:
                    self.start, self.open = t, True
                elif self.open:
                    flush()
                    text = self.tok.decode(self.text_ids, skip_special_tokens=False).strip()
                    if text and self.spk is not None:
                        self.segments.append({"spk": self.spk, "start_s": self.start, "end_s": t, "text": text,
                                              "write_ms": write_ms})
                    self.open, self.text_ids = False, []
            elif i not in (EOS_ID, WAIT_ID):
                buf.append(i)
        flush()
        return blocks

    def close(self, end_s: float, write_ms: int):
        """A segment left open at the window's end is closed there."""
        text = self.tok.decode(self.text_ids, skip_special_tokens=False).strip()
        if self.open and text and self.spk is not None:
            self.segments.append({"spk": self.spk, "start_s": self.start, "end_s": end_s, "text": text,
                                  "write_ms": write_ms})
        self.open, self.text_ids = False, []


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", choices=["wait", "la2"], required=True)
    ap.add_argument("--window", type=float, default=1.0, help="read period, s")
    ap.add_argument("--post-key", default="lowlat")
    ap.add_argument("--lookahead-ms", type=int, default=1040)
    ap.add_argument("--zero-sd", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=300)
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
    sd_proj = SDProjector(N_SLOT, thinker.config.text_config.hidden_size)
    sd_proj.load_state_dict(torch.load(Path(args.thinker) / "sd_proj.pt"))
    sd_proj = sd_proj.to(dev, torch.bfloat16).eval()
    emb_layer = thinker.get_input_embeddings()
    head_ids = tok(pg.head("gstar"), add_special_tokens=False)["input_ids"]
    pad_at = head_ids.index(AUDIO_PAD)
    win_ms = round(args.window * 1000)

    def generate(states, finals):
        """One read for a batch of windows: greedy continuation after the committed tokens."""
        wavs = [s["wav"][: s["t"] * 16] if len(s["wav"][: s["t"] * 16]) >= 1600 else
                np.pad(s["wav"][: s["t"] * 16], (0, 1600 - len(s["wav"][: s["t"] * 16]))) for s in states]
        f = proc.feature_extractor(wavs, sampling_rate=16000, padding=True, truncation=False,
                                   return_attention_mask=True, return_tensors="pt")
        n_audio = _get_feat_extract_output_lengths(f["attention_mask"].sum(1)).tolist()
        prompts = []
        for s, n in zip(states, n_audio):
            prompts.append(head_ids[:pad_at] + [AUDIO_PAD] * (n + n_speaker_tokens(n)) + head_ids[pad_at + 1:]
                           + s["committed"])
        L = max(len(p) for p in prompts)
        ids = torch.full((len(states), L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(states), L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.as_tensor(p)
            mask[i, L - len(p):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        emb = emb_layer(ids)
        a = thinker.get_audio_features(f["input_features"].to(dev, torch.bfloat16),
                                       f["attention_mask"].to(dev)).last_hidden_state.to(emb.dtype)
        parts, pos = [], 0
        for s, n, x in zip(states, n_audio, wavs):
            ai = a[pos: pos + n]
            pos += n
            post = s["post"].copy()
            post[max(0, s["t"] - args.lookahead_ms) // 80:] = 0.0
            pt = torch.as_tensor(post_to_tokens(post, n, len(x) / 16000)).to(dev, torch.bfloat16)
            sp = sd_proj(pt[None])[0]
            if args.zero_sd:
                sp = torch.zeros_like(sp)
            parts.append(interleave(ai, sp.to(emb.dtype)))
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat(parts, 0))
        o = thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=True, return_dict=True)
        pkv = o.past_key_values

        def pick(h, fin_row):
            lg = thinker.lm_head(h).float()
            lg[fin_row, WAIT_ID] = -1e9   # the last read ends the window
            if args.policy == "la2":
                lg[:, WAIT_ID] = -1e9
            return lg.argmax(-1)

        fin = torch.as_tensor(finals, device=dev)
        nxt = pick(o.last_hidden_state[:, -1], fin)
        gen = [[] for _ in states]
        done = torch.zeros(len(states), dtype=torch.bool, device=dev)
        for _ in range(args.max_new):
            for i in range(len(states)):
                if not done[i]:
                    gen[i].append(int(nxt[i]))
            done |= (nxt == EOS_ID) | (nxt == WAIT_ID)
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones((len(states), 1), dtype=mask.dtype, device=dev)], 1)
            o = thinker.model(inputs_embeds=emb_layer(nxt[:, None]), attention_mask=mask, past_key_values=pkv,
                              use_cache=True, return_dict=True)
            pkv = o.past_key_values
            nxt = pick(o.last_hidden_state[:, -1], fin)
        return gen

    t0 = time.time()
    with open(out, "x", encoding="utf-8") as fh:
        for b0 in range(0, len(windows), args.batch):
            states = []
            for w in windows[b0: b0 + args.batch]:
                x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
                assert sr == 16000
                states.append({"w": w, "wav": x.mean(1), "post": np.load(w["post"][args.post_key]).astype(np.float32),
                               "committed": [], "prev": None, "parser": Parser(tok), "blocks": [], "done": False,
                               "dur": int(round(len(x) / 16)), "r": 0})
            while any(not s["done"] for s in states):
                live = [s for s in states if not s["done"]]
                for s in live:
                    s["r"] += 1
                    s["t"] = min(s["r"] * win_ms, s["dur"])
                finals = [s["t"] >= s["dur"] for s in live]
                gens = generate(live, finals)
                for s, g, fin in zip(live, gens, finals):
                    body = [i for i in g if i not in (EOS_ID, WAIT_ID)]
                    if fin or args.policy == "wait":
                        new = body
                    else:   # LA2: what this continuation and the previous one agree on
                        k = lcp(body, s["prev"]) if s["prev"] is not None else 0
                        new, s["prev"] = body[:k], body[k:]   # the rest is what the next read is compared with
                    s["committed"] += new
                    s["blocks"] += s["parser"].feed(new, s["t"])
                    if fin:
                        s["parser"].close(s["dur"] / 1000, s["t"])
                        s["done"] = True
            for s in states:
                raw = tok.decode(s["committed"], skip_special_tokens=False)
                fh.write(json.dumps({"id": s["w"]["id"], "raw": raw,
                                     "translation": "".join(x["text"] for x in s["parser"].segments),
                                     "segments": s["parser"].segments, "blocks": s["blocks"], "reads": s["r"]},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{b0 + len(states)}/{len(windows)} windows, {time.time() - t0:.0f}s", flush=True)
    print(f"GSTAR_ONLINE_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
