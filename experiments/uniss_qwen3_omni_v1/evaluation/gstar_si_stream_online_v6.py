"""G-STAR-SI streaming inference, paper v6 copy of gstar_si_stream_online.py (unchanged behaviour for its policies)
with two additions for the review of v5:
  --policy lag   (review M04, prereg_v6 §5.2) every read generates to the end like la2 (wait barred) and commits the
                 complete segments, in order, whose predicted end time is <= t - --lag-ms (window clock); the last
                 read commits the rest. A fixed-lag baseline on the model's own timestamps.
  --time-reads   per window, the wall-clock time of each read (generation incl. audio encoding), "read_ms" (for
                 compute-aware latency; meaningful with --batch 1).

Original description:
G-STAR-SI streaming inference (docs/multi_speaker_si_plan_v8_streaming.zh-CN.md §4), HF, batched over windows.

Every --window s (and at the window's end) the Thinker reads the audio heard so far with the speaker stream of the
posteriors the diarizer has finalised by then (frames later than t - --lookahead-ms are zero, as in training), and
its reply so far as the start of its answer, then:
  --policy wait  (a model trained on streaming rows) generates until <|box_end|>; all of it is committed;
  --policy la2   (any model, e.g. the offline O-G1) generates to the end; the longest common prefix, in tokens,
                 of this continuation and the previous one is committed (LocalAgreement-2).
  --policy abc   a model trained on ABC rows: like wait, with a bias on the wait logit (--bias b0), optionally
                 state-dependent on the single-channel speech queue (--sdr) and with fluent realisation (--fr: a block
                 may not end on a dangling unit, and while the talker is idle it needs 3 Chinese characters / 4 English
                 words unless it closes a segment; the rest waits)
  --policy oracle  ABC labels (plan v9 §3): at every read the draft (audio so far, causal speaker stream) and the
                 oracle (the whole window, full speaker stream) both continue the committed tokens to the end; their
                 longest common prefix, cut back to a point where decoding and re-tokenising the committed text gives
                 the same tokens, is committed; every read is written out ("reads": t, prefix, target, draft and
                 oracle lengths). --window-set picks each window's read period from a list (by its id).
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
import math
import time
import zlib
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
    ap.add_argument("--policy", choices=["wait", "la2", "oracle", "abc", "lag"], required=True)
    ap.add_argument("--lag-ms", type=int, default=2000, help="lag: commit segments ending <= t - this")
    ap.add_argument("--time-reads", action="store_true")
    ap.add_argument("--bias", type=float, default=0.0, help="abc: b0, added to the wait logit (> 0 waits more)")
    ap.add_argument("--sdr", action="store_true", help="abc: state-dependent bias from the speech queue (plan v9 §5)")
    ap.add_argument("--fr", action="store_true", help="abc: fluent realisation, commit only admissible blocks")
    ap.add_argument("--window-set", default="", help="oracle: read periods to choose from per window, e.g. 0.5,1,2,4")
    ap.add_argument("--window", type=float, default=1.0, help="read period, s")
    ap.add_argument("--post-key", default="lowlat")
    ap.add_argument("--dir", default="en2zh", choices=["en2zh", "zh2en"], help="the instruction's direction")
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
    head_ids = tok(pg.head("gstar", args.dir), add_special_tokens=False)["input_ids"]
    pad_at = head_ids.index(AUDIO_PAD)
    win_ms = round(args.window * 1000)

    def looped(g: list[int]) -> bool:
        """The tail is one short phrase repeated (period <= 8 tokens, >= 4 times): a degenerate loop, stop it."""
        for per in range(1, 9):
            n = 4 * per
            if len(g) >= n and all(g[-1 - i] == g[-1 - i - per] for i in range(n - per)):
                return True
        return False

    def generate(states, finals, limits=None):
        """One read for a batch of windows: greedy continuation after the committed tokens (per-row token limits and
        loop stops in oracle mode)."""
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
            if not s.get("full"):
                post[max(0, s["t"] - args.lookahead_ms) // 80:] = 0.0
            pt = torch.as_tensor(post_to_tokens(post, n, len(x) / 16000)).to(dev, torch.bfloat16)
            sp = sd_proj(pt[None])[0]
            if args.zero_sd:
                sp = torch.zeros_like(sp)
            parts.append(interleave(ai, sp.to(emb.dtype)))
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat(parts, 0))
        o = thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=True, return_dict=True)
        pkv = o.past_key_values

        bias = torch.as_tensor([s.get("bias", 0.0) for s in states], device=dev, dtype=torch.float32)

        def pick(h, fin_row):
            lg = thinker.lm_head(h).float()
            if args.policy == "abc":
                lg[:, WAIT_ID] += bias
            lg[fin_row, WAIT_ID] = -1e9   # the last read ends the window
            if args.policy in ("la2", "oracle", "lag"):
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
            if args.policy == "oracle":
                for i in range(len(states)):
                    if not done[i] and ((limits and len(gen[i]) >= limits[i]) or looped(gen[i])):
                        done[i] = True
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones((len(states), 1), dtype=mask.dtype, device=dev)], 1)
            o = thinker.model(inputs_embeds=emb_layer(nxt[:, None]), attention_mask=mask, past_key_values=pkv,
                              use_cache=True, return_dict=True)
            pkv = o.past_key_values
            nxt = pick(o.last_hidden_state[:, -1], fin)
        return gen

    periods = [round(float(v) * 1000) for v in args.window_set.split(",")] if args.window_set else [win_ms]
    from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_rules import hygiene
    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import units
    lang = "eng" if args.dir == "zh2en" else "cmn"
    zh = lang == "cmn"
    RHO = {"cmn": 0.19, "eng": 0.34}   # spoken seconds per unit (the talker's training output)

    def sdr_bias(q: float, period_ms: int) -> float:
        tau0 = 1.0 / (1.0 + math.exp(-args.bias))
        tau = min(0.9975, 1.0 - (1.0 - tau0) * max(0.0, period_ms / 1000 - q) / (period_ms / 1000))
        return math.log(tau / (1.0 - tau))

    def split_tail(committed: list[int], new: list[int]):
        """new -> (index after the last closed segment, the open segment's text token positions, its committed text)."""
        open_seg, start_seen = False, False
        seg_text = []
        for i in committed:   # where the committed tokens leave off: inside a segment's text or not
            if SPK0 <= i < NEW_END:
                open_seg, start_seen, seg_text = True, False, []
            elif NEW0 <= i < SPK0:
                if open_seg and not start_seen:
                    start_seen = True
                elif open_seg:
                    open_seg, seg_text = False, []
            elif open_seg and start_seen:
                seg_text.append(i)
        safe, text_pos = 0, []
        for j, i in enumerate(new):
            if SPK0 <= i < NEW_END:
                open_seg, start_seen, seg_text, text_pos = True, False, [], []
                safe = j   # a header may be committed on its own
            elif NEW0 <= i < SPK0:
                if open_seg and not start_seen:
                    start_seen = True
                    safe = j + 1
                elif open_seg:
                    open_seg, text_pos, safe = False, [], j + 1   # a closed segment is always admissible
            elif open_seg and start_seen:
                text_pos.append(j)
        return safe, text_pos, seg_text

    def fr_cut(s: dict, new: list[int]) -> list[int]:
        """Fluent realisation: the longest admissible prefix of this read's new tokens."""
        safe, text_pos, seg_text = split_tail(s["committed"], new)
        if not text_pos:
            return new
        idle = s["q"] <= 0.0
        before = units(tok.decode(seg_text, skip_special_tokens=False), lang)
        cand = units(tok.decode([new[j] for j in text_pos], skip_special_tokens=False), lang)
        keep = hygiene(before, cand, zh, idle, 3 if zh else 4)
        if len(keep) == len(cand):
            return new
        best = safe if safe <= text_pos[0] else text_pos[0]
        for m in range(len(text_pos), 0, -1):   # the longest token prefix whose text is the kept units
            u = units(tok.decode([new[j] for j in text_pos[:m]], skip_special_tokens=False), lang)
            if len(u) <= len(keep) and u == keep[: len(u)] and len(u) == len(keep):
                best = text_pos[m - 1] + 1
                break
        return new[:best]

    def spoken_of(s: dict, new: list[int]):
        """Units of text committed now and how many times the speaker changes."""
        a, changes = 0, 0
        txt = []
        for i in new:
            if SPK0 <= i < NEW_END:
                k = i - SPK0 + 1
                if s["last_spk"] is not None and k != s["last_spk"]:
                    changes += 1
                s["last_spk"] = k
            elif not (NEW0 <= i < SPK0) and i not in (EOS_ID, WAIT_ID):
                txt.append(i)
        a = len(units(tok.decode(txt, skip_special_tokens=False), lang)) if txt else 0
        return a, changes

    def stable_cut(committed: list[int], new: list[int], k: int) -> int:
        """Largest k' <= k whose committed text re-tokenises to the same ids (a prompt that ends inside a merged token
        derails the model, and training rows store text)."""
        while k > 0:
            ids = committed + new[:k]
            if tok(tok.decode(ids, skip_special_tokens=False), add_special_tokens=False)["input_ids"] == ids:
                return k
            k -= 1
        return 0

    t0 = time.time()
    with open(out, "x", encoding="utf-8") as fh:
        for b0 in range(0, len(windows), args.batch):
            states = []
            for w in windows[b0: b0 + args.batch]:
                x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
                assert sr == 16000
                states.append({"w": w, "wav": x.mean(1), "post": np.load(w["post"][args.post_key]).astype(np.float32),
                               "committed": [], "prev": None, "parser": Parser(tok), "blocks": [], "done": False,
                               "dur": int(round(len(x) / 16)), "r": 0, "reads": [], "q": 0.0, "last_spk": None,
                               "period": periods[zlib.crc32(w["id"].encode()) % len(periods)]})
            while any(not s["done"] for s in states):
                live = [s for s in states if not s["done"]]
                for s in live:
                    s["r"] += 1
                    s["t"] = min(s["r"] * s["period"], s["dur"])
                    if args.policy == "abc":
                        s["bias"] = sdr_bias(s["q"], s["period"]) if args.sdr else args.bias
                finals = [s["t"] >= s["dur"] for s in live]
                tr = time.time()
                gens = generate(live, finals)
                if args.time_reads:
                    torch.cuda.synchronize()
                    for s in live:
                        s.setdefault("read_ms", []).append(round(1000 * (time.time() - tr)))
                if args.policy == "oracle":
                    lim = [len([i for i in g if i not in (EOS_ID, WAIT_ID)]) + 1 for g in gens]   # lcp <= the draft
                    ogens = generate([{**s, "t": s["dur"], "full": True} for s in live], [True] * len(live), lim)
                for j, (s, g, fin) in enumerate(zip(live, gens, finals)):
                    body = [i for i in g if i not in (EOS_ID, WAIT_ID)]
                    if args.policy == "oracle":
                        obody = [i for i in ogens[j] if i not in (EOS_ID, WAIT_ID)]
                        k = len(body) if fin else stable_cut(s["committed"], body, lcp(body, obody))
                        new = body[:k]
                        s["reads"].append({"t": s["t"], "prefix": tok.decode(s["committed"], skip_special_tokens=False),
                                           "target": tok.decode(new, skip_special_tokens=False), "final": fin,
                                           "draft_len": len(body), "oracle_len": len(obody), "lcp": lcp(body, obody),
                                           "k": k})
                    elif fin or args.policy == "wait":
                        new = body
                    elif args.policy == "abc":
                        new = fr_cut(s, body) if args.fr else body
                    elif args.policy == "lag":   # complete segments that ended at least lag ago
                        k = 0
                        open_seg = False
                        for j, i in enumerate(body):
                            if SPK0 <= i < NEW_END:
                                open_seg = None   # header seen, start time next
                            elif NEW0 <= i < SPK0:
                                if open_seg is None:
                                    open_seg = True
                                elif open_seg:
                                    if (i - NEW0) * 80 > s["t"] - args.lag_ms:
                                        break
                                    open_seg, k = False, j + 1
                        new = body[:k]
                    else:   # LA2: what this continuation and the previous one agree on
                        k = lcp(body, s["prev"]) if s["prev"] is not None else 0
                        new, s["prev"] = body[:k], body[k:]   # the rest is what the next read is compared with
                    s["committed"] += new
                    if args.policy == "abc":   # the single-channel queue after this read's commitment
                        a_units, n_changes = spoken_of(s, new)
                        s["q"] = max(0.0, s["q"] + RHO[lang] * a_units + 0.2 * n_changes - s["period"] / 1000)
                    s["blocks"] += s["parser"].feed(new, s["t"])
                    if fin:
                        s["parser"].close(s["dur"] / 1000, s["t"])
                        s["done"] = True
            for s in states:
                raw = tok.decode(s["committed"], skip_special_tokens=False)
                fh.write(json.dumps({"id": s["w"]["id"], "raw": raw,
                                     "translation": "".join(x["text"] for x in s["parser"].segments),
                                     "segments": s["parser"].segments, "blocks": s["blocks"], "reads": s["r"],
                                     **({"read_ms": s.get("read_ms", [])} if args.time_reads else {}),
                                     **({"read_log": s["reads"], "period_ms": s["period"]}
                                        if args.policy == "oracle" else {})},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{b0 + len(states)}/{len(windows)} windows, {time.time() - t0:.0f}s", flush=True)
    print(f"GSTAR_ONLINE_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
