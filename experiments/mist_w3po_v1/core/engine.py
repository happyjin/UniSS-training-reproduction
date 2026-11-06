"""W3PO core (docs/mist_grpo_plan.zh-CN.md §1 "一条 rollout", §3.1, §4 S0-4, §5): the G-STAR-SI Thinker read step shared
by evaluation decoding, DAgger decoding, SFT row building and RL rollouts.

A read of window w at time t: the audio heard so far (wav[:t]) is encoded, the diarizer posteriors are zeroed after
t - 1.04 s and interleaved K:1 as speaker tokens (exactly as gstar_si_stream_online_v6s.py), the prompt is the v8
chat head with an optional cross-window context (S0-4) inside the user turn, then the committed reply so far; the
model continues until <wait> (= <|box_end|>) or, at the window's last read, <eos>.

Prompt layout (identical to the v8 prompt when the context is empty, asserted at construction):
    head_a(... <|audio_pad|>*(n + ceil(n/5)) ... instruction) + ctx + head_b(<|im_end|> ... assistant\n) + committed
ctx = CTX_LEAD + " <|spk k|>text" for the last N committed segments of the previous window of the same meeting.

Token categories (plan §3.3, W3):
  when  positions between segments of a non-final read (first position of the block, and after a segment's closing
        time token): the choice <wait> (stop committing now) vs anything else (write); projected binary policy
        (p_wait, 1 - p_wait)
  who   positions whose token is a speaker token <|spk k|>: projected policy = softmax over the 8 speaker logits
  what  every other generated position (time tokens, text, <eos>)
A speaker token at a when position is both a when and a who decision.

Batching: items that share audio (same window, same t; the G rollouts of a group) share one audio-encoder pass and one
speaker-stream projection ("akey").
"""
from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import (
    AUDIO_PAD, EOS_ID, NEW0, NEW_END, N_SLOT, SPK0, SDProjector, interleave, n_speaker_tokens, post_to_tokens)

REPO = Path(__file__).resolve().parents[3]
RELEASE = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
WAIT_ID = 151649          # <|box_end|>
CTX_LEAD = " Already translated just before this audio (do not translate it again):"
LOOKAHEAD_MS = 1040


def ctx_string(segs: list[dict], n: int, max_chars: int = 240) -> str:
    """S0-4: the last n committed segments of the previous window as read-only context ("" when none)."""
    if n <= 0 or not segs:
        return ""
    parts = []
    for s in segs[-n:]:
        t = " ".join(str(s["text"]).split())[:max_chars]
        if t:
            parts.append(f" <|spk{int(s['spk'])}|>{t}")
    return CTX_LEAD + "".join(parts) if parts else ""


class Prompter:
    """Token ids of the prompt; shared by training rows and inference so both see the same ids."""

    def __init__(self, direction: str):
        from experiments.uniss_qwen3_omni_v1.training import pretrain_thinker_gstar as pg

        self.proc = pg.processor(RELEASE)
        self.tok = self.proc.tokenizer
        ins = pg.gstar_instruction(direction)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins + "@@CTX@@"}]}]
        s = self.proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        a, b = s.split("@@CTX@@")
        self.a_ids = self.tok(a, add_special_tokens=False)["input_ids"]
        self.b_ids = self.tok(b, add_special_tokens=False)["input_ids"]
        ref = self.tok(pg.head("gstar", direction), add_special_tokens=False)["input_ids"]
        assert self.a_ids + self.b_ids == ref, "the empty-context prompt must be the v8 prompt"
        self.pad_at = self.a_ids.index(AUDIO_PAD)
        self._ctx_cache: dict[str, list[int]] = {}

    def ctx_ids(self, ctx: str) -> list[int]:
        if not ctx:
            return []
        if ctx not in self._ctx_cache:
            if len(self._ctx_cache) > 50000:
                self._ctx_cache.clear()
            self._ctx_cache[ctx] = self.tok(ctx, add_special_tokens=False)["input_ids"]
        return self._ctx_cache[ctx]

    def ids(self, n_audio: int, ctx: str, prefix_ids: list[int]) -> list[int]:
        m = n_speaker_tokens(n_audio)
        return (self.a_ids[: self.pad_at] + [AUDIO_PAD] * (n_audio + m) + self.a_ids[self.pad_at + 1:]
                + self.ctx_ids(ctx) + self.b_ids + list(prefix_ids))


def load_thinker(path: str, max_gib: int = 40):
    """gstar_si_stream_online_v6s.load_thinker: the merged Thinker spread over the visible cards."""
    from transformers import Qwen3OmniMoeForConditionalGeneration

    torch.set_num_threads(int(os.environ.get("GSTAR_THREADS", max(4, (os.cpu_count() or 32) // 8))))
    mem = {i: f"{max_gib}GiB" for i in range(torch.cuda.device_count())}
    mem["cpu"] = "400GiB"
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto",
                                                                max_memory=mem)
    thinker = omni.thinker.eval()
    del omni
    gc.collect()
    torch.cuda.empty_cache()
    return thinker


def seg_state(ids: list[int]) -> str:
    """Where a token stream leaves off: "out" (between segments), "hdr" (speaker token seen, start time next) or
    "text" (inside a segment's text)."""
    st = "out"
    for i in ids:
        if SPK0 <= i < NEW_END:
            st = "hdr"
        elif NEW0 <= i < SPK0:
            st = "text" if st == "hdr" else "out"
    return st


def step_state(st: str, i: int) -> str:
    if SPK0 <= i < NEW_END:
        return "hdr"
    if NEW0 <= i < SPK0:
        return "text" if st == "hdr" else "out"
    return st


@dataclass
class ReadItem:
    """One state to advance by one read."""
    wav: np.ndarray              # the window's audio, 16 kHz (the first t ms are heard)
    post: np.ndarray             # [F80, 8] window posteriors
    t_ms: int
    final: bool
    committed: list[int]
    ctx: str = ""
    akey: str = ""               # items with the same akey share the audio encoding
    full: bool = False           # whole-window posteriors (oracle reads)
    bar_wait: bool = False       # lag / la2: generate to the end


@dataclass
class ReadOut:
    ids: list[int] = field(default_factory=list)
    when: list[bool] = field(default_factory=list)
    who: list[bool] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)       # log pi(token) under T = 1 (full vocabulary)
    logp_wait: list[float] = field(default_factory=list)  # at when positions: log p(<wait>), T = 1
    logp_who: list[float] = field(default_factory=list)   # at who positions: log softmax over the 8 speakers
    prompt_len: int = 0


class Engine:
    def __init__(self, thinker, sd_proj: SDProjector, prompter: Prompter, dev: torch.device):
        self.thinker, self.sd_proj, self.pr, self.dev = thinker, sd_proj, prompter, dev
        self.tok = prompter.tok
        self.emb_layer = thinker.get_input_embeddings()

    @classmethod
    def from_merged(cls, path: str, direction: str, max_gib: int = 40):
        thinker = load_thinker(path, max_gib)
        dev = torch.device("cuda")
        sd = SDProjector(N_SLOT, thinker.config.text_config.hidden_size)
        sd.load_state_dict(torch.load(Path(path) / "sd_proj.pt"))
        sd = sd.to(dev, torch.bfloat16).eval()
        return cls(thinker, sd, Prompter(direction), dev)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.thinker.lm_head(h).float()

    @torch.no_grad()
    def audio_embeds(self, items: list[ReadItem], emb_dtype, lookahead_ms: int = LOOKAHEAD_MS, cache: dict | None = None):
        """Per item: (n_audio, interleaved audio + speaker-stream embeddings), one encoder pass per unique akey;
        with a cache dict, akeys already there are reused and new ones stored (RL: rollout -> training)."""
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

        keys, uniq = [], {}
        for j, it in enumerate(items):
            k = it.akey or f"__{j}"
            keys.append(k)
            if cache is None or k not in cache:
                uniq.setdefault(k, j)
        order = list(uniq)
        if not order:
            return [cache[k] for k in keys]
        wavs = []
        for k in order:
            it = items[uniq[k]]
            x = it.wav[: it.t_ms * 16]
            wavs.append(x if len(x) >= 1600 else np.pad(x, (0, 1600 - len(x))))
        f = self.pr.proc.feature_extractor(wavs, sampling_rate=16000, padding=True, truncation=False,
                                           return_attention_mask=True, return_tensors="pt")
        n_audio = _get_feat_extract_output_lengths(f["attention_mask"].sum(1)).tolist()
        a = self.thinker.get_audio_features(f["input_features"].to(self.dev, torch.bfloat16),
                                            f["attention_mask"].to(self.dev)).last_hidden_state.to(emb_dtype)
        out, pos = {}, 0
        for k, n, x in zip(order, n_audio, wavs):
            it = items[uniq[k]]
            ai = a[pos: pos + n]
            pos += n
            post = it.post.copy()
            if not it.full:
                post[max(0, it.t_ms - lookahead_ms) // 80:] = 0.0
            pt = torch.as_tensor(post_to_tokens(post, n, len(x) / 16000)).to(self.dev, torch.bfloat16)
            sp = self.sd_proj(pt[None])[0]
            out[k] = (n, interleave(ai, sp.to(emb_dtype)))
        if cache is not None:
            cache.update({k: v for k, v in out.items() if not k.startswith("__")})
            return [out[k] if k in out else cache[k] for k in keys]
        return [out[k] for k in keys]

    @torch.no_grad()
    def read(self, items: list[ReadItem], max_new: int = 300, temp: float = 0.0, temp_when: float | None = None,
             gen: torch.Generator | None = None, record: bool = False, wait_bias: float = 0.0,
             cache: dict | None = None) -> list[ReadOut]:
        """Advance every item by one read. temp = 0: greedy (the evaluation decoders); otherwise sampling at temp,
        and at when positions at temp_when (default temp)."""
        emb_dtype = self.emb_layer.weight.dtype
        ae = self.audio_embeds(items, emb_dtype, cache=cache)
        prompts = [self.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(items, ae)]
        B, L = len(items), max(len(p) for p in prompts)
        ids = torch.full((B, L), self.tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.as_tensor(p)
            mask[i, L - len(p):] = 1
        ids, mask = ids.to(self.dev), mask.to(self.dev)
        emb = self.emb_layer(ids)
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
        o = self.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=True, return_dict=True)
        pkv = o.past_key_values
        h = o.last_hidden_state[:, -1]
        outs = [ReadOut(prompt_len=len(p)) for p in prompts]
        st = [seg_state(it.committed) for it in items]
        bar = torch.tensor([it.final or it.bar_wait for it in items], device=self.dev)
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        tw = temp if temp_when is None else temp_when
        spk_idx = torch.arange(SPK0, NEW_END, device=self.dev)
        for step in range(max_new):
            lg = self.logits(h).to(self.dev)
            if wait_bias:
                lg[:, WAIT_ID] += wait_bias
            lg[bar, WAIT_ID] = -1e9
            is_when = torch.tensor([s == "out" for s in st], device=self.dev) & ~bar
            if temp and temp > 0:
                tt = torch.where(is_when, torch.full((B,), tw, device=self.dev), torch.full((B,), temp, device=self.dev))
                probs = torch.softmax(lg / tt[:, None], -1)
                nxt = torch.multinomial(probs, 1, generator=gen)[:, 0]
            else:
                nxt = lg.argmax(-1)
            if record:
                lsm = torch.log_softmax(lg, -1)
                lp = lsm.gather(1, nxt[:, None])[:, 0]
                lpw = lsm[:, WAIT_ID]
                lspk = torch.log_softmax(lg[:, SPK0:NEW_END], -1)
                is_who = (nxt >= SPK0) & (nxt < NEW_END)
                lpwho = lspk.gather(1, (nxt - SPK0).clamp(0, N_SLOT - 1)[:, None])[:, 0]
                lp, lpw, lpwho, is_who_l = lp.tolist(), lpw.tolist(), lpwho.tolist(), is_who.tolist()
            nl, wl = nxt.tolist(), is_when.tolist()
            for i in range(B):
                if done[i]:
                    continue
                r = outs[i]
                r.ids.append(nl[i])
                if record:
                    r.when.append(bool(wl[i]))
                    r.who.append(bool(is_who_l[i]))
                    r.logp.append(float(lp[i]))
                    r.logp_wait.append(float(lpw[i]) if wl[i] else 0.0)
                    r.logp_who.append(float(lpwho[i]) if is_who_l[i] else 0.0)
                st[i] = step_state(st[i], nl[i])
            done |= (nxt == EOS_ID) | (nxt == WAIT_ID)
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=self.dev)], 1)
            o = self.thinker.model(inputs_embeds=self.emb_layer(nxt[:, None]), attention_mask=mask,
                                   past_key_values=pkv, use_cache=True, return_dict=True)
            pkv = o.past_key_values
            h = o.last_hidden_state[:, -1]
        del pkv, o
        return outs


class Parser:
    """gstar_si_stream_online_v6s.Parser, unchanged: committed ids -> segments and per-read blocks."""

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
                if self.open:
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
        text = self.tok.decode(self.text_ids, skip_special_tokens=False).strip()
        if self.open and text and self.spk is not None:
            self.segments.append({"spk": self.spk, "start_s": self.start, "end_s": end_s, "text": text,
                                  "write_ms": write_ms})
        self.open, self.text_ids = False, []


def lag_cut(body: list[int], t_ms: int, lag_ms: int) -> int:
    """gstar_si_stream_online_v6s --policy lag: the complete segments, in order, whose end <= t - lag."""
    k = 0
    open_seg = False
    for j, i in enumerate(body):
        if SPK0 <= i < NEW_END:
            open_seg = None
        elif NEW0 <= i < SPK0:
            if open_seg is None:
                open_seg = True
            elif open_seg:
                if (i - NEW0) * 80 > t_ms - lag_ms:
                    break
                open_seg, k = False, j + 1
    return k


def read_times(dur_ms: int, period_ms: int = 1000) -> list[int]:
    n = max(1, math.ceil(dur_ms / period_ms))
    return [min((r + 1) * period_ms, dur_ms) for r in range(n)]
