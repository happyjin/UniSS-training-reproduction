"""Plan 2's likelihood model: what text a stretch of the talker's speech says, read off its BiCodec semantic codes.

A small Transformer over the 50 Hz semantic codes with a CTC output over characters (Chinese
characters, Latin letters, digits, space and apostrophe; punctuation dropped). It is the p(y | c) the
talker's own p(c | y) lacks -- best-of-3 by the talker's likelihood kept the samples that skipped the
hard words -- and the emission model of the alignment filter: which unit of the block a code is saying.

  prep    codes and target texts from the offline pool (audio_index_clean_v1 + the UniST parquets),
          one npz with a held-out split and the character table, for training under Megatron
          (training/pretrain_ctc_head.py; exported by training/export_ctc_head.py)
  decode  greedy CTC transcripts of codes, and the per-sentence coverage of a text

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_ctc_head prep --out data/si_ctc_v1/rows.npz
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_CODES = 8192          # BiCodec semantic codebook
PAD_CODE = N_CODES      # an extra embedding row for padding
MAX_CODES = 1500        # 30 s, UniSS's own cap
CJK = re.compile(r"[一-鿿]")


def normalise(text: str, lang: str) -> str:
    """What the speech can say: punctuation gone, Latin lower-cased; English keeps word spaces."""
    t = text.lower()
    if lang == "cmn":
        return "".join(c for c in t if CJK.match(c) or c.isascii() and c.isalnum())
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r" +", " ", t).strip()


class CTCHead(nn.Module):
    def __init__(self, n_chars: int, d: int = 512, layers: int = 8, heads: int = 8, ff: int = 2048, max_len: int = MAX_CODES,
                 conv: int = 2):
        super().__init__()
        self.cfg = {"n_chars": n_chars, "d": d, "layers": layers, "heads": heads, "ff": ff, "max_len": max_len, "conv": conv}
        self.emb = nn.Embedding(N_CODES + 1, d, padding_idx=PAD_CODE)
        # local context over neighbouring codes first: without it the Transformer sat on the blank plateau
        self.conv = nn.Sequential(*[m for _ in range(conv) for m in (nn.Conv1d(d, d, 5, padding=2), nn.GELU())])
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_chars + 1)   # 0 is the CTC blank

    def forward(self, codes: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """codes [B, T] (PAD_CODE where pad), pad [B, T] bool -> log-probabilities [B, T, n_chars + 1]."""
        pos = torch.arange(codes.shape[1], device=codes.device).clamp(max=self.pos.num_embeddings - 1)
        x = self.emb(codes)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.enc(x + self.pos(pos)[None], src_key_padding_mask=pad)
        return torch.log_softmax(self.out(self.norm(h)).float(), dim=-1)


def load_ctc(path: str, device="cpu") -> tuple[CTCHead, list[str]]:
    ck = torch.load(path, map_location="cpu")
    m = CTCHead(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck["chars"]


def greedy(logp: torch.Tensor, chars: list[str]) -> str:
    ids = logp.argmax(-1).tolist()
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(chars[i - 1])
        prev = i
    return "".join(out)


def cmd_prep(args) -> None:
    from experiments.uniss_omni_s2st_v1.training.stage2_data import ParquetCodeIndex
    from experiments.uniss_qwen3_omni_v1.training.audio_index import AudioIndex

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = AudioIndex(args.audio_index)
    rng = random.Random(args.seed)
    names = sorted(idx.parquet_names())
    keep = set(rng.sample(names, min(args.parquets, len(names))))
    want = {"cmn": args.per_lang, "eng": args.per_lang}
    order = list(range(len(idx)))
    rng.shuffle(order)
    picked = []
    for i in order:
        row = idx[i]
        lang = {"zh": "cmn", "en": "eng"}.get(row.target_lang, row.target_lang)
        if row.parquet in keep and want.get(lang, 0) > 0:
            picked.append((row.parquet, row.row_index, lang, row.target_text))
            want[lang] -= 1
            if not any(want.values()):
                break
    print(f"picked {len(picked):,} rows from {len(keep)} parquets", flush=True)
    codes_ix = ParquetCodeIndex(args.parquet_root, {p for p, *_ in picked})
    texts = [normalise(t, lang) for _, _, lang, t in picked]
    counts = Counter(c for t in texts for c in t)
    chars = [c for c, n in counts.most_common() if n >= args.min_count]
    cid = {c: i + 1 for i, c in enumerate(chars)}
    code_vals, code_off, text_vals, text_off, langs, items = [], [0], [], [0], [], []
    dropped = 0
    for (parquet, ri, lang, _), t in zip(picked, texts):
        c, _ = codes_ix.get(parquet, ri)
        y = [cid[ch] for ch in t if ch in cid]
        if not len(c) or len(c) > MAX_CODES or not y or len(y) * 2 > len(c):
            dropped += 1
            continue
        code_vals.append(np.asarray(c, dtype=np.int16))
        code_off.append(code_off[-1] + len(c))
        text_vals.append(np.asarray(y, dtype=np.int32))
        text_off.append(text_off[-1] + len(y))
        langs.append(0 if lang == "cmn" else 1)
        items.append(f"{parquet}#{ri}")
    n = len(langs)
    val = np.zeros(n, dtype=bool)
    val[rng.sample(range(n), max(1, int(n * args.val_frac)))] = True
    np.savez(out, codes=np.concatenate(code_vals), code_off=np.asarray(code_off, dtype=np.int64),
             text=np.concatenate(text_vals), text_off=np.asarray(text_off, dtype=np.int64),
             lang=np.asarray(langs, dtype=np.uint8), val=val, chars=np.array(json.dumps(chars, ensure_ascii=False)),
             items=np.array(json.dumps(items)))
    frames = code_off[-1]
    print(json.dumps({"rows": n, "dropped": dropped, "chars": len(chars), "val": int(val.sum()),
                      "hours": round(frames / 50 / 3600, 1), "cmn": langs.count(0), "eng": langs.count(1),
                      "out": str(out)}, ensure_ascii=False), flush=True)


def block_nll(model, chars, codes: list[int], text: str, lang: str, dev) -> float | None:
    """-log p(text | codes) / characters: how badly this stretch of speech says this text (None: nothing to say)."""
    import torch.nn.functional as F

    cid = {c: i + 1 for i, c in enumerate(chars)}
    y = [cid[c] for c in normalise(text, lang) if c in cid]
    if not y or not codes:
        return None
    c = torch.tensor(codes[:MAX_CODES], dtype=torch.long, device=dev)[None]
    with torch.no_grad():
        lp = model(c, c == PAD_CODE)
    loss = F.ctc_loss(lp.transpose(0, 1), torch.tensor([y], device=dev), torch.tensor([c.shape[1]], device=dev),
                      torch.tensor([len(y)], device=dev), reduction="sum", zero_infinity=True)
    return float(loss) / len(y)


def cmd_check(args) -> None:
    """Does the head see what the ASR heard? Per sentence of a run made with --dump-codes: the worst block's
    normalised CTC loss against the ASR's verdict (an omission: the ASR hears under --short of the text's
    words or characters), as an AUC, and its rank correlation with the ASR's error against the text."""
    from rapidfuzz.distance import Levenshtein
    from scipy.stats import spearmanr

    from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import auc_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, chars = load_ctc(args.head, dev)
    run = Path(args.run)
    recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    asr = {json.loads(l)["id"]: json.loads(l).get("asr_text", "") for l in open(run / "asr.jsonl", encoding="utf-8")}
    worst, mean_, full_cer, label, asr_err, rows = [], [], [], [], [], []
    for r in recs:
        lang = r["tgt_lang"]
        if r["id"] not in asr or not r["blocks"] or "code_ids" not in r["blocks"][0]:
            continue
        nll = [block_nll(model, chars, b["code_ids"], b["text"], lang, dev) for b in r["blocks"]]
        nll = [x for x in nll if x is not None]
        if not nll:
            continue
        text = normalise(r["translated_text"], lang)
        heard = normalise(asr[r["id"]], lang)
        unit = (lambda s: list(s)) if lang == "cmn" else (lambda s: s.split())
        codes = [c for b in r["blocks"] for c in b["code_ids"]][:MAX_CODES]
        with torch.no_grad():
            ct = torch.tensor(codes, dtype=torch.long, device=dev)[None]
            hyp = greedy(model(ct, ct == PAD_CODE)[0], chars)
        worst.append(max(nll))
        mean_.append(float(np.mean(nll)))
        full_cer.append(Levenshtein.distance(text, hyp) / max(len(text), 1))
        label.append(float(len(unit(heard)) < args.short * len(unit(text))))
        asr_err.append(Levenshtein.distance(unit(text), unit(heard)) / max(len(unit(text)), 1))
        rows.append({"id": r["id"], "worst_nll": round(max(nll), 3), "ctc_cer": round(full_cer[-1], 3),
                     "asr_err": round(asr_err[-1], 3), "omission": int(label[-1]), "ctc_hyp": hyp, "text": text, "heard": heard})
    y = np.asarray(label)
    res = {"run": str(run), "sentences": len(y), "omissions": int(y.sum()),
           "auc_worst_block_nll": round(auc_score(y, np.asarray(worst)), 4),
           "auc_mean_block_nll": round(auc_score(y, np.asarray(mean_)), 4),
           "auc_ctc_cer": round(auc_score(y, np.asarray(full_cer)), 4),
           "spearman_worst_nll_vs_asr_err": round(float(spearmanr(worst, asr_err).correlation), 4),
           "spearman_ctc_cer_vs_asr_err": round(float(spearmanr(full_cer, asr_err).correlation), 4),
           "ctc_cer_mean": round(float(np.mean(full_cer)), 4), "asr_err_mean": round(float(np.mean(asr_err)), 4)}
    print(json.dumps(res), flush=True)
    if args.out:
        Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the head's verdict on a --dump-codes run against the ASR's")
    c.add_argument("--head", required=True)
    c.add_argument("--run", required=True, help="a realsi_<direction> directory made with --dump-codes")
    c.add_argument("--short", type=float, default=0.8)
    c.add_argument("--out", default="")
    a = sub.add_parser("prep")
    a.add_argument("--audio-index", default="data/unist_clean_v1/audio_index_clean_v1")
    a.add_argument("--parquet-root", default="data/raw/UniST")
    a.add_argument("--parquets", type=int, default=15, help="draw from this many parquets (the clean pool has 15)")
    a.add_argument("--per-lang", type=int, default=200000, help="rows a target language")
    a.add_argument("--min-count", type=int, default=5)
    a.add_argument("--val-frac", type=float, default=0.02)
    a.add_argument("--seed", type=int, default=20261022)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"prep": cmd_prep, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
