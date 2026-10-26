"""Route 4a feasibility: does the Thinker's own state predict a draft unit's stability better than the 17 features?

  extract  for reads sampled from Route 1's evidence dumps (si_thinker_stream --oracle --dump --virtual-ms-list), the
           HF Thinker re-reads the read's prompt (audio heard so far, committed prefix) and its greedy draft in one
           forward pass; each draft unit's state (last and middle layer, at the unit's last token) is kept beside the
           unit's label (draft[:u+1] == oracle[:u+1]), the 17-feature prior's score and Route 1's posterior score
  probe    an MLP on the states (and on states + the Route 1 posterior), trained on items outside the held-out fifth
           (the same split as si_bayes_lr), scored on it: unit AUC and the units a read can commit at a given risk

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_hidden_probe extract --dumps 'dir/*.jsonl' --shard 0 --num-shards 8 --out part0.npz
    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_hidden_probe probe --parts 'part*.npz' --out probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import zlib
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
MAX_UNITS = 16


def sample_reads(pattern: str, per_file: int, seed: int = 0) -> list[tuple[str, dict]]:
    out = []
    for f in sorted(glob.glob(pattern)):
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        recs = [r for r in recs if r["lp"] and "oracle_units" in r and r["draft_units"]]
        random.Random(seed).shuffle(recs)
        out += [(f, r) for r in recs[:per_file]]
    return out


def committed_text(blocks: list[dict], t: int) -> str:
    return "".join(b["text"] for b in blocks if b["write_ms"] < t)


def cmd_extract(args) -> None:
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from experiments.uniss_qwen3_omni_v1.evaluation.si_bayes_lr import load_table, token_logits, lcp
    from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import load_head
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    reads = sample_reads(args.dumps, args.per_file)[args.shard:: args.num_shards]
    blocks, audio_of = {}, {}
    for f in {f for f, _ in reads}:
        d = Path(f).parent.parent
        for b in glob.glob(str(d / "blocks_*.jsonl")):
            for line in open(b, encoding="utf-8"):
                r = json.loads(line)
                blocks[r["id"]] = r["blocks"]
        direction = "en2zh" if "_en2zh_" in f else "zh2en"
        for line in open(REPO / f"data/si_bayes_lr_v1/items_{direction}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            audio_of[r["id"]] = r["audio"]
    prior, table = load_head(args.prior), load_table(args.table)
    device = "cuda:0"
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    tok = processor.tokenizer
    config = AutoConfig.from_pretrained(QWEN3)
    config.enable_audio_output = False
    thinker = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        QWEN3, config=config, dtype=torch.bfloat16, device_map={"": device}).thinker.eval()
    import soundfile as sf

    feats, rows = [], []
    for s in range(0, len(reads), args.batch):
        chunk = reads[s: s + args.batch]
        waves, texts, conts = [], [], []
        for f, r in chunk:
            zh = "_en2zh_" in f
            x, _ = sf.read(audio_of[r["id"]], dtype="float32", always_2d=True)
            waves.append(x.mean(axis=1)[: int(r["t"]) * 16])
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                                 {"type": "text", "text": inference_instruction()["cmn" if zh else "eng"]}]}]
            head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
            texts.append(head + committed_text(blocks.get(r["id"], []), int(r["t"])))
            conts.append("".join(r["tokens"]))
        tok.padding_side = "left"
        enc = processor(text=texts, audio=waves, sampling_rate=16_000, return_tensors="pt", padding=True)
        width = enc["input_ids"].shape[1]
        cenc = [tok(c, add_special_tokens=False, return_offsets_mapping=True) for c in conts]
        longest = max(len(c["input_ids"]) for c in cenc)
        ids = torch.full((len(chunk), width + longest), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros_like(ids)
        ids[:, :width], mask[:, :width] = enc["input_ids"], enc["attention_mask"]
        for j, c in enumerate(cenc):
            ids[j, width: width + len(c["input_ids"])] = torch.as_tensor(c["input_ids"])
            mask[j, width: width + len(c["input_ids"])] = 1
        with torch.no_grad():
            hs = thinker(input_ids=ids.to(device), attention_mask=mask.to(device),
                         input_features=enc["input_features"].to(device=device, dtype=torch.bfloat16),
                         feature_attention_mask=enc["feature_attention_mask"].to(device),
                         output_hidden_states=True, return_dict=True).hidden_states
        mid = len(hs) // 2
        for j, ((f, r), c, cont) in enumerate(zip(chunk, cenc, conts)):
            zh = "_en2zh_" in f
            gu = r["draft_units"]
            spans = [(m.start(), m.end()) for m in re.finditer(r"\S" if zh else r"\S+", cont)]
            last_tok = {}   # unit -> its last token's position
            for k, (a, b) in enumerate(c["offset_mapping"]):   # a token may carry several Chinese characters
                for q, (x0, x1) in enumerate(spans):
                    if x0 < b and x1 > a:
                        last_tok[q] = k
            n = min(len(gu), len(spans), MAX_UNITS)
            if n == 0 or any(u not in last_tok for u in range(n)):
                continue
            pos = torch.as_tensor([width + last_tok[u] for u in range(n)], device=device)
            h = torch.cat([hs[-1][j, pos], hs[mid][j, pos]], dim=-1).float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
            feats.append(h.to(torch.float16).cpu().numpy())
            # the dump's own tokens for the 17-feature prior and Route 1's posterior, a unit scoring its least sure token
            virt = r.get("virt") or {}
            others = [r.get("prev_real")] + [virt.get(str(dd)) for dd in table["virtual"]]
            zp = token_logits(prior, {**table, "w_arr": np.zeros_like(table["w_arr"])}, r, gu, others,
                              int(r.get("window_ms", 1000)), r["heard_s"], r["n_committed"], zh)
            zf = token_logits(prior, table, r, gu, others, int(r.get("window_ms", 1000)), r["heard_s"], r["n_committed"], zh)
            up, uf = np.full(len(gu), 50.0), np.full(len(gu), 50.0)
            for z0, z1, u in zip(zp, zf, r["unit_of"]):
                if u < len(gu):
                    up[u], uf[u] = min(up[u], z0), min(uf[u], z1)
            for u in range(1, len(gu)):   # a character inside a multi-character token: that token's score
                if up[u] == 50.0:
                    up[u], uf[u] = up[u - 1], uf[u - 1]
            L = lcp(gu, r["oracle_units"])
            for u in range(n):
                rows.append((zlib.crc32(r["id"].encode()), s + j, u, int(u < L), float(up[u]), float(uf[u]),
                             int(zh), L, len(gu)))
        if (s // args.batch) % 20 == 0:
            print(json.dumps({"shard": args.shard, "reads": s + len(chunk), "of": len(reads), "units": len(rows)}), flush=True)
    rows = np.asarray(rows, dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, h=np.concatenate(feats), item_hash=rows[:, 0].astype(np.uint32), read=rows[:, 1].astype(np.int64),
             unit=rows[:, 2].astype(np.int32), y=rows[:, 3].astype(np.float32), z_prior=rows[:, 4].astype(np.float32),
             z_post=rows[:, 5].astype(np.float32), zh=rows[:, 6].astype(bool), read_lcp=rows[:, 7].astype(np.int32),
             read_units=rows[:, 8].astype(np.int32))
    print(json.dumps({"shard": args.shard, "units": int(len(rows)), "out": str(out)}), flush=True)


def units_curve(z: np.ndarray, d: dict, idx: np.ndarray) -> dict:
    """Commit, in each read, the leading units scoring >= tau; risk = the read commits past the oracle."""
    reads = {}
    for k in idx:
        reads.setdefault(int(d["read"][k]), []).append(k)
    curve = []
    for tau in np.linspace(-4, 6, 101):
        keep, err = [], []
        for ks in reads.values():
            ks = sorted(ks, key=lambda k: d["unit"][k])
            c = 0
            while c < len(ks) and z[ks[c]] >= tau:
                c += 1
            keep.append(c)
            err.append(c > d["read_lcp"][ks[0]])
        curve.append((float(np.mean(err)), float(np.mean(keep))))
    res = {str(eps): round(max([k for e, k in curve if e <= eps], default=0.0), 3) for eps in (0.05, 0.1, 0.2)}
    return res


def cmd_probe(args) -> None:
    from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import auc_score
    from torch import nn

    parts = [np.load(p) for p in sorted(glob.glob(args.parts))]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    # reads are numbered per shard; make them unique across parts
    off, reads = 0, []
    for p in parts:
        reads.append(p["read"] + off)
        off += int(p["read"].max()) + 1
    d["read"] = np.concatenate(reads)
    val = d["item_hash"] % 5 == 0
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = {"units_train": int(len(tr)), "units_val": int(len(va)), "pos": round(float(d["y"].mean()), 3)}
    for name, z in (("prior_17feat", d["z_prior"]), ("route1_posterior", d["z_post"])):
        res[name] = {"auc": round(auc_score(d["y"][va], z[va]), 4), "units_at_risk": units_curve(z, d, va)}
    for name, extra in (("hidden", None), ("hidden+route1", d["z_post"])):
        x = torch.from_numpy(d["h"].astype(np.float32))
        if extra is not None:
            x = torch.cat([x, torch.from_numpy(extra[:, None].astype(np.float32) / 4)], dim=1)
        y = torch.from_numpy(d["y"])
        net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        g = torch.Generator().manual_seed(0)
        for ep in range(args.epochs):
            net.train()
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]   # a full shuffle each epoch
            for s in range(0, len(perm), 1024):
                k = perm[s: s + 1024]
                loss = nn.functional.binary_cross_entropy_with_logits(net(x[k].to(dev)).squeeze(-1), y[k].to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            z = torch.cat([net(x[s: s + 8192].to(dev)).squeeze(-1).cpu() for s in range(0, len(x), 8192)]).numpy()
        res[name] = {"auc": round(auc_score(d["y"][va], z[va]), 4), "units_at_risk": units_curve(z, d, va)}
        print(json.dumps({name: res[name]}), flush=True)
    res["oracle_units_per_read"] = round(float(np.mean([d["read_lcp"][d["read"] == r][0] for r in np.unique(d["read"][va])])), 3)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dumps", required=True)
    e.add_argument("--prior", required=True)
    e.add_argument("--table", required=True)
    e.add_argument("--per-file", type=int, default=4000)
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--out", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--parts", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"extract": cmd_extract, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    main()
