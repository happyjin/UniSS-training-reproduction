"""Paper v6, review M08: how far are the deadline labels from where the source evidence for a chunk actually ends?

For a fixed sample of validation utterances (seed 20261031), each target chunk (the translation cut at punctuation, as
the labels cut it) gets
  rule        the label's own deadline before the 1 s lag: En->Zh, the source word at the chunk's character fraction
              (AMI word times); Zh->En v5, the same fraction of the utterance's duration; Zh->En v6, the source character
              at that fraction (forced-aligned times, zh_char_times.py)
  aligned     the end of the last source word (En) / character (Zh) aligned to any unit of the chunk or of an earlier
              chunk -- what has to have been heard to write the chunk -- with a SimAlign-style aligner (multilingual
              BERT, layer 8, mutual argmax of subword similarities, subwords pooled to words / characters)
Reports the error rule - aligned (s): its distribution, and the share of chunks whose rule deadline + 1 s lag still
comes before the aligned evidence ("asked for before it was heard"). Writes <out> (json).

    PYTHONPATH=. python reports/gstar_si_v6/deadline_audit.py --out <json> [--n 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import chunks_of, zh_len

D = Path("data/ami_multispk_si_v1")
MBERT = "/opt/dlami/nvme/jasonleeeli/cache/hf_v6/models--google-bert--bert-base-multilingual-cased"


def ami_words(zip_path: Path, meetings: set) -> dict:
    """meeting -> speaker -> sorted [(start, end, text)] of spoken (non-punctuation) words."""
    z = zipfile.ZipFile(zip_path)
    meet = ET.fromstring(z.read("corpusResources/meetings.xml"))
    agent = {}
    for m in meet.iter("meeting"):
        for s in m.iter("speaker"):
            agent[(m.get("observation"), s.get("nxt_agent"))] = s.get("global_name")
    out = defaultdict(lambda: defaultdict(list))
    for name in z.namelist():
        mm = re.fullmatch(r"words/([A-Z]{2}\d{4}[a-z]?)\.([A-Z])\.words\.xml", name)
        if not mm or mm.group(1) not in meetings or (mm.group(1), mm.group(2)) not in agent:
            continue
        spk = agent[(mm.group(1), mm.group(2))]
        for w in ET.fromstring(z.read(name)).iter("w"):
            if w.get("punc") or w.get("starttime") is None or w.get("endtime") is None or not (w.text or "").strip():
                continue
            out[mm.group(1)][spk].append((float(w.get("starttime")), float(w.get("endtime")), w.text.strip()))
    for m in out.values():
        for v in m.values():
            v.sort()
    return out


class Aligner:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        snap = sorted(Path(MBERT).glob("snapshots/*"))[0]
        self.tok = AutoTokenizer.from_pretrained(str(snap))
        self.model = AutoModel.from_pretrained(str(snap), output_hidden_states=True).eval()

    def embed(self, words):
        enc = self.tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=510)
        with torch.no_grad():
            h = self.model(**enc).hidden_states[8][0]
        ids = enc.word_ids()
        vec = []
        for i in range(len(words)):
            idx = [j for j, w in enumerate(ids) if w == i]
            vec.append(h[idx].mean(0) if idx else torch.zeros(h.shape[1]))
        v = torch.stack(vec)
        return torch.nn.functional.normalize(v, dim=-1)

    def align(self, src, tgt):
        """Mutual-argmax alignment of word lists -> list of (i_src, j_tgt)."""
        a, b = self.embed(src), self.embed(tgt)
        sim = a @ b.T
        f, r = sim.argmax(1), sim.argmax(0)
        return [(i, int(f[i])) for i in range(len(src)) if int(r[int(f[i])]) == i]


def audit(dr, n, seed, al, char_times=None):
    rng = random.Random(seed)
    if dr == "en2zh":
        W = [json.loads(l) for l in open(D / "e2e_gstar_stream_v1/windows_validation_ref.jsonl", encoding="utf-8")]
        words = ami_words(D / "annotations/ami_public_manual_1.6.2.zip", {w["meeting"] for w in W})
    else:
        W = [json.loads(l) for l in open(D / "e2e_gstar_zh2en_v1/windows_validation_ref.jsonl", encoding="utf-8")]
    utts = [(w, u) for w in W for u in w["utts"] if not u["hesitation"]]
    rng.shuffle(utts)
    res = defaultdict(list)
    done = 0
    for w, u in utts:
        if done >= n:
            break
        a, b = w["start_s"] + u["begin_s"], w["start_s"] + u["end_s"]
        if dr == "en2zh":
            src = [x for x in words[w["meeting"]].get(u["speaker_id"], []) if x[0] >= a - 0.05 and x[1] <= b + 0.05]
            tgt_text = u["zh"]
            if not src or not tgt_text.strip():
                continue
            src_units = [x[2] for x in src]
            src_end = [x[1] for x in src]
        else:
            ct = char_times.get(u["record_id"])
            tgt_text = u.get("ref", "")
            if not ct or not tgt_text.strip():
                continue
            src_units = re.findall(r"[一-鿿]", u["zh"])
            if len(src_units) != len(ct):
                continue
            src_end = [c[1] for c in ct]
        cs = chunks_of(tgt_text)
        if dr == "en2zh":
            tgt_units, owner = [], []
            for k, c in enumerate(cs):
                for ch in re.findall(r"[一-鿿]|[A-Za-z0-9']+", c):
                    tgt_units.append(ch)
                    owner.append(k)
        else:
            tgt_units, owner = [], []
            for k, c in enumerate(cs):
                for t in re.findall(r"[A-Za-z0-9']+", c):
                    tgt_units.append(t)
                    owner.append(k)
        if not tgt_units:
            continue
        pairs = al.align(src_units, tgt_units)
        last_src = defaultdict(lambda: -1)
        for i, j in pairs:
            last_src[owner[j]] = max(last_src[owner[j]], i)
        total = sum(zh_len(c) for c in cs)
        acc, need = 0, -1
        for k, c in enumerate(cs):
            acc += zh_len(c)
            f = acc / max(total, 1)
            need = max(need, last_src[k])   # evidence needed by this chunk and the ones before it
            if need < 0:
                continue
            aligned = src_end[need]
            rules = {}
            if dr == "en2zh":
                rules["word_fraction"] = src_end[max(0, math.ceil(f * len(src_end)) - 1)]
            else:
                rules["duration_v5"] = a + f * (b - a)
                rules["char_fraction_v6"] = src_end[max(0, math.ceil(f * len(src_end)) - 1)]
            for name, t in rules.items():
                res[name].append(t - aligned)
        done += 1
    out = {"utterances": done}
    for name, v in res.items():
        v = np.array(v)
        out[name] = {"chunks": int(len(v)), "mean_s": round(float(v.mean()), 3), "median_s": round(float(np.median(v)), 3),
                     "mae_s": round(float(np.abs(v).mean()), 3),
                     "p10_s": round(float(np.percentile(v, 10)), 3), "p90_s": round(float(np.percentile(v, 90)), 3),
                     "before_evidence_even_with_1s_lag": round(float((v + 1.0 < 0).mean()), 4),
                     "earlier_than_evidence": round(float((v < 0).mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    torch.set_num_threads(16)
    al = Aligner()
    ct = {}
    for f in (D / "zh_fa_v6").glob("*.jsonl"):
        for r in map(json.loads, open(f, encoding="utf-8")):
            ct[r["record_id"]] = r["chars"]
    res = {"en2zh": audit("en2zh", args.n, args.seed, al), "zh2en": audit("zh2en", args.n, args.seed, al, ct)}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
