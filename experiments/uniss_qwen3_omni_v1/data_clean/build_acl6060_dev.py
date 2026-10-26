"""ACL 60/60 dev (IWSLT 2023, Salesky et al.) for long-form En->Zh streaming S2TT: the five full talks and the 468 gold
reference segments with their place on each talk's clock -- not shipped with the release (only SHAS's yaml is), so each
gold segment wav (sent_<n>.wav, in FILE_ORDER talk order) is located in its talk by matching its samples (exact match,
else the best normalised cross-correlation within the next stretch of the talk).

Writes refs.jsonl {talk, seg, offset_ms, dur_ms, ref (zh), src (en)} and talks.jsonl {talk, wav, dur_ms}.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_acl6060_dev --root data/raw/acl6060/extracted/2/acl_6060/dev \
        --out data/acl6060_dev_v1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def locate(talk: np.ndarray, seg: np.ndarray, start: int) -> tuple[int, str]:
    k = min(len(seg), 4000)
    head = seg[:k]
    cand = np.flatnonzero(talk[start:] == head[0]) + start
    for c in cand[:200000]:
        if c + k <= len(talk) and np.array_equal(talk[c: c + k], head):
            return int(c), "exact"
    win = talk[start: start + 120 * 16000]   # cross-correlation over the next two minutes
    a = head - head.mean()
    best, pos = -1.0, start
    for c in range(0, max(1, len(win) - k), 160):
        b = win[c: c + k] - win[c: c + k].mean()
        r = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        if r > best:
            best, pos = r, start + c
    return int(pos), f"xcorr{best:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    order = [l.strip() for l in open(root / "FILE_ORDER") if l.strip()]
    xml = (root / "text/xml/ACL.6060.dev.en-xx.en.xml").read_text(encoding="utf-8")
    docs = re.findall(r'<doc docid="([^"]+)".*?</doc>', xml, flags=re.S)
    counts = [len(re.findall(r"<seg ", d)) for d in re.findall(r"<doc .*?</doc>", xml, flags=re.S)]
    assert docs == order, (docs, order)
    en = [l.rstrip("\n") for l in open(root / "text/txt/ACL.6060.dev.en-xx.en.txt", encoding="utf-8")]
    zh = [l.rstrip("\n") for l in open(root / "text/txt/ACL.6060.dev.en-xx.zh.txt", encoding="utf-8")]
    assert len(en) == len(zh) == sum(counts)
    out.mkdir(parents=True)
    n, methods = 0, {}
    with open(out / "refs.jsonl", "x", encoding="utf-8") as fr, open(out / "talks.jsonl", "x", encoding="utf-8") as ft:
        for talk, cnt in zip(order, counts):
            wav = root / "full_wavs" / f"{talk}.wav"
            x, sr = sf.read(wav, dtype="int16")
            assert sr == 16000
            ft.write(json.dumps({"talk": talk, "wav": str(wav.resolve()), "dur_ms": int(len(x) / 16)}) + "\n")
            pos = 0
            for _ in range(cnt):
                n += 1
                s, sr2 = sf.read(root / f"segmented_wavs/gold/sent_{n}.wav", dtype="int16")
                assert sr2 == 16000
                p, how = locate(x, s, pos)
                methods[how[:5]] = methods.get(how[:5], 0) + 1
                fr.write(json.dumps({"talk": talk, "seg": n, "offset_ms": int(p / 16), "dur_ms": int(len(s) / 16),
                                     "ref": zh[n - 1], "src": en[n - 1], "match": how}, ensure_ascii=False) + "\n")
                pos = p + len(s) // 2
    print(json.dumps({"talks": len(order), "segments": n, "match": methods, "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
