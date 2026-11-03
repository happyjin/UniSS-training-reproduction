"""Paper v8 B5 (re-review M12): do the 20 fp16 threshold flips of the causality check (Appendix K) change anything
downstream? For every cut with a flip (cuts_v2.jsonl, prefix vs whole-session dump), the flipped 80 ms frames are
located on the meeting clock and
  1. the online windows are rebuilt (build_online_windows.online_windows) with the dump's flipped frames replaced by
     the prefix run's values: window boundaries / count compared with the original;
  2. every online window holding a flipped frame gets a copy of its posterior with those frames replaced, written to
     <out>/post/<id>.npy, and a windows file <out>/<set>_windows.jsonl for re-decoding (same audio, same model).
Writes <out>/flips.json (refuses to overwrite).

    python reports/gstar_si_v8/b5_flips.py --out eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/b5
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
from build_online_windows import online_windows  # noqa: E402
from causality_compare import pool80  # noqa: E402
from causality_test import LOOKAHEAD_S, SETS  # noqa: E402

V6C = REPO / "eval_outputs/uniss_qwen3_omni_v1/gstar_v6_20261031/causality"
ON = REPO / "data/online_v6_20261031"
SETNAME = {"en": "en2zh", "zh": "zh2en", "ali": "ali"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "flips.json").exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "post").mkdir(parents=True, exist_ok=True)
    cuts = [json.loads(ln) for ln in open(V6C / "cuts_v2.jsonl")]
    flips = defaultdict(dict)   # (set, meeting) -> frame -> prefix row
    for r in cuts:
        d = r["vs_dump"]
        if round(d["frac_decision_diff"] * d["frames"]) == 0:
            continue
        st, m, t = r["set"], r["meeting"], r["t_s"]
        dump = torch.load(SETS[st][2](m), weights_only=False)["post80"].float().numpy()
        p = pool80(torch.load(V6C / "post" / f"{st}_{m}_{round(t * 1000)}.pt")["post80"].float().numpy(), t)
        n = max(0, min(int((t - LOOKAHEAD_S) / 0.08), len(p), len(dump)))
        a16 = p[:n].astype(np.float16).astype(np.float32)
        for i in np.flatnonzero(((a16 > 0.5) != (dump[:n] > 0.5)).any(1)):
            flips[(st, m)][int(i)] = a16[i]
    rep = []
    win_out = defaultdict(list)
    for (st, m), fr in sorted(flips.items()):
        dump = torch.load(SETS[st][2](m), weights_only=False)["post80"].float().numpy()
        mod = dump.copy()
        for i, row in fr.items():
            mod[i] = row
        w0, w1 = online_windows(dump), online_windows(mod)
        S = SETNAME[st]
        W = [w for w in map(json.loads, open(ON / f"windows_test_online_{S}.jsonl")) if w["meeting"] == m]
        hit = []
        for w in W:
            f0 = round(w["start_s"] / 0.08)
            post = np.load(w["post"]["lowlat"])
            idx = [i for i in fr if f0 <= i < f0 + len(post)]
            if not idx:
                continue
            q = post.astype(np.float32).copy()
            for i in idx:
                q[i - f0] = fr[i]
            pp = out / "post" / f"{w['id']}.npy"
            np.save(pp, q.astype(np.float16))
            w2 = dict(w, post={"lowlat": str(pp.resolve())})
            win_out[S].append(w2)
            hit.append(w["id"])
        rep.append({"set": S, "meeting": m, "flipped_frames": sorted(fr), "windows_same": w0 == w1,
                    "n_windows": [len(w0), len(w1)], "windows_with_flip": hit})
    for S, ws in win_out.items():
        with open(out / f"{S}_windows.jsonl", "x") as fh:
            fh.writelines(json.dumps(w) + "\n" for w in ws)
    (out / "flips.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
