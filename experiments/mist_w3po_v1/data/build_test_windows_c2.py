"""Final test (reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md): the C2 online windows of a TEST set, built with
exactly the dev C2 detector configuration (windows_zh2en_c2g06_v1): the low-latency Nemotron posteriors already on
disk (preset lowlat), build_windows_zh2en.online_windows(post, gap 0.6) -- a window opens at the first frame where any
slot is > 0.5 and closes after 0.6 s without activity or at 20 s --, audio [open - 0.2 s, last active frame + 0.3 s],
open / close detection times with the 1.04 s look-ahead. The meetings, session audio and posteriors of each set are
those of the v6 test windows (reports/gstar_si_v6/build_online_windows.py SETS: zh2en = the 16 AMI test meetings on
zh_mix_v8 + nemotron_lowlat_zh8; ali = the 20 AliMeeting test meetings on near_mix + nemotron_lowlat); every window of
every meeting is kept (no filter), the record format is the v6 test format ("split": "test"). No GPU.

--gap 1.0 --meta-only reproduces data/online_v6_20261031/windows_test_online_<set>.jsonl (the reproduction check:
--check <that file> compares every record except the audio / post paths). --meta-only writes the jsonl only (audio /
post paths in the records are where a full build would put them; nothing else is written).

    python -m experiments.mist_w3po_v1.data.build_test_windows_c2 --set zh2en|ali --gap 0.6 --out <new dir>
Refuses an existing output file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
import build_online_windows as V6  # noqa: E402

from experiments.mist_w3po_v1.data.build_windows_zh2en import CAP, FRAME, LOOKAHEAD, online_windows  # noqa: E402

assert (FRAME, LOOKAHEAD, CAP) == (V6.FRAME, V6.LOOKAHEAD, V6.CAP)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["zh2en", "ali"])
    ap.add_argument("--gap", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--check", default="", help="compare with this window file (all fields but audio / post)")
    a = ap.parse_args()
    out = Path(a.out)
    wf = out / f"windows_test_online_{a.set}.jsonl"
    for f in (wf, out / f"summary_{a.set}.json"):
        if f.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {f}")
    meetings, audio_of, post_of = V6.SETS[a.set]
    out.mkdir(parents=True, exist_ok=True)
    if not a.meta_only:
        (out / "audio" / a.set).mkdir(parents=True, exist_ok=True)
        (out / "post" / a.set).mkdir(parents=True, exist_ok=True)
    n_tot, dur_tot, per_m = 0, 0.0, {}
    with open(wf, "x", encoding="utf-8") as fh:
        for m in meetings():
            post = torch.load(post_of(m), weights_only=False)["post80"].float().numpy()
            x, sr = sf.read(str(audio_of(m)), dtype="float32", always_2d=True)
            x = x.mean(1)
            sess_s = len(x) / sr
            wl = online_windows(post, a.gap)
            for k, (st, en, j) in enumerate(wl):
                ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
                close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(post) else sess_s
                wid = f"{m}_on_{k:04d}"
                wav = out / "audio" / a.set / f"{wid}.wav"
                pp = out / "post" / a.set / f"{wid}.npy"
                if not a.meta_only:
                    sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                    f0, nf = round(ws / FRAME), int(np.ceil((we - ws) / FRAME))
                    s = post[f0: f0 + nf]
                    if len(s) < nf:
                        s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
                    np.save(pp, s.astype(np.float16))
                rec = {"id": wid, "meeting": m, "split": "test", "wtype": "online", "dir": "zh2en",
                       "start_s": round(ws, 3), "dur_ms": round((we - ws) * 1000), "audio": str(wav.resolve()),
                       "post": {"lowlat": str(pp.resolve())},
                       "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
                       "close_detect_ms": round((close - ws) * 1000), "utts": [], "strict_review": False}
                fh.write(json.dumps(rec) + "\n")
                n_tot += 1
                dur_tot += we - ws
            per_m[m] = len(wl)
            print(f"{m}: {len(wl)} windows", flush=True)
    summ = {"set": a.set, "gap": a.gap, "meetings": len(per_m), "windows": n_tot, "hours": round(dur_tot / 3600, 3),
            "per_meeting": per_m, "meta_only": a.meta_only, "args": vars(a)}
    if a.check:
        A = [json.loads(l) for l in open(wf, encoding="utf-8")]
        B = [json.loads(l) for l in open(a.check, encoding="utf-8")]
        keys = [k for k in B[0] if k not in ("audio", "post")]
        diff = sum(1 for r, s in zip(A, B) if any(r.get(k) != s.get(k) for k in keys))
        summ["check"] = {"file": a.check, "n_new": len(A), "n_ref": len(B), "records_differing": diff + abs(len(A) - len(B)),
                         "keys": keys}
    with open(out / f"summary_{a.set}.json", "x", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_meeting"}))


if __name__ == "__main__":
    main()
