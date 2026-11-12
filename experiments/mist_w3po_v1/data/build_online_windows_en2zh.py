"""En->Zh online windows (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §2): the C2 detector of the zh->en
recipe (build_windows_zh2en.online_windows(post, gap): a window opens at the first frame where any slot is > 0.5 and
closes after `gap` s without activity or at 20 s; audio [open - 0.2 s, last active frame + 0.3 s]; open / close
detection times with the 1.04 s look-ahead) on the ORIGINAL English AMI meetings: session audio ihm_mix
(<m>.Mix-Headset.wav, channel mean), low-latency Nemotron posteriors diar/nemotron_lowlat (on disk, no GPU) -- the
en2zh set of reports/gstar_si_v6/build_online_windows.py (SETS["en2zh"]). Every window of every meeting is kept (no
filter, utts empty: scoring uses the reference windows).

  --split validation  the 18 AMI dev meetings (all have scored En->Zh references; the zh->en A8 skip is a gate on the
                      Chinese re-voiced audio and does not apply) -> windows_dev_online.jsonl
  --split train16     the first 16 train meetings in alphabetical order (the zh->en PL-EM talker subset rule,
                      talker_crf_prereg_v1 §3) -> windows_train_online.jsonl
  --split test        the 16 AMI test meetings -> windows_test_online_en2zh.jsonl ("split": "test", the v6 test format)
Record format = build_windows_zh2en / build_test_windows_c2 ("dir": "en2zh").

Reproduction check (no references used): --split test --gap 1.0 --meta-only --check
data/online_v6_20261031/windows_test_online_en2zh.jsonl compares every field except audio / post.
--dev5 <dir>: also writes <dir>/windows_dev_online.jsonl with the 5 precheck meetings (ES2011a IB4003 IB4011 IS1008c
TS3004b; validation only). CPU only. Refuses existing outputs.

    python -m experiments.mist_w3po_v1.data.build_online_windows_en2zh --split validation --gap 0.6 --out <new dir> \
        [--dev5 <new dir>] [--meta-only] [--check <file>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.data.build_windows_zh2en import CAP, FRAME, LOOKAHEAD, online_windows  # noqa: F401

DATA = Path("data/ami_multispk_si_v1")
DIAR = "nemotron_lowlat"
DEV5 = ("ES2011a", "IB4003", "IB4011", "IS1008c", "TS3004b")
NAME = {"validation": "windows_dev_online.jsonl", "train16": "windows_train_online.jsonl",
        "test": "windows_test_online_en2zh.jsonl"}


def meetings(split: str) -> list[str]:
    sp = json.load(open(DATA / "meetings.json"))
    if split == "train16":
        return sorted(m for m, s in sp.items() if s == "train")[:16]
    return sorted(m for m, s in sp.items() if s == split)


def window_records(m: str, post: np.ndarray, sess_s: float, gap: float, split: str, out: Path) -> list[tuple]:
    """(record, (ws, we, f0, nf)) per window of meeting m, exactly build_test_windows_c2's arithmetic."""
    res = []
    for k, (st, en, j) in enumerate(online_windows(post, gap)):
        ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
        close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(post) else sess_s
        wid = f"{m}_on_{k:04d}"
        sub = "test" if split == "test" else split
        wav = out / "audio" / sub / f"{wid}.wav"
        pp = out / "post" / sub / f"{wid}.npy"
        rec = {"id": wid, "meeting": m, "split": "train" if split == "train16" else split, "wtype": "online",
               "dir": "en2zh", "start_s": round(ws, 3), "dur_ms": round((we - ws) * 1000), "audio": str(wav.resolve()),
               "post": {"lowlat": str(pp.resolve())}, "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
               "close_detect_ms": round((close - ws) * 1000), "utts": [], "strict_review": False}
        res.append((rec, (ws, we, round(ws / FRAME), int(np.ceil((we - ws) / FRAME)))))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(NAME))
    ap.add_argument("--gap", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--check", default="")
    ap.add_argument("--dev5", default="")
    a = ap.parse_args()
    import soundfile as sf
    import torch

    out = Path(a.out)
    wf, sf_ = out / NAME[a.split], out / f"summary_{a.split}.json"
    for f in (wf, sf_):
        if f.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {f}")
    if a.dev5:
        if a.split != "validation":
            raise SystemExit("--dev5 needs --split validation")
        if (Path(a.dev5) / NAME["validation"]).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {a.dev5}")
    out.mkdir(parents=True, exist_ok=True)
    n_tot, dur_tot, per_m, recs = 0, 0.0, {}, []
    with open(wf, "x", encoding="utf-8") as fh:
        for m in meetings(a.split):
            post = torch.load(DATA / "diar" / DIAR / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
            apath = DATA / "session_audio" / "ihm_mix" / f"{m}.Mix-Headset.wav"
            if a.meta_only:
                info = sf.info(str(apath))
                x, sr, sess_s = None, info.samplerate, info.frames / info.samplerate
            else:
                x, sr = sf.read(str(apath), dtype="float32", always_2d=True)
                x = x.mean(1)
                sess_s = len(x) / sr
            wl = window_records(m, post, sess_s, a.gap, a.split, out)
            for rec, (ws, we, f0, nf) in wl:
                if not a.meta_only:
                    wav, pp = Path(rec["audio"]), Path(rec["post"]["lowlat"])
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    pp.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                    s = post[f0: f0 + nf]
                    if len(s) < nf:
                        s = np.concatenate([s, np.zeros((nf - len(s), post.shape[1]), dtype=s.dtype)])
                    np.save(pp, s.astype(np.float16))
                fh.write(json.dumps(rec) + "\n")
                recs.append(rec)
                n_tot += 1
                dur_tot += we - ws
            per_m[m] = len(wl)
            print(f"{m}: {len(wl)} windows", flush=True)
    summ = {"split": a.split, "gap": a.gap, "meetings": len(per_m), "windows": n_tot, "hours": round(dur_tot / 3600, 3),
            "per_meeting": per_m, "meta_only": a.meta_only, "args": vars(a), "audio": "ihm_mix", "diar": DIAR}
    if a.check:
        B = [json.loads(l) for l in open(a.check, encoding="utf-8")]
        keys = [k for k in B[0] if k not in ("audio", "post")]
        diff = sum(1 for r, s in zip(recs, B) if any(r.get(k) != s.get(k) for k in keys))
        summ["check"] = {"file": a.check, "n_new": len(recs), "n_ref": len(B),
                         "records_differing": diff + abs(len(recs) - len(B)), "keys": keys}
    if a.dev5:
        d5 = Path(a.dev5)
        d5.mkdir(parents=True, exist_ok=True)
        rows = [r for r in recs if r["meeting"] in DEV5]
        with open(d5 / NAME["validation"], "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        (d5 / "README").write_text(f"dev windows of {wf} restricted to the 5 precheck meetings "
                                   f"(talker_crf_prereg_v1 §1.2): {' '.join(DEV5)}; {len(rows)} windows\n")
        summ["dev5"] = {"dir": str(d5), "windows": len(rows)}
    with open(sf_, "x", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_meeting"}))


if __name__ == "__main__":
    main()
