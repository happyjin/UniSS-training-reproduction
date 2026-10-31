"""Paper v6, review M08 (prereg_v6 §4): character times of the Chinese AMI speech, for deadline labels from real timing.

The Chinese meetings (session_audio/zh_mix_v8) are re-voiced: every utterance's Chinese speech is laid on the meeting
clock at zh_begin..zh_end (zh_mix_v8/utts/<meeting>.jsonl). Its text is known, so the FunASR timestamp predictor
(iic/speech_timestamp_prediction-v1-16k-offline, a forced aligner) gives the time of each Chinese character. Writes
<out>/<meeting>.jsonl: {"record_id", "chars": [[start_s, end_s], ...]} on the meeting clock (characters only, as the
aligner sees them; utterances without a Chinese character are skipped). Meetings already written are skipped. CPU.

    python reports/gstar_si_v6/zh_char_times.py --out data/ami_multispk_si_v1/zh_fa_v6 --procs 48 [--splits train,validation,test]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/ami_multispk_si_v1"
FA = "/opt/dlami/nvme/jasonleeeli/cache/modelscope/models_v6/models/iic--speech_timestamp_prediction-v1-16k-offline/snapshots/master"
_M = None


def one(args):
    m, out = args
    global _M
    import soundfile as sf
    import torch
    torch.set_num_threads(2)
    if _M is None:
        from funasr import AutoModel
        _M = AutoModel(model=FA, disable_update=True, device="cpu")
    dst = Path(out) / f"{m}.jsonl"
    if dst.exists():
        return m, -1
    x, sr = sf.read(str(D / f"session_audio/zh_mix_v8/{m}.wav"), dtype="float32")
    rows = []
    for u in map(json.loads, open(D / f"zh_mix_v8/utts/{m}.jsonl", encoding="utf-8")):
        if u["hes"] or "zh_begin" not in u or not u.get("placed", True):
            continue
        chars = re.findall(r"[一-鿿]", u["zh"])
        if not chars:
            continue
        seg = x[int(u["zh_begin"] * sr): int(u["zh_end"] * sr)]
        if len(seg) < sr // 10:
            continue
        try:
            r = _M.generate(input=(seg, " ".join(chars)), data_type=("sound", "text"), disable_pbar=True)
            ts = r[0].get("timestamp") or []
        except Exception as e:   # noqa: BLE001
            print(m, u["record_id"], "failed", e, flush=True)
            continue
        rows.append({"record_id": u["record_id"],
                     "chars": [[round(u["zh_begin"] + a / 1000, 3), round(u["zh_begin"] + b / 1000, 3)] for a, b in ts]})
    tmp = dst.with_suffix(".jsonl.part")
    with open(tmp, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(tmp, dst)
    return m, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--splits", default="train,validation,test")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    split = json.load(open(D / "meetings.json"))
    ms = sorted(m for m, s in split.items() if s in args.splits.split(",")
                and (D / f"zh_mix_v8/utts/{m}.jsonl").exists() and not (Path(args.out) / f"{m}.jsonl").exists())
    print(f"{len(ms)} meetings", flush=True)
    with Pool(args.procs) as p:
        for m, n in p.imap_unordered(one, [(m, args.out) for m in ms]):
            print(m, n, flush=True)
    print("ZH_CHAR_TIMES_DONE", flush=True)


if __name__ == "__main__":
    main()
