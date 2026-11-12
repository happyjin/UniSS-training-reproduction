"""Chunk-protocol windows of the En->Zh C2 windows (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1.zh-CN.md §5.3; the
user's latency standard, latency_chunk_protocol_table_v1): chunk_windows_set.py's specclose.cmd_build with --judge off
on the en2zh meetings (data/ami_multispk_si_v1, posteriors nemotron_lowlat, session audio ihm_mix, whose files are
<m>.Mix-Headset.wav: specclose.load_meeting is replaced by the same loader on that file name). Windows, audio and
posteriors are the C2 ones (cmd_build asserts row by row); only open_detect_ms / close_detect_ms (+ final_post_cut_ms,
close_kind) change to the chunk times. Additive: specclose.py / chunk_windows_set.py are unchanged. CPU only.
--dev5 <file>: also the rows of the 5 precheck meetings (dev only).

    python -m experiments.mist_w3po_v1.data.chunk_windows_en2zh --base-win <en2zh C2 windows> --out <new dir> [--dev5 <file>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.data import specclose as S
from experiments.mist_w3po_v1.data.build_online_windows_en2zh import DEV5


def load_meeting_ihm(m: str):
    import soundfile as sf
    import torch
    post = torch.load(S.DATA / "diar" / S.DIAR / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
    x, sr = sf.read(str(S.DATA / "session_audio" / S.MIX / f"{m}.Mix-Headset.wav"), dtype="float32", always_2d=True)
    return post, x.mean(1), sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-win", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev5", default="")
    a = ap.parse_args()
    if a.dev5 and Path(a.dev5).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.dev5}")
    S.DATA, S.DIAR, S.MIX = Path("data/ami_multispk_si_v1"), "nemotron_lowlat", "ihm_mix"
    S.load_meeting = load_meeting_ihm
    S.cmd_build(argparse.Namespace(judge="off", base_win=a.base_win, out=a.out))
    if a.dev5:
        rows = [l for l in open(Path(a.out) / "windows_dev_online.jsonl", encoding="utf-8")
                if json.loads(l)["meeting"] in DEV5]
        Path(a.dev5).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dev5, "x", encoding="utf-8") as fh:
            fh.writelines(rows)
        print(f"dev5 chunk windows: {len(rows)} -> {a.dev5}")


if __name__ == "__main__":
    main()
