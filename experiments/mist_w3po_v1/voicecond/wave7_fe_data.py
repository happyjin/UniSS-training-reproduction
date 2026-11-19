"""Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): the training sentences of the front-end.

5000 CoVoST 2 TRAIN sentences per direction, drawn from wave 5's already trimmed training rows
(eval_outputs/mist_w3po_v1/wave5/M/data/rows_<dir>.jsonl: held-out speakers excluded, bench_build.trim_norm, PCM16
16 kHz), ranked by sha1("20261119|w7fe|sel|<dir>|<id>"), first 5000 taken. The held-out speaker exclusion is asserted
again here. Output (refuses to overwrite): eval_outputs/mist_w3po_v1/wave7/FE/data/train_<dir>.jsonl
({id, dir, spk, ref, dur, audio}). CPU only.

    <uniss-eval python> -I experiments/mist_w3po_v1/voicecond/wave7_fe_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave1_data as W1D  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave7_fe_common import OUT, h  # noqa: E402

SRC = REPO / "eval_outputs/mist_w3po_v1/wave5/M/data"
N = 5000


def main() -> None:
    held = W1D.held_speakers()
    for d in ("zh2en", "en2zh"):
        p = OUT / "data" / f"train_{d}.jsonl"
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
        rows = [json.loads(l) for l in open(SRC / f"rows_{d}.jsonl", encoding="utf-8")]
        rows.sort(key=lambda r: h(f"20261119|w7fe|sel|{d}|{r['id']}"))
        sel = rows[:N]
        assert len(sel) == N, (d, len(sel))
        assert not {r["spk"] for r in sel} & held, "held-out speaker"
        for r in sel:
            assert (REPO / r["audio"]).exists(), r["audio"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8") as fh:
            for r in sel:
                fh.write(json.dumps({k: r[k] for k in ("id", "dir", "spk", "ref", "dur", "audio")}, ensure_ascii=False) + "\n")
        print(f"{d}: {len(sel)} of {len(rows)} rows, {len({r['spk'] for r in sel})} speakers, "
              f"{sum(r['dur'] for r in sel) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
