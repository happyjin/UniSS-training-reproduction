"""Speak a candidate selection of a si_infer --dump-cands run (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §1.2,
§2.2): per item, the chosen candidates' codes decoded as one sequence with the item's voice (BiCodec, as si_infer's
stage 3), laid on si_infer's timeline (a block plays from the later of its write time and the previous block's end),
written as a talker dir in si_infer's layout (part_<k>.jsonl with the run's own item -> shard split, wav/<id>_speech.wav)
so the speech chain (speechonly_part -> asr_transcribe -> asr_merge -> asr_hyp -> score_run) and speech_latency_lit /
speech_gap_decomp read it unchanged. GPU (BiCodec decode); refuses an existing output.

    python -m experiments.mist_w3po_v1.talker.render_selection --run <talker dir of the sampling run> \
        --sel <sel.json: id -> candidate index per block> --out <new talker dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
WAIT_READ, END_SEMANTIC = 180395, 180392


def timeline(blocks: list[dict], durs_ms: list[float]) -> list[tuple[float, float]]:
    """(start_ms, dur_ms) per block: start = max(write_ms, previous end)."""
    out, end_prev = [], 0.0
    for b, d in zip(blocks, durs_ms):
        st = max(b["write_ms"], end_prev)
        out.append((st, d))
        end_prev = st + d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "wav").mkdir(parents=True)
    sel = json.load(open(a.sel))
    from uniss import UniSSTokenizer

    device = torch.device("cuda:0")
    front = UniSSTokenizer.from_pretrained(str(REPO / "pretrained_models/UniSS"), device=device)
    run = Path(a.run)
    n_changed = n_items = 0
    for part in sorted(run.glob("part_*.jsonl")):
        k = part.stem.split("_")[1]
        cands = {r["id"]: r for r in map(json.loads, open(run / f"cands_{k}.jsonl", encoding="utf-8"))}
        rows = []
        for l in open(part, encoding="utf-8"):
            r = json.loads(l)
            c = cands.get(r["id"])
            pick = sel.get(r["id"])
            if c is None or pick is None or not r["blocks"]:
                rows.append(r)
                continue
            n_items += 1
            chosen = [c["blocks"][i]["cands"][j] for i, j in enumerate(pick)]
            n_changed += int(any(j != c["blocks"][i]["chosen"] for i, j in enumerate(pick)))
            codes = [x for ch in chosen for x in ch["codes"]]
            audio = np.zeros(0, dtype=np.float32)
            if codes:
                audio = np.asarray(front.decode(torch.tensor([*c["voice"], *codes], dtype=torch.long, device=device)),
                                   dtype=np.float32).reshape(-1)
            durs = [len(ch["codes"]) * 320 / 16 for ch in chosen]
            tl = timeline(r["blocks"], durs)
            sp = out / "wav" / f"{r['id'].replace('/', '__')}_speech.wav"
            sf.write(sp, audio if audio.size else np.zeros(320, dtype=np.float32), 16000)
            at = 0
            blocks = []
            for b, cb, ch, (st, d) in zip(r["blocks"], c["blocks"], chosen, tl):
                exp_forced = bool(ch.get("forced"))
                closing = ch["closing"] if ch.get("closing") in (WAIT_READ, END_SEMANTIC) else None
                nb = dict(b)
                nb.update(start_ms=int(st), dur_ms=int(d), codes=len(ch["codes"]), speed=ch["v"], speech_at_ms=at // 16,
                          forced=exp_forced, closed=closing is not None and not exp_forced,
                          closing="wait_read" if closing == WAIT_READ else "end")
                at += len(ch["codes"]) * 320
                blocks.append(nb)
            r = dict(r)
            r.update(blocks=blocks, codes=len(codes), speech_path=str(sp), audio_path=str(sp),
                     audio_duration_seconds=round(len(audio) / 16000, 3), error=None if codes else "no audio",
                     end_offset_ms=int(tl[-1][0] + tl[-1][1] - r["source_duration_ms"]), selection=str(a.sel))
            rows.append(r)
        with open(out / part.name, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    print(json.dumps({"out": str(out), "items": n_items, "items_changed": n_changed}))


if __name__ == "__main__":
    main()
