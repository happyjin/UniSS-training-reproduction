"""voice_onset_v1: apply a player (none / dip = cut move only, release at once / r0 / s2) to an si_infer talker run (reports/voice_onset_v1/S0_S1_S2_prereg.zh-CN.md).
Writes one row per item {id, D1, D2, NP_in, PT_in, PT_all, NC, n_pause, EndOffset, n_seg} (ms for D1 / D2) and, for the ids
in --wav-ids, the listening wav <wav-dir>/<id>.wav. CPU, multiprocessing.
    qwen3omni PYTHONPATH=. python -m experiments.voice_onset_v1.s0_apply --run <talker_dir> --dir en2zh --policy r0 --out <jsonl>
        [--model <s2 model.pt>] [--wav-ids a,b] [--wav-dir <dir>]
Refuses an existing --out.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

LANG = {"en2zh": "cmn", "zh2en": "eng"}
_G = {}


def _init(policy, model, wav_ids, wav_dir):
    from experiments.voice_onset_v1 import playout as PO
    _G.update(PO=PO, policy=policy, wav_ids=wav_ids, wav_dir=wav_dir)
    if policy == "s2":
        import torch
        _G["s2"] = PO.S2Policy(PO.mlp_numpy(torch.load(model, map_location="cpu", weights_only=False)["state"]))


def _one(args):
    r, lang = args
    PO = _G["PO"]
    it = PO.Item(r, lang)
    pol = {"none": None, "dip": None, "r0": PO.r0, "s2": _G.get("s2")}[_G["policy"]]
    segs = PO.simulate(it, pol, dip=_G["policy"] != "none")
    m = PO.metrics(it, segs)
    if r["id"] in _G["wav_ids"]:
        PO.render(it, segs, Path(_G["wav_dir"]) / f"{r['id']}.wav")
    return {"id": r["id"], **(m or {"D1": None, "D2": None})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dir", required=True, choices=list(LANG))
    ap.add_argument("--policy", required=True, choices=("none", "dip", "r0", "s2"))
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav-ids", default="")
    ap.add_argument("--wav-dir", default="")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = set(x for x in a.wav_ids.split(",") if x)
    if ids:
        Path(a.wav_dir).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(Path(a.run) / "results.jsonl", encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    with mp.get_context("fork").Pool(a.procs, _init, (a.policy, a.model, ids, a.wav_dir)) as pool, \
            open(out, "x", encoding="utf-8") as fh:
        for row in pool.imap(_one, [(r, LANG[a.dir]) for r in rows], chunksize=16):
            fh.write(json.dumps(row) + "\n")
    print(out, len(rows))


if __name__ == "__main__":
    main()
