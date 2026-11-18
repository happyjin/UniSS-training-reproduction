"""Causal check for reports/mist_w3po_v1/game_muffled_speech_analysis.zh-CN.md: the same talker streams, the same item
order (si_infer's sampling seed is seed + 1000 * item index, so arms are paired stream by stream), ONLY the voice
prompt swapped. 80 streams per set (random.Random(1)):
  game_zh2en / game_en2zh   arm O = the original prompt (raw esports audio)   arm V = its HDemucs vocals stem
                            (background music / crowd / game sounds removed; game_muffle_analysis --save-vocals)
  ami_en2zh (jtest_ben B T1c, real AMI English)   arm O = the original prompt   arm N = the original prompt + a game
                            background (a random game en2zh prompt minus its vocals stem, looped), scaled to
                            background / prompt energy = --bvr-db (default -15 dB, about the game prompts' median)
-> eval_outputs/mist_w3po_v1/speech/game_swap_<set>_<arm>/talker_in/{items,blocks}.jsonl (+ prompts/ for arm N).
Refuses existing outputs. CPU only.

    python -m experiments.mist_w3po_v1.evaluation.game_prompt_swap [--n 80] [--bvr-db -15]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
VOC = REPO / "eval_outputs/mist_w3po_v1/game_muffle_v1/vocals_prompt"
RT = "joint_v8h6k_20261113T073658Z_s06000"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def write_arm(name, items, blocks):
    d = SPD / name / "talker_in"
    if d.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {d}")
    d.mkdir(parents=True)
    with open(d / "items.jsonl", "x", encoding="utf-8") as fi, open(d / "blocks.jsonl", "x", encoding="utf-8") as fb:
        for it in items:
            fi.write(json.dumps(it, ensure_ascii=False) + "\n")
            fb.write(json.dumps(blocks[it["id"]], ensure_ascii=False) + "\n")
    return d


def subset(run, n, rng):
    S = SPD / run
    items = load(S / "talker_in/items.jsonl")
    blocks = {b["id"]: b for b in load(S / "talker_in/blocks.jsonl")}
    ok = {r["id"] for r in load(S / "talker/results.jsonl") if r.get("error") is None and r.get("speech_path")}
    items = [it for it in items if it["id"] in ok]
    return sorted(rng.sample(items, min(n, len(items))), key=lambda it: it["id"]), blocks


def read16(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == 16000, p
    return x.mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--bvr-db", type=float, default=-15.0)
    a = ap.parse_args()
    rng = random.Random(1)
    for d in ("zh2en", "en2zh"):
        items, blocks = subset(f"game_b_{d}__t1c", a.n, rng)
        write_arm(f"game_swap_{d}_O", items, blocks)
        v = [dict(it, voice_audio=str(VOC / d / f"{it['id']}.wav")) for it in items]
        assert all(Path(x["voice_audio"]).exists() for x in v)
        write_arm(f"game_swap_{d}_V", v, blocks)
        print(d, len(items))
    # game backgrounds: game en2zh prompts minus their vocals stems
    bgs = []
    for it in load(SPD / "game_b_en2zh__t1c/talker_in/items.jsonl"):
        vp = VOC / "en2zh" / f"{it['id']}.wav"
        if vp.exists():
            x, y = read16(it["voice_audio"]), read16(vp)
            m = min(len(x), len(y))
            b = x[:m] - y[:m]
            if m > 16000 * 2 and float((b ** 2).mean()) > 0:
                bgs.append(b)
    items, blocks = subset(f"jtest_ben_{RT}__t1c", a.n, rng)
    write_arm("game_swap_amien_O", items, blocks)
    pd = SPD / "game_swap_amien_N" / "prompts"
    out = []
    for it in items:
        x = read16(it["voice_audio"])
        b = bgs[rng.randrange(len(bgs))]
        b = np.tile(b, int(np.ceil(len(x) / len(b))))[: len(x)]
        g = np.sqrt((x ** 2).mean() / (b ** 2).mean() * 10 ** (a.bvr_db / 10))
        y = x + g * b
        pk = float(np.abs(y).max())
        if pk > 0.99:
            y = y * (0.99 / pk)
        pd.mkdir(parents=True, exist_ok=True)
        p = pd / f"{it['id']}.wav"
        sf.write(str(p), y.astype(np.float32), 16000)
        out.append(dict(it, voice_audio=str(p)))
    d = SPD / "game_swap_amien_N" / "talker_in"
    if d.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {d}")
    d.mkdir(parents=True)
    with open(d / "items.jsonl", "x", encoding="utf-8") as fi, open(d / "blocks.jsonl", "x", encoding="utf-8") as fb:
        for it in out:
            fi.write(json.dumps(it, ensure_ascii=False) + "\n")
            fb.write(json.dumps(blocks[it["id"]], ensure_ascii=False) + "\n")
    print("amien", len(out), "backgrounds", len(bgs))


if __name__ == "__main__":
    main()
