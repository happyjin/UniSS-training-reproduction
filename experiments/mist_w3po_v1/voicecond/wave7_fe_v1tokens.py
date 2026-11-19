"""Robust wave 7 (reports/mist_w3po_v1/robust_wave7_taskfe_prereg.zh-CN.md): the V1 (Idea 1 C) voice-token table for a
front-end arm's Talker run.

si_infer writes each item's voice prompt = the ORIGINAL item audio up to max(first write, 1 s), and the first write is
the arm's own (its blocks), so wave 3's V1 table (S1's first writes) does not match these prompts (the sha1 check of
idea1_si_infer_vo.py refuses). This script repeats wave3_v_tokens.py's V1 computation unchanged (same read / slice /
sf.write of the prompt, u = C encoder (voicecond_idea1/m2/ctrl/encoder.pt) -> FSQ ids) on the arm's first writes.
The voice method is therefore identical to V1; only the prompt length follows the arm's text timing.
Output (refuses to overwrite): wave7/FE/eval_<arm>/tokens/tokens_v1_<dir>.jsonl ({key, id, sha1, ids}) + prompts/.

    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave7_fe_v1tokens.py --arm pseudo --dir zh2en
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond.idea6_e2e_train import CKPT_C, SR  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave7_fe_common import OUT  # noqa: E402

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("pseudo", "clean"))
    ap.add_argument("--dir", required=True, choices=("zh2en", "en2zh"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    root = OUT / f"eval_{a.arm}"
    f = root / "tokens" / f"tokens_v1_{a.dir}.jsonl"
    if f.exists():
        sys.exit(f"拒绝覆盖已有输出: {f}")
    pdir = root / "tokens" / "prompts" / a.dir
    pdir.mkdir(parents=True, exist_ok=True)
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    Pc = load_encoder(BiCodecPhi(a.device), str(CKPT_C))
    Pc.model.eval()
    items = [json.loads(l) for l in open(S1 / "data" / f"items_{a.dir}.jsonl", encoding="utf-8")]
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(root / f"thinker_{a.dir}" / f"blocks_{a.dir}_w1.jsonl",
                                                                  encoding="utf-8"))}
    rows = []
    for it in items:
        if not blocks.get(it["id"]):
            continue
        x, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
        assert sr == SR
        wav = x.mean(axis=1)[0: int(it["dur_ms"]) * 16]
        h_ms = max(blocks[it["id"]][0]["write_ms"], 1000)
        p = pdir / f"{it['id'].replace('/', '__')}_voice.wav"
        sf.write(p, wav[: h_ms * 16], 16000)
        sha = hashlib.sha1(p.read_bytes()).hexdigest()
        ids = [int(v) for v in Pc.tokens(Pc.phi_wave(read_mono64(p)))]
        rows.append({"key": p.stem, "id": it["id"], "sha1": sha, "ids": ids})
    with open(f, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps({"arm": a.arm, "dir": a.dir, "n": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
