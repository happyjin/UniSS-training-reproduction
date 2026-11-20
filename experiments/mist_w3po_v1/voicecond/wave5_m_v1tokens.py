"""Robust wave 5, full-system combination (reports/mist_w3po_v1/robust_wave5_prereg.zh-CN.md, "完整系统组合"): the V1
(Idea 1 C) voice-token table for a Talker run on another Thinker's blocks. Same computation as wave7_fe_v1tokens.py /
wave3_v_tokens.py V1 (si_infer's voice prompt = original item audio up to max(first write, 1 s) of THESE blocks;
u = C encoder -> FSQ ids), only the blocks file and the output root are arguments. Refuses to overwrite.

    CUDA_VISIBLE_DEVICES=<card> python -I experiments/mist_w3po_v1/voicecond/wave5_m_v1tokens.py --dir zh2en \
        --blocks <eval>/thinker_zh2en/blocks_zh2en_w1.jsonl --out-root eval_outputs/mist_w3po_v1/wave5/M/combo
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

S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, choices=("zh2en", "en2zh"))
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    root = Path(a.out_root)
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
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(a.blocks, encoding="utf-8"))}
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
    print(json.dumps({"dir": a.dir, "n": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
