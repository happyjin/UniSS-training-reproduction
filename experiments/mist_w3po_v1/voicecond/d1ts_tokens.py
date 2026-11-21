"""Robust D1 Talker-sensitivity tokens (reports/mist_w3po_v1/robust_d1_talker_sensitivity_prereg.zh-CN.md section 2).
For every CLEAN row of the wave3 V1 table (key = <id>_voice, sha1 of the prompt si_infer writes) three override tables:
  C0  : encode_wav_to_tokens(prompts/<dir>/<key>.wav)                (B's own computation, sha1 checked)
  C10 : encode of source [10 ms, 10 ms + L)  (L = max(first write, 1 s); cut at the item end if too short)
  C45 : encode of source [0, L) + white noise 45 dB below active-speech power, rng seed crc32("wn45|<id>") (= D0's wn45 view)
Each perturbed view is written once as a PCM_16 wav (like si_infer's prompt file) and encoded from that file. Also writes the
clean-only items file per direction and stats: token match / all-32 match / d-vector cos of C10, C45 vs C0, C0 vs the oracle
table's clean rows, number of truncated C10 views.
    vllm_omni_030 python -I experiments/mist_w3po_v1/voicecond/d1ts_tokens.py --out <dir> [--device cuda:0] [--limit N]
Writes <out>/tokens_<arm>_<dir>.jsonl, <out>/items_clean_<dir>.jsonl, <out>/prompts/<arm>/<dir>/<key>.wav,
<out>/tokens_stats.json; refuses existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
VT = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V/tokens"
OT = REPO / "eval_outputs/mist_w3po_v1/oracle_voice/tokens"
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
SR = 16000
ARMS = ("C0", "C10", "C45")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def item_wav(it):
    x, sr = sf.read(it["audio"] if Path(it["audio"]).is_absolute() else str(REPO / it["audio"]), dtype="float32",
                    always_2d=True)
    assert sr == SR, it["audio"]
    return x.mean(axis=1)[0: int(it["dur_ms"]) * 16]


def speech_power(w):   # = d0_token_stability.speech_power
    f = w[: len(w) // 320 * 320].reshape(-1, 320).astype(np.float64)
    e = (f ** 2).mean(1)
    act = e[e >= e.max() * 1e-3]
    return float(act.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N clean rows per direction")
    a = ap.parse_args()
    out = Path(a.out)
    dirs = ("zh2en", "en2zh")
    files = [out / f"tokens_{arm}_{d}.jsonl" for arm in ARMS for d in dirs] + \
            [out / f"items_clean_{d}.jsonl" for d in dirs] + [out / "tokens_stats.json"]
    for f in files:
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    phi = BiCodecPhi(a.device)
    tok = phi.tok

    def enc(p):
        return [int(v) for v in tok.encode_wav_to_tokens(str(p)).flatten().tolist()[:32]]

    import torch

    @torch.no_grad()
    def dvec(ids):   # (1,1,32) as BiCodecTokenizer.detokenize / d0_token_stability
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=phi.device)[None, None, :]
        v = phi.se.detokenize(t).float().cpu().numpy().astype(np.float64).reshape(-1)
        return v / np.linalg.norm(v)

    stats = {}
    for d in dirs:
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        items = jl(S1 / "data" / f"items_{d}.jsonl")
        byid = {it["id"]: it for it in items}
        first = {r["id"]: r["blocks"][0]["write_ms"] for r in jl(S1 / f"thinker_{d}" / f"blocks_{d}_w1.jsonl") if r.get("blocks")}
        orc = {r["key"]: r["ids"] for r in jl(OT / f"tokens_oracle_{d}.jsonl")}
        src = [r for r in jl(VT / f"tokens_v1_{d}.jsonl") if meta[r["id"]]["cond"] == "clean"]
        src = src[: a.limit] if a.limit else src
        keep = {r["id"] for r in src}
        pd = {arm: out / "prompts" / arm / d for arm in ("C10", "C45")}
        for p in pd.values():
            p.mkdir(parents=True, exist_ok=True)
        rows = {arm: [] for arm in ARMS}
        tm = {"C10": [], "C45": []}
        cos = {"C10": [], "C45": []}
        c0_vs_oracle, trunc = [], 0
        for n, r in enumerate(src, 1):
            p = VT / "prompts" / d / f"{r['key']}.wav"
            assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
            ids = {"C0": enc(p)}
            L = max(first[r["id"]], 1000) * 16
            w = item_wav(byid[r["id"]])
            w10 = w[160: 160 + L]
            trunc += len(w10) < min(L, len(w))
            w45 = w[:L]
            w45 = w45 + np.random.default_rng(zlib.crc32(f"wn45|{r['id']}".encode())).standard_normal(len(w45)).astype(
                np.float32) * np.float32(np.sqrt(speech_power(w45) * 10 ** -4.5))
            for arm, ww in (("C10", w10), ("C45", w45)):
                q = pd[arm] / f"{r['key']}.wav"
                if q.exists():
                    sys.exit(f"拒绝覆盖已有输出: {q}")
                sf.write(q, np.asarray(ww, dtype=np.float32), SR)   # default subtype PCM_16, as si_infer's prompt file
                ids[arm] = enc(q)
            v0 = dvec(ids["C0"])
            for arm in ("C10", "C45"):
                tm[arm].append(float(np.mean([x == z for x, z in zip(ids[arm], ids["C0"])])))
                cos[arm].append(float(v0 @ dvec(ids[arm])))
            c0_vs_oracle.append(float(np.mean([x == z for x, z in zip(ids["C0"], orc[r["key"]])])))
            for arm in ARMS:
                rows[arm].append({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": ids[arm]})
            if n % 200 == 0:
                print(f"[d1ts-tokens] {d} {n}/{len(src)}", flush=True)
        for arm in ARMS:
            with open(out / f"tokens_{arm}_{d}.jsonl", "x", encoding="utf-8") as fh:
                fh.writelines(json.dumps(x) + "\n" for x in rows[arm])
        with open(out / f"items_clean_{d}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items if it["id"] in keep)
        st = {"n": len(src), "c10_truncated": trunc,
              "c0_vs_oracle_clean_token_match": float(np.mean(c0_vs_oracle)),
              "c0_vs_oracle_clean_all32": float(np.mean([x == 1.0 for x in c0_vs_oracle]))}
        for arm in ("C10", "C45"):
            st[arm] = {"token_match_vs_C0": float(np.mean(tm[arm])), "all32_vs_C0": float(np.mean([x == 1.0 for x in tm[arm]])),
                       "dvec_cos_vs_C0_mean": float(np.mean(cos[arm])), "dvec_cos_vs_C0_median": float(np.median(cos[arm]))}
        stats[d] = st
        print(json.dumps({d: st}), flush=True)
    with open(out / "tokens_stats.json", "x", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
