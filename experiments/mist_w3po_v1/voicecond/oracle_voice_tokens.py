"""Robust oracle-voice diagnostic (reports/mist_w3po_v1/robust_oracle_voice_prereg.zh-CN.md section 3): a perfect denoiser of
the 32 BiCodec global tokens. For every row of the wave3 V1 table (wave3_combos/V/tokens/tokens_v1_<dir>.jsonl: key = prompt
stem, id, sha1 of the prompt si_infer writes) the ids are replaced by
  clean rows : encode_wav_to_tokens(prompts/<dir>/<key>.wav)  (B's own computation, sha1 checked)
  noisy rows : encode_wav_to_tokens(clean twin prompt), the clean twin = same set + same meta `pair`, cond clean; its audio is
               read / sliced / written exactly as si_infer does, over the NOISY item's prompt span max(first write, 1 s)
               (first write from the noisy item's own Thinker blocks).
key and sha1 stay those of the row (the hook checks the noisy prompt file at run time). Also reports per direction: waveform
correlation noisy prompt vs clean-twin prompt (alignment), Hamming(oracle, B noisy), how many noisy rows have a different
prompt span than their twin, and that oracle == twin's own ids whenever the spans agree.
    vllm_omni_030 python -I experiments/mist_w3po_v1/voicecond/oracle_voice_tokens.py --out <dir> [--device cuda:0] [--limit N]
Writes <out>/tokens_oracle_<dir>.jsonl, <out>/twin_prompts/<dir>/<key>.wav, <out>/tokens_oracle_<dir>.stats.json; refuses
existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
VT = REPO / "eval_outputs/mist_w3po_v1/wave3_combos/V/tokens"
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def item_wav(it):
    x, sr = sf.read(it["audio"] if Path(it["audio"]).is_absolute() else str(REPO / it["audio"]), dtype="float32",
                    always_2d=True)
    assert sr == 16000, it["audio"]
    return x.mean(axis=1)[0: int(it["dur_ms"]) * 16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows per direction")
    a = ap.parse_args()
    out = Path(a.out)
    dirs = ("zh2en", "en2zh")
    files = {d: out / f"tokens_oracle_{d}.jsonl" for d in dirs}
    for f in files.values():
        if f.exists() or f.with_suffix(".stats.json").exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    tok = BiCodecPhi(a.device).tok

    def enc(p):
        return [int(v) for v in tok.encode_wav_to_tokens(str(p)).flatten().tolist()[:32]]

    for d in dirs:
        pdir = out / "twin_prompts" / d
        pdir.mkdir(parents=True, exist_ok=True)
        meta = {m["id"]: m for m in jl(S1 / "data" / f"meta_{d}.jsonl")}
        items = {it["id"]: it for it in jl(S1 / "data" / f"items_{d}.jsonl")}
        first = {r["id"]: r["blocks"][0]["write_ms"] for r in jl(S1 / f"thinker_{d}" / f"blocks_{d}_w1.jsonl") if r.get("blocks")}
        twin = {(m["set"], m["pair"]): i for i, m in meta.items() if m["cond"] == "clean"}
        src = jl(VT / f"tokens_v1_{d}.jsonl")
        src = src[: a.limit] if a.limit else src
        rows, corr, ham, span_diff, same_span_mism, n_noisy = [], [], [], 0, 0, 0
        own = {}
        for n, r in enumerate(src, 1):
            p = VT / "prompts" / d / f"{r['key']}.wav"
            assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
            m = meta[r["id"]]
            if m["cond"] == "clean":
                ids = own.get(r["id"]) or enc(p)
                own[r["id"]] = ids
            else:
                n_noisy += 1
                tid = twin[(m["set"], m["pair"])]
                h_ms = max(first[r["id"]], 1000)
                wt = item_wav(items[tid])
                wn = item_wav(items[r["id"]])
                assert len(wt) == len(wn), (r["id"], tid, len(wt), len(wn))
                q = pdir / f"{r['key']}.wav"
                if q.exists():
                    sys.exit(f"拒绝覆盖已有输出: {q}")
                sf.write(q, wt[: h_ms * 16], 16000)
                ids = enc(q)
                xn, xt = wn[: h_ms * 16].astype(np.float64), wt[: h_ms * 16].astype(np.float64)
                corr.append(float(np.corrcoef(xn, xt)[0, 1]))
                ham.append(sum(x != z for x, z in zip(ids, enc(p))))
                th = max(first[tid], 1000)
                if th != h_ms:
                    span_diff += 1
                else:
                    tp = VT / "prompts" / d / f"{tid}_voice.wav"
                    if tid not in own:
                        own[tid] = enc(tp)
                    same_span_mism += ids != own[tid]
            rows.append({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": ids})
            if n % 500 == 0:
                print(f"[oracle-tokens] {d} {n}/{len(src)}", flush=True)
        st = {"dir": d, "n": len(rows), "n_noisy": n_noisy, "corr_noisy_twin_median": float(np.median(corr)),
              "corr_noisy_twin_min": float(np.min(corr)), "ham_oracle_vs_B_noisy_mean": float(np.mean(ham)),
              "noisy_rows_span_differs_from_twin": span_diff, "same_span_oracle_ne_twin_ids": same_span_mism}
        with open(files[d], "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        with open(files[d].with_suffix(".stats.json"), "x", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1)
        print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
