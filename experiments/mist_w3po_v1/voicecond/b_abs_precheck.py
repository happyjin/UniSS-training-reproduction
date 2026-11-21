"""Robust B precheck (reports/mist_w3po_v1/robust_b_abs_prereg.zh-CN.md section 3), venvs/nemo_diar:
  * merges <root>/search/search_<d>_<k>.jsonl into <root>/tokens/tokens_abs_<d>.jsonl (wave3 V1 table order and format
    {key, id, sha1, ids}; every V1 table row must be present exactly once);
  * noisy rows: ECAPA2 (judge) and TitaNet (record only) cosine between each resynthesis R_noisy(s)
    (s in v1 / abs / oracle / B; search/resynth/<d>/<key>__<s>.wav) and the CLEAN twin prompt
    (oracle_voice/tokens/twin_prompts/<d>/<key>.wav);
  * stop rule: go iff mean(ECAPA2 abs) - mean(ECAPA2 v1) >= 0.02 in BOTH directions;
  * records d-vector cosines vs oracle tokens, search utility gain, accepted rounds, Hamming, latency.
    PYTHONPATH=. venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.b_abs_precheck --root <b_abs root>
        [--device cuda:0]
Writes <root>/precheck.json, <root>/precheck_rows_<d>.jsonl, tokens; refuses existing files.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import a_sv_common as C  # noqa: E402
from experiments.mist_w3po_v1.voicecond.a_sv_ecapa2_spksim import ECAPA2  # noqa: E402

ARMS = ("v1", "abs", "oracle", "B")
GATE = 0.02


def q(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true", help="code check only: restrict to the rows that were searched")
    ap.add_argument("--dirs", nargs="+", default=list(C.DIRS))
    a = ap.parse_args()
    R = Path(a.root)
    outs = [R / "precheck.json"] + [R / f"precheck_rows_{d}.jsonl" for d in C.DIRS] + \
           [R / "tokens" / f"tokens_abs_{d}.jsonl" for d in C.DIRS]
    for o in outs:
        if o.exists():
            sys.exit(f"拒绝覆盖已有输出: {o}")
    (R / "tokens").mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from experiments.mist_w3po_v1.voicecond.idea1_spksim import NEMO
    ec = torch.jit.load(ECAPA2, map_location=a.device).eval()
    tn = EncDecSpeakerLabelModel.restore_from(NEMO, map_location=a.device).eval()

    @torch.no_grad()
    def embs(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        assert sr == 16000, p
        t = torch.from_numpy(x.mean(1))[None].to(a.device)
        e1 = ec(t)[0].float().cpu().numpy()
        _, e2 = tn.forward(input_signal=t, input_signal_length=torch.tensor([t.shape[1]], device=a.device))
        e2 = e2[0].float().cpu().numpy()
        return e1 / (np.linalg.norm(e1) + 1e-9), e2 / (np.linalg.norm(e2) + 1e-9)

    res = {"gate": f"mean ECAPA2(abs) - mean ECAPA2(v1) >= {GATE} in both directions", "dirs": {}}
    go = True
    for d in a.dirs:
        rows = C.dev_rows(d)
        S = {}
        for f in sorted(glob.glob(str(R / "search" / f"search_{d}_*.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                assert r["id"] not in S, ("duplicate", r["id"])
                S[r["id"]] = r
        if a.smoke:
            rows = [r for r in rows if r["id"] in S]
        missing = [r["id"] for r in rows if r["id"] not in S]
        if missing or len(S) != len(rows):
            sys.exit(f"search rows incomplete for {d}: missing {len(missing)}, have {len(S)} of {len(rows)}")
        with open(R / "tokens" / f"tokens_abs_{d}.jsonl", "x", encoding="utf-8") as fh:
            for r in rows:
                s = S[r["id"]]
                assert s["sha1"] == r["sha1"] and s["key"] == r["key"] and s["ids_v1"] == r["ids"], r["id"]
                fh.write(json.dumps({"key": r["key"], "id": r["id"], "sha1": r["sha1"], "ids": s["ids"]}) + "\n")
        noisy = [r for r in rows if "_noisy__" in r["id"]]
        E = {k: [] for k in ARMS}
        T = {k: [] for k in ARMS}
        with open(R / f"precheck_rows_{d}.jsonl", "x", encoding="utf-8") as fh:
            for r in noisy:
                e_ref, t_ref = embs(C.twin_prompt(d, r["key"]))
                row = {"id": r["id"]}
                for k in ARMS:
                    e, t = embs(R / "search" / "resynth" / d / f"{r['key']}__{k}.wav")
                    row[f"ecapa2_{k}"] = float(e @ e_ref)
                    row[f"titanet_{k}"] = float(t @ t_ref)
                    E[k].append(row[f"ecapa2_{k}"])
                    T[k].append(row[f"titanet_{k}"])
                fh.write(json.dumps(row) + "\n")
        ss = [S[r["id"]] for r in noisy]
        allr = list(S.values())
        dd = {
            "n_noisy": len(noisy), "n_all": len(allr),
            "ecapa2_resynth_vs_twin_mean": {k: float(np.mean(E[k])) for k in ARMS},
            "ecapa2_resynth_vs_twin_median": {k: float(np.median(E[k])) for k in ARMS},
            "titanet_resynth_vs_twin_mean": {k: float(np.mean(T[k])) for k in ARMS},
            "titanet_resynth_vs_twin_median": {k: float(np.median(T[k])) for k in ARMS},
            "ecapa2_abs_minus_v1_mean": float(np.mean(E["abs"]) - np.mean(E["v1"])),
            "ecapa2_abs_minus_v1_paired_median": float(np.median(np.asarray(E["abs"]) - np.asarray(E["v1"]))),
            "frac_rows_ecapa2_abs_gt_v1": float(np.mean(np.asarray(E["abs"]) > np.asarray(E["v1"]))),
            "dcos_vs_oracle_median": {k: float(np.median([s[f"dcos_{k}"] for s in ss])) for k in ("abs", "v1", "B")},
            "dcos_vs_oracle_mean": {k: float(np.mean([s[f"dcos_{k}"] for s in ss])) for k in ("abs", "v1", "B")},
            "hamming_vs_oracle_mean": {k: float(np.mean([s[f"ham_{k}"] for s in ss])) for k in ("abs", "v1")},
            "utility_gain_mean_noisy": float(np.mean([s["u_best"] - s["u_start"] for s in ss])),
            "utility_gain_mean_all": float(np.mean([s["u_best"] - s["u_start"] for s in allr])),
            "utility_start_mean_noisy": float(np.mean([s["u_start"] for s in ss])),
            "n_accept_mean_all": float(np.mean([s["n_accept"] for s in allr])),
            "n_flips_mean_all": float(np.mean([len(s["flips"]) for s in allr])),
            "frac_rows_changed_all": float(np.mean([s["ids"] != s["ids_v1"] for s in allr])),
            "latency_s": {k: {"median": q([s[k] for s in allr], 50), "p90": q([s[k] for s in allr], 90),
                              "mean": float(np.mean([s[k] for s in allr]))} for k in ("t_search", "t_tok", "t_v1")},
        }
        dd["go"] = dd["ecapa2_abs_minus_v1_mean"] >= GATE
        go &= dd["go"]
        res["dirs"][d] = dd
        print(json.dumps({d: {k: dd[k] for k in ("ecapa2_resynth_vs_twin_mean", "ecapa2_abs_minus_v1_mean",
                                                 "dcos_vs_oracle_median", "latency_s", "go")}}), flush=True)
    res["go"] = bool(go)
    with open(R / "precheck.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"go": res["go"]}))


if __name__ == "__main__":
    main()
