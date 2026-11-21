"""AbS-C precheck (reports/mist_w3po_v1/robust_abs_c_prereg.zh-CN.md section 3), venvs/nemo_diar:
  * merges <root>/search/search_<d>_<k>.jsonl into <root>/tokens/tokens_absc_<d>.jsonl (wave3 V1 table order and format
    {key, id, sha1, ids}; every V1 table row exactly once);
  * noisy rows: ECAPA2 (judge) and TitaNet (record only) cosine between the resynthesis R_noisy(s) and the CLEAN twin
    prompt, s in v1 / absc (<root>/search/resynth) and abs (b_abs/search/resynth, record only);
  * stop rule: go iff mean(ECAPA2 absc) - mean(ECAPA2 v1) >= 0.02 in BOTH directions;
  * records SQUIM STOI / PESQ of the chosen resynthesis (absc vs abs vs v1), rejected-candidate fraction, accepted
    rounds, Hamming to AbS / oracle, d-vector cosines, latency.
    PYTHONPATH=. venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.abs_c_precheck --root <abs_c root>
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

ARMS = ("v1", "absc", "abs")
GATE = 0.02
BA_RES = REPO / "eval_outputs/mist_w3po_v1/b_abs/search/resynth"


def q(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dirs", nargs="+", default=list(C.DIRS))
    a = ap.parse_args()
    R = Path(a.root)
    outs = [R / "precheck.json"] + [R / f"precheck_rows_{d}.jsonl" for d in C.DIRS] + \
           [R / "tokens" / f"tokens_absc_{d}.jsonl" for d in C.DIRS]
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

    def wav(d, key, k):
        return (BA_RES if k == "abs" else R / "search" / "resynth") / d / f"{key}__{k}.wav"

    res = {"gate": f"mean ECAPA2(absc) - mean ECAPA2(v1) >= {GATE} in both directions", "dirs": {}}
    go = True
    for d in a.dirs:
        rows = C.dev_rows(d)
        S = {}
        for f in sorted(glob.glob(str(R / "search" / f"search_{d}_*.jsonl"))):
            for r in map(json.loads, open(f, encoding="utf-8")):
                assert r["id"] not in S, ("duplicate", r["id"])
                S[r["id"]] = r
        missing = [r["id"] for r in rows if r["id"] not in S]
        if missing or len(S) != len(rows):
            sys.exit(f"search rows incomplete for {d}: missing {len(missing)}, have {len(S)} of {len(rows)}")
        with open(R / "tokens" / f"tokens_absc_{d}.jsonl", "x", encoding="utf-8") as fh:
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
                    e, t = embs(wav(d, r["key"], k))
                    row[f"ecapa2_{k}"] = float(e @ e_ref)
                    row[f"titanet_{k}"] = float(t @ t_ref)
                    E[k].append(row[f"ecapa2_{k}"])
                    T[k].append(row[f"titanet_{k}"])
                fh.write(json.dumps(row) + "\n")
        ss = [S[r["id"]] for r in noisy]
        allr = list(S.values())
        qual = {}
        for nm, grp in (("noisy", ss), ("all", allr)):
            for m in ("stoi", "pesq"):
                qual[f"{m}_{nm}_mean"] = {k: float(np.mean([s[f"{m}_{kk}"] for s in grp]))
                                         for k, kk in (("absc", "best"), ("abs", "abs"), ("v1", "v1"))}
            qual[f"frac_abs_stoi_below_v1_{nm}"] = float(np.mean([s["stoi_abs"] < s["stoi_v1"] for s in grp]))
        dd = {
            "n_noisy": len(noisy), "n_all": len(allr),
            "ecapa2_resynth_vs_twin_mean": {k: float(np.mean(E[k])) for k in ARMS},
            "ecapa2_resynth_vs_twin_median": {k: float(np.median(E[k])) for k in ARMS},
            "titanet_resynth_vs_twin_mean": {k: float(np.mean(T[k])) for k in ARMS},
            "ecapa2_absc_minus_v1_mean": float(np.mean(E["absc"]) - np.mean(E["v1"])),
            "ecapa2_absc_minus_abs_mean": float(np.mean(E["absc"]) - np.mean(E["abs"])),
            "ecapa2_absc_minus_v1_paired_median": float(np.median(np.asarray(E["absc"]) - np.asarray(E["v1"]))),
            "frac_rows_ecapa2_absc_gt_v1": float(np.mean(np.asarray(E["absc"]) > np.asarray(E["v1"]))),
            "squim": qual,
            "rejected_frac_all": float(np.mean([s["n_rejected"] / s["n_eval"] for s in allr])),
            "dcos_vs_oracle_median": {k: float(np.median([s[f"dcos_{k}"] for s in ss])) for k in ("absc", "v1")},
            "hamming_absc_vs_abs_mean_all": float(np.mean([s["ham_absc_abs"] for s in allr])),
            "hamming_absc_vs_oracle_mean": float(np.mean([s["ham_absc_oracle"] for s in ss])),
            "frac_rows_equal_abs_all": float(np.mean([s["ids"] == s["ids_abs"] for s in allr])),
            "utility_gain_mean_noisy": float(np.mean([s["u_best"] - s["u_start"] for s in ss])),
            "utility_gain_mean_all": float(np.mean([s["u_best"] - s["u_start"] for s in allr])),
            "n_accept_mean_all": float(np.mean([s["n_accept"] for s in allr])),
            "n_flips_mean_all": float(np.mean([len(s["flips"]) for s in allr])),
            "frac_rows_changed_all": float(np.mean([s["ids"] != s["ids_v1"] for s in allr])),
            "latency_s": {k: {"median": q([s[k] for s in allr], 50), "p90": q([s[k] for s in allr], 90),
                              "mean": float(np.mean([s[k] for s in allr]))} for k in ("t_search", "t_tok", "t_v1")},
        }
        dd["go"] = dd["ecapa2_absc_minus_v1_mean"] >= GATE
        go &= dd["go"]
        res["dirs"][d] = dd
        print(json.dumps({d: {k: dd[k] for k in ("ecapa2_resynth_vs_twin_mean", "ecapa2_absc_minus_v1_mean",
                                                 "squim", "latency_s", "go")}}), flush=True)
    res["go"] = bool(go)
    with open(R / "precheck.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"go": res["go"]}))


if __name__ == "__main__":
    main()
