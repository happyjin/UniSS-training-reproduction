"""Premature commitments on RealSI text, per direction, policy and read window, against the same Thinker's own offline
translation (realsi_off_s0 for the adapted Thinker, realsi_thinker_offline_check/base for the frozen one): a sentence has
a premature error when its committed units first leave the offline translation (in units: characters for Chinese,
lower-cased words for English) inside a block written before the final read. Also: the share of units written before
the final read (how aggressive the policy is), where the first error falls (first block or later) and how often a
sentence with an error is a question in the reference. Read-only; prints a table and writes the json given.

    PYTHONPATH=. python3 reports/uniss_si_v4/commit_errors.py out.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "reports/uniss_si_v4")
sys.path.insert(0, "paper/bayes_simul_s2st_v4/scripts")
from endpoints import ROOT, realsi_refs  # noqa: E402
from realsi_text import POLICIES  # noqa: E402
from zh2en_diag import files  # noqa: E402

O = ROOT / "eval_outputs/uniss_qwen3_omni_v1"
OFF = {"frozen": O / "realsi_thinker_offline_check/base", "adapted": O / "si_v4/realsi_off_s0"}
THINKER = {"LA2": "frozen", "LA2-adapted": "adapted", "A0": "adapted"}


def units(t, d):
    return re.findall(r"[一-鿿]|[A-Za-z0-9]+", t) if d == "en2zh" else [w.lower() for w in re.findall(r"[A-Za-z0-9']+", t)]


def offline(m, d):
    f = sorted(Path(OFF[m]).glob(f"blocks_{d}_w60*.jsonl"))[0]
    return {r["id"]: units(r["translation"], d) for r in map(json.loads, open(f, encoding="utf-8"))}


def main():
    res = {}
    for d in ("en2zh", "zh2en"):
        refs = realsi_refs(d)
        offs = {m: offline(m, d) for m in OFF}
        for name in THINKER:
            off = offs[THINKER[name]]
            for w, f in files(POLICIES[name], d).items():
                n = early = err = err_first = err_q = n_q = 0
                for r in map(json.loads, open(f, encoding="utf-8")):
                    if r["id"] not in refs or r["id"] not in off:
                        continue
                    n += 1
                    q = refs[r["id"]][0].rstrip().endswith("?")
                    n_q += q
                    o, pos, bad = off[r["id"]], 0, None
                    for bi, b in enumerate(r["blocks"]):
                        u = units(b["text"], d)
                        if not b.get("final"):
                            early += len(u)
                        for x in u:
                            if bad is None and (pos >= len(o) or o[pos] != x):
                                bad = (bi, b.get("final", False))
                            pos += 1
                    if bad is not None and not bad[1]:
                        err += 1
                        err_first += bad[0] == 0
                        err_q += q
                total = sum(len(units(x["translation"], d)) for x in map(json.loads, open(f, encoding="utf-8"))
                            if x["id"] in refs and x["id"] in off)
                res.setdefault(d, {}).setdefault(name, {})[f"{w:g}"] = {
                    "early_share": round(early / max(total, 1), 3), "premature": round(err / n, 3),
                    "first_block_of_errors": round(err_first / max(err, 1), 3),
                    "question_of_errors": round(err_q / max(err, 1), 3), "question_base": round(n_q / n, 3), "n": n}
    with open(sys.argv[1], "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for d in res:
        print(f"== {d}")
        print("policy        w    early  premature  1st-block  question(err/base)")
        for name in THINKER:
            for w, x in res[d][name].items():
                if float(w) <= 2.5:
                    print(f"{name:12s} {w:>4s}  {x['early_share']:.3f}  {x['premature']:.3f}      {x['first_block_of_errors']:.3f}"
                          f"      {x['question_of_errors']:.3f}/{x['question_base']:.3f}")


if __name__ == "__main__":
    main()
