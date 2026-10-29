"""D1 selection on dev only (prereg_dirsel.zh-CN.md): per direction and candidate (posterior, b0), E1 and E2 against LA2
on the UniST dev and on the spontaneous dev (endpoints.py, B = 200); the score is the mean E1 of the two. The default is
(s0, b0 = 1); another candidate replaces it only if its score is >= 0.3 BLEU higher. Writes the json given.

    PYTHONPATH=. python3 reports/uniss_si_v4/select_d1.py reports/uniss_si_v4/diag/d1_selection.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "paper/bayes_simul_s2st_v4/scripts")
from endpoints import ROOT, endpoints, items_refs, text_run  # noqa: E402

V4 = ROOT / "eval_outputs/uniss_qwen3_omni_v1/si_v4"
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
DEVS = {"un": ("data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl", "dev_{m}", "dev_la2"),
        "sp": ("data/si_spont_v1/dev_{d}_ref.jsonl", "devsp_{m}", "devsp_la2")}
MARGIN = 0.3


def family(folder, d, tag, refs, lang):
    fs = sorted(Path(folder).glob(f"blocks_{d}_w*{tag}.jsonl"), key=lambda f: float(f.name.split("_w")[1].split("_")[0].removesuffix(".jsonl")))
    if not tag:
        fs = [f for f in fs if "_sw" not in f.name]
    return [[text_run(f, refs, lang)] for f in fs]


def main():
    out = {}
    for d in ("zh2en", "en2zh"):
        lang = "cmn" if d == "en2zh" else "eng"
        res = {}
        for dev, (items, folder, la2dir) in DEVS.items():
            refs = items_refs(str(ROOT / items.format(d=d)))
            la2 = family(V4 / la2dir, d, "", refs, lang)
            for m in ("s0", "bma3", "a2"):
                for b in (0, 1, 2, 3):
                    ours = family(V4 / folder.format(m=m), d, f"_sw{m}b{b}q{RATE[d]}", refs, lang)
                    if len(ours) < 8:
                        continue
                    r = endpoints({"LA2": la2, "x": ours}, "x", "LA2", B=200)
                    res.setdefault(f"{m}_b{b}", {})[dev] = {k: r[k] for k in ("E1", "E1_ci", "E2", "E2_ci")}
                    print(d, dev, m, b, r["E1"], r["E2"], flush=True)   # None: the curves share no latency range
        for k, v in res.items():
            e1 = [x["E1"] for x in v.values()]
            v["score"] = sum(e1) / 2 if len(e1) == 2 and None not in e1 else None
        best = max((k for k in res if res[k]["score"] is not None), key=lambda k: res[k]["score"])
        default = res["s0_b1"]["score"]
        chosen = best if res[best]["score"] >= default + MARGIN else "s0_b1"
        out[d] = {"candidates": res, "best": best, "default_score": default, "chosen": chosen}
        print(d, "best", best, round(res[best]["score"], 2), "default", round(default, 2), "->", chosen, flush=True)
    with open(sys.argv[1], "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
