"""Paper v9 P1-3: full-coverage Zh->En references. The Zh->En references are the human AMI English transcripts,
case/punctuation-normalised (data/ami_multispk_si_v1/en_norm_v1.jsonl, "en"); every non-hesitation utterance placed in
the re-voiced Chinese mix (zh_mix_v8/utts, placed) becomes a scored utterance on the mix clock. One pseudo-window per
meeting, as reports/gstar_si_v8/b1_full_refs.py. Writes <out> (refuses to overwrite) and prints the coverage.

    python3 reports/gstar_si_v9/zh2en_full_refs.py --out data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_fullref_v9.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/ami_multispk_si_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    N = {json.loads(ln)["record_id"]: json.loads(ln) for ln in open(D / "en_norm_v1.jsonl")}
    meetings = sorted({json.loads(ln)["meeting"] for ln in open(D / "e2e_gstar_zh2en_v1/windows_test_ref.jsonl")})
    c = Counter()
    with open(out, "x", encoding="utf-8") as fh:
        for m in meetings:
            us = []
            for u in map(json.loads, open(D / f"zh_mix_v8/utts/{m}.jsonl", encoding="utf-8")):
                c["utterances"] += 1
                if not u.get("placed", True):
                    c["not_placed"] += 1
                    continue
                n = N.get(u["record_id"])
                ref = (n or {}).get("en", "").strip()
                if u["hes"]:
                    c["hesitation"] += 1
                elif not ref:
                    c["no_normalised_text"] += 1
                else:
                    c["scored"] += 1
                us.append({"record_id": u["record_id"], "speaker_id": u["speaker_id"], "begin_s": u["zh_begin"],
                           "end_s": u["zh_end"], "zh": u.get("zh", ""), "ref": ref, "hesitation": bool(u["hes"])})
            fh.write(json.dumps({"id": f"{m}_full", "meeting": m, "split": "test", "wtype": "fullref", "start_s": 0.0,
                                 "utts": sorted(us, key=lambda u: u["begin_s"])}, ensure_ascii=False) + "\n")
    kept = sum(1 for ln in open(D / "e2e_gstar_zh2en_v1/windows_test_ref.jsonl") for u in json.loads(ln)["utts"]
               if not u["hesitation"] and u.get("ref", "").strip())
    c["scored_reference_windows"] = kept
    print(json.dumps(c))


if __name__ == "__main__":
    main()
