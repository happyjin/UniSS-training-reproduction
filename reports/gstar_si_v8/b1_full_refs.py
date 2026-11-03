"""Paper v8 B1: full-coverage En->Zh references for the 16 AMI test meetings. Every non-hesitation test utterance with
a selected translation (data/ami_multispk_si_v1/index/chain_en_zh_ihm, the same 3-MT + QE selection as the reference
windows, Appendix A) becomes a scored utterance; one pseudo-window per meeting (start 0) in the reference-window format,
so overlap_score.py / meeting_score.py can use it as --ref-windows. Writes <out> (refuses to overwrite) and prints the
coverage against the reference windows.

    python reports/gstar_si_v8/b1_full_refs.py --out data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_fullref_v8.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
D = REPO / "data/ami_multispk_si_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import pyarrow.dataset as pads
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import is_hesitation
    meetings = sorted({json.loads(ln)["meeting"] for ln in open(D / "e2e_gstar_v1/windows_test_ref.jsonl")})
    tr = {r["record_id"]: (r["translation_text"] or "").strip() for r in pads.dataset(
        D / "index/chain_en_zh_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "translation_text", "split"]).to_pylist() if r["split"] == "test"}
    src = pads.dataset(D / "index/source_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript"]).to_pylist()
    by = defaultdict(list)
    n_all = n_hes = n_tr = 0
    for u in src:
        if u["meeting_id"] not in meetings:
            continue
        n_all += 1
        hes = is_hesitation(u["transcript"])
        n_hes += hes
        t = tr.get(u["record_id"], "")
        n_tr += bool(t) and not hes
        by[u["meeting_id"]].append({"record_id": u["record_id"], "speaker_id": u["speaker_id"],
                                    "begin_s": float(u["begin_time"]), "end_s": float(u["end_time"]),
                                    "en": u["transcript"], "zh": t, "ref": t, "hesitation": bool(hes)})
    with open(out, "x", encoding="utf-8") as fh:
        for m in meetings:
            fh.write(json.dumps({"id": f"{m}_full", "meeting": m, "split": "test", "wtype": "fullref", "start_s": 0.0,
                                 "utts": sorted(by[m], key=lambda u: u["begin_s"])}, ensure_ascii=False) + "\n")
    kept = sum(1 for ln in open(D / "e2e_gstar_v1/windows_test_ref.jsonl") for u in json.loads(ln)["utts"]
               if not u["hesitation"] and u["zh"].strip())
    print(json.dumps({"meetings": len(meetings), "utterances": n_all, "hesitations": n_hes,
                      "scored_full": n_tr, "scored_reference_windows": kept}))


if __name__ == "__main__":
    main()
