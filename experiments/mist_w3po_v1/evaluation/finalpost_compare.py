"""Secondary metrics of the final-read causal P test (scripts/finalpost_dev_orch.py), CPU only. Per decode dir
(e.g. ref / off / cand): the score_text.json headline, speech tcpBLEU per TTS seed (score_asr.json of the seed-1 dir and
of the seed-2 dir), short-unit recall (<= 0.5 s = the le0.5 bin; <= 1 s = the le0.5 and 0.5-1 bins pooled by count;
1spk / ov as the scorer splits them), the MISLABEL rate of <= 1 s / <= 0.5 s units (diag_order_attr check A summary:
MISLABEL / n_short, isolated / overlapped), runaway windows (loop_stats), and per pair of dirs the windows whose final
read (last read_log entry) / any earlier read / whole output changed. Writes a new json (refuses an existing one).

    python -m experiments.mist_w3po_v1.evaluation.finalpost_compare --dirs ref=<dir> off=<dir> cand=<dir> \
        --seed2 ref=<dir> off=<dir> cand=<dir> --diag ref=<check A dir> ... --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mist_w3po_v1.core.engine import LOOKAHEAD_MS
from experiments.mist_w3po_v1.evaluation.loop_stats import loop_stats
from experiments.mist_w3po_v1.evaluation.online_decode_finalpost import final_cut_ms

WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_v1/windows_dev_online.jsonl"
HEAD = ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk", "CR", "collapse", "duplicate_rate", "omitted_ov", "omitted_1spk",
        "inserted_ov", "inserted_1spk", "pre_end_share")


def kv(xs: list[str]) -> dict:
    return dict(x.split("=", 1) for x in xs)


def text_metrics(d: Path) -> dict:
    s = json.loads((d / "score_text.json").read_text())
    out = {k: s[k] for k in HEAD}
    for r in ("ov", "1spk"):
        out[f"lag_{r}_p50"], out[f"lag_{r}_mean"], out[f"lag_{r}_n"] = s[f"lag_{r}"]["p50"], s[f"lag_{r}"]["mean"], s[f"lag_{r}"]["n"]
    n = out["lag_ov_n"] + out["lag_1spk_n"]
    out["lag_all_mean"] = round((out["lag_ov_mean"] * out["lag_ov_n"] + out["lag_1spk_mean"] * out["lag_1spk_n"]) / n, 3)
    sr = s["short_recall"]
    for r in ("1spk", "ov"):
        a, b = sr[f"le0.5_{r}"], sr[f"0.5-1_{r}"]
        out[f"short_le0.5_{r}"] = a["recall"]
        out[f"short_le1_{r}"] = round((a["recall"] * a["n"] + b["recall"] * b["n"]) / (a["n"] + b["n"]), 4)
        out[f"short_n_le1_{r}"] = a["n"] + b["n"]
    return out


def speech_metrics(d: Path) -> dict | None:
    p = d / "score_asr.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {k: s[k] for k in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk", "CR", "collapse")}


def diag_metrics(d: Path) -> dict:
    g = json.loads((d / "check_a_summary.json").read_text())["groups"]
    out = {}
    for k, v in g.items():
        out[k] = {"n_short": v["n_short"], "miss_rate": v["miss_rate"],
                  "mislabel_rate": round(v["counts"]["MISLABEL"] / v["n_short"], 4), "mislabel": v["counts"]["MISLABEL"],
                  "order_omission": v["counts"]["ORDER_OMISSION"]}
    return out


def recs(d: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, open(d / "hyp.jsonl", encoding="utf-8"))}


def changes(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    fin, early, raw, spk = [], 0, 0, 0
    for i in ids:
        la, lb = a[i]["read_log"], b[i]["read_log"]
        early += any(x["new"] != y["new"] for x, y in zip(la[:-1], lb[:-1])) or len(la) != len(lb)
        if la[-1]["new"] != lb[-1]["new"]:
            fin.append(i)
        raw += a[i]["raw"] != b[i]["raw"]
        spk += [s["spk"] for s in a[i]["segments"]] != [s["spk"] for s in b[i]["segments"]]
    return {"windows": len(ids), "final_read_changed": len(fin), "earlier_read_changed": early, "output_changed": raw,
            "speaker_sequence_changed": spk, "final_changed_ids": fin}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--seed2", nargs="*", default=[])
    ap.add_argument("--diag", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out_p}")
    dirs, s2, dg = kv(a.dirs), kv(a.seed2), kv(a.diag)
    res = {"dirs": dirs, "seed2": s2, "per_run": {}, "pairs": {}}
    W = [json.loads(l) for l in open(WIN, encoding="utf-8")]
    ext = sum(final_cut_ms(w["dur_ms"], w["close_detect_ms"]) // 80 > max(0, w["dur_ms"] - LOOKAHEAD_MS) // 80 for w in W)
    full = sum(final_cut_ms(w["dur_ms"], w["close_detect_ms"]) >= w["dur_ms"] for w in W)
    res["windows"] = {"n": len(W), "final_P_extended": ext, "final_P_whole_window": full}
    R = {}
    for name, d in dirs.items():
        d = Path(d)
        r = {"text": text_metrics(d), "speech_seed1": speech_metrics(d),
             "speech_seed2": speech_metrics(Path(s2[name])) if name in s2 else None,
             "runaway_windows": loop_stats(d)["runaway_windows"]}
        if r["speech_seed1"] and r["speech_seed2"]:
            r["speech_mean2"] = {k: round((r["speech_seed1"][k] + r["speech_seed2"][k]) / 2, 4) for k in r["speech_seed1"]}
        if name in dg:
            r["diag_a"] = diag_metrics(Path(dg[name]))
        res["per_run"][name] = r
        R[name] = recs(d)
    names = list(dirs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res["pairs"][f"{names[j]}_vs_{names[i]}"] = changes(R[names[i]], R[names[j]])
    if "cand" in R and "ref" in R and "off" in R:
        c = set(res["pairs"]["cand_vs_ref"]["final_changed_ids"]) if "cand_vs_ref" in res["pairs"] else set()
        o = set(res["pairs"]["off_vs_ref"]["final_changed_ids"]) if "off_vs_ref" in res["pairs"] else set()
        res["final_changed_cand_not_off"] = len(c - o)
    with open(out_p, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    for name, r in res["per_run"].items():
        print(name, json.dumps({k: r["text"][k] for k in ("tcpBLEU_all", "tcpBLEU_ov", "short_le1_1spk", "short_le1_ov")}))
    for k, v in res["pairs"].items():
        print(k, {x: y for x, y in v.items() if x != "final_changed_ids"})


if __name__ == "__main__":
    main()
