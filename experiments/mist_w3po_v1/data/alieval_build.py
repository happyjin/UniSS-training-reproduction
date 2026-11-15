"""v11 P0-2 (docs/mist_v11_plan.zh-CN.md §3; reports/mist_w3po_v1/v11_p02_alieval_prereg.zh-CN.md): the AliMeeting
Eval set (8 meetings, never used for training, development or testing) as a clean confirmatory Zh->En test set, built
exactly like the AliMeeting test set except where noted. Test only: nothing is trained or tuned on it.

Data dir data/alimeeting_eval_si_v1 (build_alimeeting_test.py prep on data/raw/alimeeting/Eval_Ali; diarizer
posteriors by dump_nemotron_posteriors.py, presets offline / lowlat, as make_v3_jobs.py's ali_nemo_* jobs).
  refs     data/diarist_alimeeting_refs_v1/dev.txt (DiariST dev = AliMeeting Eval, human English) matched to our
           utterances by meeting, speaker, begin, end (10 ms) and identical Chinese, as align_diarist_refs.py
           -> <data>/refs.jsonl ({"id", "zh", "en"} for every mt_items.jsonl item; "" without a match)
           DEVIATION: the test set's refs.jsonl was Hy-MT2 output (the model is not on this server); here the human
           translation takes its place in the ref windows' "en" / "ref" fields.
  audit    <data>/refs_audit.json with NO failures: the test audit removed broken MT output (4 of 14,148 items);
           human references are not filtered (DEVIATION, so the reference windows are not selected on the references).
           The counts the MT audit rules would give on the human text are recorded for information.
  (then build_alimeeting_test.py windows --out <data>: reference windows, unchanged)
  align    align_diarist_refs.py on the Eval set -> data/diarist_alimeeting_refs_v1/{windows_eval_ref_human.jsonl,
           windows_eval_fullref_human.jsonl, align_report_eval.json} ("ref" = human, "ref_mt" = "", "human")
  c2win    build_test_windows_c2.py --set ali --gap 0.6 with the Eval meetings / audio / lowlat posteriors
           -> eval_outputs/mist_w3po_v1/data/windows_alieval_c2g06_v1/windows_test_online_ali.jsonl
  chunk    chunk_windows_set.py --set ali on those windows with the Eval posteriors / audio
           -> eval_outputs/mist_w3po_v1/data/windows_alieval_c2g06_chunk_v1/
Every step refuses to overwrite. Additive: the wrapped modules are patched in this process only.

    python -m experiments.mist_w3po_v1.data.alieval_build refs|audit|align|c2win|chunk
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AE = REPO / "data/alimeeting_eval_si_v1"
DR = REPO / "data/diarist_alimeeting_refs_v1"
DD = REPO / "eval_outputs/mist_w3po_v1/data"
C2 = DD / "windows_alieval_c2g06_v1"
CH = DD / "windows_alieval_c2g06_chunk_v1"


def x_open(p: Path):
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    return open(p, "x", encoding="utf-8")


def human_map() -> tuple[dict, Counter, dict]:
    rows = [ln.rstrip("\n").split("\t") for ln in open(DR / "dev.txt", encoding="utf-8")]
    key = {}
    for m, spk, b, e, zh, en in rows:
        key[(m, spk.removeprefix("N_"), round(float(b), 2), round(float(e), 2))] = (zh, en)
    human, rep, utts = {}, Counter(), defaultdict(list)
    for m in sorted(json.load(open(AE / "meetings.json"))):
        for u in map(json.loads, open(AE / f"utts/{m}.jsonl", encoding="utf-8")):
            utts[m].append(u)
            k = (m, u["speaker_id"], round(u["begin_s"], 2), round(u["end_s"], 2))
            rep["utterances"] += 1
            if k not in key:
                rep["no_row"] += 1
                continue
            zh, en = key[k]
            if zh.strip() != u["zh"].strip():
                rep["zh_mismatch"] += 1
                continue
            if not en.strip():
                rep["empty_en"] += 1
                continue
            human[u["record_id"]] = en.strip()
            rep["matched"] += 1
    rep["diarist_rows"] = len(rows)
    return human, rep, utts


def refs() -> None:
    human, rep, _ = human_map()
    items = [json.loads(l) for l in open(AE / "mt_items.jsonl", encoding="utf-8")]
    n = 0
    with x_open(AE / "refs.jsonl") as fh:
        for i in items:
            en = human.get(i["id"], "")
            n += bool(en)
            fh.write(json.dumps({"id": i["id"], "zh": i["zh"], "en": en, "source": "diarist_dev_human"},
                                ensure_ascii=False) + "\n")
    print(json.dumps({**rep, "items": len(items), "items_with_human_ref": n}))


def audit() -> None:
    from experiments.uniss_qwen3_omni_v1.data_clean.build_alimeeting_test import ref_fail
    zh = {r["id"]: r["zh"] for r in map(json.loads, open(AE / "mt_items.jsonl", encoding="utf-8"))}
    ref = {r["id"]: r["en"] for r in map(json.loads, open(AE / "refs.jsonl", encoding="utf-8"))}
    would = Counter(ref_fail(z, ref.get(k, "")) for k, z in zh.items())
    would.pop(None, None)
    res = {"items": len(zh), "failed": 0, "by_reason": {}, "fails": {},
           "note": "human references (DiariST dev): not filtered; the MT audit rules would flag the counts below",
           "mt_rules_on_human_refs": dict(would)}
    with x_open(AE / "refs_audit.json") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res))


def align() -> None:
    human, rep, utts = human_map()
    outs = [DR / "windows_eval_ref_human.jsonl", DR / "windows_eval_fullref_human.jsonl", DR / "align_report_eval.json"]
    for o in outs:
        if o.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o}")
    n_w = n_s = n_h = 0
    with x_open(outs[0]) as fh:
        for w in map(json.loads, open(AE / "windows_test_ref.jsonl", encoding="utf-8")):
            for u in w["utts"]:
                if u["hesitation"]:
                    continue
                n_s += 1
                h = human.get(u["record_id"])
                u["ref_mt"] = ""
                u["ref"] = h or ""
                u["human"] = h is not None
                n_h += h is not None
            fh.write(json.dumps(w, ensure_ascii=False) + "\n")
            n_w += 1
    rep.update(ref_windows=n_w, scored_utterances_in_ref_windows=n_s, of_which_human_matched=n_h)
    n_full = 0
    with x_open(outs[1]) as fh:
        for m in sorted(utts):
            us = []
            for u in sorted(utts[m], key=lambda u: u["begin_s"]):
                h = human.get(u["record_id"])
                us.append({"record_id": u["record_id"], "speaker_id": u["speaker_id"], "begin_s": u["begin_s"],
                           "end_s": u["end_s"], "zh": u["zh"], "ref": h or "", "hesitation": bool(u["hesitation"])})
                n_full += bool(h) and not u["hesitation"]
            fh.write(json.dumps({"id": f"{m}_full", "meeting": m, "split": "test", "wtype": "fullref", "start_s": 0.0,
                                 "utts": us}, ensure_ascii=False) + "\n")
    rep["full_scored_utterances"] = n_full
    with x_open(outs[2]) as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep))


def c2win() -> None:
    sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
    import build_online_windows as V6
    from experiments.mist_w3po_v1.data import build_test_windows_c2 as B
    V6.SETS["ali"] = (lambda: sorted(json.load(open(AE / "meetings.json"))),
                      lambda m: AE / f"session_audio/near_mix/{m}.wav",
                      lambda m: AE / f"diar/nemotron_lowlat/{m}.post.pt")
    sys.argv = [sys.argv[0], "--set", "ali", "--gap", "0.6", "--out", str(C2.relative_to(REPO))]
    B.main()


def chunk() -> None:
    from experiments.mist_w3po_v1.data import chunk_windows_set as C
    C.SETS["ali"] = (str(AE.relative_to(REPO)), "nemotron_lowlat", "near_mix")
    sys.argv = [sys.argv[0], "--set", "ali", "--base-win", str((C2 / "windows_test_online_ali.jsonl").relative_to(REPO)),
                "--out", str(CH.relative_to(REPO))]
    C.main()


if __name__ == "__main__":
    {"refs": refs, "audit": audit, "align": align, "c2win": c2win, "chunk": chunk}[sys.argv[1]]()
