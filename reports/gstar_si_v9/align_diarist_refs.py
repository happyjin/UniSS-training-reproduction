"""Paper v9 §2.7: attach the DiariST human English references (data/diarist_alimeeting_refs_v1/test.txt) to our
AliMeeting test utterances. A row matches an utterance of data/alimeeting_si_v1/utts/<meeting>.jsonl when meeting,
speaker (DiariST "N_<spk>"), begin and end (to 10 ms) agree; the Chinese text must then be identical (else counted as a
text mismatch and not used). Writes, never overwriting:
  windows_test_ref_human.jsonl   our reference windows with "ref" = the human translation (utterances without a match
                                 keep no reference: "ref" = "" and "human": false, so they are scored as don't-care)
  windows_test_fullref_human.jsonl  one pseudo-window per meeting with every non-hesitation utterance (P1-3 full coverage)
  align_report.json              match counts and reasons

    python3 reports/gstar_si_v9/align_diarist_refs.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/diarist_alimeeting_refs_v1"
A = REPO / "data/alimeeting_si_v1"


def main():
    outs = [D / "windows_test_ref_human.jsonl", D / "windows_test_fullref_human.jsonl", D / "align_report.json"]
    for o in outs:
        if o.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o}")
    rows = [ln.rstrip("\n").split("\t") for ln in open(D / "test.txt", encoding="utf-8")]
    key = {}
    for m, spk, b, e, zh, en in rows:
        key[(m, spk.removeprefix("N_"), round(float(b), 2), round(float(e), 2))] = (zh, en)
    meetings = sorted(json.load(open(A / "meetings.json")))
    human, rep = {}, Counter()
    utts = defaultdict(list)
    for m in meetings:
        for u in map(json.loads, open(A / f"utts/{m}.jsonl", encoding="utf-8")):
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
    n_w = n_s = n_h = 0
    with open(outs[0], "x", encoding="utf-8") as fh:
        for w in map(json.loads, open(A / "windows_test_ref.jsonl", encoding="utf-8")):
            for u in w["utts"]:
                if u["hesitation"]:
                    continue
                n_s += 1
                h = human.get(u["record_id"])
                u["ref_mt"] = u.get("ref", "")
                u["ref"] = h or ""
                u["human"] = h is not None
                n_h += h is not None
            fh.write(json.dumps(w, ensure_ascii=False) + "\n")
            n_w += 1
    rep["ref_windows"] = n_w
    rep["scored_utterances_in_ref_windows"] = n_s
    rep["of_which_human_matched"] = n_h
    n_full = 0
    with open(outs[1], "x", encoding="utf-8") as fh:
        for m in meetings:
            us = []
            for u in sorted(utts[m], key=lambda u: u["begin_s"]):
                h = human.get(u["record_id"])
                us.append({"record_id": u["record_id"], "speaker_id": u["speaker_id"], "begin_s": u["begin_s"],
                           "end_s": u["end_s"], "zh": u["zh"], "ref": h or "", "hesitation": bool(u["hesitation"])})
                n_full += bool(h) and not u["hesitation"]
            fh.write(json.dumps({"id": f"{m}_full", "meeting": m, "split": "test", "wtype": "fullref", "start_s": 0.0,
                                 "utts": us}, ensure_ascii=False) + "\n")
    rep["full_scored_utterances"] = n_full
    outs[2].write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
