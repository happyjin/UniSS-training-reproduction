"""Paper v6, review M01 / M13 (prereg_v6 §1): whole-meeting scores of a system's output, from windows of any kind
(the reference windows of Tables 1-2, or the online windows of build_online_windows.py) on the meeting clock.

Scored references: the non-hesitation utterances of the reference windows (--ref-windows), i.e. the same utterances
as Tables 1-2. Every other utterance of the meeting (hesitations, utterances the windows dropped) is "don't care".
Each output segment (window start + its predicted span) goes to the utterance it overlaps most within a --collar:
  a scored utterance of the speaker its slot is matched to   -> that utterance's hypothesis
  a scored utterance of another speaker                      -> the matched speaker's own scored utterance overlapping
                                                                it if any, else an insertion
  a don't-care utterance                                     -> dropped
  nothing                                                    -> insertion
Slots are matched to speakers once per meeting (chrF Hungarian on the whole meeting's text, as gstar_score).
  tcpBLEU        corpus BLEU over (assigned text, reference) per scored unit plus (insertion, "")
  mcpBLEU        corpus BLEU over (all text of a slot, all references of its speaker) per meeting and speaker
  units          a speaker's scored utterances less than --merge s apart form one reference unit
  omitted        scored units with no text; inserted: share of kept segments that are insertions
  end lag        streaming output: commit time of the last segment assigned to a unit minus its end (s)
--online: commit times are clamped to the window's open_detect_ms, and the last read's commits are moved to its
close_detect_ms (the moment the window is known to be over); offline output has no commit times.
Writes <out> (json) with per-meeting BLEU statistics for the cluster bootstrap.

    python reports/gstar_si_v6/meeting_score.py --dir en2zh --windows <decoded windows> --hyp <hyp.jsonl> \
        --ref-windows <windows_test_ref.jsonl> [--online] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper/gstar_si_v6/scripts"))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match  # noqa: E402

D = REPO / "data/ami_multispk_si_v1"
A = REPO / "data/alimeeting_si_v1"


def universe(dr: str, meetings: set) -> dict:
    """meeting -> every annotated utterance [(record_id, speaker, begin, end, hesitation)] on that meeting's clock."""
    out = defaultdict(list)
    if dr == "en2zh":
        import pyarrow.dataset as pads
        from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import is_hesitation
        src = pads.dataset(D / "index/source_ihm", format="parquet", partitioning="hive").to_table(
            columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript"]).to_pylist()
        for u in src:
            if u["meeting_id"] in meetings:
                out[u["meeting_id"]].append((u["record_id"], u["speaker_id"], u["begin_time"], u["end_time"],
                                             is_hesitation(u["transcript"])))
    elif dr == "zh2en":
        for m in meetings:
            for u in map(json.loads, open(D / f"zh_mix_v8/utts/{m}.jsonl", encoding="utf-8")):
                if u.get("placed", True):
                    out[m].append((u["record_id"], u["speaker_id"], u["zh_begin"], u["zh_end"], u["hes"]))
    else:
        for m in meetings:
            for u in map(json.loads, open(A / f"utts/{m}.jsonl", encoding="utf-8")):
                out[m].append((u["record_id"], u["speaker_id"], u["begin_s"], u["end_s"], u["hesitation"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, choices=["en2zh", "zh2en", "ali"])
    ap.add_argument("--windows", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref-windows", required=True)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--collar", type=float, default=2.0)
    ap.add_argument("--merge", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from sacrebleu.metrics import BLEU

    eng = args.dir != "en2zh"
    sep = " " if eng else ""
    bleu = BLEU(tokenize="13a" if eng else "zh")
    H = {h["id"]: h for h in map(json.loads, open(args.hyp, encoding="utf-8"))}
    W = [w for w in map(json.loads, open(args.windows, encoding="utf-8")) if w["id"] in H]
    meetings = {w["meeting"] for w in W}
    scored = {}
    for w in map(json.loads, open(args.ref_windows, encoding="utf-8")):
        if w["meeting"] not in meetings:
            continue
        for u in w["utts"]:
            if not u["hesitation"] and u.get("ref", u["zh"]).strip():
                scored[u["record_id"]] = {"meeting": w["meeting"], "spk": u["speaker_id"], "a": w["start_s"] + u["begin_s"],
                                          "b": w["start_s"] + u["end_s"], "ref": u.get("ref", u["zh"]).strip()}
    U = universe(args.dir, meetings)
    # output segments on the meeting clock
    segs = defaultdict(list)
    n_final = n_units_moved = 0
    for w in W:
        h = H[w["id"]]
        for s in h["segments"]:
            c = None
            if "write_ms" in s:
                c = s["write_ms"]
                if args.online:
                    c = max(c, w["open_detect_ms"])
                    if s["write_ms"] >= w["dur_ms"]:
                        c = max(c, w["close_detect_ms"])
                        n_final += 1
                c = w["start_s"] + c / 1000
            segs[w["meeting"]].append({"spk": s["spk"], "a": w["start_s"] + s["start_s"],
                                       "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08), "text": s["text"].strip(),
                                       "commit": c, "win": w["id"]})
    pairs_tcp, pairs_m, per_meeting = [], [], {}
    omitted = n_scored = n_kept = n_ins = n_drop = 0
    lags = []
    for m in sorted(meetings):
        us = [(rid, spk, a, b, hes) for rid, spk, a, b, hes in U[m]]
        sc = {rid: v for rid, v in scored.items() if v["meeting"] == m}
        hyp_t, ref_t = defaultdict(str), defaultdict(str)
        for s in segs[m]:
            hyp_t[s["spk"]] += sep + s["text"]
        for v in sorted(sc.values(), key=lambda v: v["a"]):
            ref_t[v["spk"]] += sep + v["ref"]
        mp = match(dict(hyp_t), dict(ref_t))
        # reference units: a speaker's scored utterances chained while < --merge s apart (a model may write two
        # consecutive utterances of one person as one segment; scoring them apart would count that as an error)
        unit_of, units_ = {}, {}
        for spk in {v["spk"] for v in sc.values()}:
            cur = None
            for rid, v in sorted(((r, v) for r, v in sc.items() if v["spk"] == spk), key=lambda kv: kv[1]["a"]):
                if cur is not None and v["a"] - units_[cur]["b"] < args.merge:
                    units_[cur]["b"] = max(units_[cur]["b"], v["b"])
                    units_[cur]["ref"] += sep + v["ref"]
                else:
                    cur = rid
                    units_[cur] = {"spk": spk, "a": v["a"], "b": v["b"], "ref": v["ref"]}
                unit_of[rid] = cur
        got = defaultdict(list)
        ins, kept_text = [], defaultdict(list)
        for s in sorted(segs[m], key=lambda s: s["a"]):
            if not s["text"]:
                continue
            a, b = s["a"] - args.collar, s["b"] + args.collar
            best, bu = 0.0, None
            for u in us:
                ov = min(b, u[3]) - max(a, u[2])
                if ov > best:
                    best, bu = ov, u
            if bu is None:
                ins.append(s)
                continue
            if bu[0] not in sc:   # a hesitation or an utterance the reference windows dropped
                n_drop += 1
                continue
            n_kept += 1
            spk = mp.get(s["spk"])
            target = unit_of[bu[0]] if sc[bu[0]]["spk"] == spk else None
            if target is None and spk is not None:
                best2 = 0.0
                for rid, v in units_.items():
                    if v["spk"] != spk:
                        continue
                    ov = min(b, v["b"]) - max(a, v["a"])
                    if ov > best2:
                        best2, target = ov, rid
            if target is None:
                ins.append(s)
                continue
            got[target].append(s)
            kept_text[spk].append(s["text"])
        n_ins += len(ins)
        tcp = []
        for rid, v in sorted(units_.items(), key=lambda kv: kv[1]["a"]):
            ss = sorted(got.get(rid, []), key=lambda s: s["a"])
            tcp.append((sep.join(s["text"] for s in ss).strip(), v["ref"]))
            n_scored += 1
            omitted += not ss
            cs = [s["commit"] for s in ss if s["commit"] is not None]
            if cs:
                lags.append(max(cs) - v["b"])
        tcp += [(s["text"], "") for s in ins]
        mc = [(sep.join(kept_text.get(spk, [])).strip(), ref_t[spk].strip()) for spk in ref_t]
        mc += [(sep.join(s["text"] for s in ins), "")] if ins else []
        pairs_tcp += tcp
        pairs_m += mc
        st = bleu._extract_corpus_statistics([x for x, _ in tcp], [[y for _, y in tcp]])
        per_meeting[m] = {"tcp_stats": np.sum(np.array(st, dtype=np.float64), 0).tolist() if st else None,
                          "tcpBLEU": round(bleu.corpus_score([x for x, _ in tcp], [[y for _, y in tcp]]).score, 2)}
    res = {"dir": args.dir, "windows": len(W), "meetings": len(meetings), "scored_units": n_scored, "scored_utterances": len(scored),
           "collar_s": args.collar, "online": args.online,
           "tcpBLEU": round(bleu.corpus_score([x for x, _ in pairs_tcp], [[y for _, y in pairs_tcp]]).score, 2),
           "mcpBLEU": round(bleu.corpus_score([x for x, _ in pairs_m], [[y for _, y in pairs_m]]).score, 2),
           "omitted": round(omitted / max(n_scored, 1), 4), "inserted": round(n_ins / max(n_kept + n_ins, 1), 4),
           "dropped_dont_care_segments": n_drop, "final_read_segments_moved": n_final}
    if lags:
        lags = np.array(lags)
        res["end_lag_s"] = {"mean": round(float(lags.mean()), 3), "p50": round(float(np.median(lags)), 3),
                            "p90": round(float(np.percentile(lags, 90)), 3), "p95": round(float(np.percentile(lags, 95)), 3)}
    res["per_meeting"] = per_meeting
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "per_meeting"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
