"""E2E-v0 after the Thinker (docs/multi_speaker_si_plan_v2.zh-CN.md): speech per speaker, one channel, scores.

  from-blocks  a single-speaker run (si_thinker_stream blocks) as E2E output: every block S1 -- the B0 baseline
  talker-prep  one talker stream per speaker label of a piece (si_infer --items-file / --blocks-file): the label's
               blocks only, its voice the audio from the label's first start to its first commit (heard by then)
  stitch       the speakers' speech back on one channel in commit order (a block starts when it is committed or
               when the one before has ended, 0.2 s later on a change of speaker); stereo with the source
  score        cpBLEU / cpchrF (labels matched to reference speakers one-to-one, Hungarian on chrF), speaker-blind
               BLEU, label count, attribution, LAAL; per group (all / single / multi / overlap / strict review);
               with --asr the same on what the speech says (paraformer on each speaker's speech)

    python -m experiments.uniss_qwen3_omni_v1.evaluation.e2e_ms_post score --pieces <pieces_test.jsonl> --hyp <hyp>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

LABEL = re.compile(r"\[S(\d+)@(\d+(?:\.\d+)?)\]")
PUNCT = "，。？！；、,.?!;：:“”\"'…—- "


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def from_blocks(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    with open(out, "x", encoding="utf-8") as fh:
        for r in load(args.blocks):
            bl = [{"write_ms": b["write_ms"], "spk": "S1", "anchor_ms": 0, "text": b["text"].strip(), "final": b["final"]}
                  for b in r["blocks"] if b["text"].strip()]
            if bl:
                bl[-1]["final"] = True
            fh.write(json.dumps({"id": r["id"], "raw": "".join(b["text"] for b in bl),
                                 "translation": "".join(b["text"] for b in bl), "blocks": bl}, ensure_ascii=False) + "\n")


def talker_prep(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "voice").mkdir(parents=True)
    pieces = {p["id"]: p for p in load(args.items)}
    n = 0
    with open(out / "items.jsonl", "x", encoding="utf-8") as fi, open(out / "blocks.jsonl", "x", encoding="utf-8") as fb:
        for h in load(args.hyp):
            p = pieces[h["id"]]
            wav, sr = sf.read(p["audio"], dtype="float32")
            by = defaultdict(list)
            for b in h["blocks"]:
                if b["text"].strip():
                    by[b["spk"]].append(b)
            for spk, bl in by.items():
                first = bl[0]["write_ms"]
                a = bl[0]["anchor_ms"] if 0 <= bl[0]["anchor_ms"] < first - 500 else max(0, first - 1500)
                ref = wav[a * 16: first * 16]
                vp = out / "voice" / f"{h['id']}__{spk}.wav"
                sf.write(str(vp), ref if len(ref) >= 8000 else wav[max(0, first - 1500) * 16: max(first, 1000) * 16], sr)
                blocks = [{"write_ms": b["write_ms"], "text": b["text"], "final": False} for b in bl]
                blocks[-1]["final"] = True
                iid = f"{h['id']}__{spk}"
                fi.write(json.dumps({"id": iid, "audio": p["audio"], "dur_ms": p["dur_ms"],
                                     "translation": "".join(b["text"] for b in bl), "voice_audio": str(vp)},
                                    ensure_ascii=False) + "\n")
                fb.write(json.dumps({"id": iid, "blocks": blocks}, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"streams": n, "out": str(out)}))


def stitch(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "wav").mkdir(parents=True)
    pieces = {p["id"]: p for p in load(args.items)}
    streams = defaultdict(list)
    for r in load(args.results):
        pid, spk = r["id"].rsplit("__", 1)
        streams[pid].append((spk, r))
    with open(out / "transcript.jsonl", "x", encoding="utf-8") as ft:
        for pid, ss in sorted(streams.items()):
            p = pieces[pid]
            src, sr = sf.read(p["audio"], dtype="float32")
            segs = []
            for spk, r in ss:
                speech, ssr = sf.read(r["speech_path"], dtype="float32")
                assert ssr == 16000
                for k, b in enumerate(r["blocks"]):
                    a = b["speech_at_ms"] * 16
                    segs.append((b["write_ms"], spk, k, b["text"], speech[a: a + b["dur_ms"] * 16]))
            segs.sort(key=lambda s: (s[0], s[1], s[2]))
            t_end, last, placed = 0, None, []
            for w, spk, _, text, x in segs:
                start = max(w, t_end + (200 if last not in (None, spk) else 0))
                placed.append({"spk": spk, "write_ms": w, "start_ms": start, "dur_ms": len(x) // 16, "text": text})
                t_end, last = start + len(x) // 16, spk
            total = max(p["dur_ms"], t_end) + 300
            right = np.zeros(total * 16, dtype=np.float32)
            for pl, (_, _, _, _, x) in zip(placed, segs):
                right[pl["start_ms"] * 16: pl["start_ms"] * 16 + len(x)] += x
            left = np.zeros_like(right)
            left[: len(src)] = src[: len(right)]
            sf.write(str(out / "wav" / f"{pid}.wav"), right, 16000)
            sf.write(str(out / "wav" / f"{pid}_stereo.wav"), np.stack([0.8 * left, right], 1), 16000)
            ft.write(json.dumps({"id": pid, "dur_ms": p["dur_ms"], "blocks": placed}, ensure_ascii=False) + "\n")
    print(json.dumps({"pieces": len(streams), "out": str(out)}))


def chars(s: str) -> int:
    return len([c for c in s if c not in PUNCT and not c.isspace()])


def match(hyp: dict, ref: dict):
    """Labels -> reference speakers, one-to-one, maximising the summed chrF."""
    from sacrebleu.metrics import CHRF
    from scipy.optimize import linear_sum_assignment

    hs, rs = [k for k in hyp if hyp[k].strip()], [k for k in ref if ref[k].strip()]
    if not hs or not rs:
        return {}, hs, rs
    chrf = CHRF()
    cost = np.array([[-chrf.sentence_score(hyp[h], [ref[r]]).score for r in rs] for h in hs])
    i, j = linear_sum_assignment(cost)
    m = {hs[a]: rs[b] for a, b in zip(i, j)}
    return m, [h for h in hs if h not in m], [r for r in rs if r not in m.values()]


def score(args):
    from sacrebleu.metrics import BLEU, CHRF

    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units

    pieces = {p["id"]: p for p in load(args.pieces)}
    hyps = {h["id"]: h for h in load(args.hyp)}
    asr = {}
    if args.asr:
        for r in load(args.asr):
            asr[r["id"]] = r.get("asr_text", "")
    groups = defaultdict(list)
    rows = []
    for pid, p in pieces.items():
        h = hyps.get(pid, {"blocks": [], "translation": ""})
        hyp_by = defaultdict(str)
        for b in h["blocks"]:
            hyp_by[b["spk"]] += b["text"]
        ref_by = p["ref_by_label"]
        m, extra_h, extra_r = match(hyp_by, ref_by)
        pairs = [(hyp_by[a], ref_by[b]) for a, b in m.items()] + [(hyp_by[a], "") for a in extra_h] + \
                [("", ref_by[b]) for b in extra_r]
        ref_plain = LABEL.sub("", p["reference"])
        # attribution: the reference speaker talking where the block's turn starts
        spans = [(u["begin_ms"], u["end_ms"], f"S{u['label']}") for u in p["utts"] if u["label"] and not u["hesitation"]]
        ok = tot = 0
        for b in h["blocks"]:
            a = b["anchor_ms"]
            best = max(spans, key=lambda s: (min(s[1], a + 500) - max(s[0], a), -abs(s[0] - a)), default=None)
            n = chars(b["text"])
            tot += n
            ok += n * (best is not None and m.get(b["spk"]) == best[2])
        delays = [b["write_ms"] for b in h["blocks"] for _ in units(b["text"], "cmn")]
        lag = laal(delays, p["dur_ms"], len(units(ref_plain, "cmn"))) if delays else None
        row = {"id": pid, "map": m, "pairs": pairs, "hyp_plain": h["translation"], "ref_plain": ref_plain,
               "n_hyp": len([k for k in hyp_by if hyp_by[k].strip()]), "n_ref": len([k for k in ref_by if ref_by[k]]),
               "attr_ok": ok, "attr_tot": tot, "laal_ms": lag}
        if args.asr:
            asr_by = {spk: asr.get(f"{pid}__{spk}", "") for spk in hyp_by}
            row["asr_pairs"] = [(asr_by.get(a, ""), ref_by[b]) for a, b in m.items()] + \
                [(asr_by.get(a, ""), "") for a in extra_h] + [("", ref_by[b]) for b in extra_r]
            order = sorted(hyp_by, key=lambda s: min(b["write_ms"] for b in h["blocks"] if b["spk"] == s))
            row["asr_plain"] = "".join(asr_by.get(s, "") for s in order)
        rows.append(row)
        groups["all"].append(row)
        groups["single_spk" if p["n_spk"] == 1 else "multi_spk"].append(row)
        if p["overlap_ms"] > 0:
            groups["overlap"].append(row)
        if p.get("strict_review"):
            groups["strict_review"].append(row)
    bleu, chrf = BLEU(tokenize="zh"), CHRF()
    res = {}
    for g, rs in groups.items():
        H = [a for r in rs for a, _ in r["pairs"]]
        R = [b for r in rs for _, b in r["pairs"]]
        lags = [r["laal_ms"] for r in rs if r["laal_ms"] is not None]
        d = {"pieces": len(rs),
             "cpBLEU": round(bleu.corpus_score(H, [R]).score, 2),
             "cpchrF": round(chrf.corpus_score(H, [R]).score, 2),
             "BLEU_speaker_blind": round(bleu.corpus_score([r["hyp_plain"] for r in rs], [[r["ref_plain"] for r in rs]]).score, 2),
             "count_exact": round(sum(r["n_hyp"] == r["n_ref"] for r in rs) / len(rs), 4),
             "count_over": round(sum(r["n_hyp"] > r["n_ref"] for r in rs) / len(rs), 4),
             "count_under": round(sum(r["n_hyp"] < r["n_ref"] for r in rs) / len(rs), 4),
             "attribution_acc": round(sum(r["attr_ok"] for r in rs) / max(sum(r["attr_tot"] for r in rs), 1), 4),
             "empty_output": round(sum(not r["hyp_plain"].strip() for r in rs) / len(rs), 4),
             "LAAL_s": round(float(np.mean(lags)) / 1000, 3) if lags else None}
        if args.asr:
            HA = [a for r in rs for a, _ in r["asr_pairs"]]
            RA = [b for r in rs for _, b in r["asr_pairs"]]
            d["cpASR_BLEU"] = round(bleu.corpus_score(HA, [RA]).score, 2)
            d["ASR_BLEU_speaker_blind"] = round(bleu.corpus_score([r["asr_plain"] for r in rs], [[r["ref_plain"] for r in rs]]).score, 2)
        res[g] = d
    out = Path(args.out) if args.out else None
    if out:
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        with open(out.with_suffix(".rows.jsonl"), "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def score_single(args):
    """Single-speaker regression (RealSI items): BLEU of the reply without its labels, and LAAL."""
    from sacrebleu.metrics import BLEU

    from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units

    items = {r["id"]: r for r in load(args.items)}
    hyps = {h["id"]: h for h in load(args.hyp)}
    H, R, lags = [], [], []
    for iid, it in items.items():
        h = hyps.get(iid, {"blocks": [], "translation": ""})
        H.append(LABEL.sub("", h["translation"]).strip())
        R.append(it["translation"])
        delays = [b["write_ms"] for b in h["blocks"] for _ in units(LABEL.sub("", b["text"]), "cmn")]
        if delays:
            lags.append(laal(delays, it["dur_ms"], len(units(it["translation"], "cmn"))))
    res = {"items": len(items), "BLEU": round(BLEU(tokenize="zh").corpus_score(H, [R]).score, 2),
           "LAAL_s": round(float(np.mean(lags)) / 1000, 3) if lags else None,
           "labels_other_than_S1": sum(any(b["spk"] != "S1" for b in hyps.get(i, {"blocks": []})["blocks"]) for i in items)}
    print(json.dumps(res, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("score-single")
    a.add_argument("--items", required=True)
    a.add_argument("--hyp", required=True)
    a = sub.add_parser("from-blocks")
    a.add_argument("--blocks", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("talker-prep")
    a.add_argument("--hyp", required=True)
    a.add_argument("--items", required=True, help="pieces items (id, audio, dur_ms)")
    a.add_argument("--out", required=True)
    a = sub.add_parser("stitch")
    a.add_argument("--results", required=True, help="si_infer results.jsonl of the per-speaker streams")
    a.add_argument("--items", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("score")
    a.add_argument("--pieces", required=True)
    a.add_argument("--hyp", required=True)
    a.add_argument("--asr", default="", help="asr_speech.jsonl of the per-speaker streams")
    a.add_argument("--out", default="")
    args = ap.parse_args()
    {"from-blocks": from_blocks, "talker-prep": talker_prep, "stitch": stitch, "score": score,
     "score-single": score_single}[args.cmd](args)


if __name__ == "__main__":
    main()
