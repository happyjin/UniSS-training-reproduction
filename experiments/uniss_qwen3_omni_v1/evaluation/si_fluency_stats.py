"""Fluency of an si_infer run's output speech, measured on the timeline audio as a listener hears it.

Pauses come from the timeline wav itself (energy VAD: 10 ms frames, silence below the utterance's speech level - 30 dB,
pauses >= 0.15 s; the wait before the first sound and the silence after the last are not pauses). Each pause is put
on the text: a pause that overlaps the seam between two blocks is a seam pause, the rest are inside a block. A seam is

  S  sentence     block ends with 。？！.?!
  C  clause       block ends with ，、；：,;: (or the next block starts with one)
  P  phrase       between words, no punctuation (English: after a function word is counted apart as P-fw)
  W  word         inside a word (Chinese: inside a jieba word of the whole translation)

and its pause is excess if longer than the tolerance a listener grants there (W 0.15, P 0.30, C 0.50, S 1.00 s), the
same tolerances as the fluency package's scorer. Reported per run:

  speed      articulation rate (units / s of sound), speech rate (units / s from first to last sound), phonation
             ratio, mean length of run (units between pauses >= 0.25 s); per block (>= 2 units), its seconds per unit
             against the run's median and the jump between adjacent blocks
  breakdown  pauses >= 0.25 / 0.5 / 1 / 2 s per 100 units, their seam types, excess-pause burden (s / 100 units)
  queue      blocks that waited for the previous one (start > write), how long, and the lag that is left at the end
  blocks     units per block, one-unit and punctuation-only blocks, blocks the guard had to close
  repair     the Talker against its text (CER / WER of the speech-only ASR), repeats the speech has and the text has not
             (stutter), and repeats the text has and the reference has not

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats --run m2_en2zh=<si_infer dir> ... --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

CJK = re.compile(r"[一-鿿]")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
TOL = {"W": 0.15, "P": 0.30, "P-fw": 0.30, "C": 0.50, "S": 1.00}
SENT, CLAUSE = set("。？！.?!"), set("，、；：,;:")
FUNCTION = set("a an the of to in on at for with by from and or but that which who whose is are was were be been "
               "this these those it its as than so if then not no our your their his her my we you they i he she "
               "very more most can will would could should do does did have has had into about".split())


def units(text: str, tgt: str) -> list[str]:
    return CJK.findall(text) + WORD.findall(text) if tgt == "cmn" else WORD.findall(text)


def n_units(text: str, tgt: str) -> int:
    return len(CJK.findall(text)) + len(WORD.findall(text)) if tgt == "cmn" else len(WORD.findall(text))


def vad_pauses(path: str, min_pause: float = 0.15) -> tuple[list[tuple[float, float]], float, float, float]:
    """Silent stretches between the first and the last sound, (start, end) in s; also first/last sound, sounding s."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    hop = sr // 100
    n = len(x) // hop
    if n == 0:
        return [], 0.0, 0.0, 0.0
    db = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loud = db[db > -70]
    if not len(loud):
        return [], 0.0, 0.0, 0.0
    on = db > max(np.percentile(loud, 90) - 30, -60)
    idx = np.flatnonzero(on)
    if not len(idx):
        return [], 0.0, 0.0, 0.0
    first, last = idx[0], idx[-1]
    pauses, i = [], first
    while i <= last:
        if not on[i]:
            j = i
            while j <= last and not on[j]:
                j += 1
            if (j - i) / 100 >= min_pause:
                pauses.append((i / 100, j / 100))
            i = j
        else:
            i += 1
    return pauses, first / 100, (last + 1) / 100, float(on[first: last + 1].sum()) / 100


def seam_types(blocks: list[dict], tgt: str, translation: str) -> list[str]:
    """The type of each seam between block k and k+1."""
    out = []
    if tgt == "cmn":
        import jieba
        words = [w for w in jieba.lcut(re.sub(r"\s+", "", translation)) if w.strip()]
        word_end, pos = set(), 0       # character offsets (in the concatenated block text) where a word ends
        for w in words:
            pos += len(w)
            word_end.add(pos)
        pos = 0
        for a, b in zip(blocks, blocks[1:]):
            pos += len(re.sub(r"\s+", "", a["text"]))
            ta, tb = a["text"].rstrip(), b["text"].lstrip()
            if ta and ta[-1] in SENT:
                out.append("S")
            elif (ta and ta[-1] in CLAUSE) or (tb and tb[0] in CLAUSE | SENT):
                out.append("C")
            elif pos in word_end:
                out.append("P")
            else:
                out.append("W")
        return out
    for a, b in zip(blocks, blocks[1:]):
        ta, tb = a["text"].rstrip(), b["text"].lstrip()
        last = WORD.findall(ta)[-1].lower() if WORD.findall(ta) else ""
        if ta and ta[-1] in SENT:
            out.append("S")
        elif (ta and ta[-1] in CLAUSE) or (tb and tb[0] in CLAUSE | SENT):
            out.append("C")
        elif ta and ta[-1].isalpha() and tb and tb[0].isalpha() and not a["text"].endswith(" ") and not b["text"].startswith(" "):
            out.append("W")        # a word cut in two
        else:
            out.append("P-fw" if last in FUNCTION else "P")
    return out


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of hyp against ref."""
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(len(ref) + 1), np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, s, dl, ins = len(ref), len(hyp), 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl, i = dl + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return s, dl, ins


def repeats(seq: list[str], max_n: int = 4) -> Counter:
    """Immediately repeated n-grams (x x), counted by their text."""
    c = Counter()
    for n in range(1, max_n + 1):
        for i in range(len(seq) - 2 * n + 1):
            if seq[i: i + n] == seq[i + n: i + 2 * n] and (n > 1 or len(seq[i]) > 0):
                c["".join(seq[i: i + n]) if n > 1 else seq[i]] += 1
    return c


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def analyse(run_dir: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(l) for l in open(run_dir / "results.jsonl")]
    asr = {}
    if (run_dir / "asr_speech.jsonl").exists():
        asr = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(run_dir / "asr_speech.jsonl"))}
    per, seams, blocks_all = [], [], []
    for r in rows:
        if r.get("error") or not r.get("blocks"):
            continue
        tgt, b = r["tgt_lang"], r["blocks"]
        text = r.get("translated_text") or "".join(x["text"] for x in b)
        nu = n_units(text, tgt)
        pauses, first, last, sounding = vad_pauses(r["audio_path"])
        types = seam_types(b, tgt, text)
        seam_t = [(x["start_ms"] + x["dur_ms"]) / 1000 for x in b[:-1]]
        nxt_t = [x["start_ms"] / 1000 for x in b[1:]]
        seam_pause = [0.0] * len(types)
        inner = []
        for p0, p1 in pauses:
            k = next((k for k, (a, c) in enumerate(zip(seam_t, nxt_t)) if p0 <= c + 0.05 and p1 >= a - 0.05), None)
            if k is None:
                inner.append(p1 - p0)
            else:
                seam_pause[k] = max(seam_pause[k], p1 - p0)
        for k, t in enumerate(types):
            seams.append({"type": t, "pause": seam_pause[k], "excess": max(0.0, seam_pause[k] - TOL[t]),
                          "after_source": nxt_t[k] > r["source_duration_ms"] / 1000})
        q_end = 0.0
        for k, x in enumerate(b):
            u = n_units(x["text"], tgt)
            queued = x["start_ms"] - x["write_ms"]
            blocks_all.append({"utt": len(per), "units": u, "queued_s": queued / 1000,
                               "s_per_unit": (x["dur_ms"] / 1000) / u if u else None, "forced": x.get("forced", False),
                               "closed": x.get("closed", True), "final": x["final"]})
            q_end = queued / 1000
        all_p = [p1 - p0 for p0, p1 in pauses]
        long_p = [p for p in all_p if p >= 0.25]
        toks = units(text, tgt)
        hyp_asr = asr.get(r["id"])
        rep = {}
        if hyp_asr is not None:
            norm = (lambda s: CJK.findall(s) + [w.lower() for w in WORD.findall(s)]) if tgt == "cmn" else \
                   (lambda s: [w.lower() for w in WORD.findall(s)])
            ref_t, hyp_t = norm(text), norm(hyp_asr)
            s, dl, ins = edit_ops(ref_t, hyp_t)
            extra = repeats(hyp_t) - repeats(ref_t)
            rep = {"talker_err": (s + dl + ins) / max(1, len(ref_t)), "talker_del": dl / max(1, len(ref_t)),
                   "talker_ins": ins / max(1, len(ref_t)), "stutter": sum(extra.values()), "stutter_ex": list(extra)[:3]}
        ref_rep = repeats(units(r.get("translation_ref", ""), tgt)) if r.get("translation_ref") else Counter()
        text_rep = repeats([w.lower() for w in toks]) - ref_rep
        per.append({
            "id": r["id"], "units": nu, "blocks": len(b), "source_s": r["source_duration_ms"] / 1000,
            "first_s": first, "last_s": last, "sounding_s": sounding, "span_s": last - first,
            "pauses": all_p, "inner_pauses": inner, "n_pause_25": len(long_p),
            "excess_s": sum(max(0.0, sp - TOL[t]) for sp, t in zip(seam_pause, types)) + sum(max(0.0, p - TOL["P"]) for p in inner),
            "tail_s": last - r["source_duration_ms"] / 1000, "q_end_s": q_end,
            "text_repeat": sum(text_rep.values()), "text_repeat_ex": list(text_rep)[:3], **rep,
            "seam_types": types, "seam_pause": [round(p, 2) for p in seam_pause],
            "block_text": [x["text"] for x in b], "translation": text, "ref": r.get("translation_ref", "")})
    return summarise(per, seams, blocks_all), per


def summarise(per: list[dict], seams: list[dict], blocks: list[dict]) -> dict:
    U = sum(p["units"] for p in per)
    pauses = [x for p in per for x in p["pauses"]]
    inner = [x for p in per for x in p["inner_pauses"]]
    # a block's pace against the run's own: seconds per unit over the median of blocks with >= 3 units (> 1: slower)
    pace = float(np.median([b["s_per_unit"] for b in blocks if b["units"] >= 3]))
    rel = [b["s_per_unit"] / pace for b in blocks if b["units"] >= 2]
    jumps = [abs(np.log(a["s_per_unit"] / c["s_per_unit"])) for a, c in zip(blocks, blocks[1:])
             if a["utt"] == c["utt"] and a["units"] >= 2 and c["units"] >= 2]
    queued = [b["queued_s"] for b in blocks if b["queued_s"] > 0.01]
    st = Counter(s["type"] for s in seams)
    seam_out = {}
    for t in ("S", "C", "P", "P-fw", "W"):
        xs = [s for s in seams if s["type"] == t]
        if xs:
            seam_out[t] = {"share": len(xs) / len(seams), "pause_ge_tol": sum(s["pause"] >= TOL[t] for s in xs) / len(xs),
                           "mean_pause_s": float(np.mean([s["pause"] for s in xs])),
                           "excess_s_per_100u": 100 * sum(s["excess"] for s in xs) / U}
    talker = [p for p in per if "talker_err" in p]
    return {
        "utterances": len(per), "units": U, "n_blocks": len(blocks),
        "speed": {
            "block_s_per_unit_p50": pace,
            "articulation_u_per_s": U / sum(p["sounding_s"] for p in per),
            "speech_rate_u_per_s": U / sum(p["span_s"] for p in per),
            "phonation_ratio": sum(p["sounding_s"] for p in per) / sum(p["span_s"] for p in per),
            "mean_length_of_run_u": U / sum(p["n_pause_25"] + 1 for p in per),
            "block_rel_dur_p10_p50_p90": [pct(rel, 10), pct(rel, 50), pct(rel, 90)],
            "blocks_rushed_lt0.75": float(np.mean([x < 0.75 for x in rel])),
            "blocks_dragged_gt1.33": float(np.mean([x > 1.33 for x in rel])),
            "adjacent_rate_jump_gt1.5x": float(np.mean([j > np.log(1.5) for j in jumps])) if jumps else None,
        },
        "breakdown": {
            **{f"pauses_ge{t}_per_100u": 100 * sum(x >= t for x in pauses) / U for t in (0.25, 0.5, 1.0, 2.0)},
            "pause_s_per_100u": 100 * sum(pauses) / U,
            "inner_block_pauses_ge0.25_per_100u": 100 * sum(x >= 0.25 for x in inner) / U,
            "excess_pause_burden_s_per_100u": 100 * sum(p["excess_s"] for p in per) / U,
            "utt_with_pause_ge1s": float(np.mean([any(x >= 1.0 for x in p["pauses"]) for p in per])),
            "seams": seam_out, "seam_after_source_share": float(np.mean([s["after_source"] for s in seams])) if seams else None,
        },
        "queue": {
            "blocks_queued": len(queued) / len(blocks), "queued_s_p50_p90_max": [pct(queued, 50), pct(queued, 90), pct(queued, 100)],
            "utt_lag_left_at_end_ge1s": float(np.mean([p["q_end_s"] >= 1.0 for p in per])),
            "tail_after_source_s_p50_p90": [pct([p["tail_s"] for p in per], 50), pct([p["tail_s"] for p in per], 90)],
            "first_sound_s_p50": pct([p["first_s"] for p in per], 50),
        },
        "blocks": {
            "per_utt": len(blocks) / len(per), "units_per_block_p50": pct([b["units"] for b in blocks], 50),
            "one_unit": float(np.mean([b["units"] == 1 for b in blocks])),
            "no_unit": float(np.mean([b["units"] == 0 for b in blocks])),
            "forced_close": float(np.mean([bool(b["forced"]) for b in blocks])),
        },
        "repair": {
            "talker_err": float(np.mean([p["talker_err"] for p in talker])) if talker else None,
            "talker_del": float(np.mean([p["talker_del"] for p in talker])) if talker else None,
            "talker_ins": float(np.mean([p["talker_ins"] for p in talker])) if talker else None,
            "utt_talker_err_gt0.2": float(np.mean([p["talker_err"] > 0.2 for p in talker])) if talker else None,
            "stutter_per_100u": 100 * sum(p.get("stutter", 0) for p in per) / U,
            "text_repeat_per_100u": 100 * sum(p["text_repeat"] for p in per) / U,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=si_infer output dir (holding results.jsonl)")
    ap.add_argument("--out", required=True, help="summary json; per-utterance rows go next to it as <name>.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for spec in args.run:
        name, d = spec.split("=", 1)
        summary[name], per = analyse(Path(d))
        with open(out.parent / f"{name}.jsonl", "x") as fh:
            for p in per:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(name, json.dumps(summary[name], ensure_ascii=False)[:400], flush=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
