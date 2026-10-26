"""Where each commit policy's errors come from, on RealSI's text side and through the talker (CPU only).

Every committed unit (a Chinese character, an English word) is set against two references that do not depend
on the policy: the frozen Thinker's own full-context translation (realsi_thinker_offline_check, the same model
with the whole sentence heard) and RealSI's reference translation with the time its source support was heard
(realsi_align_v1). A unit committed before the sentence ends is

  divergent    -- absent from the full-context translation: the model with more audio would not have said it;
  ahead        -- its place in the full-context translation runs past the share of source words heard (overrun);
  hallucinated -- divergent, absent from the reference too, and ahead: a completion of a sentence not yet heard.

Beside these, each policy's fragments (units per early block), sentence closures made while more than a second of
source is still to come, and -- through every talker seed of its full-system run -- the committed units the
talker's speech leaves out (ASR against the text it was given).

    python si_commit_diag.py --out <json>
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
E = REPO / "eval_outputs/uniss_qwen3_omni_v1"
STREAM = E / "realsi_thinker_stream_v1"
OFFLINE = E / "realsi_thinker_offline_check/base"
ALIGN = E / "realsi_align_v1"
SYSTEMS = E / "si_t1c_20261021T181440Z/full_iter0001500"
# policy: (block-file suffix at m1, full-system run kind); the main CRISP point differs by direction
POLICIES = {
    "la2": ("", "la2"),
    "hold": ("_hold", "hold"),
    "vote8g0.6": ("_vote8g0.6", "vote8g0.6"),
    "vote8g0.7": ("_vote8g0.7", "vote8g0.7"),
    "head-s0-t0.25": ("_headhead_cons_s0t0.25", "head-cons-s0-t0.25"),
    "head-all-t0.35": ("_headhead_cons_allt0.35", "head-cons-all-t0.35"),
}
END_PUNCT = tuple("。！？.!?")


def units(text: str, zh: bool) -> list[str]:
    if zh:
        return re.findall(r"[一-鿿]|[A-Za-z0-9]+", text)
    return re.findall(r"[a-z0-9']+", text.lower())


def matched(a: list[str], b: list[str]) -> list[int]:
    """Index into b of each unit of a that the alignment keeps, -1 for the rest."""
    out = [-1] * len(a)
    for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(m.size):
            out[m.a + k] = m.b + k
    return out


def jl(path: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def text_side(d: str, suffix: str, offline: dict, words: dict, align: dict, zh: bool) -> tuple[dict, dict]:
    path = STREAM / f"blocks_{d}_w1{suffix}.jsonl"
    early = defaultdict(float)
    per_item = {}
    for it in map(json.loads, open(path, encoding="utf-8")):
        dur = it["drafts"][-1]["t"]
        off = units(offline[it["id"]]["translation"], zh)
        src = [w["end_ms"] for w in words.get(it["id"], {}).get("source_words", [])]
        ref, sup = [], []
        a = align.get(it["id"])
        if a:
            for w, s in zip(a["target_words"], a["support_ms"]):
                u = units(w, zh)
                ref += u
                sup += [s] * len(u)
        out, when, blocks = [], [], []
        for b in it["blocks"]:
            u = units(b["text"], zh)
            out += u
            when += [b["write_ms"]] * len(u)
            if b["write_ms"] < dur and u:
                blocks.append((b["write_ms"], len(u), b["text"].rstrip()))
        m_off, m_ref = matched(out, off), matched(out, ref)
        n_early = 0
        for i, (u, t) in enumerate(zip(out, when)):
            early["units"] += 1
            early["time_sum"] += t / 1000
            if t >= dur:
                continue
            n_early += 1
            heard = sum(e <= t for e in src) / len(src) if src else t / dur
            ahead = (i + 1) / max(len(off), 1) - heard > 0.1
            div = m_off[i] < 0
            early["early"] += 1
            early["div"] += div
            early["ahead"] += ahead
            early["ahead_div"] += ahead and div
            early["halluc"] += div and m_ref[i] < 0 and ahead
            if m_ref[i] >= 0:
                early["ref_matched"] += 1
                early["anticipated"] += t < sup[m_ref[i]]
        later = [b for b in it["blocks"] if b["text"].strip()]
        for k, (t, n, txt) in enumerate(blocks):
            early["blocks"] += 1
            early["block_units"] += n
            early["tiny"] += n <= (3 if zh else 2)
            early["closure"] += txt.endswith(END_PUNCT) and t <= dur - 1000 and later[-1]["write_ms"] > t
        early["items"] += 1
        early["first_s"] += (it["blocks"][0]["write_ms"] if it["blocks"] else dur) / 1000
        per_item[it["id"]] = {"early_units": n_early, "blocks": len(blocks),
                              "mean_block": statistics.mean([n for _, n, _ in blocks]) if blocks else 0.0}
    return dict(early), per_item


def speech_side(d: str, kind: str, zh: bool) -> list[dict]:
    rows = []
    for run in sorted(SYSTEMS.glob(f"system_{kind}*_m1")):
        tail = run.name[len(f"system_{kind}"):-len("_m1")]
        if tail and not re.fullmatch(r"-seed\d+", tail):
            continue
        asr = run / f"realsi_{d}/asr.jsonl"
        if not asr.exists():
            continue
        for r in map(json.loads, open(asr, encoding="utf-8")):
            said = units(r.get("translated_text") or "", zh)
            heard = units(r.get("asr_text") or "", zh)
            if said:
                kept = sum(j >= 0 for j in matched(said, heard))
                rows.append({"id": r["id"], "seed": tail or "-seed0", "omit": 1 - kept / len(said)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from evaluation.text_metrics import corpus_bleu

    report = {}
    for d in ("en2zh", "zh2en"):
        zh = d.endswith("zh")
        offline = jl(OFFLINE / f"blocks_{d}_w60.jsonl")
        words, align = jl(ALIGN / f"words_{d}.jsonl"), jl(ALIGN / f"align_{d}.jsonl")
        refs = {}
        for f in (SYSTEMS / f"system_la2_m1/realsi_{d}/asr.jsonl",):
            refs = {r["id"]: r["translation_ref"] for r in map(json.loads, open(f, encoding="utf-8"))}
        print(f"\n## {d}\n")
        print("| 策略 | 文本BLEU | 语音ASR-BLEU | 首次提交 s | 单位平均提交 s | 提前提交单位 | 与全上下文译文不一致 | "
              "超前于源(>10%) | 超前单位中不一致 | 脑补(不一致∧参考里没有∧超前) | 参考匹配单位中先于源支撑 | "
              "早期块平均单位数 | 碎块(≤2词/≤3字) | 提前断句/句 | talker漏读 |")
        print("|" + "---|" * 15)
        frag_omit = []
        for name, (suffix, kind) in POLICIES.items():
            if not (STREAM / f"blocks_{d}_w1{suffix}.jsonl").exists():
                continue
            c, per_item = text_side(d, suffix, offline, words, align, zh)
            hyps = {r["id"]: r["translation"] for r in map(json.loads, open(STREAM / f"blocks_{d}_w1{suffix}.jsonl", encoding="utf-8"))}
            ids = [i for i in hyps if i in refs]
            tb = corpus_bleu([hyps[i] for i in ids], [refs[i] for i in ids], language="cmn" if zh else "eng")["score"]
            sp = speech_side(d, kind, zh)
            seeds = sorted({r["seed"] for r in sp})
            asr_b = []
            for s in seeds:
                f = SYSTEMS / f"system_{kind}{'' if s == '-seed0' else s}_m1/realsi_{d}/metrics.json"
                if f.exists():
                    asr_b.append(json.load(open(f))["asr_bleu"])
            omit = statistics.mean(r["omit"] for r in sp) if sp else float("nan")
            for r in sp:
                if r["id"] in per_item and per_item[r["id"]]["blocks"]:
                    frag_omit.append((per_item[r["id"]]["mean_block"], r["omit"], name))
            e = max(c.get("early", 0), 1)
            row = {"text_bleu": round(tb, 2), "asr_bleu": [round(x, 2) for x in asr_b],
                   "first_commit_s": c["first_s"] / c["items"], "unit_commit_s": c["time_sum"] / c["units"],
                   "early_share": c.get("early", 0) / c["units"], "divergent": c.get("div", 0) / e,
                   "ahead": c.get("ahead", 0) / e, "div_given_ahead": c.get("ahead_div", 0) / max(c.get("ahead", 0), 1),
                   "div_given_not_ahead": (c.get("div", 0) - c.get("ahead_div", 0)) / max(e - c.get("ahead", 0), 1),
                   "hallucinated": c.get("halluc", 0) / e,
                   "anticipated": c.get("anticipated", 0) / max(c.get("ref_matched", 0), 1),
                   "units_per_block": c.get("block_units", 0) / max(c.get("blocks", 0), 1),
                   "tiny_blocks": c.get("tiny", 0) / max(c.get("blocks", 0), 1),
                   "closure_per_item": c.get("closure", 0) / c["items"], "talker_omit": omit, "seeds": len(seeds)}
            report[f"{d}/{name}"] = row
            ab = f"{statistics.mean(asr_b):.2f}±{statistics.stdev(asr_b):.2f}" if len(asr_b) > 1 else (f"{asr_b[0]:.2f}" if asr_b else "—")
            print(f"| {name} | {tb:.2f} | {ab} | {row['first_commit_s']:.2f} | {row['unit_commit_s']:.2f} | "
                  f"{row['early_share']:.1%} | {row['divergent']:.1%} | {row['ahead']:.1%} | {row['div_given_ahead']:.1%}"
                  f"（非超前 {row['div_given_not_ahead']:.1%}） | {row['hallucinated']:.1%} | {row['anticipated']:.1%} | "
                  f"{row['units_per_block']:.1f} | {row['tiny_blocks']:.1%} | {row['closure_per_item']:.2f} | {omit:.1%} |")
        bins = [(0, 2), (2, 4), (4, 8), (8, 1e9)] if zh else [(0, 1.5), (1.5, 3), (3, 6), (6, 1e9)]
        print("\n平均早期块长度 → talker 漏读（所有策略、所有种子合并）")
        for lo, hi in bins:
            v = [o for m, o, _ in frag_omit if lo <= m < hi]
            if v:
                print(f"  [{lo:g},{hi:g}) 单位: n={len(v)} 漏读 {statistics.mean(v):.1%}")
                report[f"{d}/omit_by_block_{lo:g}_{hi:g}"] = {"n": len(v), "omit": statistics.mean(v)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
