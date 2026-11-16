"""E9 cascade (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S3): causal speaker relabelling of a
speaker-blind decode, as a new decode dir that the existing scorers / Talker chain read unchanged.

  decode/<src>/hyp.jsonl  ->  decode/<src>_rel/hyp.jsonl (+ relabel_stats.json)

Rule: reports/gstar_si_v8/overlap_score.relabel_causal, unchanged (prereg_v8 §3), called exactly as
overlap_score.score(relabel=True, online=True) calls it -- the window's post.lowlat diarizer posterior; segment span
a = window start + start_s, b = window start + max(end_s, start_s + 0.08); commit = the online-protocol commit
(write_ms, >= open_detect_ms, >= close_detect_ms when written at / after the window's last read) -- so each segment gets
the slot with the highest mean posterior over its span up to commit - 1.04 s, else the most active slot over
[a - 2 s, commit - 1.04 s], else "unk" (scored as an unmatched output speaker: overlap_score / score_ext exclude "unk"
from the speaker matching).

Per window: every segment's "spk" becomes its relabel (the model's own speaker token is kept as "spk_model"); every
streaming block (the Talker speaks blocks) takes the label of the segment it belongs to (same model speaker and start
time; the segment whose close is the first at or after the block's write; blocks at a segment's close read go to that
segment), so text and speech carry the same speaker. A block without a segment (no closing segment with its key) is
relabelled on its own (span [start_s, start_s + 0.08], its own commit). "raw", "translation", read logs: unchanged.
Counts (relabel_stats.json): segments, unk, label changes, blocks, blocks spoken before their segment's close (their
label was fixed at the close: the only place where a spoken block's label uses frames after its own commit), orphans.

    python -m experiments.mist_w3po_v1.evaluation.relabel_causal_run --src <decode name> --windows <online windows> \
        [--dst <name>] [--dec-root <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/gstar_si_v8"))
sys.path.insert(0, str(REPO / "reports/gstar_si_v6"))
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"


def commit_s(w: dict, write_ms: int) -> float:
    """overlap_score.score's online commit (meeting time, s)."""
    c = max(write_ms, w["open_detect_ms"])
    if write_ms >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return w["start_s"] + c / 1000


def relabel_window(w: dict, h: dict, post, rc, st: Counter) -> dict:
    segs = []
    for s in h["segments"]:
        seg = {"a": w["start_s"] + s["start_s"], "b": w["start_s"] + max(s["end_s"], s["start_s"] + 0.08)}
        k = rc(seg, post, w["start_s"], commit_s(w, s["write_ms"]))
        lab = k if k else "unk"
        st["segments"] += 1
        st["unk_segments"] += lab == "unk"
        st["changed_vs_model"] += lab != s["spk"]
        segs.append(dict(s, spk=lab, spk_model=s["spk"]))
    by_key = defaultdict(list)
    for i, s in enumerate(h["segments"]):
        by_key[(s["spk"], s["start_s"])].append(i)
    blocks = []
    for b in h.get("blocks", []):
        cand = by_key.get((b["spk"], b["start_s"]), [])
        j = next((i for i in cand if h["segments"][i]["write_ms"] >= b["write_ms"]), None)
        st["blocks"] += 1
        if j is None:
            a = w["start_s"] + (b["start_s"] or 0.0)
            k = rc({"a": a, "b": a + 0.08}, post, w["start_s"], commit_s(w, b["write_ms"]))
            lab = k if k else "unk"
            st["orphan_blocks"] += 1
        else:
            lab = segs[j]["spk"]
            st["blocks_before_segment_close"] += b["write_ms"] < h["segments"][j]["write_ms"]
            st["ambiguous_block_keys"] += len({segs[i]["spk"] for i in cand}) > 1
        st["unk_blocks"] += lab == "unk"
        blocks.append(dict(b, spk=lab, spk_model=b["spk"]))
    return dict(h, segments=segs, blocks=blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--dst", default="")
    ap.add_argument("--dec-root", default=str(DEC))
    a = ap.parse_args()
    from overlap_score import relabel_causal

    dec = Path(a.dec_root)
    src, dst = dec / a.src, dec / (a.dst or f"{a.src}_rel")
    if dst.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {dst}")
    W = {w["id"]: w for w in map(json.loads, open(a.windows, encoding="utf-8"))}
    H = [json.loads(l) for l in open(src / "hyp.jsonl", encoding="utf-8")]
    miss = [h["id"] for h in H if h["id"] not in W]
    if miss or len(H) != len(W):
        raise SystemExit(f"{src}/hyp.jsonl: {len(H)} lines for {len(W)} windows; {len(miss)} ids not in --windows")
    st = Counter()
    out = []
    for h in H:
        w = W[h["id"]]
        pp = (w.get("post") or {}).get("lowlat")
        post = np.load(pp) if pp and Path(pp).exists() else None
        st["windows"] += 1
        st["windows_without_post"] += post is None
        out.append(relabel_window(w, h, post, relabel_causal, st))
    dst.mkdir(parents=True)
    with open(dst / "hyp.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(h, ensure_ascii=False) + "\n" for h in out)
    spk = Counter(str(s["spk"]) for h in out for s in h["segments"])
    res = {"src": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src), "windows_file": a.windows,
           "rule": "reports/gstar_si_v8/overlap_score.relabel_causal (prereg_v8 §3), online commit, post.lowlat",
           "stats": dict(st), "spk_segments": dict(sorted(spk.items()))}
    with open(dst / "relabel_stats.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
