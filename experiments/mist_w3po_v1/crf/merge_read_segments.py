"""Read-level talker input (reports/mist_w3po_v1/thinker_joint_crf_prereg_v1.zh-CN.md §1.7; block_merge_bound_v1):
in every window, each speaker's non-empty segments written at the same read (same write_ms) and consecutive in that
speaker's own sequence become ONE segment (texts joined with a space), and likewise the hyp "blocks" the talker prep
reads (gstar_talker_v6 prep --blocks). The talker then plays one block per (read, speaker) -- J1's training block
B_{k,r} -- instead of one block per <= 4-word piece. Nothing else changes: write_ms, the first member's start_s, the
last member's end_s; raw / translation / read_log untouched; empty segments stay where they are; segments of other
speakers in between do not break a run (the talker's streams are per speaker).

Additive tool: a NEW decode dir <dst> with the merged hyp.jsonl and the source's score_text.json copied (the text
scores stay those of the original decode; thinker_t_eval_orch --speech-from <dst> runs only the speech chain on it).
Refuses to overwrite anything.

    python -m experiments.mist_w3po_v1.crf.merge_read_segments --src <decode name> --dst <new decode name> [--sep '']

--sep (additive, 2026-11-12, en2zh plan §3): the string that joins the member texts; default " " (unchanged); '' for
Chinese output (en2zh: the talker prep joins a slot's Chinese segments with '' as well).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEC = Path("eval_outputs/mist_w3po_v1/decode")


def merge_entries(entries: list[dict], end_key: str | None, sep: str = " ") -> list[dict]:
    """Merge, per speaker, the non-empty entries with the same write_ms that are consecutive in the list restricted to
    that speaker (a "run"). A run becomes one entry at its first member's list position; its members are ordered by
    (start_s or 0) (stable; the talker prep's order inside a read), text = the stripped member texts joined with one
    space, start_s = the first ordered member's, end_key (if given) = the members' maximum; other fields from the first
    ordered member. Empty-text entries are kept unchanged and do not break a run. Returns a new list (inputs untouched).
    A merged entry carries "_n" (members) for statistics; callers drop it."""
    runs: list[list[int]] = []
    cur: dict = {}             # spk -> index in runs of the speaker's current run
    slot: list = []            # per list position: ("e", i) empty entry, ("r", run index) first member of a run
    for i, e in enumerate(entries):
        if not e["text"].strip():
            slot.append(("e", i))
            continue
        k = e["spk"]
        j = cur.get(k)
        if j is not None and entries[runs[j][-1]]["write_ms"] == e["write_ms"]:
            runs[j].append(i)
            continue
        runs.append([i])
        cur[k] = len(runs) - 1
        slot.append(("r", len(runs) - 1))
    out = []
    for kind, x in slot:
        if kind == "e":
            out.append(dict(entries[x]))
            continue
        mem = sorted(runs[x], key=lambda i: (entries[i].get("start_s") or 0.0, i))
        m = dict(entries[mem[0]])
        m["text"] = sep.join(entries[i]["text"].strip() for i in mem)
        if end_key is not None:
            m[end_key] = max(entries[i][end_key] for i in mem)
        m["_n"] = len(mem)
        out.append(m)
    return out


def merge_hyp(h: dict, sep: str = " ") -> tuple[dict, dict]:
    """-> (merged hyp record, stats). Segments and blocks are merged by the same rule, each in its own list order (a
    speaker's same-read entries are consecutive in both), so a merged block has the merged segment with the same
    (spk, start_s) and text whenever the blocks are the segments (check_pair)."""
    h2 = dict(h)
    segs = merge_entries(h.get("segments", []), "end_s", sep)
    st = {"seg_in": sum(1 for s in h.get("segments", []) if s["text"].strip()),
          "seg_out": sum(1 for s in segs if s["text"].strip()),
          "seg_merged_members": sum(s.get("_n", 1) - 1 for s in segs)}
    for s in segs:
        s.pop("_n", None)
    h2["segments"] = segs
    if h.get("blocks"):
        bm = merge_entries(h["blocks"], None, sep)
        for b in bm:
            b.pop("_n", None)
        h2["blocks"] = bm
        st["blk_in"] = sum(1 for b in h["blocks"] if b["text"].strip())
        st["blk_out"] = sum(1 for b in bm if b["text"].strip())
    return h2, st


def check_pair(h2: dict) -> bool:
    """every non-empty merged block has a merged segment with the same (spk, start_s) and text (what the talker prep
    and speech_latency_lit rely on)."""
    seg = {(s["spk"], s["start_s"]): s["text"] for s in h2["segments"] if s["text"].strip()}
    return all(seg.get((b["spk"], b["start_s"])) == b["text"] for b in h2.get("blocks", []) if b["text"].strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sep", default=" ", help="joins the member texts (default ' '; en2zh: '')")
    a = ap.parse_args()
    src, dst = DEC / a.src, DEC / a.dst
    if dst.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {dst}")
    if not (src / "hyp.jsonl").exists() or not (src / "score_text.json").exists():
        raise SystemExit(f"{src}: hyp.jsonl / score_text.json missing")
    dst.mkdir(parents=True)
    tot = {"windows": 0, "pair_ok": 0, "src_pair_ok": 0, "pair_lost": 0}
    with open(dst / "hyp.jsonl", "x", encoding="utf-8") as fh:
        for l in open(src / "hyp.jsonl", encoding="utf-8"):
            h2, st = merge_hyp(json.loads(l), a.sep)
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
            tot["windows"] += 1
            ok, ok0 = check_pair(h2), check_pair(json.loads(l))
            tot["pair_ok"] += ok
            tot["src_pair_ok"] += ok0
            tot["pair_lost"] += ok0 and not ok
            fh.write(json.dumps(h2, ensure_ascii=False) + "\n")
    with open(src / "score_text.json", "rb") as fi, open(dst / "score_text.json", "xb") as fo:
        shutil.copyfileobj(fi, fo)
    tot["src"], tot["note"] = str(src), "read-level talker input (merge_read_segments.py); text scores = the source's"
    if a.sep != " ":
        tot["sep"] = a.sep
    with open(dst / "merge_read_segments.json", "x", encoding="utf-8") as fh:
        json.dump(tot, fh, indent=1)
    print(json.dumps(tot))
    if tot["pair_lost"]:   # windows whose source blocks pair with its segments must still pair after the merge
        raise SystemExit(f"block / segment pairing lost in {tot['pair_lost']} windows")


if __name__ == "__main__":
    main()
