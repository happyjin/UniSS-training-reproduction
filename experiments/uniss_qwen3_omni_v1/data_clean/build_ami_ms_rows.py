"""E2E-v0 rows (docs/multi_speaker_si_plan_v2.zh-CN.md): the Thinker taught to interpret a multi-speaker meeting
mixture, writing each speaker's translation behind a label it assigns itself.

Pieces are AMI utterance groups (utterances that overlap or touch, at most --max-piece s), cut from the IHM-mix
session audio, kept only when every utterance in them passed the data review. --review text (default): a
translation exists and its translated wav decodes and is not silent -- the text side, all E2E-v0 consumes;
--review strict also requires the wav not to be over-long and Whisper to hear it (CER <= --max-cer), which drops
most multi-speaker pieces (one misheard short back-channel fails the piece). Every piece records whether it passes
the strict review, so the strict subset can be scored on its own. Pure hesitations (UM, UH, MM-HMM ...) are not
translated and need no review. Each translation is cut at its punctuation into chunks; a chunk
is committed --lag s after the source word that ends it is heard (AMI word times; proportional inside the
utterance when the words are missing). The output serialises the chunks in commit order; each change of speaker
opens with the label "[S<k>@<t>]" -- speakers numbered by first appearance in the piece, t the second (piece
clock) where the source of that chunk starts.

Rows are the selfwait rows of build_thinker_selfdistill with one more field: {"id", "audio", "dir", "t_ms",
"dur_ms", "prefix", "target", "end": "wait" | "eos", "kind": "read" | "offline", "inst": "ms"}. A share of
pretrain rows of Plan 4A ("inst": "sd") keeps the single-speaker behaviour.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows --out data/ami_multispk_si_v1/e2e_v0
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import soundfile as sf

HESITATIONS = {"UM", "UH", "HMM", "MM", "MM-HMM", "UH-HUH", "HM", "ER", "ERM", "MHM", "UM-HUM", "UH-UH", "MM-MM", "AH"}
PUNCT = "，。？！；、,.?!;"
TAG = "[S{k}@{t:.1f}]"


def is_hesitation(text: str) -> bool:
    w = re.findall(r"[A-Z'\-]+", (text or "").upper())
    return bool(w) and all(x in HESITATIONS for x in w)


def zh_len(s: str) -> int:
    return len(re.sub(rf"[\s{re.escape(PUNCT)}]", "", s or ""))


def chunks_of(zh: str) -> list[str]:
    out = re.findall(rf"[^{re.escape(PUNCT)}]+[{re.escape(PUNCT)}]*", zh.strip())
    return [c for c in out if zh_len(c)] or ([zh.strip()] if zh.strip() else [])


def load_words(manual_zip: Path) -> dict:
    """meeting -> speaker_id -> sorted [(start, end)] of spoken (non-punctuation) words."""
    z = zipfile.ZipFile(manual_zip)
    meet = ET.fromstring(z.read("corpusResources/meetings.xml"))
    agent = {}
    for m in meet.iter("meeting"):
        for s in m.iter("speaker"):
            agent[(m.get("observation"), s.get("nxt_agent"))] = s.get("global_name")
    words = defaultdict(lambda: defaultdict(list))
    for name in z.namelist():
        mm = re.fullmatch(r"words/([A-Z]{2}\d{4}[a-z]?)\.([A-Z])\.words\.xml", name)
        if not mm or (mm.group(1), mm.group(2)) not in agent:
            continue
        spk = agent[(mm.group(1), mm.group(2))]
        for w in ET.fromstring(z.read(name)).iter("w"):
            if w.get("punc") or w.get("starttime") is None or w.get("endtime") is None:
                continue
            words[mm.group(1)][spk].append((float(w.get("starttime")), float(w.get("endtime"))))
    for m in words.values():
        for v in m.values():
            v.sort()
    return words


def review(data: Path, max_cer: float) -> dict:
    """record_id -> (passed, translation, reason)."""
    tg = {r["record_id"]: r for r in pads.dataset(data / "index/chain_en_zh_ihm", format="parquet", partitioning="hive")
          .to_table(columns=["record_id", "translation_text"]).to_pylist()}
    chk = {}
    for line in open(data / "reports/audio_check_rows.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["side"] == "target":
            chk[x["record_id"]] = x
    cer = {}
    for r in pads.dataset(data / "annotations/ihm_all/tts_whisper_error_rates", format="parquet", partitioning="hive") \
            .to_table(columns=["record_id", "payload_json"]).to_pylist():
        try:
            cer[r["record_id"]] = json.loads(r["payload_json"]).get("whisper_cer")
        except (TypeError, ValueError):
            pass
    out = {}
    for rid, r in tg.items():
        zh = (r["translation_text"] or "").strip()
        c, a = cer.get(rid), chk.get(rid, {})
        if not zh:
            out[rid] = (False, zh, "no_text")
        elif not a.get("ok"):
            out[rid] = (False, zh, "silent_or_bad_audio")
        elif a.get("dur_ms", 0) > 3 * 250 * max(zh_len(zh), 1) + 1500:
            out[rid] = (False, zh, "over_long")
        elif c is None:
            out[rid] = (False, zh, "no_cer")
        elif c > max_cer:
            out[rid] = (False, zh, "cer")
        else:
            out[rid] = (True, zh, "ok")
    return out


def utt_chunks(u: dict, words: list, lag_ms: int, t0: float) -> list[dict]:
    """Chunks of one utterance with the piece-clock source start/end of each and its commit time."""
    ws = [w for w in words if w[0] >= u["begin_time"] - 0.05 and w[1] <= u["end_time"] + 0.05]
    cs = chunks_of(u["zh"])
    total = sum(zh_len(c) for c in cs)
    out, acc, prev_end = [], 0, u["begin_time"]
    for c in cs:
        acc += zh_len(c)
        f = acc / total
        if ws:
            end = ws[max(0, math.ceil(f * len(ws)) - 1)][1]
        else:
            end = u["begin_time"] + f * (u["end_time"] - u["begin_time"])
        end = max(end, prev_end)
        out.append({"text": c, "src_start_ms": round((prev_end - t0) * 1000), "src_end_ms": round((end - t0) * 1000),
                    "commit_ms": round((end - t0) * 1000) + lag_ms})
        prev_end = end
    return out


def serialise(chunks: list[dict]) -> list[dict]:
    """Chunks in commit order, each with the text it adds to the reply (a label where the speaker changes)."""
    order = sorted(chunks, key=lambda c: (c["commit_ms"], c["src_start_ms"]))
    last, out = None, []
    for c in order:
        head = TAG.format(k=c["label"], t=c["src_start_ms"] / 1000) if c["label"] != last else ""
        out.append({**c, "piece_text": head + c["text"]})
        last = c["label"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--channel", default="ihm_mix", choices=["ihm_mix"])
    ap.add_argument("--manual", default="/opt/dlami/nvme/jasonleeeli/datasets/auden_ami_sdm/ami_public_manual_1.6.2.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=1.0, help="read period, s")
    ap.add_argument("--lag", type=float, default=1.0, help="commit this long after the chunk's last source word, s")
    ap.add_argument("--max-piece", type=float, default=20.0)
    ap.add_argument("--max-cer", type=float, default=0.3)
    ap.add_argument("--review", choices=["text", "strict"], default="text")
    ap.add_argument("--offline-share", type=float, default=0.2, help="pieces that also get a whole-piece row")
    ap.add_argument("--keep-empty", type=float, default=0.5, help="share of reads with nothing to commit that are kept")
    ap.add_argument("--sd-rows", default="data/si_thinker_selfdistill_v2/rows_train.jsonl")
    ap.add_argument("--sd-share", type=float, default=0.15, help="Plan 4A rows added, as a share of the AMI rows")
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    split_of = json.load(open(data / "meetings.json"))
    src = pads.dataset(data / "index/source_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "original_split", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript"]).to_pylist()
    rev = review(data, args.max_cer)
    words = load_words(Path(args.manual))
    lag_ms, win_ms = round(args.lag * 1000), round(args.window * 1000)
    by_meet = defaultdict(list)
    for r in src:
        by_meet[r["meeting_id"]].append(r)
    (out / "audio").mkdir(parents=True)
    stats = defaultdict(lambda: defaultdict(int))
    pieces = defaultdict(list)
    for meeting in sorted(by_meet):
        split = split_of[meeting]
        sess = data / "session_audio" / args.channel / f"{meeting}.Mix-Headset.wav"
        if not sess.exists():
            stats[split]["meetings_without_audio"] += 1
            continue
        info = sf.info(str(sess))
        assert info.samplerate == 16000, sess
        us = sorted(by_meet[meeting], key=lambda r: r["begin_time"])
        groups, cur, end = [], [], -1.0
        for u in us:
            if cur and (u["begin_time"] > end or u["end_time"] - cur[0]["begin_time"] > args.max_piece):
                groups.append(cur)
                cur = []
            cur.append(u)
            end = max(end, u["end_time"]) if len(cur) > 1 else u["end_time"]
        if cur:
            groups.append(cur)
        x = None
        for gi, g in enumerate(groups):
            s = stats[split]
            s["pieces"] += 1
            for u in g:
                u["skip"] = is_hesitation(u["transcript"])
                ok, zh, why = rev.get(u["record_id"], (False, "", "no_translation"))
                u["strict"], u["zh"], u["why"] = ok, zh, why
                u["passed"] = ok if args.review == "strict" else why in ("ok", "cer", "over_long", "no_cer")
                if not u["skip"]:
                    s["utt_" + why] += 1
            if not any(not u["skip"] for u in g):
                s["pieces_only_hesitation"] += 1
                continue
            if any(not u["skip"] and not u["passed"] for u in g):
                s["pieces_failed_review"] += 1
                continue
            g_start, g_end = min(u["begin_time"] for u in g), max(u["end_time"] for u in g)
            if g_end - g_start > args.max_piece * 1.5:
                s["pieces_too_long"] += 1
                continue
            prev_end = max(v["end_time"] for v in groups[gi - 1]) if gi else 0.0
            next_beg = min(v["begin_time"] for v in groups[gi + 1]) if gi + 1 < len(groups) else info.duration
            t0 = max(0.0, g_start - min(0.3, max(0.0, g_start - prev_end) / 2))
            t1 = min(info.duration, g_end + min(0.5, max(0.0, next_beg - g_end) / 2))
            if x is None:
                x, _ = sf.read(str(sess), dtype="float32", always_2d=True)
                x = x.mean(axis=1)
            pid = f"{meeting}_{gi:04d}"
            wav = out / "audio" / split / f"{pid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav), x[round(t0 * 16000): round(t1 * 16000)], 16000)
            dur_ms = round((t1 - t0) * 1000)
            talk = [u for u in g if not u["skip"]]
            first = {}
            for u in sorted(talk, key=lambda u: u["begin_time"]):
                first.setdefault(u["speaker_id"], len(first) + 1)
            chunks = []
            for u in talk:
                for c in utt_chunks(u, words.get(meeting, {}).get(u["speaker_id"], []), lag_ms, t0):
                    chunks.append({**c, "label": first[u["speaker_id"]], "speaker_id": u["speaker_id"],
                                   "record_id": u["record_id"]})
            for c in chunks:
                c["commit_ms"] = min(c["commit_ms"], dur_ms)
            ser = serialise(chunks)
            spans = [(round((u["begin_time"] - t0) * 1000), round((u["end_time"] - t0) * 1000)) for u in g]
            overlap = sum(max(0, min(a[1], b[1]) - max(a[0], b[0])) for i, a in enumerate(spans) for b in spans[i + 1:])
            pieces[split].append({
                "id": pid, "meeting": meeting, "split": split, "audio": str(wav.resolve()), "dur_ms": dur_ms,
                "offset_ms": round(t0 * 1000), "n_spk": len(first), "overlap_ms": overlap,
                "strict_review": all(u["strict"] for u in talk),
                "speakers": {f"S{k}": spk for spk, k in first.items()},
                "utts": [{"record_id": u["record_id"], "speaker_id": u["speaker_id"], "label": first.get(u["speaker_id"]),
                          "begin_ms": round((u["begin_time"] - t0) * 1000), "end_ms": round((u["end_time"] - t0) * 1000),
                          "en": u["transcript"], "zh": "" if u["skip"] else u["zh"], "hesitation": u["skip"]} for u in g],
                "serialised": [{k: c[k] for k in ("label", "speaker_id", "text", "piece_text", "src_start_ms",
                                                  "src_end_ms", "commit_ms")} for c in ser],
                "reference": "".join(c["piece_text"] for c in ser),
                "ref_by_label": {f"S{k}": "".join(c["text"] for c in sorted(
                    (c for c in ser if c["label"] == k), key=lambda c: c["src_start_ms"])) for k in first.values()},
            })
            s["pieces_kept"] += 1
            s["kept_multi_spk"] += len(first) > 1
            s["kept_ms"] += dur_ms
        print(f"{meeting} {split}: {dict(stats[split])}", flush=True)

    def rows_of(p: dict) -> list[dict]:
        ser, dur = p["serialised"], p["dur_ms"]
        reads = list(range(win_ms, dur, win_ms)) + [dur]
        rows, prev = [], 0
        for t in reads:
            done = [c for c in ser if c["commit_ms"] <= prev]
            new = [c for c in ser if prev < c["commit_ms"] <= t] if t < dur else [c for c in ser if c["commit_ms"] > prev]
            target = "".join(c["piece_text"] for c in new)
            last = t >= dur
            if target or last or rng.random() < args.keep_empty:
                rows.append({"id": p["id"], "audio": p["audio"], "dir": "en2zh", "t_ms": t, "dur_ms": dur,
                             "prefix": "".join(c["piece_text"] for c in done), "target": target,
                             "end": "eos" if last else "wait", "kind": "read", "inst": "ms"})
            prev = t
        if rng.random() < args.offline_share:
            rows.append({"id": p["id"], "audio": p["audio"], "dir": "en2zh", "t_ms": dur, "dur_ms": dur, "prefix": "",
                         "target": p["reference"], "end": "eos", "kind": "offline", "inst": "ms"})
        return rows

    summary = {}
    for split in ("train", "validation", "test"):
        with open(out / f"pieces_{split}.jsonl", "x", encoding="utf-8") as fh:
            for p in pieces[split]:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(out / f"items_{split}.jsonl", "x", encoding="utf-8") as fh:   # si_infer / stream items
            for p in pieces[split]:
                fh.write(json.dumps({"id": p["id"], "audio": p["audio"], "dur_ms": p["dur_ms"],
                                     "translation": p["reference"]}, ensure_ascii=False) + "\n")
        summary[split] = {**stats[split], "kept_h": round(stats[split]["kept_ms"] / 3.6e6, 2)}
    train = [r for p in pieces["train"] for r in rows_of(p)]
    val = [r for p in pieces["validation"] for r in rows_of(p)]
    n_sd = 0
    if args.sd_share > 0 and Path(args.sd_rows).exists():
        sd = [json.loads(l) for l in open(args.sd_rows, encoding="utf-8")]
        pick = rng.sample(sd, min(len(sd), round(args.sd_share * len(train))))
        train += [{**r, "inst": "sd"} for r in pick]
        n_sd = len(pick)
    rng.shuffle(train)
    with open(out / "rows_train.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
    with open(out / "rows_val.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in val)
    summary["rows"] = {"train": len(train), "train_sd": n_sd, "val": len(val),
                       "train_wait_empty": sum(1 for r in train if r.get("inst") == "ms" and not r["target"]),
                       "train_offline": sum(1 for r in train if r.get("kind") == "offline" and r.get("inst") == "ms")}
    summary["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
