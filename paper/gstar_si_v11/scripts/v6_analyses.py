"""Paper v6 analyses that need no new decoding (review of v5: M01.4, M06.1, M06.4, M09.1, M10.1, M13, T01, T03, T10).

  flush      share of committed units written at a window's last read (the forced end-of-window commit), per
             streaming system; and in the deadline labels, the share of chunks whose deadline fell after the window end
  filter     which test speech the reference windows keep: utterances / duration kept, by kind (<= 1 s, overlapped,
             in a dropped > 20 s group or straddling a window, failed review)
  sameb      T01: segments whose start and end time tokens coincide
  relabel    M06.1: the no-speaker-stream model's outputs relabelled after the fact -- each segment gets the diarizer
             slot with the highest mean posterior over its predicted span (same posteriors as MIST reads, no reference)
             -> eval_outputs/.../gstar_v6_20261031/<dir>_O2-nosd-relabel/hyp.jsonl, scored with gstar_score
  segments   M13 / M06.4 / T03 / T10: per reference utterance, the output segments overlapping it (0.5 s collar):
             covered (one of them is the right speaker's, after the session matching), misattributed (only others'),
             omitted (none); attribution precision = segments overlapping an utterance of their matched speaker;
             nearest onset / offset error (median); repeated segments; insertions (segments overlapping no utterance); per-speaker LAAL and end lag of streaming output; identity by meeting third and on re-entry
             after >= 60 s of silence; end lag of utterances nested in another speaker's turn (head-of-line)
  voices     M09.1: causal voice prompts of the streaming output -- available (>= 1 s heard alone before the slot's
             first commit) or fallback, length, wait from the slot's first activity

Writes generated/v6_extras.json (keys per analysis; an existing key is kept unless --redo names it).

    python scripts/v6_analyses.py flush filter sameb segments voices      (from the paper folder)
    python scripts/v6_analyses.py relabel
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems import hyp_dir, windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units  # noqa: E402

GEN = Path(__file__).resolve().parents[1] / "generated"
EV = ROOT / "eval_outputs/uniss_qwen3_omni_v1"
V6 = EV / "gstar_v6_20261031"
D = ROOT / "data/ami_multispk_si_v1"
A = ROOT / "data/alimeeting_si_v1"
COLLAR = 2.0
LANG = {"en2zh": "cmn", "zh2en": "eng", "ali": "eng"}
SEP = {"en2zh": "", "zh2en": " ", "ali": " "}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def ref_text(u):
    return u.get("ref", u["zh"]).strip()


def system(dr, name):
    d = hyp_dir(dr, name)
    if d is None or not (d / "hyp.jsonl").exists():
        return None, None
    H = {h["id"]: h for h in load(d / "hyp.jsonl")}
    W = [w for w in load(windows(dr, name)) if w["id"] in H]
    return W, H


def session_map(dr, W, H):
    """Per meeting one matching of output slots to reference speakers (as gstar_score)."""
    hm, rm = defaultdict(lambda: defaultdict(str)), defaultdict(lambda: defaultdict(str))
    for w in W:
        for s in H[w["id"]]["segments"]:
            hm[w["meeting"]][s["spk"]] += SEP[dr] + s["text"]
        for u in sorted(w["utts"], key=lambda u: u["begin_s"]):
            if not u["hesitation"]:
                rm[w["meeting"]][u["speaker_id"]] += SEP[dr] + ref_text(u)
    return {m: match(dict(hm[m]), dict(rm[m])) for m in rm}


# ---------------------------------------------------------------------------------------------------------- flush
STREAM_SYS = {"en2zh": ["S2dl_d1", "S2dl_d2", "S2dl_d0.5", "S2dlm_d1", "O2_LA2_d1", "S2_b0_d1"],
              "zh2en": ["S2dl_d1", "S2dl_d2", "O2_LA2_d1"], "ali": ["S2dl_d1", "O2_LA2_d1"]}


def flush():
    out = {}
    for dr, names in STREAM_SYS.items():
        for name in names:
            W, H = system(dr, name)
            if W is None:
                continue
            tot = fin = 0
            for w in W:
                for b in H[w["id"]].get("blocks", []):
                    n = len(units(b["text"], LANG[dr]))
                    tot += n
                    fin += n if b["write_ms"] >= w["dur_ms"] else 0
            out[f"{dr}:{name}"] = {"units": tot, "final_read_share": round(fin / max(tot, 1), 4)}
    # deadline labels: chunks whose deadline (source end + 1 s lag) is after the window end, i.e. only the last
    # read can commit them; recomputed on the training windows with the builders' rules
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import utt_chunks
    for dr, src in (("zh2en", D / "e2e_gstar_zh2en_v1/windows_train_ref.jsonl"),):
        n = late = 0
        for w in load(src):
            for u in w["utts"]:
                if u["hesitation"] or not u.get("ref", "").strip():
                    continue
                uu = {"begin_time": w["start_s"] + u["begin_s"], "end_time": w["start_s"] + u["end_s"], "zh": u["ref"]}
                for c in utt_chunks(uu, [], 1000, w["start_s"]):
                    n += 1
                    late += c["commit_ms"] > w["dur_ms"]
        out[f"labels:{dr}"] = {"chunks": n, "deadline_after_window_end": round(late / max(n, 1), 4)}
    rows = D / "e2e_gstar_stream_v1/rows_train.jsonl"   # En->Zh: chunks in the final read's target vs all read targets
    import re
    fin = tot = 0
    for r in map(json.loads, open(rows, encoding="utf-8")):
        if r.get("kind") != "read":
            continue
        k = len(units(re.sub(r"<\|[^|]*\|>", "", r["target"]), "cmn"))
        tot += k
        fin += k if r["end"] == "eos" else 0
    out["labels:en2zh_final_read_target_share"] = round(fin / max(tot, 1), 4)
    return out


# --------------------------------------------------------------------------------------------------------- filter
def utterances(dr):
    """meeting -> non-hesitation test utterances [(record_id, speaker, begin, end)] on that meeting's clock."""
    out = defaultdict(list)
    if dr == "en2zh":
        import pyarrow.dataset as pads
        from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import is_hesitation
        split = json.load(open(D / "meetings.json"))
        src = pads.dataset(D / "index/source_ihm", format="parquet", partitioning="hive").to_table(
            columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript"]).to_pylist()
        for u in src:
            if split.get(u["meeting_id"]) == "test" and not is_hesitation(u["transcript"]):
                out[u["meeting_id"]].append((u["record_id"], u["speaker_id"], u["begin_time"], u["end_time"]))
    elif dr == "zh2en":
        split = json.load(open(D / "meetings.json"))
        for m, s in split.items():
            if s != "test":
                continue
            for u in load(D / f"zh_mix_v8/utts/{m}.jsonl"):
                if not u["hes"] and u.get("placed", True):
                    out[m].append((u["record_id"], u["speaker_id"], u["zh_begin"], u["zh_end"]))
    else:
        for f in sorted((A / "utts").glob("*.jsonl")):
            for u in load(f):
                if not u["hesitation"]:
                    out[f.stem].append((u["record_id"], u["speaker_id"], u["begin_s"], u["end_s"]))
    return out


def filt():
    out = {}
    wins = {"en2zh": D / "e2e_gstar_v1/windows_test_ref.jsonl", "zh2en": D / "e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
            "ali": A / "windows_test_ref.jsonl"}
    for dr, wf in wins.items():
        kept = {u["record_id"] for w in load(wf) for u in w["utts"] if not u["hesitation"]}
        U = utterances(dr)
        rows = []
        for m, us in U.items():
            for rid, spk, a, b in us:
                ov = any(min(b, b2) > max(a, a2) for rid2, s2, a2, b2 in us if s2 != spk)
                rows.append({"kept": rid in kept, "dur": b - a, "short": b - a <= 1.0, "overlap": ov})
        def agg(rs):
            return {"utts": len(rs), "hours": round(sum(r["dur"] for r in rs) / 3600, 2),
                    "kept_utts": round(np.mean([r["kept"] for r in rs]), 4) if rs else None,
                    "kept_dur": round(sum(r["dur"] for r in rs if r["kept"]) / max(sum(r["dur"] for r in rs), 1e-9), 4)}
        out[dr] = {"all": agg(rows), "short_le1s": agg([r for r in rows if r["short"]]),
                   "overlapped": agg([r for r in rows if r["overlap"]]),
                   "short_overlapped": agg([r for r in rows if r["short"] and r["overlap"]])}
    return out


# ---------------------------------------------------------------------------------------------------------- sameb
def sameb():
    out = {}
    for dr, names in (("en2zh", ["O2", "S2dl_d1", "O2_LA2_d1"]), ("zh2en", ["O2", "S2dl_d1"]), ("ali", ["O2", "S2dl_d1"])):
        for name in names:
            W, H = system(dr, name)
            if W is None:
                continue
            segs = [s for w in W for s in H[w["id"]]["segments"]]
            out[f"{dr}:{name}"] = {"segments": len(segs),
                                   "start_eq_end": round(np.mean([s["end_s"] <= s["start_s"] for s in segs]), 4)}
    return out


# -------------------------------------------------------------------------------------------------------- relabel
def relabel():
    out = {}
    import subprocess
    for dr in ("en2zh", "zh2en", "ali"):
        W, H = system(dr, "O2-nosd")
        if W is None:
            continue
        d = V6 / f"{dr}_O2-nosd-relabel"
        if (d / "hyp.jsonl").exists():
            out[dr] = json.load(open(d / "score.json"))["all"] if (d / "score.json").exists() else None
            continue
        d.mkdir(parents=True)
        n_moved = n = 0
        with open(d / "hyp.jsonl", "x", encoding="utf-8") as fh:
            for w in W:
                post = np.load(w["post"]["offline"]).astype(np.float32)
                h = dict(H[w["id"]])
                segs = []
                for s in h["segments"]:
                    a = int(s["start_s"] / 0.08)
                    b = max(a + 1, int(np.ceil(s["end_s"] / 0.08)))
                    sl = post[a:b] if a < len(post) else post[-1:]
                    if not sl.size or sl.max() <= 0:   # no activity in the span: the slot most active in the window
                        sl = post
                    k = int(sl.mean(0).argmax()) + 1
                    n += 1
                    n_moved += k != s["spk"]
                    segs.append({**s, "spk": k, "spk_model": s["spk"]})
                h["segments"] = segs
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
        sc = subprocess.run([str(Path("/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python")), "-m",
                             "experiments.uniss_qwen3_omni_v1.evaluation.gstar_score", "--windows", str(windows(dr, "O2-nosd")),
                             "--hyp", str(d / "hyp.jsonl"), "--out", str(d / "score.json")],
                            cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
        if sc.returncode != 0:
            raise SystemExit(sc.stderr[-2000:])
        out[dr] = {**json.load(open(d / "score.json"))["all"], "relabelled_share": round(n_moved / max(n, 1), 4)}
    return out


# ------------------------------------------------------------------------------------------------------- segments
SEG_SYS = {"en2zh": ["B-off", "C-ft2", "O2-nosd", "O2", "S2dl_d1", "O2_LA2_d1", "S2dlm_d1"],
           "zh2en": ["C-ft2", "O2-nosd", "O2", "S2dl_d1", "O2_LA2_d1"],
           "ali": ["C-ft2", "O2-nosd", "O2", "S2dl_d1", "O2_LA2_d1"]}


def assign(w, segs):
    """Output segment -> index of the window's non-hesitation utterance it overlaps most (collar), or None."""
    talk = [u for u in w["utts"] if not u["hesitation"]]
    out = []
    for s in segs:
        a, b = s["start_s"] - COLLAR, max(s["end_s"], s["start_s"] + 0.08) + COLLAR
        best, bi = 0.0, None
        for i, u in enumerate(talk):
            ov = min(b, u["end_s"]) - max(a, u["begin_s"])
            if ov > best:
                best, bi = ov, i
        out.append(bi)
    return talk, out


def segments():
    res = {}
    meet_dur = {}
    for dr, names in SEG_SYS.items():
        for name in names:
            W, H = system(dr, name)
            if W is None:
                continue
            smap = session_map(dr, W, H)
            if not meet_dur:
                pass
            lang = LANG[dr]
            # meeting clock positions for thirds and re-entry
            last_end = defaultdict(lambda: -1e9)
            mdur = defaultdict(float)
            for w in W:
                for u in w["utts"]:
                    mdur[w["meeting"]] = max(mdur[w["meeting"]], w["start_s"] + u["end_s"])
            per_u = []
            n_seg = n_dup = n_ins = n_seg_right = 0
            spk_laal, nest_lag, plain_lag = [], [], []
            seen = defaultdict(lambda: defaultdict(lambda: -1e9))
            for w in sorted(W, key=lambda w: (w["meeting"], w["start_s"])):
                h = H[w["id"]]
                mp = smap.get(w["meeting"], {})
                talk = [u for u in w["utts"] if not u["hesitation"]]
                texts = set()
                C = 0.5   # time collar of coverage, s

                def ov(s, u):
                    return min(max(s["end_s"], s["start_s"] + 0.08), u["end_s"] + C) - max(s["start_s"], u["begin_s"] - C)

                for s in h["segments"]:
                    n_seg += 1
                    key = (s["spk"], s["text"].strip())
                    if key in texts and s["text"].strip():
                        n_dup += 1
                    texts.add(key)
                    if not any(ov(s, u) > 0 for u in talk):
                        n_ins += 1
                    n_seg_right += any(ov(s, u) > 0 and mp.get(s["spk"]) == u["speaker_id"] for u in talk)
                for u in talk:
                    t_abs = w["start_s"] + u["begin_s"]
                    prev = seen[w["meeting"]][u["speaker_id"]]
                    reentry = prev > -1e9 and t_abs - prev >= 60.0
                    seen[w["meeting"]][u["speaker_id"]] = max(prev, w["start_s"] + u["end_s"])
                    ss = [s for s in h["segments"] if ov(s, u) > 0]
                    right = [s for s in ss if mp.get(s["spk"]) == u["speaker_id"]]
                    nested = any(v is not u and v["speaker_id"] != u["speaker_id"] and v["begin_s"] <= u["begin_s"]
                                 and v["end_s"] >= u["end_s"] for v in talk)
                    rec = {"meeting": w["meeting"], "third": min(2, int(3 * t_abs / max(mdur[w["meeting"]], 1e-9))),
                           "reentry": reentry, "covered": bool(right), "omitted": not ss,
                           "misattributed": bool(ss) and not right, "short": u["end_s"] - u["begin_s"] <= 1.0,
                           "nested": nested}
                    if right:
                        rec["onset_err"] = min(abs(s["start_s"] - u["begin_s"]) for s in right)
                        rec["offset_err"] = min(abs(max(s["end_s"], s["start_s"]) - u["end_s"]) for s in right)
                        if "write_ms" in right[0]:
                            lag = max(s["write_ms"] for s in right) / 1000 - u["end_s"]
                            rec["end_lag"] = lag
                            (nest_lag if nested else plain_lag).append(lag)
                    per_u.append(rec)
                if h.get("blocks"):   # T10: LAAL per output speaker against that speaker's own reference
                    inv = defaultdict(list)
                    for b in h["blocks"]:
                        inv[mp.get(b["spk"])].append(b)
                    for spk in {u["speaker_id"] for u in talk}:
                        us = [u for u in talk if u["speaker_id"] == spk]
                        delays = [b["write_ms"] for b in inv.get(spk, []) for _ in units(b["text"], lang)]
                        if delays:
                            ref_len = len(units(SEP[dr].join(ref_text(u) for u in us), lang))
                            spk_laal.append(laal(delays, 1000 * max(u["end_s"] for u in us), ref_len))
            P = n_seg_right / max(n_seg, 1)
            R = np.mean([r["covered"] for r in per_u])
            out = {"utterances": len(per_u), "segments": n_seg,
                   "attr_precision": round(P, 4), "attr_recall": round(float(R), 4),
                   "attr_F1": round(2 * P * R / max(P + R, 1e-9), 4),
                   "omitted": round(np.mean([r["omitted"] for r in per_u]), 4),
                   "misattributed": round(np.mean([r["misattributed"] for r in per_u]), 4),
                   "insertion_segments": round(n_ins / max(n_seg, 1), 4),
                   "duplicate_segments": round(n_dup / max(n_seg, 1), 4),
                   "onset_med_s": round(float(np.median([r["onset_err"] for r in per_u if "onset_err" in r])), 3),
                   "offset_med_s": round(float(np.median([r["offset_err"] for r in per_u if "offset_err" in r])), 3),
                   "covered_by_third": [round(float(np.mean([r["covered"] for r in per_u if r["third"] == k])), 4)
                                        for k in range(3)],
                   "covered_reentry": round(float(np.mean([r["covered"] for r in per_u if r["reentry"]])), 4)
                   if any(r["reentry"] for r in per_u) else None,
                   "n_reentry": sum(r["reentry"] for r in per_u),
                   "covered_short": round(float(np.mean([r["covered"] for r in per_u if r["short"]])), 4)}
            if spk_laal:
                out["per_speaker_LAAL_s"] = round(float(np.nanmean(spk_laal)) / 1000, 3)
            if plain_lag:
                out["end_lag_s"] = {"plain_p50": round(float(np.median(plain_lag)), 3),
                                    "nested_p50": round(float(np.median(nest_lag)), 3) if nest_lag else None,
                                    "plain_p90": round(float(np.percentile(plain_lag, 90)), 3),
                                    "nested_p90": round(float(np.percentile(nest_lag, 90)), 3) if nest_lag else None,
                                    "n_nested": len(nest_lag)}
            res[f"{dr}:{name}"] = out
            print(dr, name, json.dumps(out), flush=True)
    return res


# --------------------------------------------------------------------------------------------------------- voices
def voices():
    from experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker import solo_runs
    import torch
    res = {}
    sess = {"en2zh": lambda m: D / f"diar/nemotron_lowlat/{m}.post.pt",
            "zh2en": lambda m: D / f"diar/nemotron_lowlat_zh8/{m}.post.pt",
            "ali": lambda m: A / f"diar/nemotron_lowlat/{m}.post.pt"}
    for dr in ("en2zh", "zh2en", "ali"):
        W, H = system(dr, "S2dl_d1")
        if W is None:
            continue
        cache = {}
        n = avail = 0
        lens, waits, solo_any, purity = [], [], [], []
        for w in W:
            m = w["meeting"]
            if m not in cache:
                cache.clear()
                cache[m] = torch.load(sess[dr](m), weights_only=False)["post80"].float().numpy()
            post = cache[m]
            first = {}
            for b in H[w["id"]].get("blocks", []):
                first.setdefault(b["spk"], b["write_ms"])
            for k, wm in first.items():
                cut = w["start_s"] + wm / 1000
                runs = solo_runs(post, k - 1, int(cut / 0.08))
                tot, frames = 0.0, []
                for a, b in reversed(runs):
                    tot += (b - a) * 0.08
                    frames += list(range(a, b))
                    if tot >= 8.0:
                        break
                if frames:   # purity: the strongest other slot over the prompt's frames (< 0.5 by construction)
                    oth = np.delete(post[frames], k - 1, axis=1).max(1)
                    purity.append(float(oth.mean()))
                n += 1
                avail += tot >= 1.0
                lens.append(min(tot, 8.0))
                solo_any.append(bool(solo_runs(post, k - 1)))
                act = np.flatnonzero(post[:, k - 1] > 0.5)
                if len(act) and tot >= 1.0:   # wait from the slot's first activity until 1 s heard alone
                    acc, t1 = 0.0, None
                    for a, b in solo_runs(post, k - 1):
                        acc += (b - a) * 0.08
                        if acc >= 1.0:
                            t1 = b * 0.08 - (acc - 1.0)
                            break
                    if t1 is not None:
                        waits.append(t1 - act[0] * 0.08)
        res[dr] = {"streams": n, "causal_prompt_available": round(avail / max(n, 1), 4),
                   "fallback": round(1 - avail / max(n, 1), 4), "prompt_s_mean": round(float(np.mean(lens)), 2),
                   "slot_ever_alone_in_session": round(float(np.mean(solo_any)), 4),
                   "wait_to_1s_alone_p50_s": round(float(np.median(waits)), 2) if waits else None,
                   "prompt_other_slot_mean": round(float(np.mean(purity)), 4) if purity else None}
        print(dr, res[dr], flush=True)
    return res


def main() -> None:
    f = GEN / "v6_extras.json"
    res = json.load(open(f)) if f.exists() else {}
    for name in sys.argv[1:]:
        fn = {"flush": flush, "filter": filt, "sameb": sameb, "relabel": relabel, "segments": segments,
              "voices": voices}[name]
        res[name] = fn()
        f.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
