"""zh->en G-STAR-SI data on the Chinese AMI meetings (docs/multi_speaker_si_plan_v9_bidir_abc.zh-CN.md §2, §11.5), in the
format of build_ami_gstar_si_rows.py (en->zh), so one model trains on both directions.

  utterances  zh_mix_v3/utts (Chinese clock, quality flags) + the Paraformer recheck of the clips Whisper rejected +
              the normalised English references (normalize_ami_en.py)
  A8          meetings listed as failing in <mix>/audit_diar.json (diarizer DER > English + 10 points) are left out
  gate        an utterance whose clip is missing, unintelligible to both ASRs (CER > 0.3), has an inner pause > 1 s or
              is over three times the English length was left out of the v4 meeting (build_ami_zh_mix --exclude);
              it is skipped here (v5 = v4 with a quieter background and a 4x gain cap). Any remaining utterance failing the gate drops its window
  slots       reference speaker -> Nemotron slot on the Chinese meeting, per preset (Hungarian on the whole meeting)
  windows     ref: the English utterance groups, cut on the Chinese clock (padded by up to 0.3 s / 0.5 s, half the gap to
              the neighbouring speech); dropped if Chinese speech of another group reaches into it, if it is longer
              than 20.5 s, or if the gate fails. diar_offline (training only): causal windows on Nemotron activity
              keeping whole, good utterances, as for en->zh
  target      <|spk k|><|t a|>English<|t b|> per utterance by Chinese start; a / b on the Chinese clock
  rows        offline rows ("dir": "zh2en", "inst": "gstar"), ref windows with the offline or the streaming preset's
              posteriors (half each), diar windows with their own

Utterance fields keep en->zh's names for the scorer: "zh" is the source (Chinese) text, "en" and "ref" the English
reference; gstar_score reads "ref" when present.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_zh2en_rows --out data/ami_multispk_si_v1/e2e_gstar_zh2en_v1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import (
    FRAME, N_SLOT, diar_windows, save_slice, seg, slot_map)
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix import bad_reason, cer_of, removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--mix", default="zh_mix_v5")
    ap.add_argument("--recheck", default="data/ami_multispk_si_v1/zh_mix_v3_recheck")
    ap.add_argument("--en-norm", default="data/ami_multispk_si_v1/en_norm_v1.jsonl")
    ap.add_argument("--diar", default="nemotron_{}_zh5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--presets", default="offline,lowlat")
    ap.add_argument("--max-win", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--inside", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    presets = args.presets.split(",")
    split_of = json.load(open(data / "meetings.json"))
    audit = data / args.mix / "audit_diar.json"   # A8: meetings the diarizer tracks much worse than English are not used
    skip = set(json.load(open(audit))["summary"]["failing_meetings"]) if audit.exists() else set()
    if not audit.exists():
        print(f"warning: no {audit}; every meeting is used", flush=True)
    asr = {}
    for f in sorted(Path(args.recheck).glob("asr_s*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            asr[r["id"]] = r.get("asr_text", "")
    en_norm = {r["record_id"]: r["en"] for r in map(json.loads, open(args.en_norm, encoding="utf-8"))}
    stats = defaultdict(lambda: defaultdict(int))
    windows = defaultdict(list)
    slot_maps = {p: {} for p in presets}
    for f in sorted((data / args.mix / "utts").glob("*.jsonl")):
        m = f.stem
        split = split_of[m]
        if m in skip:
            stats[split]["meetings_failing_A8"] += 1
            continue
        us = [json.loads(l) for l in open(f, encoding="utf-8")]
        talk = []
        for u in us:
            if u["hes"] or removed(u):   # removed: failed the gate, left out of the meeting (not heard, not translated)
                stats[split]["utt_removed"] += removed(u)
                continue
            if u["record_id"] in asr:
                u["cer2"] = round(cer_of(asr[u["record_id"]], u["zh"]), 4)
            u["ref"] = en_norm.get(u["record_id"], "")
            u["_bad"] = bad_reason(u) or (None if u["ref"].strip() else "no_reference")
            talk.append(u)
            stats[split]["utt_" + (u["_bad"] or "ok")] += 1
        placed = sorted((u for u in talk if u["placed"]), key=lambda u: u["zh_begin"])
        posts = {p: torch.load(data / "diar" / args.diar.format(p) / f"{m}.post.pt", weights_only=False)["post80"]
                 .float().numpy() for p in presets}
        T = max(len(v) for v in posts.values())
        act = defaultdict(lambda: np.zeros(T, dtype=bool))
        for u in placed:
            act[u["speaker_id"]][int(u["zh_begin"] / FRAME): int(np.ceil(u["zh_end"] / FRAME))] = True
        act = dict(act)
        oracle = {}
        for p in presets:
            mp = slot_map(act, posts[p])
            slot_maps[p][m] = mp
            o = np.zeros((len(posts[p]), N_SLOT), dtype=np.float16)
            for s, k in mp.items():
                n = min(len(o), len(act[s]))
                o[:n, k - 1] = np.maximum(o[:n, k - 1], act[s][:n])
            oracle[p] = o
        x, sr = sf.read(str(data / "session_audio" / args.mix / f"{m}.wav"), dtype="float32")
        dur = len(x) / sr

        def cut(ws, we, wtype, wid, members):
            wav = out / "audio" / split / f"{wid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
            w = {"id": wid, "meeting": m, "split": split, "wtype": wtype, "dir": "zh2en", "start_s": round(ws, 3),
                 "dur_ms": round((we - ws) * 1000), "audio": str(wav.resolve()),
                 "strict_review": all((u.get("cer") or 9) <= 0.3 for u in members),
                 "utts": [{"record_id": u["record_id"], "speaker_id": u["speaker_id"],
                           "begin_s": round(max(0.0, u["zh_begin"] - ws), 3), "end_s": round(min(we, u["zh_end"]) - ws, 3),
                           "zh": u["zh"], "en": u["ref"], "ref": u["ref"], "hesitation": False} for u in members]}
            w["post"] = {}
            for p in presets:
                w["post"][p] = save_slice(out, f"{p}_{wtype}" if wtype != "ref" else p, w, posts[p])
                w["post"][f"oracle_{p}"] = save_slice(out, f"oracle_{p}", w, oracle[p])
            w["target"] = {p: "".join(seg(slot_maps[p][m][u["speaker_id"]], u["zh_begin"] - ws, u["zh_end"] - ws,
                                          u["ref"].strip())
                                      for u in sorted(members, key=lambda u: (u["zh_begin"], u["zh_end"])))
                           for p in presets}
            return w

        # ref windows: the English utterance groups, on the Chinese clock
        from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix import groups_of
        groups = groups_of(talk, "en_begin", "en_end")
        spans = [(min(u["zh_begin"] for u in g if u["placed"]), max(u["zh_end"] for u in g if u["placed"]))
                 if any(u["placed"] for u in g) else None for g in groups]
        for gi, g in enumerate(groups):
            st = stats[(split, "ref")]
            bad = [u["_bad"] for u in g if u["_bad"]]
            if bad:
                st["drop_gate_" + bad[0]] += 1
                continue
            a, b = spans[gi]
            if b - a > args.max_win + 0.5:
                st["drop_too_long"] += 1
                continue
            prev_end = max([s[1] for s in spans[:gi] if s] + [0.0])
            next_beg = min([s[0] for s in spans[gi + 1:] if s] + [dur])
            ws = max(0.0, a - min(0.3, max(0.0, a - prev_end) / 2))
            we = min(dur, b + min(0.5, max(0.0, next_beg - b) / 2))
            ids = {u["record_id"] for u in g}
            if any(u["record_id"] not in ids and min(we, u["zh_end"]) - max(ws, u["zh_begin"]) > 0.05 for u in placed):
                st["drop_straddle"] += 1
                continue
            windows[(split, "ref")].append(cut(ws, we, "ref", f"{m}_zh_{gi:04d}", g))
            st["kept"] += 1
        # diar windows (training only), on the offline preset's activity
        if split == "train":
            for k, (ws, we) in enumerate(diar_windows(posts["offline"], args.max_win, args.gap)):
                ws, we = max(0.0, ws - 0.2), min(dur, we + 0.3)
                st = stats[(split, "diar_offline")]
                inside, ok = [], True
                for u in placed:
                    ov = min(we, u["zh_end"]) - max(ws, u["zh_begin"])
                    if ov <= 0:
                        continue
                    if ov < args.inside * (u["zh_end"] - u["zh_begin"]):
                        ok = False
                        st["drop_straddle"] += 1
                        break
                    inside.append(u)
                if not ok:
                    continue
                if not inside:
                    st["drop_no_speech"] += 1
                    continue
                if any(u["_bad"] for u in inside):
                    st["drop_gate"] += 1
                    continue
                windows[(split, "diar_offline")].append(cut(ws, we, "diar_offline", f"{m}_zhd_{k:04d}", inside))
                st["kept"] += 1
        print(f"{m} {split}: " + " ".join(f"{k[1]}={stats[k]['kept']}" for k in stats if isinstance(k, tuple)
                                          and k[0] == split), flush=True)
    rows = {"train": [], "validation": []}
    for (split, wtype), ws in sorted(windows.items()):
        with open(out / f"windows_{split}_{wtype}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in ws)
        if split not in rows:
            continue
        for w in ws:
            p = rng.choice(presets) if wtype == "ref" else "offline"
            rows[split].append({"id": f"{w['id']}|{p}", "audio": w["audio"], "dir": "zh2en", "t_ms": w["dur_ms"],
                                "dur_ms": w["dur_ms"], "post": w["post"][p], "prefix": "", "target": w["target"][p],
                                "end": "eos", "kind": "offline", "inst": "gstar", "wtype": wtype, "preset": p})
    for split, rs in rows.items():
        rng.shuffle(rs)
        with open(out / f"rows_{split}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    (out / "slot_map.json").write_text(json.dumps(slot_maps, indent=0), encoding="utf-8")
    summ = {"/".join(k) if isinstance(k, tuple) else k: dict(v) for k, v in stats.items()}
    summ["rows"] = {k: len(v) for k, v in rows.items()}
    summ["args"] = vars(args)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
