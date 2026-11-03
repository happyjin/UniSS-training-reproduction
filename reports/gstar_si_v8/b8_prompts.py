"""Paper v8 B8 (re-review M17): voice-prompt length and prompt cold start, on the 8 whole AMI En->Zh meetings that v7
synthesised as on_en2zh_S2dl_rest (3,318 streams). Nothing else changes (same text, blocks, Talker, speed policy).

  prep    for every stream whose causal prompt (gstar_talker_v6 prep --causal) had >= 1 s heard alone, write a 1 s
          prompt: the most recent 1 s of solo audio before the slot's first commit (the 8 s prompt is v7's, already
          synthesised); streams with less than 1 s keep v7's fallback prompt. Writes <out>/items.jsonl, blocks.jsonl
          (copied), voice_1s/*.wav and prompts.json (per stream: solo seconds available, time until 1 s was heard
          alone, measured from the slot's first activity in the window, and the first commit time).
  wait    cold start: holding a slot's blocks until 1 s of it has been heard alone (instead of speaking at once with
          the fallback prompt) delays them by max(0, t_1s - commit); reports the delayed share of streams and blocks
          and the added delay (p50 / p90), from prompts.json and the v7 blocks.

    python reports/gstar_si_v8/b8_prompts.py prep --out eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/b8/talker_in
    python reports/gstar_si_v8/b8_prompts.py wait --prep <that dir> --out <json>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker_v6 import session_of, solo_runs  # noqa: E402

V7S = REPO / "eval_outputs/uniss_qwen3_omni_v1/gstar_v7_speech_20261101/on_en2zh_S2dl_rest"
WIN = REPO / "eval_outputs/uniss_qwen3_omni_v1/gstar_v7_20261101/on_en2zh_S2dl_rest/todo.windows.jsonl"


def prep(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "voice_1s").mkdir(parents=True)
    shutil.copy(V7S / "talker_in/blocks.jsonl", out / "blocks.jsonl")
    W = {w["id"]: w for w in map(json.loads, open(WIN))}
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(V7S / "talker_in/blocks.jsonl"))}
    sargs = argparse.Namespace(preset="lowlat", diar="", session="ihm_mix")
    sess, meta, n1 = {}, {}, 0
    with open(out / "items.jsonl", "x", encoding="utf-8") as fi:
        for it in map(json.loads, open(V7S / "talker_in/items.jsonl")):
            wid, k = it["id"].rsplit("__S", 1)
            k = int(k)
            w = W[wid]
            if w["meeting"] not in sess:
                sess.clear()
                sess[w["meeting"]] = session_of(sargs, w["meeting"])
            post, x, sr = sess[w["meeting"]]
            commit = w["start_s"] + blocks[it["id"]][0]["write_ms"] / 1000
            runs = solo_runs(post, k - 1, int(commit / 0.08))
            tot = sum((b - e0) * 0.08 for e0, b in runs) if runs else 0.0
            # time 1 s of the slot had been heard alone (over the whole session, causal frames only)
            acc, t1 = 0.0, None
            for e0, b in solo_runs(post, k - 1):
                acc += (b - e0) * 0.08
                if acc >= 1.0:
                    t1 = b * 0.08 - (acc - 1.0)
                    break
            meta[it["id"]] = {"solo_s": round(tot, 2), "t_1s": None if t1 is None else round(t1, 2),
                              "commit": round(commit, 2), "meeting": w["meeting"]}
            if tot >= 1.0:
                pieces, got = [], 0.0
                for e0, b in reversed(runs):
                    pieces.insert(0, x[int(e0 * 0.08 * sr): int(b * 0.08 * sr)])
                    got += (b - e0) * 0.08
                    if got >= 1.0:
                        break
                v = out / "voice_1s" / f"{it['id']}.wav"
                sf.write(str(v), np.concatenate(pieces)[-int(1.0 * sr):], sr)
                it = dict(it, voice_audio=str(v))
                n1 += 1
            fi.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "prompts.json").write_text(json.dumps(meta))
    print(json.dumps({"streams": len(meta), "with_1s_prompt": n1}))


def wait(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    meta = json.load(open(Path(a.prep) / "prompts.json"))
    blocks = {r["id"]: r["blocks"] for r in map(json.loads, open(V7S / "talker_in/blocks.jsonl"))}
    W = {w["id"]: w for w in map(json.loads, open(WIN))}
    add, n_b, n_bd, n_s, n_sd, never = [], 0, 0, 0, 0, 0
    for sid, m in meta.items():
        wid = sid.rsplit("__S", 1)[0]
        n_s += 1
        if m["t_1s"] is None:
            never += 1
        delayed = False
        for b in blocks[sid]:
            c = W[wid]["start_s"] + b["write_ms"] / 1000
            n_b += 1
            d = 0.0 if m["t_1s"] is None else max(0.0, m["t_1s"] - c)   # a slot never heard alone keeps the fallback
            if d > 0:
                n_bd += 1
                add.append(d)
                delayed = True
        n_sd += delayed
    add = np.array(add) if add else np.zeros(1)
    res = {"streams": n_s, "streams_delayed": round(n_sd / max(n_s, 1), 4), "blocks_delayed": round(n_bd / max(n_b, 1), 4),
           "added_delay_p50": round(float(np.median(add)), 2), "added_delay_p90": round(float(np.percentile(add, 90)), 2),
           "slots_never_1s_alone": never,
           "prompt_1s_available_at_commit": round(sum(m["solo_s"] >= 1.0 for m in meta.values()) / max(n_s, 1), 4),
           "prompt_8s_full_at_commit": round(sum(m["solo_s"] >= 8.0 for m in meta.values()) / max(n_s, 1), 4)}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--out", required=True)
    p = sub.add_parser("wait")
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": prep, "wait": wait}[a.cmd](a)
