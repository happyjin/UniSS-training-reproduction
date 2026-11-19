"""OpenBroadcast-SI v1 listening A/B data (plan §7.2f; reports/mist_w3po_v1/voicecond_bench.zh-CN.md §7): the SAME 18-stream
subset as scripts/voicecond_bench_listen.py (3 streams per direction, chosen without system output, each in id2 / ood1 /
clean) for SEVERAL arms, in build_game_demo_v2.py's data.json schema so that game_demo_ab_template.html works unchanged:
samples[] = {id, dir, title (condition + direction + stream), url, video, part, t0, dur, windows, speakers, src (shared:
CoVoST source text + reference, real speaker index), arms: {key: {label, run, audio, tgt}}}, arm_order, arm_labels,
stats {arm: {sample id: {metric: value}}} (bench_eval per-stream means of the arm's TTS-seed-1 run, when evaluated).
Per sample and arm one STEREO MP3: left = the condition's source mix, right = the arm's simultaneous translation (every
talker stream at its window start), the same normalisation for every arm (build_game_demo.norm). Arms are talker runs
(TTS seed 1) named by a pattern with {cond} / {dir}; an arm is skipped for a set whose run is not complete (no
talker/results.jsonl, which the speech chain writes only when all 8 talker shards are present). Read-only on all inputs.
Writes <out>/audio/<m>__<arm>.mp3 and <out>/data.json. Refuses an existing <out>.

    python experiments/mist_w3po_v1/scripts/voicecond_bench_listen_ab.py --out demo_packages/voicecond_bench_listen_ab_v1 \\
        [--arm key:label:pattern ...]     (default: B, idea2, z, c below)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_game_demo import SR, TAIL_S, norm, read16  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from voicecond_bench_listen import pick  # noqa: E402  (the same three streams per direction)

BD = ROOT / "data/open_voicecond_v1/bench"
CONDS = ("id2", "ood1", "clean")
COND_LABEL = {"id2": "ID-2 领域内背景（留出的 MUSAN / FSD50K）", "ood1": "OOD-1 领域外背景（FMA / DEMAND / ESC-50）",
              "clean": "无背景对照"}
DIR_LABEL = {"zh2en": "中→英（长流）", "en2zh": "英→中（短流）"}
ARMS = ["B:现有（baseline B）:obsi_b_{cond}_{dir}__t1c",
        "idea2:Idea 2 收缩感知身份后验（shrink + Vfb）:obsi_i2_{cond}_{dir}__t1c",
        "z:Idea 1 ZNE 端到端编码器:obsi_i1z_{cond}_{dir}__t1c",
        "c:对照：去噪编码器:obsi_i1c_{cond}_{dir}__t1c"]
STAT_KEYS = {"pesq": "PESQ", "contrast": "对比度 dB", "sim": "相似度", "blind_asr": "盲 ASR-BLEU"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", default=[], help="key:label:run pattern with {cond} and {dir}")
    a = ap.parse_args()
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        sys.exit(f"拒绝覆盖已有输出: {out}")
    arms = [tuple(x.split(":", 2)) for x in (a.arm or ARMS)]
    (out / "audio").mkdir(parents=True)
    samples, stats, skipped = [], defaultdict(dict), []
    for d in ("zh2en", "en2zh"):
        keys = pick(d)
        for cond in CONDS:
            M = json.load(open(BD / cond / d / "meetings.json"))
            by_key = {v["stream"]: m for m, v in M.items()}
            wins = [json.loads(l) for l in open(ROOT / f"eval_outputs/mist_w3po_v1/data/windows_obsi_{cond}_{d}_c2g06_v1/"
                                                       f"windows_test_online_{d}.jsonl")]
            refs = defaultdict(list)
            for l in open(BD / cond / d / "refs_windows.jsonl", encoding="utf-8"):
                r = json.loads(l)
                refs[r["meeting"]].append(r)
            streams = {}
            for key, label, pat in arms:
                run = pat.format(cond=cond, dir=d)
                rp = ROOT / f"eval_outputs/mist_w3po_v1/speech/{run}/talker/results.jsonl"
                if not rp.exists():
                    skipped.append(run)
                    print("skip arm (run not complete):", key, run, flush=True)
                    continue
                st = defaultdict(list)
                for l in open(rp):
                    r = json.loads(l)
                    if r.get("error") is None and r.get("audio_path"):
                        st[r["id"].rsplit("__S", 1)[0]].append(r)
                streams[key] = (label, run, st)
                mf = ROOT / f"eval_outputs/mist_w3po_v1/voicecond_bench/{run}/metrics.json"
                if mf.exists():
                    ps = json.load(open(mf))["per_stream"].get(run, {})
                    for m, v in ps.items():
                        stats[key][m] = {STAT_KEYS[k]: v[k] for k in STAT_KEYS if v.get(k) is not None}
            sdir = "mix" if d == "zh2en" else "ihm_mix"
            for skey in keys:
                m = by_key[skey]
                info = M[m]
                src = read16(BD / cond / d / "session_audio" / sdir / (f"{m}.wav" if d == "zh2en" else f"{m}.Mix-Headset.wav"))
                dur = len(src) / SR
                ws = sorted((w for w in wins if w["meeting"] == m), key=lambda w: w["start_s"])
                spk_ix = {s: i for i, s in enumerate(info["speakers"])}
                sl = [{"t": round(r["start_s"], 2), "e": round(r["start_s"] + r["utts"][0]["end_s"], 2),
                       "spk": spk_ix[r["utts"][0]["speaker_id"]], "text": r["utts"][0]["zh" if d == "zh2en" else "en"],
                       "ref": r["utts"][0]["ref"]} for r in sorted(refs[m], key=lambda r: r["start_s"])]
                n_max = int(round((dur + TAIL_S) * SR))
                tracks = {}
                for key, (label, run, st) in streams.items():
                    tr = np.zeros(n_max, np.float32)
                    tl = []
                    for w in ws:
                        for r in st.get(w["id"], []):
                            y = read16(r["audio_path"])
                            i = int(round(w["start_s"] * SR))
                            j = min(n_max, i + len(y))
                            if j > i:
                                tr[i:j] += y[: j - i]
                            k = int(r["id"].rsplit("__S", 1)[1])
                            for b in r.get("blocks") or []:
                                if (b.get("text") or "").strip():
                                    s = w["start_s"] + b["start_ms"] / 1000.0
                                    tl.append({"t": round(s, 2), "e": round(s + b["dur_ms"] / 1000.0, 2), "spk": k,
                                               "text": b["text"].strip()})
                    tracks[key] = (tr, sorted(tl, key=lambda x: x["t"]), label, run)
                last = max([x["e"] for (tr, tl, _, _) in tracks.values() for x in tl] + [dur])
                n = min(n_max, int(round(min(last + 0.5, dur + TAIL_S) * SR)))
                left = norm(np.pad(src, (0, max(0, n - len(src))))[:n])
                arm_out = {}
                for key, (tr, tl, label, run) in tracks.items():
                    x = np.stack([left, norm(tr[:n])], 1)
                    pk = float(np.max(np.abs(x))) if x.size else 0.0
                    if pk > 0.97:
                        x = x * (0.97 / pk)
                    p = out / "audio" / f"{m}__{key}.mp3"
                    sf.write(str(p), x.astype(np.float32), SR, format="MP3")
                    arm_out[key] = {"label": label, "run": run, "audio": f"audio/{p.name}", "tgt": tl}
                samples.append({"id": m, "dir": d, "cond": cond, "stream": skey,
                                "title": f"{COND_LABEL[cond]} · {DIR_LABEL[d]} · 流 {skey}（{info['K']} 人，"
                                         f"BVR {info.get('bvr_db', '—')} dB）",
                                "url": "https://github.com/facebookresearch/covost?", "video": f"OpenBroadcast-SI {skey}",
                                "part": 1, "t0": 0.0, "dur": round(n / SR, 2), "windows": len(ws), "speakers": info["K"],
                                "twins": info["twins"], "src": sl, "arms": arm_out})
                print(m, d, cond, skey, "arms", list(arm_out), flush=True)
    ids = {x["id"] for x in samples}
    stats = {k: {m: v for m, v in d2.items() if m in ids} for k, d2 in stats.items()}
    (out / "data.json").write_text(json.dumps({"samples": samples, "arm_order": [k for k, _, _ in arms],
                                               "arm_labels": {k: l for k, l, _ in arms}, "stats": stats,
                                               "skipped_runs": skipped, "layout": "left = source mix, right = translation"},
                                              ensure_ascii=False), encoding="utf-8")
    print(out, "samples", len(samples), "skipped runs", len(skipped))


if __name__ == "__main__":
    main()
