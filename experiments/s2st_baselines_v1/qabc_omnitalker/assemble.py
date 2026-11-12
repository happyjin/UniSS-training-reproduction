"""After synthesis (registration paper/baselines_run_plan_qabc_omnitalker_v1.md §2-§3): timelines, thinker-text
fidelity, speech-only ASR.

assemble <point> <seed>: for every RealSI sentence of the point, block k starts playing at
  start_k = max(write_ms_k, start_{k-1} + dur_{k-1}) and lasts its Omni audio (16 kHz; a punctuation-only block has no
  request and no audio), the paper's queue model with no compute time. Writes, as si_infer (refuses an existing run):
    <OUT>/runs/<point>_s<seed>/realsi_<dir>/wav/<id>_simul.wav   target channel on the source clock
                                           wav/<id>_speech.wav  the blocks' speech back to back (what the ASR hears)
                                           results.jsonl        si_infer's fields (blocks: write_ms, text, final,
                                                                start_ms, dur_ms, + thinker_text / tts)
fidelity <point>: per block, the Omni Thinker's text against the text sent (normalised as BLEU), all seeds ->
  <OUT>/fidelity/<point>.json (exact-match share, token WER with S / D / I, seeds agreeing, worst examples).
pipeline --asr-cards ...: waits for each seed's 8 synthesis shards, assembles that seed's 8 runs, writes fidelity once
  all seeds are in, and runs reports/uniss_si_v4/speech_only.py + evaluation.asr_transcribe --batch-size 4 (uniss-eval)
  per run, at most --per-card ASR processes a card, each started only where nvidia-smi shows >= MIN_FREE_GIB under
  the 135 GiB cap (and >= 30 s apart); then the secondary block-level ASR of seed 0 (each block's own wav
  against its text, <OUT>/block_asr/). Prints PIPELINE_DONE.

    PYTHONPATH=. uniss-eval/bin/python experiments/s2st_baselines_v1/qabc_omnitalker/assemble.py pipeline --asr-cards 4 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.text_metrics import normalize_for_bleu  # noqa: E402
from experiments.s2st_baselines_v1.qabc_omnitalker.common import (  # noqa: E402
    OUT, POINTS, REPO, SEEDS, omni_run, point_name)

PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
SHARDS_PER_SEED = 8
PT = {point_name(p[0], p[1]): p for p in POINTS}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def synth_rows() -> dict:
    out = {}
    for f in sorted((OUT / "synth").glob("part_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            assert r["key"] not in out, r["key"]
            out[r["key"]] = r
    return out


def assemble(name: str, seed: int, synth: dict) -> dict:
    p = PT[name]
    d = p[0]
    out = omni_run(p, seed)
    out.mkdir(parents=True, exist_ok=False)
    (out / "wav").mkdir()
    records, used, missing = [], 0, 0
    for r in map(json.loads, open(OUT / "blocks" / f"{name}.jsonl", encoding="utf-8")):
        X = r["source_duration_ms"]
        pieces, blocks, end_prev = [], [], 0
        for k, b in enumerate(r["blocks"]):
            s = synth.get(f"{name}_s{seed}__{r['id']}__{k}")
            piece = np.zeros(0, dtype=np.float32)
            if s is not None and s.get("wav16"):
                piece = sf.read(OUT / s["wav16"], dtype="float32")[0]
                used += 1
            elif s is not None:
                missing += 1
            dur = int(round(len(piece) / 16))
            start = max(int(b["write_ms"]), end_prev)
            pieces.append((start, piece))
            end_prev = start + dur
            blocks.append({"write_ms": b["write_ms"], "text": b["text"], "final": b["final"], "start_ms": start,
                           "dur_ms": dur, "tts": s is not None, "thinker_text": s["thinker_text"] if s else None,
                           "wav16": s.get("wav16") if s else None})
        total = int(max(X, end_prev)) + 300
        right = np.zeros(total * 16, dtype=np.float32)
        for start, piece in pieces:
            a = start * 16
            right[a: a + len(piece)] += piece
        speech = np.concatenate([q for _, q in pieces]) if pieces else np.zeros(0, dtype=np.float32)
        tgt_path, speech_path = out / "wav" / f"{r['id']}_simul.wav", out / "wav" / f"{r['id']}_speech.wav"
        sf.write(tgt_path, right, 16000)
        sf.write(speech_path, speech if speech.size else np.zeros(320, dtype=np.float32), 16000)
        first = next((st for st, q in pieces if len(q)), None)
        records.append({"id": r["id"], "mode": "si", "name": r["id"], "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
                        "transcription_ref": r["transcription_ref"], "translation_ref": r["translation_ref"],
                        "translated_text": "".join(b["text"] for b in r["blocks"]), "audio_path": str(tgt_path),
                        "speech_path": str(speech_path), "audio_duration_seconds": total / 1000,
                        "source_duration_ms": X, "system": "QABC committed blocks + Qwen3-Omni-30B-A3B-Instruct native talker",
                        "talker_seed": seed, "error": None if speech.size else "no audio",
                        "start_offset_ms": first, "end_offset_ms": end_prev - X, "blocks": blocks})
    with open(out / "results.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in records)
    return {"run": str(out.relative_to(REPO)), "sentences": len(records), "blocks_with_audio": used,
            "requests_without_audio": missing}


def fidelity(name: str, synth: dict) -> dict:
    from experiments.s2st_baselines_v1.qabc_omnitalker.score import edit_ops
    lang = PT[name][0] == "en2zh" and "cmn" or "eng"
    per_seed, texts = {}, {}
    for s in SEEDS:
        rows = [r for k, r in synth.items() if k.startswith(f"{name}_s{s}__")]
        S = D = I = N = exact = 0
        bad = []
        for r in rows:
            ref, hyp = normalize_for_bleu(r["text_sent"], lang).split(), normalize_for_bleu(r["thinker_text"], lang).split()
            a, b, c = edit_ops(ref, hyp)
            S, D, I, N = S + a, D + b, I + c, N + len(ref)
            exact += ref == hyp
            if ref != hyp:
                bad.append((a + b + c, r["text_sent"], r["thinker_text"]))
            texts.setdefault(r["key"].split("__", 1)[1], set()).add(r["thinker_text"])
        bad.sort(key=lambda x: -x[0])
        per_seed[s] = {"blocks": len(rows), "exact_share": round(exact / max(len(rows), 1), 4),
                       "thinker_wer": round(100 * (S + D + I) / max(N, 1), 2), "sub": round(100 * S / max(N, 1), 2),
                       "del": round(100 * D / max(N, 1), 2), "ins": round(100 * I / max(N, 1), 2), "ref_tokens": N,
                       "no_audio": sum(not r.get("wav16") for r in rows),
                       "dur_ms_mean": round(float(np.mean([r["dur_ms"] for r in rows])), 1),
                       "dur_ms_max": max(r["dur_ms"] for r in rows),
                       "worst": [{"sent": t, "said": h} for _, t, h in bad[:15]]}
    return {"point": name, "unit": "word" if lang == "eng" else "char", "per_seed": per_seed,
            "thinker_text_same_across_seeds": round(sum(len(v) == 1 for v in texts.values()) / max(len(texts), 1), 4)}


MIN_FREE_GIB = 20.0   # an ASR process (Whisper-large-v3 / Paraformer, batch 4) starts only on a card with this much real room


def smi_free() -> dict:
    """GiB under the 135 GiB per-card cap, by real nvidia-smi use."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    return {int(i): 135.0 - float(u) / 1024 for i, u in (l.split(",") for l in out.strip().splitlines())}


def asr_cmd(inp: Path, outp: Path, logf: Path) -> str:
    return (f"{PYE} -m evaluation.asr_transcribe --input {inp} --output {outp} --device cuda:0 --batch-size 4 "
            f"> {logf} 2>&1")


def pipeline(cards: list[int], per_card: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    env0 = {**os.environ, "PYTHONPATH": ".", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
            "HF_HOME": "/opt/dlami/nvme/jasonleeeli/cache/huggingface",
            "MODELSCOPE_CACHE": "/opt/dlami/nvme/jasonleeeli/cache/modelscope"}
    runs = [(n, s) for s in SEEDS for n in PT]
    asr = []   # (Popen, card, label)
    pool = ProcessPoolExecutor(8)
    assembling, synth, fid_done = {}, None, (OUT / "fidelity").exists()
    block_asr_started = (OUT / "block_asr").exists()
    while True:
        for x in list(asr):
            if x[0].poll() is not None:
                log(f"asr {'done' if x[0].returncode == 0 else 'FAILED'} {x[2]} (card {x[1]}, rc {x[0].returncode})")
                asr.remove(x)
        for key, fut in list(assembling.items()):
            if fut.done():
                log(f"assembled {key}: {fut.result()}")
                del assembling[key]
        seeds_ready = [s for s in SEEDS if all((OUT / "synth" / f"done_{s * SHARDS_PER_SEED + i:03d}").exists()
                                               for i in range(SHARDS_PER_SEED))]
        if seeds_ready and any(not omni_run(PT[n], s).exists() for n, s in runs if s in seeds_ready):
            synth = synth_rows()
        for n, s in runs:
            if s in seeds_ready and not omni_run(PT[n], s).exists() and (n, s) not in assembling:
                assembling[(n, s)] = pool.submit(assemble, n, s, synth)
                log(f"assembling {n} seed {s}")
        if not fid_done and len(seeds_ready) == len(SEEDS):
            synth = synth or synth_rows()
            (OUT / "fidelity").mkdir()
            for n in PT:
                with open(OUT / "fidelity" / f"{n}.json", "x", encoding="utf-8") as fh:
                    json.dump(fidelity(n, synth), fh, ensure_ascii=False, indent=1)
            fid_done = True
            log("fidelity written")
        load = Counter(c for _, c, _ in asr)
        todo = []
        for n, s in runs:
            o = omni_run(PT[n], s)
            if (o / "results.jsonl").exists() and not (o / "asr_speech.jsonl").exists() and not (o / "asr.log").exists():
                todo.append((f"{n}_s{s}", o))
        if not block_asr_started and 0 in seeds_ready:   # secondary: seed 0 blocks, each its own wav
            ba = OUT / "block_asr"
            ba.mkdir()
            synth = synth or synth_rows()
            for d, lang in (("en2zh", "cmn"), ("zh2en", "eng")):
                with open(ba / f"blocks_{d}_s0.jsonl", "x", encoding="utf-8") as fh:
                    for k, r in synth.items():
                        if k.startswith(f"{d}_") and r["seed"] == 0 and r.get("wav16"):
                            fh.write(json.dumps({"id": k, "mode": "block", "tgt_lang": lang, "audio_path": str(OUT / r["wav16"]),
                                                 "audio_duration_seconds": r["dur_ms"] / 1000, "text_sent": r["text_sent"],
                                                 "error": None}, ensure_ascii=False) + "\n")
            block_asr_started = True
        for d in ("en2zh", "zh2en"):
            ba = OUT / "block_asr"
            if block_asr_started and not (ba / f"asr_{d}_s0.jsonl").exists() and not (ba / f"asr_{d}_s0.log").exists():
                todo.append((f"block_asr_{d}", ba / d))
        for label, o in todo:
            room = smi_free()
            free = [c for c in cards if load[c] < per_card and room.get(c, 0) >= MIN_FREE_GIB]
            if not free:
                break
            c = max(free, key=lambda c: room[c])
            env = {**env0, "CUDA_VISIBLE_DEVICES": str(c)}
            if label.startswith("block_asr_"):
                d = label.rsplit("_", 1)[1]
                ba = OUT / "block_asr"
                (ba / f"asr_{d}_s0.log").touch(exist_ok=False)
                cmd = asr_cmd(ba / f"blocks_{d}_s0.jsonl", ba / f"asr_{d}_s0.jsonl", ba / f"asr_{d}_s0.log")
            else:
                (o / "asr.log").touch(exist_ok=False)
                cmd = (f"{PYE} reports/uniss_si_v4/speech_only.py {o} > /dev/null && "
                       + asr_cmd(o / "results_speechonly.jsonl", o / "asr_speech.jsonl", o / "asr.log"))
            asr.append((subprocess.Popen(cmd, shell=True, cwd=REPO, env=env), c, label))
            load[c] += 1
            log(f"asr started {label} on card {c} ({room[c]:.0f} GiB room)")
            time.sleep(30)   # let its memory show before the next placement
        all_runs = all((omni_run(PT[n], s) / "asr_speech.jsonl").exists() for n, s in runs)
        blocks_done = all((OUT / "block_asr" / f"asr_{d}_s0.jsonl").exists() for d in ("en2zh", "zh2en"))
        if all_runs and blocks_done and fid_done and not asr and not assembling:
            log("PIPELINE_DONE")
            return
        time.sleep(20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("assemble")
    o.add_argument("point")
    o.add_argument("seed", type=int)
    f = sub.add_parser("fidelity")
    f.add_argument("point")
    p = sub.add_parser("pipeline")
    p.add_argument("--asr-cards", type=int, nargs="+", required=True)
    p.add_argument("--per-card", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "assemble":
        print(json.dumps(assemble(a.point, a.seed, synth_rows())))
    elif a.cmd == "fidelity":
        print(json.dumps(fidelity(a.point, synth_rows()), ensure_ascii=False, indent=1))
    else:
        pipeline(a.asr_cards, a.per_card)


if __name__ == "__main__":
    main()
