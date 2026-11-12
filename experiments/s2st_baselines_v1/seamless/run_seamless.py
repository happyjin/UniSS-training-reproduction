"""SeamlessStreaming (Meta, official SimulEval S2ST agent SeamlessStreamingS2STAgent) on RealSI sentence level, or on a
dev jsonl for grid calibration. R1 runner of paper/baselines_run_plan_seamless_v1.md.

The loop is SimulEval 1.1.4's SentenceLevelEvaluator verbatim (agent.reset(); while not finished: send 320 ms of source,
agent.pushpop, receive) with the agent built by simuleval.utils.agent.build_system_args from the official
streaming_evaluate s2st config (seamless_communication/cli/streaming/evaluate.py), only decision_threshold (and, if the
registered grid says so, min_starting_wait_w2vbert) changed. Two thin subclasses only RECORD which text tokens each
emitted speech segment speaks (the unit decoder emits units for all text tokens received since its previous write);
they do not change any decision.

Per sentence one si_infer-style results.jsonl row (paper protocol): blocks = the emitted speech segments, each with
write_ms (source ms read when emitted = SimulEval's delay, computation-unaware), start_ms (playback start, queued:
max(write_ms, end of previous segment)), dur_ms, text (the segment's text; a word split across segments belongs to the
segment of its first piece). speech_path = the segments concatenated without waits (for ASR, the paper's speech-only
ASR); audio_path = the timeline wav.

    PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/venvs/s2st_baselines/bin/python \
        experiments/s2st_baselines_v1/seamless/run_seamless.py --direction en2zh --decision-threshold 0.5 \
        --out <run>/realsi_en2zh --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
REALSI = REPO / "data/raw/realsi/RealSI_data/data"
MODEL_DIR = Path("/opt/dlami/nvme/jasonleeeli/models/seamless-streaming")
LANGS = {"en2zh": ("eng", "cmn"), "zh2en": ("cmn", "eng")}
CJK = re.compile(r"[一-鿿]|[A-Za-z0-9']+")   # si_metrics.units
EN = re.compile(r"[A-Za-z0-9']+")
SR = 16000
# official streaming_evaluate s2st config (cli/streaming/evaluate.py, commit 9a081e93)
OFFICIAL = dict(source_segment_size=320, device="cuda:0", dtype="fp16", min_starting_wait_w2vbert=192,
                decision_threshold=0.5, no_early_stop=True, max_len_a=0, max_len_b=100, min_unit_chunk_size=50)


def realsi_items(direction: str) -> list[dict]:
    """experiments.uniss_qwen3_omni_v1.evaluation.si_infer.realsi_items, verbatim (copied: that module needs vLLM)."""
    items = []
    for f in sorted(glob.glob(str(REALSI / direction / "json" / "*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        for k, s in enumerate(j["segment"]):
            subs = [u for u in s.get("utterance", []) if u.get("sub_trg_text", "").strip()]
            if not subs or not s.get("trg_text", "").strip():
                continue
            items.append({"id": f"{j['vid']}_{k:03d}", "wav": str(REALSI / direction / "wav" / f"{j['vid']}.wav"),
                          "start": int(s["start_time"]), "end": int(s["end_time"]), "src_text": s["src_text"],
                          "trg_text": s["trg_text"], "subs": [(int(u["sub_end_time"]), u["sub_trg_text"].strip())
                                                              for u in subs]})
    return items


def dev_items(path: str) -> list[dict]:
    """data/si_spont_v1/dev_<d>_ref.jsonl rows {id, audio, dur_ms, transcript, translation} (not a test set)."""
    return [{"id": r["id"], "audio": r["audio"], "src_text": r["transcript"], "trg_text": r["translation"]}
            for r in map(json.loads, open(path, encoding="utf-8"))]


def register_local_weights() -> None:
    """Point fairseq2's asset cards at the pre-downloaded HF facebook/seamless-streaming files (the '@user' override
    fairseq2 0.2 provides for local checkpoints). The text tokenizer: the HF seamless-streaming tokenizer.model (sha256
    026a7682...), the same file (same xet hash 6b69da23...) as the card's seamless-m4t-large/tokenizer.model."""
    from fairseq2.assets import InProcAssetMetadataProvider, asset_store
    import seamless_communication  # noqa: F401  (registers the cards)
    f = lambda n: "file://" + str(MODEL_DIR / n)  # noqa: E731
    asset_store.user_metadata_providers.append(InProcAssetMetadataProvider([
        {"name": "seamless_streaming_unity@user", "checkpoint": f("seamless_streaming_unity.pt"),
         "char_tokenizer": f("spm_char_lang38_tc.model"), "tokenizer": f("tokenizer.model")},
        {"name": "seamless_streaming_monotonic_decoder@user", "checkpoint": f("seamless_streaming_monotonic_decoder.pt")},
        {"name": "vocoder_v2@user", "checkpoint": f("vocoder_v2.pt")},
        # the T2U's NAR decoder loads seamlessM4T_v2_large's char tokenizer: same file (sha256 9e7f2075...) as ours
        {"name": "seamlessM4T_v2_large@user", "char_tokenizer": f("spm_char_lang38_tc.model")},
    ]))


REC: dict = {}


def build_agent(tgt: str, cfg: dict):
    from simuleval import cli as se_cli
    from simuleval.agents.actions import WriteAction
    from simuleval.data.segments import SpeechSegment, TextSegment
    from simuleval.utils.agent import build_system_args
    from seamless_communication.streaming.agents.offline_w2v_bert_encoder import OfflineWav2VecBertEncoderAgent
    from seamless_communication.streaming.agents.online_feature_extractor import OnlineFeatureExtractorAgent
    from seamless_communication.streaming.agents.online_text_decoder import UnitYMMATextDecoderAgent
    from seamless_communication.streaming.agents.online_unit_decoder import NARUnitYUnitDecoderAgent
    from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
    from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent

    class RecUnitDecoder(NARUnitYUnitDecoderAgent):
        def policy(self, states):
            if len(states.source_token_list) < REC["n_spoken"]:   # pipeline restarted after an early stop
                REC["n_spoken"] = 0
                REC["restarts"] += 1
            act = super().policy(states)
            if (isinstance(act, WriteAction) and isinstance(act.content, TextSegment)
                    and hasattr(act.content.content, "numel") and act.content.content.numel() > 0):
                REC["pending"].append(list(states.source_token_list[REC["n_spoken"]:]))
                REC["n_spoken"] = len(states.source_token_list)
            return act

    class RecVocoder(VocoderAgent):
        def policy(self, states):
            act = super().policy(states)
            if isinstance(act, WriteAction) and isinstance(act.content, SpeechSegment) and len(act.content.content) > 0:
                REC["speech_tokens"].append(REC["pending"].pop(0) if REC["pending"] else [])
            return act

    class RecSeamlessStreamingS2STAgent(SeamlessStreamingS2STAgent):
        pipeline = [OnlineFeatureExtractorAgent, OfflineWav2VecBertEncoderAgent, UnitYMMATextDecoderAgent,
                    RecUnitDecoder, RecVocoder]

    se_cli.EVALUATION_SYSTEM_LIST.append(RecSeamlessStreamingS2STAgent)
    config = {"task": "s2st", "tgt_lang": tgt, **cfg, "output": cfg["_tmp_output"],
              "latency_metrics": "StartOffset EndOffset"}   # no dataloader: the source is fed by run_sentence
    config.pop("_tmp_output")
    argv, sys.argv = sys.argv, sys.argv[:1]   # simuleval also parses sys.argv; give it only the config
    try:
        agent, args = build_system_args(config)
    finally:
        sys.argv = argv
    return agent, args


def block_texts(seg_tokens: list[list[str]], tgt: str) -> list[str]:
    """each segment's text; a unit (si_metrics.units) cut by a segment boundary goes to the segment of its start."""
    raws = ["".join(x for x in t if x != "<unk>").replace("▁", " ") for t in seg_tokens]   # <unk>: no text
    full = "".join(raws)
    b = [0]
    for r in raws:
        b.append(b[-1] + len(r))
    rx = CJK if tgt == "cmn" else EN
    for m in rx.finditer(full):
        for k in range(1, len(b) - 1):
            if m.start() < b[k] < m.end():
                b[k] = m.end()
    for k in range(1, len(b)):
        b[k] = max(b[k], b[k - 1])
    out = [full[b[k]: b[k + 1]] for k in range(len(raws))]
    if out:
        out[0] = out[0].lstrip()
    return out


def run_sentence(agent, args, wav: np.ndarray, tgt: str) -> dict:
    """SimulEval SentenceLevelEvaluator + SpeechToSpeechInstance (send_source / receive_prediction), verbatim logic."""
    from simuleval.data.segments import EmptySegment, SpeechSegment
    REC.update(n_spoken=0, pending=[], speech_tokens=[], restarts=0)
    agent.reset()
    samples = wav.tolist()
    step, finish_prediction, source_finished_reading = 0, False, False
    num = math.ceil(args.source_segment_size / 1000 * SR)
    segs = []
    t0 = time.time()
    while not finish_prediction:
        if step < len(samples):
            is_finished = step + num >= len(samples)
            chunk = samples[step:] if is_finished else samples[step: step + num]
            source_finished_reading = source_finished_reading or is_finished
            step = min(step + num, len(samples))
            seg = SpeechSegment(index=step * 1000 / SR, content=chunk, sample_rate=SR, finished=is_finished, tgt_lang=tgt)
        else:
            seg = EmptySegment(index=step * 1000 / SR, finished=True)
            source_finished_reading = True
        out = agent.pushpop(seg)
        if finish_prediction and source_finished_reading:
            break
        finish_prediction = out.finished
        if out.is_empty or len(out.content) == 0:
            continue
        segs.append({"delay_ms": step * 1000 / SR, "wav": np.asarray(out.content, dtype=np.float32),
                     "sr": out.sample_rate, "tokens": REC["speech_tokens"][-1] if REC["speech_tokens"] else []})
    n_tok_spoken = sum(len(s["tokens"]) for s in segs)
    return {"segs": segs, "elapsed_s": time.time() - t0, "restarts": REC["restarts"],
            "n_speech_records": len(REC["speech_tokens"]), "unspoken_tokens": REC["pending"],
            "n_tok_spoken": n_tok_spoken}


def set_knobs(agent, cfg: dict) -> None:
    """set the three latency knobs on a built pipeline exactly where the agents' __init__ put them from args (dev
    calibration loops several configs on one loaded model; RealSI runs build a fresh agent per config)."""
    from seamless_communication.streaming.agents.offline_w2v_bert_encoder import OfflineWav2VecBertEncoderAgent
    from seamless_communication.streaming.agents.online_text_decoder import UnitYMMATextDecoderAgent
    from seamless_communication.streaming.agents.online_unit_decoder import NARUnitYUnitDecoderAgent
    seen = set()
    for m in agent.module_list:
        if isinstance(m, OfflineWav2VecBertEncoderAgent):
            m.min_starting_wait = cfg["min_starting_wait_w2vbert"]
            seen.add("enc")
        elif isinstance(m, UnitYMMATextDecoderAgent):
            m.decision_threshold = cfg["decision_threshold"]
            seen.add("txt")
        elif isinstance(m, NARUnitYUnitDecoderAgent):
            m.min_unit_chunk_size = cfg["min_unit_chunk_size"]
            seen.add("unit")
    assert seen == {"enc", "txt", "unit"}, seen


def load_wav(it: dict, cache: dict) -> np.ndarray:
    if "wav" in it:   # RealSI: the sentence cut of the talk, as si_infer
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == SR, it["wav"]
            cache.clear()
            cache[it["wav"]] = x.mean(axis=1)
        return cache[it["wav"]][it["start"] * 16: it["end"] * 16]
    x, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
    wav = x.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR).astype(np.float32)
    return wav


def make_record(it: dict, wav: np.ndarray, res: dict, err, out: Path, src_lang: str, tgt: str, run_cfg: dict) -> dict:
    name = it["id"].replace("/", "__")
    src_path = out / "src" / f"{name}.wav"
    sf.write(src_path, wav, SR)
    X = len(wav) // 16
    segs = res["segs"]
    for s in segs:
        assert s["sr"] == SR, s["sr"]
    unspoken = sum(res["unspoken_tokens"], [])
    texts = block_texts([s["tokens"] for s in segs] + ([unspoken] if unspoken else []), tgt)
    blocks, end_prev = [], 0.0
    for k, s in enumerate(segs):
        dur = len(s["wav"]) / 16
        start = max(s["delay_ms"], end_prev)
        end_prev = start + dur
        blocks.append({"write_ms": int(round(s["delay_ms"])), "start_ms": int(round(start)),
                       "dur_ms": int(round(dur)), "text": texts[k], "n_tokens": len(s["tokens"])})
    unspoken_text = texts[len(segs)] if unspoken else ""
    speech = np.concatenate([s["wav"] for s in segs]) if segs else np.zeros(320, dtype=np.float32)
    total = int(max(X, end_prev)) + 300
    right = np.zeros(total * 16, dtype=np.float32)
    for b, s in zip(blocks, segs):
        a0 = b["start_ms"] * 16
        right[a0: a0 + len(s["wav"])] += s["wav"][: len(right) - a0]
    sim_path = out / "wav" / f"{name}_simul.wav"
    speech_path = out / "wav" / f"{name}_speech.wav"
    sf.write(sim_path, right, SR)
    sf.write(speech_path, speech, SR)
    full = "".join(b["text"] for b in blocks) + unspoken_text
    return {"id": it["id"], "mode": "si", "name": it["id"], "src_lang": src_lang, "tgt_lang": tgt,
            "transcription_ref": it["src_text"], "translation_ref": it["trg_text"],
            "translated_text": full.strip(), "audio_path": str(sim_path), "speech_path": str(speech_path),
            "audio_duration_seconds": round(len(right) / SR, 3), "source_audio_path": str(src_path),
            "source_duration_ms": X, "error": err,
            "start_offset_ms": blocks[0]["start_ms"] if blocks else None,
            "end_offset_ms": int(end_prev - X) if blocks else None,
            "blocks": blocks, "unspoken_text": unspoken_text, "restarts": res["restarts"],
            "speech_records": res["n_speech_records"], "elapsed_s": round(res["elapsed_s"], 3), "config": run_cfg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", choices=sorted(LANGS), required=True)
    ap.add_argument("--decision-threshold", type=float, default=OFFICIAL["decision_threshold"])
    ap.add_argument("--min-starting-wait-w2vbert", type=int, default=OFFICIAL["min_starting_wait_w2vbert"])
    ap.add_argument("--min-unit-chunk-size", type=int, default=OFFICIAL["min_unit_chunk_size"])
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32",
                    help="fp32: the precision of the Seamless paper's numbers (cli/streaming/README.md); CLI default fp16")
    ap.add_argument("--out", required=True, help="the run's realsi_<dir> (or dev_<dir>) folder")
    ap.add_argument("--items", default="realsi", help="realsi, or a dev jsonl (data/si_spont_v1/dev_<d>_ref.jsonl)")
    ap.add_argument("--dev-configs", default="",
                    help="DEV CALIBRATION ONLY (refused with --items realsi): thr:wait:chunk,... run in turn on one "
                         "loaded model, each into <out>/t<thr>_w<wait>_c<chunk>/")
    ap.add_argument("--limit", type=int, default=0, help="first N items (after sharding); dev calibration only")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="append to an existing part file, skipping its ids")
    a = ap.parse_args()
    if a.items == "realsi" and (a.dev_configs or a.limit):
        sys.exit("refuse: --dev-configs / --limit are for dev calibration, not RealSI")
    src_lang, tgt = LANGS[a.direction]
    base = dict(OFFICIAL, decision_threshold=a.decision_threshold, min_starting_wait_w2vbert=a.min_starting_wait_w2vbert,
                min_unit_chunk_size=a.min_unit_chunk_size, dtype=a.dtype)
    if a.dev_configs:
        plan = []
        for c in a.dev_configs.split(","):
            t, w, ch = c.split(":")
            cfg = dict(base, decision_threshold=float(t), min_starting_wait_w2vbert=int(w), min_unit_chunk_size=int(ch))
            plan.append((cfg, Path(a.out) / f"t{float(t):g}_w{int(w)}_c{int(ch)}"))
    else:
        plan = [(base, Path(a.out))]

    items = realsi_items(a.direction) if a.items == "realsi" else dev_items(a.items)
    n_all = len(items)
    items = items[a.shard:: a.num_shards]
    if a.limit:
        items = items[: a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp_out = Path(a.out) / f"simuleval_tmp_{a.shard}"
    tmp_out.mkdir(exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] {a.direction} items {n_all}, shard {a.shard}/{a.num_shards}: {len(items)}; "
          f"configs {[c for c, _ in plan]}", flush=True)
    register_local_weights()
    agent, args = build_agent(tgt, dict(plan[0][0], _tmp_output=str(tmp_out)))
    cache: dict = {}
    for cfg, out in plan:
        set_knobs(agent, cfg)
        run_cfg = {"system": "SeamlessStreaming (SeamlessStreamingS2STAgent)", "direction": a.direction,
                   "code": "seamless_communication 9a081e93, simuleval 1.1.4, fairseq2 0.2.1", **cfg}
        part = out / f"part_{a.shard}.jsonl"
        done = set()
        if part.exists():
            if not a.resume:
                sys.exit(f"refuse: {part} exists (use --resume)")
            done = {json.loads(l)["id"] for l in open(part, encoding="utf-8")}
        for d in ("wav", "src"):
            (out / d).mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        with open(part, "a", encoding="utf-8") as fh:
            for n, it in enumerate(items):
                if it["id"] in done:
                    continue
                wav = load_wav(it, cache)
                err = None
                try:
                    res = run_sentence(agent, args, wav, tgt)
                except Exception as e:  # recorded; the sentence then scores as empty
                    err = f"{type(e).__name__}: {e}"
                    res = {"segs": [], "elapsed_s": 0.0, "restarts": 0, "n_speech_records": 0, "unspoken_tokens": [],
                           "n_tok_spoken": 0}
                rec = make_record(it, wav, res, err, out, src_lang, tgt, run_cfg)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if n % 20 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {out.name} {n + 1}/{len(items)} {it['id']} "
                          f"X={rec['source_duration_ms']} segs={len(rec['blocks'])} t={res['elapsed_s']:.2f}s err={err}",
                          flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] CONFIG_DONE {out} {len(items)} items in {time.time() - t_start:.0f}s",
              flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
