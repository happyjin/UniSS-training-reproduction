"""E7 (reports/mist_w3po_v1/e7_seamless_multispk_prereg.zh-CN.md): SeamlessStreaming (official SeamlessStreamingS2STAgent,
official defaults tau 0.5 / W 192 / C 50, fp32) on our ONLINE multi-speaker test windows, one window at a time.

A copy of experiments/s2st_baselines_v1/seamless/run_seamless.py (that file is not changed): the same agent build, the
same SimulEval SentenceLevelEvaluator loop (agent.reset() per window, 320 ms source steps, finished flag on the last
chunk, then EmptySegment(finished=True)), the same record-only subclasses. One addition, record only: the time (source
ms read so far) at which every text token reached the unit decoder, i.e. left the text decoder in that same pushpop.

Per window one row of <out>/part_<shard>.jsonl:
  segs      the emitted (non-empty) speech segments in order: text (the segment's text, single-speaker rule: a word
            cut by a segment boundary belongs to the segment of its start, <unk> no text), text_ms (source ms read when
            the segment's last token left the text decoder), write_ms (source ms read when the vocoder emitted the
            segment; SimulEval's delay, computation-unaware), dur_ms, wav (the segment's 16 kHz audio)
  unspoken  text the text decoder wrote but nobody spoke by the end of the window, with its text_ms
Speaker labels, streams, the scorer formats: build_e7.py (CPU).

    source experiments/s2st_baselines_v1/seamless/env.sh; CUDA_VISIBLE_DEVICES=<c> $PYS \
        experiments/s2st_baselines_v1/seamless_multispk/run_e7.py --windows <online windows jsonl> --tgt eng|cmn \
        --out <dir> --shard k --num-shards n [--limit N] [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_DIR = Path("/opt/dlami/nvme/jasonleeeli/models/seamless-streaming")
CJK = re.compile(r"[一-鿿]|[A-Za-z0-9']+")   # si_metrics.units
EN = re.compile(r"[A-Za-z0-9']+")
SR = 16000
# official streaming_evaluate s2st config (cli/streaming/evaluate.py, commit 9a081e93); fp32 as the Seamless paper
OFFICIAL = dict(source_segment_size=320, device="cuda:0", dtype="fp32", min_starting_wait_w2vbert=192,
                decision_threshold=0.5, no_early_stop=True, max_len_a=0, max_len_b=100, min_unit_chunk_size=50)


def register_local_weights() -> None:
    """run_seamless.register_local_weights, verbatim."""
    from fairseq2.assets import InProcAssetMetadataProvider, asset_store
    import seamless_communication  # noqa: F401  (registers the cards)
    f = lambda n: "file://" + str(MODEL_DIR / n)  # noqa: E731
    asset_store.user_metadata_providers.append(InProcAssetMetadataProvider([
        {"name": "seamless_streaming_unity@user", "checkpoint": f("seamless_streaming_unity.pt"),
         "char_tokenizer": f("spm_char_lang38_tc.model"), "tokenizer": f("tokenizer.model")},
        {"name": "seamless_streaming_monotonic_decoder@user", "checkpoint": f("seamless_streaming_monotonic_decoder.pt")},
        {"name": "vocoder_v2@user", "checkpoint": f("vocoder_v2.pt")},
        {"name": "seamlessM4T_v2_large@user", "char_tokenizer": f("spm_char_lang38_tc.model")},
    ]))


REC: dict = {}


def build_agent(tgt: str, cfg: dict):
    """run_seamless.build_agent; RecUnitDecoder additionally stamps each arriving text token with REC['now_ms']."""
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
            if len(states.source_token_list) < len(REC["tok_ms"]):
                del REC["tok_ms"][len(states.source_token_list):]
            while len(REC["tok_ms"]) < len(states.source_token_list):   # tokens pushed in this pushpop
                REC["tok_ms"].append(REC["now_ms"])
            act = super().policy(states)
            if (isinstance(act, WriteAction) and isinstance(act.content, TextSegment)
                    and hasattr(act.content.content, "numel") and act.content.content.numel() > 0):
                REC["pending"].append((list(states.source_token_list[REC["n_spoken"]:]),
                                       list(REC["tok_ms"][REC["n_spoken"]:len(states.source_token_list)])))
                REC["n_spoken"] = len(states.source_token_list)
            return act

    class RecVocoder(VocoderAgent):
        def policy(self, states):
            act = super().policy(states)
            if isinstance(act, WriteAction) and isinstance(act.content, SpeechSegment) and len(act.content.content) > 0:
                REC["speech_tokens"].append(REC["pending"].pop(0) if REC["pending"] else ([], []))
            return act

    class RecSeamlessStreamingS2STAgent(SeamlessStreamingS2STAgent):
        pipeline = [OnlineFeatureExtractorAgent, OfflineWav2VecBertEncoderAgent, UnitYMMATextDecoderAgent,
                    RecUnitDecoder, RecVocoder]

    se_cli.EVALUATION_SYSTEM_LIST.append(RecSeamlessStreamingS2STAgent)
    config = {"task": "s2st", "tgt_lang": tgt, **cfg, "output": cfg["_tmp_output"],
              "latency_metrics": "StartOffset EndOffset"}
    config.pop("_tmp_output")
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        agent, args = build_system_args(config)
    finally:
        sys.argv = argv
    return agent, args


def block_texts(seg_tokens: list[list[str]], tgt: str) -> list[str]:
    """run_seamless.block_texts, verbatim."""
    raws = ["".join(x for x in t if x != "<unk>").replace("▁", " ") for t in seg_tokens]
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


ZH_SP = re.compile(r"\s+(?=[一-鿿　-〿＀-￯])|(?<=[一-鿿　-〿＀-￯])\s+")


def zh_nospace(t: str) -> str:
    """prereg §3: Chinese output, whitespace next to a CJK character / CJK punctuation removed (Latin spacing kept)."""
    return ZH_SP.sub("", t)


def run_window(agent, args, wav: np.ndarray, tgt: str) -> dict:
    """run_seamless.run_sentence (SimulEval SentenceLevelEvaluator + SpeechToSpeechInstance), plus REC['now_ms']."""
    from simuleval.data.segments import EmptySegment, SpeechSegment
    REC.update(n_spoken=0, pending=[], speech_tokens=[], restarts=0, tok_ms=[], now_ms=0.0)
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
        REC["now_ms"] = step * 1000 / SR
        out = agent.pushpop(seg)
        if finish_prediction and source_finished_reading:
            break
        finish_prediction = out.finished
        if out.is_empty or len(out.content) == 0:
            continue
        tok, tms = REC["speech_tokens"][-1] if REC["speech_tokens"] else ([], [])
        segs.append({"delay_ms": step * 1000 / SR, "wav": np.asarray(out.content, dtype=np.float32),
                     "sr": out.sample_rate, "tokens": tok, "tok_ms": tms})
    return {"segs": segs, "elapsed_s": time.time() - t0, "restarts": REC["restarts"],
            "n_speech_records": len(REC["speech_tokens"]), "unspoken": REC["pending"]}


def make_record(w: dict, wav: np.ndarray, res: dict, err, out: Path, tgt: str) -> dict:
    segs = res["segs"]
    for s in segs:
        assert s["sr"] == SR, s["sr"]
    un_tok = sum((t for t, _ in res["unspoken"]), [])
    un_ms = sum((m for _, m in res["unspoken"]), [])
    texts = block_texts([s["tokens"] for s in segs] + ([un_tok] if un_tok else []), tgt)
    if tgt == "cmn":
        texts = [zh_nospace(t) for t in texts]
    rows = []
    for k, s in enumerate(segs):
        p = out / "wav" / f"{w['id']}_{k:03d}.wav"
        sf.write(p, s["wav"], SR)
        rows.append({"text": texts[k], "text_ms": int(round(max(s["tok_ms"]))) if s["tok_ms"] else int(round(s["delay_ms"])),
                     "write_ms": int(round(s["delay_ms"])), "dur_ms": int(round(len(s["wav"]) / 16)),
                     "n_tokens": len(s["tokens"]), "wav": str(p)})
    return {"id": w["id"], "meeting": w["meeting"], "dur_ms": int(w["dur_ms"]), "src_samples": int(len(wav)),
            "segs": rows, "unspoken": {"text": texts[len(segs)], "text_ms": int(round(max(un_ms)))} if un_tok else None,
            "restarts": res["restarts"], "speech_records": res["n_speech_records"],
            "elapsed_s": round(res["elapsed_s"], 3), "error": err}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--tgt", choices=["eng", "cmn"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N windows of the shard (smoke test only)")
    ap.add_argument("--resume", action="store_true", help="append to an existing part file, skipping its ids")
    a = ap.parse_args()
    W = [json.loads(l) for l in open(a.windows, encoding="utf-8")]
    items = W[a.shard:: a.num_shards]
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    part = out / f"part_{a.shard}.jsonl"
    done = set()
    if part.exists():
        if not a.resume:
            sys.exit(f"refuse: {part} exists (use --resume)")
        for l in open(part, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except json.JSONDecodeError:
                pass
    tmp_out = out / f"simuleval_tmp_{a.shard}"
    tmp_out.mkdir(exist_ok=True)
    cfg = dict(OFFICIAL)
    print(f"[{time.strftime('%H:%M:%S')}] windows {len(W)}, shard {a.shard}/{a.num_shards}: {len(items)} "
          f"(done {len(done)}); tgt {a.tgt}; config {cfg}", flush=True)
    register_local_weights()
    agent, args = build_agent(a.tgt, dict(cfg, _tmp_output=str(tmp_out)))
    t_start = time.time()
    with open(part, "a", encoding="utf-8") as fh:
        for n, w in enumerate(items):
            if w["id"] in done:
                continue
            x, sr = sf.read(w["audio"], dtype="float32", always_2d=True)
            assert sr == SR, w["audio"]
            wav = x.mean(axis=1)
            err = None
            try:
                res = run_window(agent, args, wav, a.tgt)
            except Exception as e:  # recorded; the window then scores as empty
                err = f"{type(e).__name__}: {e}"
                res = {"segs": [], "elapsed_s": 0.0, "restarts": 0, "n_speech_records": 0, "unspoken": []}
            rec = make_record(w, wav, res, err, out, a.tgt)
            rec["config"] = cfg
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {n + 1}/{len(items)} {w['id']} dur={w['dur_ms']} "
                      f"segs={len(rec['segs'])} t={res['elapsed_s']:.2f}s err={err}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] E7_SHARD_DONE {part} {len(items)} windows in {time.time() - t_start:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
