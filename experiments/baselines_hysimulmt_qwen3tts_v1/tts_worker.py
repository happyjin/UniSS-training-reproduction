"""One Qwen3-TTS-12Hz-0.6B-Base worker (registration: paper/baselines_run_plan_hysimulmt_qwen3tts_v1.md).

Loads the model once (bf16, sdpa, as SimulInfer's tts_backends/qwen.py), then claims batch jobs of jobs.jsonl in order
(O_EXCL claim files under <jobs folder>/claims, up to 3 attempts per job) and synthesises every block of the batch in
one batched call that is generate_voice_clone's own code path (tokenise, model.generate, speech_tokenizer.decode):
  voice    x-vector only (speaker embedding of the job's reference clip, resampled to the speaker encoder's rate, as
           SimulInfer's make_prompt(embedding_only=True)); no ICL reference text, no history
  text     the block text, language Chinese / English, non_streaming_mode=True (SimulInfer's vllm_omni Base request
           with a speaker-only reference)
  decoding the checkpoint's generation_config.json sampling values (nothing overridden); torch.manual_seed(job
           torch_seed) per batch
  length   amendment 1: each block may have at most cap = min(1800, max(180, 18 * len(text))) codec frames, the
           per-request limit of SimulInfer's own qwen-tts backend (tts_backends/qwen.py); the batch generates up to the
           largest cap of its blocks and each block's codes are cut BEFORE decoding to what one request with
           max_new_tokens = cap returns: qwen-tts hands back max_new_tokens - 1 codec frames when it stops at the limit
           (the last sampled token gets no residual codebooks), so a block keeps at most cap - 1 frames (rows of a batch
           do not interact, so this equals one request per block); hit_cap = no codec EOS within those cap - 1 frames
Outputs (never overwritten): <out>/wav24/<job>_a<n>/<id>__b<k>.wav (24 kHz PCM16) and <out>/done/<job>.json.
--reserve-gib: the worker takes that much from the CUDA caching allocator once and frees it into the cache, so the
card's guard sees one process above its small-process limit while the worker runs (no placeholder stacks on top).

    CUDA_VISIBLE_DEVICES=<card> venvs/tts_schemes/bin/python experiments/baselines_hysimulmt_qwen3tts_v1/tts_worker.py \
        eval_outputs/baselines_hysimulmt_qwen3tts_v1/a1/jobs.jsonl [--reserve-gib 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
MODEL = "/opt/dlami/nvme/jasonleeeli/models/Qwen3-TTS-12Hz-0.6B-Base"
LOGS = REPO / "logs/baselines_hysimulmt_qwen3tts"
CLAIMS = Path()   # <jobs folder>/claims, set in main
STOP = Path()     # logs/baselines_hysimulmt_qwen3tts/STOP_<jobs folder name>, set in main


def frame_cap(text: str) -> int:
    """SimulInfer tts_backends/qwen.py: max_new_tokens=min(1800, max(180, len(text) * 18))."""
    return min(1800, max(180, len(text) * 18))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def claim(name: str) -> bool:
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(CLAIMS / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--reserve-gib", type=float, default=10.0)
    ap.add_argument("--only", default="", help="comma-separated job-name prefixes to take (default: all)")
    a = ap.parse_args()
    global CLAIMS, STOP
    CLAIMS = Path(a.jobs).resolve().parent / "claims"
    STOP = LOGS / f"STOP_{Path(a.jobs).resolve().parent.name}"
    jobs = [json.loads(l) for l in open(a.jobs, encoding="utf-8")]
    if a.only:
        jobs = [j for j in jobs if any(j["job"].startswith(p) for p in a.only.split(","))]

    import librosa
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    torch.set_num_threads(4)
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
                                          local_files_only=True)
    spk_sr = int(model.model.speaker_encoder_sample_rate)
    log(f"loaded {MODEL}; speaker encoder {spk_sr} Hz; generate kwargs {model._merge_generate_kwargs()}")
    if a.reserve_gib > 0:
        x = torch.empty(int(a.reserve_gib * 2**30), dtype=torch.uint8, device="cuda:0")
        del x
        log(f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB in the caching allocator")
    spk_cache: dict[str, object] = {}

    def prompt(ref: str):
        if ref not in spk_cache:
            audio, sr = sf.read(REPO / ref, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != spk_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=spk_sr)
            emb = model.model.extract_speaker_embedding(audio=audio, sr=spk_sr)
            spk_cache[ref] = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=emb, x_vector_only_mode=True,
                                                  icl_mode=False, ref_text=None)
        return spk_cache[ref]

    def run_job(j: dict, n: int, out: Path) -> bool:
        t0 = time.time()
        try:
            with torch.inference_mode():
                items = [prompt(b["ref"]) for b in j["blocks"]]
            texts, langs = [b["text"] for b in j["blocks"]], [j["language"]] * len(j["blocks"])
            caps = [frame_cap(t) for t in texts]
            model._validate_languages(langs)
            input_ids = model._tokenize_texts([model._build_assistant_text(t) for t in texts])
            torch.manual_seed(j["torch_seed"])
            with torch.inference_mode():
                codes, _ = model.model.generate(input_ids=input_ids, ref_ids=[None] * len(texts),
                                                voice_clone_prompt=model._prompt_items_to_voice_clone_prompt(items),
                                                languages=langs, non_streaming_mode=True,
                                                **model._merge_generate_kwargs(max_new_tokens=max(caps)))
                assert len(codes) == len(texts)
                lens = [int(c.shape[0]) for c in codes]
                codes = [c[: cap - 1] for c, cap in zip(codes, caps)]
                wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes])
            wdir = out / "wav24" / f"{j['job']}_a{n}"
            wdir.mkdir(parents=True, exist_ok=False)
            rows = []
            for b, w, c, cap, ln in zip(j["blocks"], wavs, codes, caps, lens):
                w = np.asarray(w, dtype=np.float32).reshape(-1)
                p = wdir / f"{b['id']}__b{b['k']}.wav"
                with open(p, "xb") as fh:
                    sf.write(fh, w, sr, format="WAV", subtype="PCM_16")
                rows.append({"id": b["id"], "k": b["k"], "text": b["text"], "wav": str(p.relative_to(REPO)),
                             "sr": int(sr), "samples": int(w.size), "frames": int(c.shape[0]), "cap": cap,
                             "hit_cap": ln >= cap - 1})
            (out / "done").mkdir(parents=True, exist_ok=True)
            rec = {"job": j["job"], "attempt": n, "torch_seed": j["torch_seed"], "seconds": round(time.time() - t0, 2),
                   "audio_seconds": round(sum(r["samples"] for r in rows) / sr, 2), "batch_max_new_tokens": max(caps),
                   "n_hit_cap": sum(r["hit_cap"] for r in rows), "blocks": rows,
                   "card": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid()}
            with open(out / "done" / f"{j['job']}.json", "x", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False)
            log(f"done {j['job']} a{n}: {len(rows)} blocks, {rec['audio_seconds']:.1f} s audio in {rec['seconds']:.1f} s")
            return True
        except Exception as e:  # noqa: BLE001
            claim(f"failed_{j['job']}_a{n}")
            log(f"FAILED {j['job']} a{n}: {e!r}")
            return False

    n_done, claimed = 0, True
    while claimed:   # passes over the job list until one claims nothing (a failed job is retried in a later pass)
        claimed = False
        for j in jobs:
            if STOP.exists():
                log("STOP file: exiting")
                return
            out = REPO / j["out"]
            if (out / "done" / f"{j['job']}.json").exists():
                continue
            n = next((n for n in (1, 2, 3) if not (CLAIMS / f"{j['job']}_a{n}").exists()), None)
            if n is None or (n > 1 and not (CLAIMS / f"failed_{j['job']}_a{n - 1}").exists()):
                continue   # three attempts used, or another worker holds the current attempt
            if not claim(f"{j['job']}_a{n}"):
                continue
            claimed = True
            n_done += run_job(j, n, out)
    log(f"WORKER_PASS_DONE ({n_done} jobs)")


if __name__ == "__main__":
    main()
