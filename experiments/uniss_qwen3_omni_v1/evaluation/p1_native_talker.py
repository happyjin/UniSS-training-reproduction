"""P1: Qwen3-Omni's own talker on the same CVSS-T sentences, no training.

The end-to-end route loses a quarter of its Text-BLEU between the text and
the waveform, almost all of it as misread words, and the talker that does
it was transplanted from Qwen2.5-Omni to read Qwen3's states. Qwen3-Omni
ships a talker of its own, trained on exactly these states, and its input
is built differently: the reply reaches it as token embeddings only, and
what the thinker heard enters as context from the user turn's audio
positions. This runs that talker untouched, to measure what a native
talker built that way gets on the sentences phase_h and the cascade were
scored on.

Two systems come out of one generation:

    p1a  the official pipeline as shipped -- thinker, talker, residual-code
         predictor, code2wav -- in a preset voice, resampled 24 -> 16 kHz.
    p1b  p1a's speech re-encoded by BiCodec, its 32 global tokens replaced
         by the source speaker's, and decoded again. BiCodec keeps content
         in its semantic tokens and voice in its global ones, which is how
         UniSS keeps the speaker; this asks whether the official talker's
         content survives being put back in the speaker's voice.

The thinker prompt is phase_h's, built by the same code path and decoded
greedily to the same end-of-turn token, so the translation matches and
whatever differs is the speech side. The talker decodes with the official
defaults: this is a measurement of the model as released.

At full test-set size two things change, both opt-in. --batch-size runs
several utterances per generate call (left-padded, so every prompt ends on
the same assistant turn, which the native generate assumes); the work is
taken longest first so a batch holds similar lengths. --claims replaces the
fixed shard with a shared directory where each utterance is claimed by
creating its file, so a worker that finishes early takes the next free one
instead of idling while the slowest shard finishes. A batch that fails is
redone one utterance at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from experiments.uniss_qwen3_omni_v1.evaluation.phase_d_e2e import INSTRUCTION

QWEN3 = "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
SYSTEMS = ("p1a", "p1b")


def record(row, index, hypothesis, audio_path, codes, extra):
    """phase_d_e2e's record shape, so the full-set metric harness reads it."""
    name = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
    return {
        "index": index, "name": name, "id": row["id"], "mode": "tts",
        "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"],
        "dataset_name": "CVSS-T",
        "transcription_ref": row["transcription"],
        "translation_ref": row["translation"],
        "generated_text_raw": hypothesis, "generated_text_clean": hypothesis,
        "translated_text": hypothesis, "reference_text": row["translation"],
        "source_audio_path": row["source_audio_path"],
        "reference_audio_path": row["reference_audio_path"],
        "source_audio_duration_seconds": row["source_audio_duration_seconds"],
        "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
        "synthetic_source": bool(row["synthetic_source"]),
        "synthetic_reference": bool(row["synthetic_reference"]),
        "codes": codes, "gold_codes": len(row["target_bicodec"]),
        "audio_path": audio_path, "error": None if audio_path else "no audio",
        **extra,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", required=True,
                    help="name:parquet[:subset_jsonl], e.g. cmn_eng:/.../zh_en_test.parquet:/.../subset.jsonl; "
                         "without a subset, every row of the parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0, help="this worker's number; with --claims it only names the output file")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--claims", default="",
                    help="a directory shared by one round's workers; utterances are taken first come, not by shard")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="talker sampling seed, offset by the batch's first index")
    ap.add_argument("--speaker", default="Ethan")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--talker-max-new-tokens", type=int, default=4096,
                    help="the official default; a batch runs until its longest row stops, so a runaway costs all of it")
    ap.add_argument("--limit", type=int, default=0, help="per direction, for smoke runs")
    ap.add_argument("--speech-tokenizer", default="pretrained_models/UniSS")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from uniss import UniSSTokenizer

    out = Path(args.output)
    work = []
    for spec in args.pair:
        name, parquet, *subset = spec.split(":")
        rows = pq.read_table(parquet, columns=[
            "id", "translation", "transcription", "src_lang", "tgt_lang",
            "bicodec_global", "target_bicodec", "source_audio_path",
            "reference_audio_path", "synthetic_source", "synthetic_reference",
            "source_audio_duration_seconds", "reference_audio_duration_seconds",
        ]).to_pylist()
        if subset and subset[0]:
            wanted = {json.loads(l)["id"] for l in Path(subset[0]).read_text().splitlines() if l.strip()}
            rows = [r for r in rows if r["id"] in wanted]
        rows.sort(key=lambda r: r["id"])
        if args.limit:
            rows = rows[: args.limit]
        mine = list(enumerate(rows))
        if not args.claims:
            mine = mine[args.shard :: args.num_shards]
        done = set()
        for system in SYSTEMS:
            d = out / system / name
            (d / "wav").mkdir(parents=True, exist_ok=True)
            # Claimed work can land in any worker's file, so all of them count.
            parts = sorted(d.glob("p1_*.jsonl")) if args.claims else [d / f"p1_{args.shard}.jsonl"]
            ids = set()
            for part in parts:
                if not part.exists():
                    continue
                for line in part.read_text(encoding="utf-8").splitlines():
                    try:
                        ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass  # a torn last line: that utterance is redone
            done = ids if system == SYSTEMS[0] else done & ids
        (out / "p1a" / name / "wav24k").mkdir(parents=True, exist_ok=True)
        work += [(name, index, row) for index, row in mine if row["id"] not in done]
    if args.claims or args.batch_size > 1:
        work.sort(key=lambda w: -w[2]["source_audio_duration_seconds"])
    print(f"{len(work)} utterances to do{' (shared)' if args.claims else ' in this shard'}", flush=True)
    if not work:
        return

    claims = Path(args.claims) if args.claims else None
    if claims:
        claims.mkdir(parents=True, exist_ok=True)

    def batches():
        """Consecutive free utterances, claimed only as the worker gets to them."""
        batch = []
        for name, index, row in work:
            if claims:
                try:
                    os.close(os.open(claims / f"{name}_{index:05d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
            batch.append((name, index, row))
            if len(batch) == args.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    device = torch.device(args.device)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        QWEN3, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    if not getattr(model, "has_talker", False):
        raise SystemExit("the checkpoint loaded without its talker; P1 needs it")
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    processor.tokenizer.padding_side = "left"  # a batch's prompts must all end where generation starts
    vocoder = UniSSTokenizer.from_pretrained(args.speech_tokenizer, device=device)
    resample = torchaudio.transforms.Resample(24_000, 16_000).to(device)
    frame = 1920  # code2wav samples per 12.5 Hz frame at 24 kHz

    @torch.no_grad()
    def speak(batch):
        """(hypothesis, 24 kHz waveform, error, seconds) per utterance, in batch order."""
        rows = [row for _, _, row in batch]
        conversations = [[{"role": "user", "content": [
            {"type": "audio", "audio": row["source_audio_path"]},
            {"type": "text", "text": INSTRUCTION[row["tgt_lang"]]},
        ]}] for row in rows]
        torch.cuda.synchronize(device)
        t0 = time.time()
        try:
            text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
            audio = [sf.read(row["source_audio_path"], dtype="float32")[0] for row in rows]
            inputs = processor(text=text, audio=audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            inputs = {key: (v.to(device) if hasattr(v, "to") else v) for key, v in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]
            if args.seed is not None:
                torch.manual_seed(args.seed + batch[0][1])
            sequences, wavs = model.generate(
                **inputs, speaker=args.speaker, return_audio=True,
                thinker_max_new_tokens=args.max_new_tokens, thinker_do_sample=False,
                talker_max_new_tokens=args.talker_max_new_tokens)
            if torch.is_tensor(wavs):
                wavs = [wavs]  # a batch of one comes back as one [1, 1, n] tensor
            hypotheses = [processor.tokenizer.decode(s[prompt_length:], skip_special_tokens=True).strip()
                          for s in sequences]
            torch.cuda.synchronize(device)
            seconds = time.time() - t0
            return [(h, w.reshape(-1).float(), None, seconds, len(batch)) for h, w in zip(hypotheses, wavs)]
        except Exception as e:  # recorded, not fatal: one bad utterance must not end the worker
            torch.cuda.empty_cache()
            if len(batch) > 1:
                return [r for item in batch for r in speak([item])]
            return [("", None, f"{type(e).__name__}: {e}"[:300], time.time() - t0, 1)]

    spoken = ((item, said) for batch in batches() for item, said in zip(batch, speak(batch)))
    for k, ((name, index, row), (hypothesis, wav24, err, gen_s, batch_size)) in enumerate(spoken):
        stem = f"{index:05d}_{row['src_lang']}2{row['tgt_lang']}"
        wav16, frames = None, 0
        if wav24 is not None:
            frames = int(wav24.numel() // frame)
            wav16 = resample(wav24.to(device)).cpu().numpy()
            sf.write(out / "p1a" / name / "wav24k" / f"{stem}.wav", wav24.cpu().numpy(), 24_000)

        a_path = None
        if wav16 is not None and wav16.size:
            a_path = str(out / "p1a" / name / "wav" / f"{stem}.wav")
            sf.write(a_path, wav16, 16_000)
        ra = record(row, index, hypothesis, a_path, frames, {
            "system": "p1a_native_preset_voice", "speaker": args.speaker.lower(),
            "gen_seconds": round(gen_s, 3), "batch_size": batch_size,
            "audio_path_24k": str(out / "p1a" / name / "wav24k" / f"{stem}.wav") if a_path else None})
        if err:
            ra["error"] = err

        # p1b: BiCodec re-encode, keep the semantic tokens, swap in the source
        # speaker's 32 global tokens -- the ones UniSS decodes every route with.
        b_path, n_sem, swap_s = None, 0, 0.0
        if a_path:
            t1 = time.time()
            try:
                _, bic = vocoder.tokenize(a_path)
                bic = [int(x) for x in (bic.flatten().tolist() if hasattr(bic, "flatten") else bic)]
                semantic = bic[32:]
                n_sem = len(semantic)
                if semantic:
                    tokens = torch.tensor([*row["bicodec_global"], *semantic], dtype=torch.long, device=device)
                    b_path = str(out / "p1b" / name / "wav" / f"{stem}.wav")
                    vocoder.save_audio(vocoder.decode(tokens), b_path, sample_rate=16_000)
            except Exception as e:
                ra.setdefault("p1b_error", f"{type(e).__name__}: {e}"[:300])
            swap_s = time.time() - t1
        rb = record(row, index, hypothesis, b_path, n_sem, {
            "system": "p1b_bicodec_source_voice", "speaker": "source(bicodec_global)",
            "gen_seconds": round(gen_s, 3), "batch_size": batch_size, "swap_seconds": round(swap_s, 3)})

        for system, rec in (("p1a", ra), ("p1b", rb)):
            with (out / system / name / f"p1_{args.shard}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if k % 10 == 0:
            print(f"  {k}/{len(work)}  {gen_s:.1f}s gen x{batch_size}  {swap_s:.1f}s swap  {frames} frames  "
                  f"{torch.cuda.max_memory_allocated(device) / 2**30:.0f} GiB  "
                  f"{hypothesis[:40]}{'  ERR ' + err if err else ''}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
