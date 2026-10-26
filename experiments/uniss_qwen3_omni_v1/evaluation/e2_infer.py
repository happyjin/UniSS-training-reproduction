"""E2 inference: the trained chain, source speech in, target speech out.

Same chain as E1, with E2's three added inputs:
  1. the Thinker translates (vLLM), or --reuse takes the token ids an E1
     run recorded -- the Thinker is frozen, so they are the same tokens;
  2. the Thinker reads prompt + translation once more (Hugging Face) for
     its last-layer state at each translation token, exactly as training
     read it; gate * W(h) is computed from export_e2's added.pt;
  3. UniSS's front end gives the GLM-4 and BiCodec global tokens; pitch and
     energy are measured at the GLM-4 rate as in training; the length to
     speak for is the source duration times --length-ratio (0 for free);
  4. UniSS's embeddings plus the three inputs go to vLLM as prompt
     embeddings, and UniSS speaks with P2 step zero's sampling and guard;
  5. BiCodec decodes in the source speaker's voice.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
THINKER_EOS = 151645
SEMANTIC_0, SEMANTIC_LAST, END_SEMANTIC = 155761, 163952, 180392
ADD_SCALE = 0.03
# How long untrained UniSS speaks a translation, fitted on its full CVSS-T run
# (e2_untrained_full_r2, 4,897 sentences a direction): seconds from the
# translation's syllables, its punctuation and the source's duration.
# Predicted over actual, 10th to 90th percentile: 0.92-1.09 into Chinese,
# 0.86-1.22 into English. E2 at step 4000 follows the length it is asked
# for closely, and asked for the source's length on every sentence it
# stretched short English over long Chinese sources and filled the time with
# loops; --length-policy natural-clip asks for this length instead, held
# within --clip of the source's.
NATURAL = {"cmn": (0.0551, -0.0120, 0.7139, 0.0519), "eng": (0.0454, 0.1591, 0.7910, -0.3992)}
CJK = re.compile(r"[一-鿿]")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
PUNCT = re.compile(r"[,.;:!?，。；：！？、—]")


def natural_seconds(text: str, tgt: str, source_seconds: float) -> float:
    syllables = sum(2 * len(w) if w.isdigit() else max(1, len(re.findall(r"[aeiouy]+", w.lower())))
                    for w in WORD.findall(text))
    if tgt == "cmn":
        syllables += len(CJK.findall(text))
    a, b, c, d = NATURAL[tgt]
    return max(0.3, a * syllables + b * len(PUNCT.findall(text)) + c * source_seconds + d)
COLUMNS = ["id", "translation", "transcription", "src_lang", "tgt_lang", "target_bicodec",
           "source_audio_path", "reference_audio_path", "synthetic_source", "synthetic_reference",
           "source_audio_duration_seconds", "reference_audio_duration_seconds"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, help="export_e2's output directory, or 'untrained' for step zero")
    ap.add_argument("--pair", action="append", required=True, help="name:parquet[:ids_jsonl]")
    ap.add_argument("--reuse", default="", help="an E1 run directory whose <name>/results.jsonl hold reply_token_ids")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--length-ratio", type=float, default=1.0, help="target length / source length; 0 = free")
    ap.add_argument("--length-policy", choices=("source", "natural-clip"), default="source",
                    help="source: --length-ratio times the source, <|speed_9|>; natural-clip: the length "
                         "untrained UniSS would speak (natural_seconds), held within --clip of the source's, "
                         "with the speed token for that ratio")
    ap.add_argument("--clip", default="0.85,1.15", help="natural-clip's bounds on target/source duration")
    ap.add_argument("--loop-ngram", type=int, default=0,
                    help="ban a speech-code run of this length from repeating (StopGuard ngram); 0 = off")
    ap.add_argument("--extra-stop", default="",
                    help="more token ids that end the speech, comma-separated (T1's <|wait_read|>, 180395: a "
                         "streaming talker may close an offline sentence with it); empty = only the end token")
    ap.add_argument("--si-guard", action="store_true",
                    help="use evaluation/si_guard.py as for the last block of a stream: <|wait_read|> pooled into the "
                         "end token, so the length rules hold for either way of stopping (T1's talkers)")
    ap.add_argument("--min-codes-frac", type=float, default=0.0,
                    help="bar the end token until this share of the expected length is spoken (StopGuard min_frac)")
    ap.add_argument("--drop", default="",
                    help="diagnostics only: added inputs to leave out, comma-separated from hidden, prosody, "
                         "duration; UniSS keeps its trained weights")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--batch", type=int, default=16, help="Thinker re-read batch")
    ap.add_argument("--chunk", type=int, default=256, help="UniSS requests per generate call")
    ap.add_argument("--states-cache", default="",
                    help="directory of the Thinker's normalised translation-token states, one file a sentence; "
                         "the Thinker is frozen, so they hold for every checkpoint and are computed once")
    args = ap.parse_args()
    drop = {d for d in args.drop.split(",") if d}
    extra_stop = [int(t) for t in args.extra_stop.split(",") if t.strip()]
    if drop - {"hidden", "prosody", "duration"}:
        ap.error(f"--drop: unknown {sorted(drop - {'hidden', 'prosody', 'duration'})}")

    import pyarrow.parquet as pq
    import torchaudio
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from uniss import UniSSTokenizer

    from experiments.uniss_qwen3_omni_v1.evaluation.e1_pipeline import read_with_soundfile
    from experiments.uniss_qwen3_omni_v1.evaluation.uniss_stop_guard import StopGuard
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import (
        LANG3, SPEED_0, SPEED_9, duration_features, inference_instruction, speed_token, uniss_header, uniss_prefix)
    from experiments.uniss_qwen3_omni_v1.training.e2_data import PROSODY_DIM, prosody

    torchaudio.load = read_with_soundfile
    instruction = inference_instruction()
    work = []
    for spec in args.pair:
        name, parquet, *ids = spec.split(":")
        rows = pq.read_table(parquet, columns=COLUMNS).to_pylist()
        if ids and ids[0]:
            wanted = {json.loads(l)["id"] for l in Path(ids[0]).read_text().splitlines() if l.strip()}
            rows = [r for r in rows if r["id"] in wanted]
        rows.sort(key=lambda r: r["id"])
        if args.limit:
            rows = rows[: args.limit]
        reused = {}
        if args.reuse:
            reused = {json.loads(l)["id"]: json.loads(l)["reply_token_ids"]
                      for l in open(Path(args.reuse) / name / "results.jsonl")}
        work += [(name, i, r, reused.get(r["id"])) for i, r in list(enumerate(rows))[args.shard :: args.num_shards]]
    out = Path(args.output).resolve()
    timing = {}
    device = torch.device("cuda:0")
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)

    # 1. Translation token ids.
    missing = [k for k, w in enumerate(work) if w[3] is None]
    if missing:
        thinker = LLM(model=QWEN3, gpu_memory_utilization=0.55, max_model_len=8192,
                      limit_mm_per_prompt={"audio": 1}, seed=args.seed)
        prompts = []
        for k in missing:
            name, i, row, _ = work[k]
            wav, sr = sf.read(row["source_audio_path"], dtype="float32")
            conv = [{"role": "user", "content": [{"type": "audio", "audio": row["source_audio_path"]},
                                                 {"type": "text", "text": instruction[row["tgt_lang"]]}]}]
            prompts.append({"prompt": processor.apply_chat_template([conv], add_generation_prompt=True,
                                                                    tokenize=False)[0],
                            "multi_modal_data": {"audio": (wav, sr)}})
        t0 = time.time()
        said = thinker.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256,
                                                        stop_token_ids=[THINKER_EOS]), use_tqdm=False)
        timing["thinker"] = time.time() - t0
        for k, s in zip(missing, said):
            work[k] = (*work[k][:3], [t for t in s.outputs[0].token_ids if t != THINKER_EOS])
        del thinker
        torch.cuda.empty_cache()

    # 2. The Thinker's state at each translation token, as training read it.
    if args.export == "untrained":
        added = None
        uniss_path = str(REPO / "pretrained_models/UniSS_full")
    else:
        added = torch.load(Path(args.export) / "added.pt", map_location="cpu")["added"]
        uniss_path = str(Path(args.export) / "uniss")
    hidden_add = [None] * len(work)
    if added and "hidden" not in drop and float(added["hid_gate"].float().abs().max()) > 0:
        t0 = time.time()
        cache = Path(args.states_cache) if args.states_cache else None
        states = [None] * len(work)
        if cache:
            for k, (name, _, row, _) in enumerate(work):
                f = cache / name / f"{row['id']}.pt"
                if f.exists():
                    states[k] = torch.load(f)
        todo = [k for k, s in enumerate(states) if s is None]
        if todo:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(QWEN3)
            config.enable_audio_output = False   # the Thinker only: no talker, no code2wav to load
            omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                QWEN3, config=config, dtype=torch.bfloat16, device_map={"": device})
            thinker = omni.thinker.eval()
            tok = processor.tokenizer
            tok.padding_side = "left"
        for s in range(0, len(todo), args.batch):
            chunk = [work[k] for k in todo[s : s + args.batch]]
            waves = [sf.read(row["source_audio_path"], dtype="float32")[0] for _, _, row, _ in chunk]
            convs = [[{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                                   {"type": "text", "text": instruction[row["tgt_lang"]]}]}]
                     for _, _, row, _ in chunk]
            enc = processor(text=list(processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)),
                            audio=waves, sampling_rate=16_000, return_tensors="pt", padding=True)
            width = enc["input_ids"].shape[1]
            replies = [r for _, _, _, r in chunk]
            longest = max(len(r) for r in replies)
            ids = torch.full((len(chunk), width + longest), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros_like(ids)
            ids[:, :width], mask[:, :width] = enc["input_ids"], enc["attention_mask"]
            for j, r in enumerate(replies):
                ids[j, width : width + len(r)] = torch.as_tensor(r)
                mask[j, width : width + len(r)] = 1
            with torch.no_grad():
                h = thinker(input_ids=ids.to(device), attention_mask=mask.to(device),
                            # the Thinker is bf16; Megatron cast this for training, here it is done by hand
                            input_features=enc["input_features"].to(device=device, dtype=torch.bfloat16),
                            feature_attention_mask=enc["feature_attention_mask"].to(device),
                            output_hidden_states=True, return_dict=True).hidden_states[-1].float()
            for j, r in enumerate(replies):
                x = h[j, width : width + len(r)]
                x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.float16).cpu()
                k = todo[s + j]
                states[k] = x
                if cache:
                    name, _, row, _ = work[k]
                    (cache / name).mkdir(parents=True, exist_ok=True)
                    torch.save(x, cache / name / f"{row['id']}.pt")
        if todo:
            del thinker, omni
            torch.cuda.empty_cache()
        proj = added["hid_proj.weight"].to(device=device, dtype=torch.float32)
        gate = added["hid_gate"].to(device=device, dtype=torch.float32)
        for k, x in enumerate(states):
            hidden_add[k] = (gate * (ADD_SCALE * (x.to(device=device, dtype=torch.float32) @ proj.T))).cpu()
        timing["thinker_states"] = time.time() - t0
        timing["states_computed"] = float(len(todo))

    # 3. Front end, prosody, length.
    front = UniSSTokenizer.from_pretrained(str(REPO / "pretrained_models/UniSS"), device=device)
    t0 = time.time()
    fronts = []
    for _, _, row, _ in work:
        glm, bic = front.tokenize(row["source_audio_path"])
        glm = [int(x) for x in (glm.flatten().tolist() if hasattr(glm, "flatten") else glm)]
        bic = [int(x) for x in (bic.flatten().tolist() if hasattr(bic, "flatten") else bic)]
        wav = sf.read(row["source_audio_path"], dtype="float32")[0]
        fronts.append((glm, bic[:32], prosody(wav, len(glm))))
    timing["front_end"] = time.time() - t0

    # 4. Prompt embeddings, then UniSS speaks.
    from safetensors import safe_open
    with safe_open(str(Path(uniss_path) / "model.safetensors"), "pt", device="cpu") as fh:
        embed = fh.get_tensor("model.embed_tokens.weight").to(device=device, dtype=torch.float32)
    pros_w = pros_b = dur_w = dur_b = None
    if added:   # empty for a run trained without the added inputs
        pros_w, pros_b = added["pros_proj.weight"].float().to(device), added["pros_proj.bias"].float().to(device)
        dur_w, dur_b = added["dur_proj.weight"].float().to(device), added["dur_proj.bias"].float().to(device)
    lo, hi = (float(v) for v in args.clip.split(","))
    requests, params, plans = [], [], []
    for k, ((name, i, row, reply), (glm, glob, pros)) in enumerate(zip(work, fronts)):
        tgt = LANG3[row["tgt_lang"]]
        src_s = float(row["source_audio_duration_seconds"])
        if args.length_policy == "natural-clip":
            said = processor.tokenizer.decode(reply, skip_special_tokens=True)
            ratio = min(max(natural_seconds(said, tgt, src_s) / src_s, lo), hi)
            target_codes = max(1, round(50 * src_s * ratio))
            speed, expected = speed_token(src_s, target_codes), target_codes
        else:
            ratio = args.length_ratio
            target_codes = max(1, round(50 * src_s * ratio)) if ratio > 0 else None
            speed, expected = SPEED_9, max(1, round(50 * src_s))
        plans.append((ratio, speed - SPEED_0))
        prefix = uniss_prefix(tgt, glob, glm, speed)
        seq = prefix + list(reply) + uniss_header(tgt, speed)
        x = embed[torch.as_tensor(seq, device=device)].clone()
        if hidden_add[k] is not None:
            x[len(prefix) : len(prefix) + len(reply)] += hidden_add[k].to(device)
        if pros_w is not None and "prosody" not in drop:
            x[37 : 37 + len(glm)] += ADD_SCALE * (torch.as_tensor(pros, device=device) @ pros_w.T + pros_b)
        if dur_w is not None and "duration" not in drop:
            feats = [0.0] * dur_w.shape[1] if target_codes is None else duration_features(target_codes, src_s)
            x[-1] += ADD_SCALE * (torch.as_tensor(feats, device=device) @ dur_w.T + dur_b)
        # The ids ride along, every position still read from the embeddings: given
        # embeddings alone, vLLM leaves a reused batch slot's old token buffer under
        # the prompt, and the repetition penalty indexed by it hit a device-side
        # assert once the full set (1,226 requests a shard) reused slots. With the
        # ids the penalty also sees the prompt E1's token prompts gave it.
        requests.append({"prompt_embeds": x.to(torch.bfloat16).cpu(), "prompt_token_ids": list(seq),
                         "prompt_is_token_ids": [False] * len(seq)})
        guard = {"expected": expected, **({"ngram": args.loop_ngram} if args.loop_ngram else {}),
                 **({"min_frac": args.min_codes_frac} if args.min_codes_frac else {})}
        params.append(SamplingParams(
            # with --si-guard the guard applies the 1.3 penalty itself, to the output (the prompt holds no codes)
            temperature=0.7, top_p=0.8, repetition_penalty=1.0 if args.si_guard else 1.3, max_tokens=1500,
            stop_token_ids=extra_stop or None,
            seed=args.seed + i + (0 if name == "cmn_eng" else 500_000), skip_special_tokens=False,
            extra_args={"si_guard": {**guard, "final": True}} if args.si_guard else {"stop_guard": guard}))
    del embed
    torch.cuda.empty_cache()
    if args.si_guard:
        from experiments.uniss_qwen3_omni_v1.evaluation.si_guard import SIGuard
    talker = LLM(model=uniss_path, gpu_memory_utilization=0.3, max_model_len=4096, seed=args.seed,
                 enable_prompt_embeds=True, logits_processors=[SIGuard if args.si_guard else StopGuard])
    t0 = time.time()
    # In chunks only so the log shows how far a shard has got.
    spoken = []
    for s in range(0, len(requests), args.chunk):
        print(f"talker requests {s}-{min(s + args.chunk, len(requests))} of {len(requests)}", flush=True)
        spoken += talker.generate(requests[s : s + args.chunk], params[s : s + args.chunk], use_tqdm=False)
    timing["talker"] = time.time() - t0
    vocab = talker.get_tokenizer()

    # 5. BiCodec.
    t0 = time.time()
    files = {}
    for (name, i, row, reply), (glm, glob, _), res, (ratio, speed) in zip(work, fronts, spoken, plans):
        ids = list(res.outputs[0].token_ids)
        codes = [t - SEMANTIC_0 for t in ids if SEMANTIC_0 <= t <= SEMANTIC_LAST]
        (out / name / "wav").mkdir(parents=True, exist_ok=True)
        path = None
        if codes:
            path = str(out / name / "wav" / f"{i:05d}_{row['src_lang']}2{row['tgt_lang']}.wav")
            front.save_audio(front.decode(torch.tensor([*glob, *codes], dtype=torch.long, device=device)),
                             path, sample_rate=16_000)
        text = vocab.decode(reply, skip_special_tokens=True).strip()
        files.setdefault(name, []).append({
            "index": i, "name": f"{i:05d}_{row['src_lang']}2{row['tgt_lang']}", "id": row["id"], "mode": "tts",
            "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"], "dataset_name": "CVSS-T",
            "transcription_ref": row["transcription"], "translation_ref": row["translation"],
            "generated_text_raw": text, "generated_text_clean": text, "translated_text": text,
            "reference_text": row["translation"], "source_audio_path": row["source_audio_path"],
            "reference_audio_path": row["reference_audio_path"],
            "source_audio_duration_seconds": row["source_audio_duration_seconds"],
            "reference_audio_duration_seconds": row["reference_audio_duration_seconds"],
            "synthetic_source": bool(row["synthetic_source"]), "synthetic_reference": bool(row["synthetic_reference"]),
            "codes": len(codes), "gold_codes": len(row["target_bicodec"]), "audio_path": path,
            "error": None if path else "no audio", "system": f"e2:{args.export}", "length_ratio": args.length_ratio,
            "length_policy": args.length_policy, "requested_ratio": ratio, "speed_token": speed,
            "loop_ngram": args.loop_ngram, "ended": END_SEMANTIC in ids or any(t in ids for t in extra_stop),
            "reply_token_ids": list(reply),
            "code_ids": codes})
    timing["decode"] = time.time() - t0
    for name, recs in files.items():
        with open(out / name / f"part_{args.shard}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print("timing " + "  ".join(f"{k} {v:.1f}s" for k, v in timing.items())
          + f"  not ended {sum(not r['ended'] for v in files.values() for r in v)}", flush=True)


if __name__ == "__main__":
    main()
