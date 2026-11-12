"""Speak QABC's committed blocks with Qwen3-Omni-30B-A3B-Instruct's NATIVE Thinker -> Talker -> code2wav (vLLM-Omni
0.30.0), one request per block (registration paper/baselines_run_plan_qabc_omnitalker_v1.md §2).

Prompt (common.SYSTEM_PROMPT as the system turn, the block text alone as the user turn), processor-built chat template
as qwen3omni_native_cvss.py, preset voice Ethan; sampling as the native CVSS-T run: thinker greedy (max 256 tokens),
talker temperature 0.9 / top_k 50 / top_p 1.0 / repetition_penalty 1.05 (the official deploy defaults), code2wav greedy,
the same seed (SEED_BASE + talker seed) on every stage of every request of a shard (a shard holds one seed).

Work: <root>/jobs/shard_<k>.jsonl (make_jobs.py). One engine process loads the model once, then claims shards one at a
time (<root>/claims/shard_<k>, O_EXCL; a claim whose process is dead and whose shard has no done marker may be taken
over) and writes <root>/synth/part_<k>.jsonl (appended; keys already there are skipped, so a killed shard resumes),
<root>/synth/wav24/<key>.wav (the model's 24 kHz output) and wav16/<key>.wav (torchaudio resample to 16 kHz, as
qwen3omni_native_cvss.py), and <root>/synth/done_<k> when the shard is complete. Exits when no shard is left.
Runs in venvs/vllm_omni_030 (its bin on PATH).

    python -m experiments.s2st_baselines_v1.qabc_omnitalker.synth --root <dir> --deploy-config <yaml> --tag <engine>
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from experiments.s2st_baselines_v1.qabc_omnitalker.common import CODEC_EOS, QWEN3, SEED_BASE, SPEAKER, SYSTEM_PROMPT, THINKER_EOS


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim(root: Path, shard: str, tag: str) -> bool:
    c = root / "claims" / shard
    if (root / "synth" / f"done_{shard.split('_', 1)[1]}").exists():
        return False
    body = f"{socket.gethostname()} {os.getpid()} {tag} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        old = c.read_text().split()
        if len(old) >= 2 and old[1].isdigit() and not alive(int(old[1])):   # its process died mid-shard: take over
            stale = c.with_name(f"{c.name}.stale_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{old[1]}")
            os.rename(c, stale)
            return claim(root, shard, tag)
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--deploy-config", required=True)
    ap.add_argument("--tag", default="e")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    root = Path(args.root)
    for sub in ("claims", "synth/wav24", "synth/wav16"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    shards = sorted(p.stem for p in (root / "jobs").glob("shard_*.jsonl"))
    omni = processor = None
    for shard in shards:
        if not claim(root, shard, args.tag):
            continue
        k = shard.split("_", 1)[1]
        jobs = [json.loads(l) for l in open(root / "jobs" / f"{shard}.jsonl", encoding="utf-8")]
        part = root / "synth" / f"part_{k}.jsonl"
        done = set()
        if part.exists():
            for line in part.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
        work = [j for j in jobs if j["key"] not in done]
        seeds = {j["seed"] for j in jobs}
        assert len(seeds) == 1, (shard, seeds)
        seed = SEED_BASE + seeds.pop()
        print(f"[{time.strftime('%H:%M:%S')}] {args.tag} {shard}: {len(jobs)} requests, {len(work)} to do, seed {seed}",
              flush=True)
        if work:
            if omni is None:
                from transformers import Qwen3OmniMoeProcessor
                from vllm_omni.entrypoints.omni import Omni
                processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
                t = time.time()
                omni = Omni(model=QWEN3, async_chunk=False, deploy_config=args.deploy_config)
                print(f"load {time.time() - t:.0f}s", flush=True)
            from vllm import SamplingParams
            prompts = []
            for j in work:
                conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                                {"role": "user", "content": [{"type": "text", "text": j["text"]}]}]
                prompts.append({
                    "prompt": processor.apply_chat_template([conversation], add_generation_prompt=True, tokenize=False)[0],
                    "additional_information": {"speaker": [SPEAKER.lower()]},
                })
            sampling = [
                SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=[THINKER_EOS], seed=seed),
                SamplingParams(temperature=0.9, top_k=50, top_p=1.0, max_tokens=4096, repetition_penalty=1.05,
                               stop_token_ids=[CODEC_EOS], detokenize=False, seed=seed),
                SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=4096 * 16, detokenize=True,
                               repetition_penalty=1.1, seed=seed),
            ][: omni.num_stages]

            def write(i, text, wav24, elapsed):
                j = work[i]
                rec = {"key": j["key"], "seed": j["seed"], "text_sent": j["text"], "thinker_text": text,
                       "wav24": None, "wav16": None, "dur_ms": 0.0, "speaker": SPEAKER, "vllm_seed": seed,
                       "engine": f"vllm-omni 0.30.0 {args.tag}", "gen_seconds_since_start": round(elapsed, 3),
                       "error": None}
                if wav24 is not None and wav24.numel():
                    p24 = root / "synth" / "wav24" / f"{j['key']}.wav"
                    p16 = root / "synth" / "wav16" / f"{j['key']}.wav"
                    sf.write(p24, wav24.numpy(), 24_000)
                    wav16 = torchaudio.functional.resample(wav24, 24_000, 16_000)
                    sf.write(p16, wav16.numpy(), 16_000)
                    rec.update(wav24=str(p24.relative_to(root)), wav16=str(p16.relative_to(root)),
                               dur_ms=round(wav16.numel() / 16, 3))
                else:
                    rec["error"] = "no audio"
                with part.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            texts, wavs, written = {}, {}, set()
            t0 = time.time()
            for output in omni.generate(prompts, sampling, py_generator=True, use_tqdm=False):
                i = int(output.request_id.split("_", 1)[0])
                if output.final_output_type == "text":
                    texts[i] = output.outputs[0].text.strip()
                elif output.final_output_type == "audio":
                    audio = output.outputs[0].multimodal_output["audio"]
                    if isinstance(audio, list):
                        audio = torch.cat([(a if isinstance(a, torch.Tensor) else torch.tensor(a)).flatten() for a in audio])
                    wavs[i] = audio.float().detach().cpu().reshape(-1)
                if i in texts and i in wavs and i not in written:
                    write(i, texts.pop(i), wavs.pop(i), time.time() - t0)
                    written.add(i)
                    if len(written) % 500 == 0:
                        print(f"  {len(written)}/{len(work)}  {len(written) / (time.time() - t0):.2f}/s", flush=True)
            wall = time.time() - t0
            for i in range(len(work)):   # text without speech (or the reverse) is still recorded, as an error row
                if i not in written:
                    write(i, texts.get(i, ""), wavs.get(i), wall)
            print(f"  generate {wall:.0f}s for {len(work)} ({len(work) / max(wall, 1e-6):.2f}/s), "
                  f"incomplete {len(work) - len(written)}", flush=True)
        n = sum(1 for _ in open(part, encoding="utf-8")) if part.exists() else 0
        if n >= len(jobs):
            (root / "synth" / f"done_{k}").write_text(f"{n}\n")
        print(f"[{time.strftime('%H:%M:%S')}] {args.tag} {shard} finished ({n}/{len(jobs)} rows)", flush=True)
    if omni is not None:
        omni.close()
    print(f"QABC_OMNITALKER_ENGINE_DONE {args.tag}", flush=True)


if __name__ == "__main__":
    main()
