"""Plan 2's self-monitoring: the system's own Thinker (Qwen3-Omni) listens to the talker's speech and says how likely
it is that the speech says the committed text -- the likelihood p(y | c) of the posterior over what to speak, from
the comprehension side of the same model rather than an outside recogniser (Levelt's perceptual loop: speakers
monitor their own speech with the system they understand others with, before it is articulated).

The score is -log p(text | speech, transcription prompt, prefix) per token, read from vLLM's prompt log-probabilities:
the prompt is the Thinker's chat template with the speech and a transcription instruction, the reply teacher-forced
as prefix + text, and only the text's tokens are scored.

  runs   score whole sentences of finished runs (results.jsonl) -> jsonl, for the offline checks
  serve  the same over HTTP for si_infer --select-k (the API of si_asr_scorer.py)

    CUDA_VISIBLE_DEVICES=0 venvs/vllm_omni_030/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_self_scorer runs --out nll.jsonl <run>/realsi_zh2en ...
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
INSTRUCTION = {"English": "Transcribe the English speech into text.", "Chinese": "请将这段中文语音转写成文字。"}


class SelfScorer:
    def __init__(self, gpu_mem: float):
        from transformers import AutoTokenizer, Qwen3OmniMoeProcessor
        from vllm import LLM

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
        self.tok = AutoTokenizer.from_pretrained(QWEN3)
        self.llm = LLM(model=QWEN3, gpu_memory_utilization=gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
                       seed=20261022)
        self.head = {}
        for lang, ins in INSTRUCTION.items():
            conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": ins}]}]
            self.head[lang] = self.processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        self.lock = threading.Lock()

    def score(self, items: list[dict]) -> list[float | None]:
        """items: {"wav": float32 array 16 kHz, "prefix": str, "text": str, "lang": "English" | "Chinese"}"""
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        reqs, n_text = [], []
        for it in items:
            wav = it["wav"] if len(it["wav"]) >= 1600 else np.pad(it["wav"], (0, 1600 - len(it["wav"])))
            reqs.append({"prompt": self.head[it["lang"]] + it.get("prefix", "") + it["text"],
                         "multi_modal_data": {"audio": (wav, 16000)}})
            # the text's tokens as they fall after the prefix: the difference of the two tokenisations
            n_text.append(len(self.tok(it.get("prefix", "") + it["text"], add_special_tokens=False)["input_ids"])
                          - len(self.tok(it.get("prefix", ""), add_special_tokens=False)["input_ids"]))
        with self.lock:
            res = self.llm.generate(reqs, params, use_tqdm=False)
        out = []
        for r, n in zip(res, n_text):
            if n <= 0 or r.prompt_logprobs is None:
                out.append(None)
                continue
            ids = r.prompt_token_ids[-n:]
            lps = [lp[t].logprob for lp, t in zip(r.prompt_logprobs[-n:], ids) if lp is not None and t in lp]
            out.append(float(-np.mean(lps)) if lps else None)
        return out


def cmd_runs(args) -> None:
    import soundfile as sf

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    scorer = SelfScorer(args.gpu_mem)
    with open(out, "w") as fh:
        for run in args.runs:
            lang = "English" if run.rstrip("/").endswith("zh2en") else "Chinese"
            recs = [json.loads(l) for l in open(Path(run) / "results.jsonl", encoding="utf-8")]
            items, keep = [], []
            for r in recs:
                wav, _ = sf.read(r["speech_path"], dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                if r["translated_text"].strip():
                    items.append({"wav": wav, "text": r["translated_text"].strip(), "lang": lang})
                    keep.append(r["id"])
            for i, x in zip(keep, scorer.score(items)):
                fh.write(json.dumps({"run": run, "id": i, "nll": x}) + "\n")
            fh.flush()
            print("scored", run, flush=True)


def cmd_serve(args) -> None:
    scorer = SelfScorer(args.gpu_mem)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            items = [{**it, "wav": np.frombuffer(base64.b64decode(it.pop("pcm16")), dtype=np.int16).astype(np.float32) / 32768.0}
                     for it in body["items"]]
            res = json.dumps({"nll": scorer.score(items)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res)))
            self.end_headers()
            self.wfile.write(res)

        def log_message(self, *a):
            pass

    print(f"SCORER_READY port {args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("runs")
    a.add_argument("--out", required=True)
    a.add_argument("--gpu-mem", type=float, default=0.85)
    a.add_argument("runs", nargs="+")
    b = sub.add_parser("serve")
    b.add_argument("--port", type=int, default=18766)
    b.add_argument("--gpu-mem", type=float, default=0.6)
    args = ap.parse_args()
    cmd_runs(args) if args.cmd == "runs" else cmd_serve(args)


if __name__ == "__main__":
    main()
