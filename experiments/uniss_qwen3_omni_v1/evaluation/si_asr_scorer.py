"""Plan 2's likelihood p(y | c): how likely an independent ASR (Qwen3-ASR-1.7B) finds it that a stretch of the
talker's speech says a given text -- served over HTTP so the talker (vLLM venv, transformers 5) can ask the
ASR (align venv, transformers 4.57) without either environment changing.

POST /score  {"items": [{"pcm16": <base64 int16 mono 16 kHz>, "prefix": "...", "text": "...", "lang": "English"}]}
          -> {"nll": [mean -log p over the text's tokens, given the speech and the prefix text, or null]}

The prompt is Qwen3-ASR's own (system, the audio, "language X<asr_text>"); the prefix -- the blocks already
spoken -- is teacher-forced before the scored text, so a block is judged in its sentence, not alone (alone, a
block's score detected omissions with AUC 0.62-0.78; a whole sentence's, 0.84-0.95).

    CUDA_VISIBLE_DEVICES=0 venvs/uniss-formal-align-v1/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_asr_scorer --port 18765
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

ASR = "/opt/dlami/nvme/jasonleeeli/projects/multi_speaker_pipeline_share/multi_speaker_pipeline_share/model_weights/Qwen3-ASR-1.7B"


class Scorer:
    def __init__(self, path: str, batch: int):
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

        self.asr = Qwen3ASRModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0")
        self.proc, self.model, self.batch = self.asr.processor, self.asr.model, batch
        self.lock = threading.Lock()

    def _inputs(self, texts, wavs):
        x = self.proc(text=texts, audio=wavs, return_tensors="pt", padding=True)
        return {k: (v.to(self.model.device).to(self.model.dtype) if v.is_floating_point() else v.to(self.model.device))
                for k, v in x.items()}

    def score(self, items: list[dict]) -> list[float | None]:
        out = []
        with self.lock, torch.no_grad():
            for s in range(0, len(items), self.batch):
                chunk = items[s: s + self.batch]
                wavs = [np.frombuffer(base64.b64decode(it["pcm16"]), dtype=np.int16).astype(np.float32) / 32768.0
                        for it in chunk]
                wavs = [w if len(w) >= 1600 else np.pad(w, (0, 1600 - len(w))) for w in wavs]
                base = [self.asr._build_text_prompt(context="", force_language=it["lang"]) + it.get("prefix", "") for it in chunk]
                # the prefix's length per item, with the same audio so its placeholder expands the same; the text's
                # tokens end where the row's real tokens end, whichever side the processor pads
                a = self.proc(text=base, audio=wavs, return_tensors="pt", padding=True)
                b = self._inputs([p + it["text"] for p, it in zip(base, chunk)], wavs)
                logits = self.model.thinker(**b).logits.float()
                ids, mask_a = b["input_ids"], a["attention_mask"]
                for k, it in enumerate(chunk):
                    n_text = int(b["attention_mask"][k].sum() - mask_a[k].sum())
                    if n_text <= 0:
                        out.append(None)
                        continue
                    end = int(b["attention_mask"][k].nonzero().max()) + 1
                    lp = torch.log_softmax(logits[k, end - n_text - 1: end - 1], -1)
                    tgt = ids[k, end - n_text: end]
                    out.append(float(-lp.gather(-1, tgt[:, None])[:, 0].mean()))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--model", default=ASR)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    scorer = Scorer(args.model, args.batch)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            res = json.dumps({"nll": scorer.score(body["items"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res)))
            self.end_headers()
            self.wfile.write(res)

        def log_message(self, *a):
            pass

    print(f"SCORER_READY port {args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
