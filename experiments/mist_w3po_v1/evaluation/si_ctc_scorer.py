"""Speech-gap plan B1 (reports/mist_w3po_v1/text_to_speech_gap_plan.zh-CN.md §5): the selector for si_infer
--select-k, served with si_asr_scorer.py's HTTP interface, but scored by a CTC ASR (wav2vec 2.0 Large LV-60
self-trained, 960 h LibriSpeech) instead of Qwen3-ASR-1.7B (not on this server since the 2026-11 move).

The CTC model is independent of the evaluation ASR (Whisper large-v3), so choosing a block's candidate by it does
not tune the speech to the scorer. English only (zh2en).

POST /score  {"items": [{"pcm16": <base64 int16 mono 16 kHz>, "prefix": "...", "text": "...", "lang": "English"}]}
          -> {"nll": [CTC -log p(prefix + text | speech) per target character, or null when no letters]}
     an item with "cover": true (talker scheme 2) adds "cover": [share of the block's words in the CTC best-path
     transcript, LCS against prefix + text] to the response; without it the response is unchanged

The speech is the stream so far (blocks already chosen + this candidate); the prefix (their text) is the same for
every candidate of a block, so the argmin over candidates is the argmin of -log p(text | speech, prefix) under CTC.

    CUDA_VISIBLE_DEVICES=0 conda_envs/qwen3omni/bin/python -m experiments.mist_w3po_v1.evaluation.si_ctc_scorer \
        --port 18765 --model /opt/dlami/nvme/jasonleeeli/evaluation_models/wav2vec2-large-960h-lv60-self
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

CTC = "/opt/dlami/nvme/jasonleeeli/evaluation_models/wav2vec2-large-960h-lv60-self"


def ctc_text(prefix: str, text: str) -> str:
    """The CTC model's alphabet: upper-case letters and the apostrophe, words split by single spaces; everything
    else (digits, punctuation) becomes a word break."""
    s = (prefix + " " + text).upper().replace("’", "'")
    s = re.sub(r"[^A-Z']+", " ", s)
    return re.sub(r" +", " ", s).strip()


def lcs_matched(ref: list[str], hyp: list[str]) -> list[bool]:
    """Per ref word: matched in one longest common subsequence of ref and hyp (exact word equality)."""
    n, m = len(ref), len(hyp)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if ref[i] == hyp[j] else max(L[i + 1][j], L[i][j + 1])
    out, i, j = [False] * n, 0, 0
    while i < n and j < m:
        if ref[i] == hyp[j]:
            out[i] = True
            i, j = i + 1, j + 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def block_cover(prefix: str, text: str, transcript: str) -> float | None:
    """Share of the block's words (ctc_text of text) matched, in order, by the CTC transcript of the stream so far
    (LCS against prefix + text); None when the block has no letters."""
    blk = ctc_text("", text).split()
    if not blk:
        return None
    pre = ctc_text("", prefix).split() if prefix.strip() else []
    m = lcs_matched(pre + blk, ctc_text("", transcript).split())
    return round(sum(m[len(pre):]) / len(blk), 4)


class Scorer:
    def __init__(self, path: str):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).eval().to("cuda:0")
        self.blank = self.model.config.pad_token_id
        self.lock = threading.Lock()

    def score(self, items: list[dict], covers: list | None = None) -> list[float | None]:
        torch = self.torch
        out = []
        with self.lock, torch.no_grad():
            for it in items:
                tgt = ctc_text(it.get("prefix", ""), it["text"])
                ids = self.proc.tokenizer(tgt).input_ids if tgt else []
                if not ids:
                    out.append(None)
                    if covers is not None:
                        covers.append(None)
                    continue
                w = np.frombuffer(base64.b64decode(it["pcm16"]), dtype=np.int16).astype(np.float32) / 32768.0
                w = w if len(w) >= 1600 else np.pad(w, (0, 1600 - len(w)))
                x = self.proc(w, sampling_rate=16000, return_tensors="pt").input_values.to("cuda:0")
                lp = torch.log_softmax(self.model(x).logits.float(), -1)[0]          # T x V
                loss = torch.nn.functional.ctc_loss(
                    lp[:, None, :], torch.tensor([ids], device=lp.device), torch.tensor([lp.shape[0]]),
                    torch.tensor([len(ids)]), blank=self.blank, reduction="sum", zero_infinity=False)
                v = float(loss)   # infinite: the speech is too short for the text -- the worst candidate
                out.append(v / len(ids) if np.isfinite(v) else 1e9)
                if covers is not None:
                    covers.append(block_cover(it.get("prefix", ""), it["text"], self.greedy(lp)))
        return out

    def greedy(self, lp) -> str:
        """CTC best-path transcript of a T x V log-posterior: argmax, repeats merged, blanks dropped."""
        best = lp.argmax(-1).tolist()
        keep = [t for i, t in enumerate(best) if t != self.blank and (i == 0 or t != best[i - 1])]
        return self.proc.tokenizer.decode(keep, group_tokens=False) if keep else ""   # repeats already merged

    def score_cover(self, items: list[dict]) -> tuple[list, list]:
        """Talker scheme 2 (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §2.1): score() plus, per item, the
        block's word coverage (block_cover) on the same CTC output. score() alone is unchanged."""
        covers: list = []
        nll = self.score(items, covers)
        return nll, covers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--model", default=CTC)
    args = ap.parse_args()
    scorer = Scorer(args.model)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if any(it.get("cover") for it in body["items"]):   # talker scheme 2: coverage too (opt-in per request)
                nll, cov = scorer.score_cover(body["items"])
                data = json.dumps({"nll": nll, "cover": cov}).encode()
            else:
                data = json.dumps({"nll": scorer.score(body["items"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    print(f"SCORER_READY port {args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
