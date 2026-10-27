"""T1's streaming probe: the talker speaks RealSI sentence by sentence, block by block.

The text is RealSI's own: each sentence is split into sub-utterances with
their source times and reference translations, and a sub-utterance's text is
handed to the talker at the end of the first read window (--window seconds,
SimulS2ST-Omni's m) that reaches its source end + delta -- an oracle
translator with a phrase-level policy, so what is measured is the talker:
whether it speaks each piece, closes it, continues in the next, and how the
speech lies against the source in time.

Per block, as T1 trained (training/si_batch.py):
  * the source heard so far is tokenised again (GLM-4 attends both ways, so
    a longer prefix revises earlier tokens) and the blocks read it up to a
    holdback of --holdback tokens behind the audio (all of it once the source
    has ended);
  * the voice is the source's 32 BiCodec global tokens from the audio heard
    by the first write;
  * the talker continues one sequence: the earlier blocks with the codes it
    spoke, then this block's GLM, text and header; it stops at <|wait_read|>
    (more to come) or <|end_semantic_token|> (the last block).
Blocks of all sentences are generated a round at a time. The codes are then
decoded as one sequence and laid on a timeline: a block plays from the later
of its write time and the end of the previous block (no compute time added).

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model <uniss dir> --direction zh2en \
        --window 3 --limit 100 --output <dir> [--shard k --num-shards n]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import time
from pathlib import Path

import os
import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
REALSI = REPO / "data/raw/realsi/RealSI_data/data"
SEMANTIC_0, SEMANTIC_LAST, END_SEMANTIC, SILENCE_CODE = 155761, 163952, 180392, 7645
WAIT_READ, STREAMING_MODE = 180395, 180406
LANGS = {"zh2en": ("cmn", "eng"), "en2zh": ("eng", "cmn")}
CJK = re.compile(r"[一-鿿]")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
GLM_MS = 80


def flat(x) -> list[int]:
    x = x.flatten().tolist() if hasattr(x, "flatten") else x
    out = []
    for v in x:
        out += flat(v) if isinstance(v, (list, tuple)) or hasattr(v, "flatten") else [int(v)]
    return out


def expected_codes(text: str, tgt: str) -> int:
    """A rough spoken length at 50 codes a second: 0.26 s a Chinese character, 0.22 s an English syllable."""
    if tgt == "cmn":
        units, rate = len(CJK.findall(text)) + sum(len(w) for w in WORD.findall(text)) * 0.6, 0.26
    else:
        units = sum(2 * len(w) if w.isdigit() else max(1, len(re.findall(r"[aeiouy]+", w.lower())))
                    for w in WORD.findall(text))
        rate = 0.22
    return max(30, int(50 * rate * max(units, 1)))


def spoken_units(text: str, tgt: str) -> int:
    """Chinese characters (and Latin words in them), or English syllables."""
    if tgt == "cmn":
        return len(CJK.findall(text)) + len(WORD.findall(text))
    return sum(2 * len(w) if w.isdigit() else max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in WORD.findall(text))


class VoiceMemory:
    """A talk's speaker as a running estimate of BiCodec's speaker latent (before its FSQ), quantised per sentence."""

    def __init__(self, tok, alpha_min: float, reset: float):
        self.tok, self.alpha_min, self.reset = tok, alpha_min, reset
        self.state = {}   # talk -> (latent, count)
        self.resets = 0

    @torch.no_grad()
    def update(self, talk: str, wav_path: str) -> list[int]:
        enc = self.tok.model.speaker_encoder
        _, ref = self.tok.process_audio(wav_path)
        mel = self.tok.model.mel_transformer(ref.to(self.tok.device)).squeeze(1)
        _, feats = enc.speaker_encoder(mel.transpose(1, 2), True)
        x = enc.perceiver_sampler(feats.transpose(1, 2)).transpose(1, 2)
        s, n = self.state.get(talk, (None, 0))
        if s is not None and torch.nn.functional.cosine_similarity(x.flatten(), s.flatten(), dim=0) < self.reset:
            s, n = None, 0   # another speaker: start again
            self.resets += 1
        if s is None:
            s, n = x, 1
        else:
            n += 1
            lam = max(self.alpha_min, 1.0 / n)
            s = (1 - lam) * s + lam * x
        self.state[talk] = (s, n)
        _, idx = enc.quantizer(s)
        return flat(idx)[:32]


class BayesVoiceMemory:
    """A talk's speaker as a precision-weighted posterior mean of BiCodec's pre-quantisation speaker latent, with locked
    tokens (--voice-bayes). Every observation is the talk's audio heard since the previous one, voiced frames only, encoded
    in pieces of at most BiCodec's 6 s reference (shorter ones are tiled by BiCodec, so observations wait for --voice-min-obs-ms
    of speech once an estimate exists); each piece weighs its seconds of speech, down-weighted the further it lies from the
    current estimate (a Cauchy weight on the cosine distance, scaled by the talk's median distance); the evidence is capped at
    --voice-max-sec, beyond which old evidence fades. The 32 tokens change only when re-quantising the estimate would change
    at least --voice-lock of them, and two consecutive pieces of 2 s or more below --voice-reset cosine start a new speaker."""

    def __init__(self, tok, tmp: Path, min_obs_ms: int, max_sec: float, lock: float, reset: float, freeze_sec: float = 0.0):
        self.tok, self.tmp, self.min_obs, self.max_sec, self.lock, self.reset = tok, tmp, min_obs_ms * 16, max_sec, lock, reset
        self.freeze = freeze_sec   # > 0: once this much speech supports the estimate, the tokens stay until a new speaker
        self.state = {}
        self.stats = {"pieces": 0, "token_updates": 0, "resets": 0, "kept": 0}

    @staticmethod
    def voiced(x: np.ndarray, floor_db: float = -50.0, rel_db: float = 30.0) -> np.ndarray:
        n = len(x) // 320
        if n == 0:
            return x[:0]
        fr = x[: n * 320].reshape(n, 320)
        db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
        keep = db > max(floor_db, float(np.percentile(db, 95)) - rel_db)
        return fr[keep].reshape(-1)

    @torch.no_grad()
    def latent(self, x: np.ndarray, name: str) -> torch.Tensor:
        p = self.tmp / f"{name}.wav"
        sf.write(p, x, 16000)
        enc = self.tok.model.speaker_encoder
        _, ref = self.tok.process_audio(str(p))
        mel = self.tok.model.mel_transformer(ref.to(self.tok.device)).squeeze(1)
        _, feats = enc.speaker_encoder(mel.transpose(1, 2), True)
        return enc.perceiver_sampler(feats.transpose(1, 2)).transpose(1, 2)

    def tokens_of(self, s: torch.Tensor) -> list[int]:
        _, idx = self.tok.model.speaker_encoder.quantizer(s)
        return flat(idx)[:32]

    def update(self, talk: str, audio: np.ndarray, end: int, name: str) -> list[int]:
        st = self.state.setdefault(talk, {"sum": None, "w": 0.0, "tokens": None, "pending": audio[:0], "last": 0, "d": [], "miss": 0})
        new = audio[st["last"]: end]
        st["last"] = max(st["last"], end)
        st["pending"] = np.concatenate([st["pending"], self.voiced(new)])
        if st["sum"] is None and len(st["pending"]) < 8000:   # nothing usable yet: the raw audio heard, as before
            st["pending"] = np.concatenate([st["pending"], new]) if len(st["pending"]) == 0 else st["pending"]
        if st["sum"] is not None and len(st["pending"]) < self.min_obs:
            self.stats["kept"] += 1
            return st["tokens"]
        piece_len = 6 * 16000
        pieces = [st["pending"][i: i + piece_len] for i in range(0, len(st["pending"]), piece_len)]
        pieces = [p for p in pieces if len(p) >= 4000] or pieces[:1]
        st["pending"] = audio[:0]
        for k, piece in enumerate(pieces):
            x = self.latent(piece, f"{name}_v{k}")
            w = len(piece) / 16000
            self.stats["pieces"] += 1
            if st["sum"] is not None:
                mean = st["sum"] / st["w"]
                d = 1 - float(torch.nn.functional.cosine_similarity(x.flatten(), mean.flatten(), dim=0))
                if d > 1 - self.reset and w >= 2:
                    st["miss"] += 1
                    if st["miss"] >= 2:   # another speaker: start again from this piece
                        st.update(sum=None, w=0.0, tokens=None, d=[], miss=0, frozen=False)
                        self.stats["resets"] += 1
                    else:
                        continue
                else:
                    st["miss"] = 0
                if st["sum"] is not None:
                    scale = 2 * float(np.median(st["d"])) if len(st["d"]) >= 3 else 0.1
                    w = w / (1 + (d / max(scale, 0.02)) ** 2)
                    st["d"].append(d)
            st["sum"] = x * w if st["sum"] is None else st["sum"] + x * w
            st["w"] += w
            if st["w"] > self.max_sec:   # bounded evidence: older pieces fade
                st["sum"] *= self.max_sec / st["w"]
                st["w"] = self.max_sec
        self.current = st["sum"] / st["w"]   # the posterior mean after this update, for --select-voice
        if self.freeze > 0 and st["tokens"] is not None and st.get("frozen"):
            return st["tokens"]
        q = self.tokens_of(st["sum"] / st["w"])
        if self.freeze > 0 and st["w"] >= self.freeze:
            st["frozen"] = True
        if st["tokens"] is None or np.mean(np.asarray(q) != np.asarray(st["tokens"])) >= self.lock:
            if st["tokens"] is not None:
                self.stats["token_updates"] += 1
            st["tokens"] = q
        return st["tokens"]


def realsi_items(direction: str) -> list[dict]:
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


def plan_blocks(item: dict, tgt: str, window_ms: int, delta_ms: int) -> list[dict]:
    dur = item["end"] - item["start"]
    blocks = []
    for end, text in item["subs"]:
        write = min(math.ceil((end - item["start"] + delta_ms) / window_ms) * window_ms, dur)
        if blocks and blocks[-1]["write_ms"] == write:
            blocks[-1]["texts"].append(text)
        else:
            blocks.append({"write_ms": write, "texts": [text]})
    join = "" if tgt == "cmn" else " "
    for k, b in enumerate(blocks):
        b["text"] = (join if k else "") + join.join(b.pop("texts"))
        b["final"] = k == len(blocks) - 1
    return blocks


def plan_word_blocks(item: dict, align: dict, window_ms: int, delta_ms: int, min_words: int) -> list[dict]:
    """--text-policy word: the translation released word by word (realsi_align.py's supports, the omni rule).
    At the end of each read window the words whose tau (running max of support + delta) has passed are
    handed over once there are min_words of them; when the source ends, all that is left."""
    dur = item["end"] - item["start"]
    text, starts = item["trg_text"], align["word_start"]
    tau, run = [], 0
    for s in align["support_ms"]:
        run = max(run, s + delta_ms)
        tau.append(min(run, dur + delta_ms))

    def boundary(w: int) -> int:
        if w >= len(starts):
            return len(text)
        b = starts[w]
        while b > 0 and text[b - 1].isspace():
            b -= 1
        return b

    blocks, done, t = [], 0, window_ms
    while done < len(tau):
        t = min(t, dur)
        ready = sum(1 for x in tau if x <= t) if t < dur else len(tau)
        if ready - done >= min_words or (t >= dur and ready > done):
            blocks.append({"write_ms": t, "text": text[boundary(done) if done else 0: boundary(ready)], "final": False})
            done = ready
        t += window_ms
    blocks[-1]["final"] = True
    return blocks



def select_by_asr(results, owners, r, front, device, tgt, url, per_request=16):
    """Plan 2's posterior choice for round r: per block, the candidate whose speech -- after the blocks already
    chosen -- an independent ASR finds likeliest to say the block's text. Returns, per owner, (index, its score,
    every candidate's score)."""
    import base64
    import urllib.request

    lang = "Chinese" if tgt == "cmn" else "English"
    items, where = [], []
    for k, (it, res) in enumerate(zip(owners, results)):
        prev = [c for b in it["blocks"][:r] for c in b["codes"]]
        prefix = "".join(b["text"] for b in it["blocks"][:r])
        for j, o in enumerate(res.outputs):
            cand = [t - SEMANTIC_0 for t in o.token_ids if SEMANTIC_0 <= t <= SEMANTIC_LAST]
            codes = prev + cand
            wav = np.zeros(1600, dtype=np.float32)
            if codes:
                wav = np.asarray(front.decode(torch.tensor([*it["voice"], *codes], dtype=torch.long, device=device)),
                                 dtype=np.float32).reshape(-1)
            pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()
            items.append({"pcm16": base64.b64encode(pcm).decode(), "prefix": prefix, "text": it["blocks"][r]["text"],
                          "lang": lang})
            where.append((k, j))
    scores = []
    for s in range(0, len(items), per_request):
        req = urllib.request.Request(url, data=json.dumps({"items": items[s: s + per_request]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            scores += json.loads(resp.read())["nll"]
    per = {}
    for (k, j), x in zip(where, scores):
        per.setdefault(k, []).append(float("inf") if x is None else x)
    out = []
    for k in range(len(owners)):
        c = per[k]
        j = int(np.argmin(c))
        out.append((j, None if c[j] == float("inf") else round(c[j], 4), [None if x == float("inf") else round(x, 4) for x in c]))
    return out

def select_by_voice(results, owners, r, front, device, memory, tgt, tail_s: float = 6.0):
    """--select-voice: per block, among the candidates that close properly, have a plausible length and sit in the upper
    half by mean log-probability, the one whose speech -- after the blocks already chosen, its last tail_s seconds --
    BiCodec's speaker encoder places closest (cosine) to the talk's posterior voice. WavLM, the evaluation model, is not
    used, so the choice is not tuned to the metric. Returns, per owner, (index, its cosine, every candidate's cosine)."""
    out = []
    for k, (it, res) in enumerate(zip(owners, results)):
        prev = [c for b in it["blocks"][:r] for c in b["codes"]]
        exp = expected_codes(it["blocks"][r]["text"], tgt)
        cands = []
        for j, o in enumerate(res.outputs):
            ids = list(o.token_ids)
            cand = [t - SEMANTIC_0 for t in ids if SEMANTIC_0 <= t <= SEMANTIC_LAST]
            ok = bool(ids) and ids[-1] in (WAIT_READ, END_SEMANTIC) and 0.5 * exp <= len(cand) <= 1.8 * exp
            lp = (o.cumulative_logprob or 0.0) / max(len(ids), 1)
            cands.append((j, cand, ok, lp))
        pool = [c for c in cands if c[2]] or cands
        cut = sorted((c[3] for c in pool), reverse=True)[(len(pool) - 1) // 2]
        pool = [c for c in pool if c[3] >= cut]
        sims = {}
        for j, cand, _, _ in pool:
            codes = prev + cand
            if not codes:
                sims[j] = -1.0
                continue
            wav = np.asarray(front.decode(torch.tensor([*it["voice"], *codes], dtype=torch.long, device=device)),
                             dtype=np.float32).reshape(-1)[-int(tail_s * 16000):]
            x = memory.latent(wav, f"selvoice_p{os.getpid()}_{k % 4}")   # shards share tmp/: one file name per process
            sims[j] = float(torch.nn.functional.cosine_similarity(x.flatten(), it["voice_latent"].flatten(), dim=0))
        j = max(sims, key=sims.get)
        out.append((j, round(sims[j], 4), {str(a): round(v, 4) for a, v in sims.items()}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="a UniSS directory (an export's uniss/)")
    ap.add_argument("--direction", choices=sorted(LANGS), required=True)
    ap.add_argument("--window", type=float, default=3.0, help="read window, seconds (SimulS2ST-Omni's m)")
    ap.add_argument("--delta-ms", type=int, default=200)
    ap.add_argument("--holdback", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="this many sentences, evenly spaced over the set")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--speed", type=int, default=9, help="speed token index (9: the source's pace)")
    ap.add_argument("--text-policy", choices=("clause", "word", "file"), default="clause",
                    help="clause: RealSI's sub-utterances, each once heard; word: the translation word by word "
                         "(realsi_align.py, --align); file: blocks written by si_thinker_stream.py (--blocks-file), "
                         "the whole system with the frozen Thinker")
    ap.add_argument("--blocks-file", default="")
    ap.add_argument("--align", default=str(REPO / "eval_outputs/uniss_qwen3_omni_v1/realsi_align_v1"))
    ap.add_argument("--min-words", type=int, default=2, help="word policy: words a block waits for (the last excepted)")
    ap.add_argument("--speed-policy", choices=("fixed", "block", "queue"), default="fixed",
                    help="fixed: --speed for every block; block: per block, the source since the last write over the "
                         "block's natural length (--codes-per-unit), as T1c's speed tokens are trained; queue (fluency plan "
                         "v3): the block pace, then faster by --queue-shift-per-s steps a second of speech still queued when "
                         "the block is written (from the earlier blocks' codes, >= 0.5 s), at most --queue-shift-max, and at "
                         "least --final-shift for a sentence's last block")
    ap.add_argument("--queue-shift-per-s", type=float, default=3.0)
    ap.add_argument("--queue-shift-max", type=int, default=6)
    ap.add_argument("--final-shift", type=int, default=2)
    ap.add_argument("--speed-shift", type=int, default=0,
                    help="block policy: ask every block this many speed steps faster (+) or slower (-) than its pace")
    ap.add_argument("--codes-per-unit", type=float, default=12.0,
                    help="natural codes a Chinese character / English syllable (training targets: median 12.8 with pauses)")
    ap.add_argument("--history-ban", type=int, default=0, help="ban an n-code run the earlier blocks spoke (0 = off)")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="the talker's context; long low-latency utterances (many small blocks) can pass 8192")
    ap.add_argument("--temperature", type=float, default=0.7, help="the talker's sampling temperature (top-p 0.8)")
    ap.add_argument("--select-k", type=int, default=1,
                    help="Plan 2: sample k continuations a block and keep the one an independent ASR (si_asr_scorer.py at "
                         "--scorer-url) finds likeliest to say the block's text, given the speech and text before it "
                         "-- the posterior over what was said, not the talker's own likelihood (--best-of)")
    ap.add_argument("--scorer-url", default="http://127.0.0.1:18765/score")
    ap.add_argument("--select-voice", type=int, default=1,
                    help="with --voice-bayes: sample k continuations a block and keep the one closest to the talk's voice "
                         "(BiCodec speaker latent) among the well-formed, more likely half (select_by_voice)")
    ap.add_argument("--items-file", default="",
                    help="jsonl of {id, audio, dur_ms, translation} (training items) instead of RealSI; with --text-policy file")
    ap.add_argument("--dump-seq", action="store_true",
                    help="keep, per block, the talker's whole input at that round and what it said (codes + closing): "
                         "training rows for distilling the chosen speech back into the talker (Plan 2)")
    ap.add_argument("--dump-codes", action="store_true", help="keep each block's semantic codes in results (Plan 2's CTC check)")
    ap.add_argument("--best-of", type=int, default=1,
                    help="sample this many continuations a block and keep the one the talker finds likeliest "
                         "(mean log-probability a token): a block that skips words fits its text worse")
    ap.add_argument("--lookahead", action="store_true",
                    help="file policy: show each block the text its blocks-file entry holds back (\"look\": tentative, "
                         "not to be spoken) between the lookahead brackets, as a talker trained with --si-lookahead-max reads it")
    ap.add_argument("--voice-ema", type=float, default=0.0,
                    help="speaker memory across a talk's sentences (talk = id up to its last '_', items in order): the "
                         "voice is the running estimate s = (1-l) s + l x of BiCodec's pre-quantisation speaker latent x, "
                         "l = max(this, 1/n) -- a running mean that turns into an EMA; quantised to the 32 global tokens. 0: off")
    ap.add_argument("--voice-context-ms", type=int, default=0,
                    help="the voice reference is the talk heard so far: this many ms of the recording up to the sentence's "
                         "first commit, earlier sentences included (RealSI sentences are cut from one talk); 0: the sentence's own prefix")
    ap.add_argument("--voice-bayes", action="store_true",
                    help="speaker memory as a precision-weighted posterior over the talk's voiced audio, tokens locked (BayesVoiceMemory)")
    ap.add_argument("--voice-min-obs-ms", type=int, default=3000, help="with --voice-bayes: speech gathered before an update")
    ap.add_argument("--voice-max-sec", type=float, default=60.0, help="with --voice-bayes: cap on the evidence, in seconds of speech")
    ap.add_argument("--voice-lock", type=float, default=0.25,
                    help="with --voice-bayes: share of the 32 tokens that must change before the voice is updated (0 = always)")
    ap.add_argument("--voice-freeze-sec", type=float, default=0.0,
                    help="with --voice-bayes: freeze the tokens once this many seconds of speech support them (0 = never)")
    ap.add_argument("--voice-reset", type=float, default=0.5,
                    help="with --voice-ema: a sentence whose latent is below this cosine to the memory starts a new speaker")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from uniss import UniSSTokenizer
    from experiments.uniss_qwen3_omni_v1.evaluation.si_guard import SIGuard
    from experiments.uniss_qwen3_omni_v1.training.si_batch import LOOK_CLOSE, LOOK_OPEN
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import (
        BALANCE, END_CONTENT, END_GLOBAL, GLM_0, GLOBAL_0, LANG_TOKEN, SPEED_0, START_CONTENT, START_GLOBAL,
        START_SEMANTIC, TASK_S2S, TASK_S2T, TASK_TTS, WRITE, speed_token)

    src_lang, tgt = LANGS[args.direction]
    items = realsi_items(args.direction)
    if args.items_file:
        items = [{"id": r["id"], "wav": r["audio"], "start": 0, "end": int(r["dur_ms"]), "src_text": "",
                  "trg_text": r.get("translation", ""), "subs": [], "voice_audio": r.get("voice_audio"),
                  "talk_wav": r.get("talk_wav"), "talk_offset_ms": int(r.get("offset_ms", 0))}
                 for r in map(json.loads, open(args.items_file, encoding="utf-8"))]
    aligned = {}
    if args.text_policy == "word":
        aligned = {json.loads(l)["id"]: json.loads(l)
                   for l in open(Path(args.align) / f"align_{args.direction}.jsonl", encoding="utf-8")}
        items = [it for it in items if it["id"] in aligned]
    given = {}
    if args.text_policy == "file":
        given = {json.loads(l)["id"]: json.loads(l)["blocks"] for l in open(args.blocks_file, encoding="utf-8")}
        items = [it for it in items if given.get(it["id"])]
    if args.limit and args.limit < len(items):
        items = [items[round(i * (len(items) - 1) / (args.limit - 1))] for i in range(args.limit)]
    items = items[args.shard:: args.num_shards]
    out = Path(args.output).resolve()
    for d in ("wav", "src", "tmp"):
        (out / d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    timing = {}
    front = UniSSTokenizer.from_pretrained(str(REPO / "pretrained_models/UniSS"), device=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"))
    lang, speed = LANG_TOKEN[tgt], SPEED_0 + args.speed
    window_ms = int(round(args.window * 1000))

    # 1. Source, blocks, voice, and the GLM tokens each block may read.
    t0 = time.time()
    voice_memory = VoiceMemory(front.bicodec_tokenizer, args.voice_ema, args.voice_reset) if args.voice_ema > 0 else None
    bayes_memory = (BayesVoiceMemory(front.bicodec_tokenizer, out / "tmp", args.voice_min_obs_ms, args.voice_max_sec, args.voice_lock,
                                     args.voice_reset, args.voice_freeze_sec) if args.voice_bayes else None)
    cache, talk_cache = {}, {}
    for it in items:
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache = {it["wav"]: x.mean(axis=1)}
        wav = cache[it["wav"]][it["start"] * 16: it["end"] * 16]
        it["src_path"] = str(out / "src" / f"{it['id'].replace('/', '__')}.wav")
        sf.write(it["src_path"], wav, 16000)
        dur = len(wav) // 16
        it["dur_ms"] = dur
        it["blocks"] = ([dict(b) for b in given[it["id"]]] if args.text_policy == "file" else
                        plan_word_blocks(it, aligned[it["id"]], window_ms, args.delta_ms, args.min_words)
                        if args.text_policy == "word" else plan_blocks(it, tgt, window_ms, args.delta_ms))
        first = it["blocks"][0]["write_ms"]
        p = out / "tmp" / f"{it['id'].replace('/', '__')}_voice.wav"
        if it.get("voice_audio"):   # a speaker reference prepared by the caller (e.g. the talk heard so far)
            p = Path(it["voice_audio"])
        elif args.voice_context_ms:   # the talk heard so far, up to this sentence's first commit
            talk = cache[it["wav"]]
            end = (it["start"] + max(first, 1000)) * 16
            sf.write(p, talk[max(0, end - args.voice_context_ms * 16): end], 16000)
        else:
            sf.write(p, wav[: max(first, 1000) * 16], 16000)
        if bayes_memory is not None:   # the talk heard so far, up to this sentence's first commit
            if it.get("talk_wav"):   # pieces of one long recording (items file with talk_wav + offset_ms): the whole talk
                if it["talk_wav"] not in talk_cache:
                    xt, srt = sf.read(it["talk_wav"], dtype="float32", always_2d=True)
                    assert srt == 16000, it["talk_wav"]
                    talk_cache.clear()
                    talk_cache[it["talk_wav"]] = xt.mean(axis=1)
                audio_t, end_t = talk_cache[it["talk_wav"]], (it["talk_offset_ms"] + max(first, 1000)) * 16
            else:
                audio_t, end_t = cache[it["wav"]], (it["start"] + max(first, 1000)) * 16
            it["voice"] = bayes_memory.update(it["id"].rsplit("_", 1)[0], audio_t, end_t, it["id"].replace("/", "__"))
            it["voice_latent"] = bayes_memory.current.detach().clone()
        elif args.voice_ema > 0:
            it["voice"] = voice_memory.update(it["id"].rsplit("_", 1)[0], str(p))
        else:
            it["voice"] = flat(front.bicodec_tokenizer.encode_wav_to_tokens(str(p)))[:32]
        g_prev = 0
        for b in it["blocks"]:
            heard = wav[: b["write_ms"] * 16]
            glm = flat(front.glm4.tokenize(speech=torch.from_numpy(heard).unsqueeze(0), sr=16000))
            b["glm"] = glm      # this prefix's tokenisation: the blocks up to here read it
            n = len(glm) if b["write_ms"] >= dur else max(0, min(len(glm), round(b["write_ms"] / GLM_MS)) - args.holdback)
            b["glm_end"] = g_prev = max(n, g_prev)
            b["ids"] = tok(b["text"], add_special_tokens=False)["input_ids"]
            b["look_ids"] = (tok(b["look"], add_special_tokens=False)["input_ids"]
                             if args.lookahead and b.get("look") and not b["final"] else [])
        w_prev = 0
        for b in it["blocks"]:   # the speed token each block asks with
            b["speed"] = speed if args.speed_policy == "fixed" else min(max(speed_token(
                max(b["write_ms"] - w_prev, 1) / 1000, max(10, round(args.codes_per_unit * spoken_units(b["text"], tgt))))
                + args.speed_shift, SPEED_0), SPEED_0 + 34)
            w_prev = b["write_ms"]
            b["speed_base"] = b["speed"]
    timing["front_end"] = time.time() - t0
    if voice_memory is not None:
        print(f"voice memory: {len(voice_memory.state)} talks, {voice_memory.resets} speaker resets", flush=True)
    if bayes_memory is not None:
        print(f"bayes voice memory: {len(bayes_memory.state)} talks, {bayes_memory.stats}", flush=True)

    # 2. The talker, a round of blocks at a time.
    talker = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, seed=args.seed,
                 logits_processors=[SIGuard])
    t0 = time.time()
    for r in range(max(len(it["blocks"]) for it in items)):
        reqs, params, owners = [], [], []
        for n_it, it in enumerate(items):
            if r >= len(it["blocks"]):
                continue
            if args.speed_policy == "queue":   # the queue this block is written into, from the codes spoken so far
                end_prev = 0
                for x in it["blocks"][:r]:
                    end_prev = max(x["write_ms"], end_prev) + len(x["codes"]) * 20
                bq = it["blocks"][r]
                q = max(0, end_prev - bq["write_ms"]) / 1000
                shift = min(args.queue_shift_max, int(round(args.queue_shift_per_s * q))) if q >= 0.5 else 0
                if bq["final"]:
                    shift = max(shift, args.final_shift)
                bq["speed"] = min(max(bq["speed_base"] + shift, SPEED_0), SPEED_0 + 34)
            glm = it["blocks"][r]["glm"]
            seq = [TASK_S2S, BALANCE, lang, START_GLOBAL] + [GLOBAL_0 + g for g in it["voice"]] + [END_GLOBAL]
            g_prev = 0
            for k, b in enumerate(it["blocks"][: r + 1]):
                seq += [GLM_0 + t for t in glm[g_prev: b["glm_end"]]]
                seq += [WRITE, TASK_S2T, lang, b["speed"], START_CONTENT] + b["ids"] + [END_CONTENT]
                if b["look_ids"]:
                    seq += [LOOK_OPEN] + b["look_ids"] + [LOOK_CLOSE]
                seq += [WRITE, TASK_TTS, lang, b["speed"]]
                seq += ([] if b["final"] else [STREAMING_MODE]) + [START_SEMANTIC]
                if k < r:
                    seq += [SEMANTIC_0 + c for c in b["codes"]] + [b["closing"]]
                g_prev = b["glm_end"]
            b = it["blocks"][r]
            b["prompt_len"] = len(seq)
            if args.dump_seq:
                b["prompt"] = seq
            reqs.append({"prompt_token_ids": seq})
            params.append(SamplingParams(n=max(args.best_of, args.select_k, args.select_voice), temperature=args.temperature,
                                         top_p=0.8, max_tokens=1500, seed=args.seed + 1000 * n_it + r,
                                         logprobs=0 if max(args.best_of, args.select_voice) > 1 else None,
                                         stop_token_ids=[WAIT_READ, END_SEMANTIC], skip_special_tokens=False,
                                         extra_args={"si_guard": {"expected": expected_codes(b["text"], tgt),
                                                                  **({"history": [c for x in it["blocks"][:r] for c in x["codes"]],
                                                                      "hist_n": args.history_ban} if args.history_ban else {}),
                                                                  "final": b["final"], "ngram": 8, "min_frac": 0.3}}))
            owners.append(it)
        print(f"round {r}: {len(reqs)} blocks", flush=True)
        results = talker.generate(reqs, params, use_tqdm=False)
        chosen = select_by_asr(results, owners, r, front, device, tgt, args.scorer_url) if args.select_k > 1 else None
        vchosen = (select_by_voice(results, owners, r, front, device, bayes_memory, tgt)
                   if args.select_voice > 1 and bayes_memory is not None else None)
        for k, (it, res) in enumerate(zip(owners, results)):
            if vchosen is not None:
                best = res.outputs[vchosen[k][0]]
                it["blocks"][r]["select_voice_cos"], it["blocks"][r]["cand_voice_cos"] = vchosen[k][1], vchosen[k][2]
            elif chosen is not None:
                best = res.outputs[chosen[k][0]]
                it["blocks"][r]["select_nll"], it["blocks"][r]["cand_nll"] = chosen[k][1], chosen[k][2]
            else:
                best = max(res.outputs, key=lambda o: (o.cumulative_logprob or 0.0) / max(len(o.token_ids), 1))
            ids = list(best.token_ids)
            b = it["blocks"][r]
            b["codes"] = [t - SEMANTIC_0 for t in ids if SEMANTIC_0 <= t <= SEMANTIC_LAST]
            b["closing"] = ids[-1] if ids and ids[-1] in (WAIT_READ, END_SEMANTIC) else (END_SEMANTIC if b["final"] else WAIT_READ)
            # at the guard's hard limit the closing token is forced, not chosen
            exp = expected_codes(b["text"], tgt)
            b["forced"] = len(b["codes"]) >= max(int(2.2 * exp), exp + 100)
            b["closed"] = bool(ids) and ids[-1] in (WAIT_READ, END_SEMANTIC) and not b["forced"]
    timing["talker"] = time.time() - t0

    # 3. One decode per sentence, then the timeline.
    t0 = time.time()
    records = []
    for it in items:
        codes = [c for b in it["blocks"] for c in b["codes"]]
        audio = np.zeros(0, dtype=np.float32)
        if codes:
            audio = np.asarray(front.decode(torch.tensor([*it["voice"], *codes], dtype=torch.long, device=device)),
                               dtype=np.float32).reshape(-1)
        src = sf.read(it["src_path"], dtype="float32")[0]
        pieces, end_prev, at = [], 0, 0
        for b in it["blocks"]:
            n = len(b["codes"]) * 320
            b["speech_at_ms"] = at // 16          # where the block starts in the speech alone
            piece = audio[at: at + n]
            at += n
            start = max(b["write_ms"], end_prev)
            pieces.append((start, piece))
            end_prev = start + len(piece) / 16
            b.update(start_ms=int(start), dur_ms=int(len(piece) / 16))
        total = int(max(it["dur_ms"], end_prev)) + 300
        right = np.zeros(total * 16, dtype=np.float32)
        for start, piece in pieces:
            a = int(start * 16)
            right[a: a + len(piece)] += piece
        left = np.zeros_like(right)
        left[: len(src)] = src[: len(right)]
        tgt_path = out / "wav" / f"{it['id'].replace('/', '__')}_simul.wav"
        sf.write(tgt_path, right, 16000)
        speech_path = out / "wav" / f"{it['id'].replace('/', '__')}_speech.wav"   # the speech alone, no waits: for A.PCP, voice, alignment
        sf.write(speech_path, audio if audio.size else np.zeros(320, dtype=np.float32), 16000)
        sf.write(out / "wav" / f"{it['id'].replace('/', '__')}_stereo.wav", np.stack([left * 0.8, right], axis=1), 16000)
        lead = 0
        for c in codes:
            if c != SILENCE_CODE:
                break
            lead += 1
        speech = sum(len(p) / 16 for _, p in pieces)
        gaps = sum(max(0.0, s - (ps + len(pp) / 16)) for (s, _), (ps, pp) in zip(pieces[1:], pieces[:-1]))
        records.append({
            "id": it["id"], "mode": "si", "name": it["id"], "src_lang": src_lang, "tgt_lang": tgt,
            "transcription_ref": it["src_text"], "translation_ref": it["trg_text"],
            "translated_text": "".join(b["text"] for b in it["blocks"]), "audio_path": str(tgt_path),
            "speech_path": str(speech_path),
            "audio_duration_seconds": total / 1000, "source_audio_path": it["src_path"],
            "source_duration_ms": it["dur_ms"], "window_s": args.window, "text_policy": args.text_policy,
            "speed_policy": args.speed_policy, "talker_temperature": args.temperature, "best_of": args.best_of,
            "speed_shift": args.speed_shift, "select_k": args.select_k,
            "codes": len(codes), "error": None if codes else "no audio",
            "start_offset_ms": int(pieces[0][0] + lead * 20) if codes else None,
            "end_offset_ms": int(end_prev - it["dur_ms"]),
            "silence_ratio": round(gaps / max(speech + gaps, 1e-6), 4),
            "blocks": [{**{k: b[k] for k in ("write_ms", "glm_end", "text", "final", "start_ms", "dur_ms", "closed", "forced", "speed", "speech_at_ms")},
                        **({"code_ids": [int(c) for c in b["codes"]]} if args.dump_codes else {}),
                        **{k: b[k] for k in ("select_nll", "cand_nll", "select_voice_cos", "cand_voice_cos") if k in b},
                        **({"train_ids": b["prompt"] + [SEMANTIC_0 + c for c in b["codes"]] + [b["closing"]],
                            "label_from": len(b["prompt"])} if args.dump_seq else {})}
                       | {"codes": len(b["codes"]), "closing": "wait_read" if b["closing"] == WAIT_READ else "end"}
                       for b in it["blocks"]]})
    timing["decode"] = time.time() - t0
    with open(out / f"part_{args.shard}.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    print("timing " + "  ".join(f"{k} {v:.1f}s" for k, v in timing.items()) + f"  sentences {len(records)}", flush=True)


if __name__ == "__main__":
    main()
