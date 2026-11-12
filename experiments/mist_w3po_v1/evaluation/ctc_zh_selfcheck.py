"""CPU self-check of the Chinese CTC scorer (si_ctc_scorer --lang zh; reports/mist_w3po_v1/en2zh_full_plan_prereg_v1
.zh-CN.md §4.2) before it selects any en2zh talker candidate: are its NLL / character values sane, and does it rank a
correctly read block above obviously misread ones?

Audio: UniSS-talker Chinese speech of AMI TRAIN meetings (the zh_revoice_v1 talker clips, placed in the zh_mix_v8
sessions; the clips themselves are no longer on disk, so each utterance is cut from the session at [zh_begin - 0.1,
zh_end + 0.1]); only utterances that are placed, not hesitations, unflagged, 1.5-8 s, tempo 0.9-1.2, >= 6 scorable
characters, and with no other placed utterance within 0.3 s (single talker). Up to --n per meeting, first meetings of
the train split in alphabetical order. No dev / test data.

Per utterance, the NLL / character of five targets on the same audio:
  correct    its own text
  wrong      another selected utterance's text (closest character count)
  omission   the first half of its characters (the audio says more than the text)
  insertion  its text + the wrong text's first half (the text says more than the audio)
  sub30      30 % of its characters replaced by random vocabulary characters (seeded)
Reports means, P(correct < variant) per variant, and the best-path CER against the text. Writes --out (json; refuses
to overwrite).

    python -m experiments.mist_w3po_v1.evaluation.ctc_zh_selfcheck --out <json> [--n 12] [--meetings 2]
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path

import numpy as np

from experiments.mist_w3po_v1.evaluation.si_ctc_scorer import CTC_ZH, Scorer, ctc_text_zh

DATA = Path("data/ami_multispk_si_v1")


def cer(ref: str, hyp: str) -> float:
    n, m = len(ref), len(hyp)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return d[m] / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--meetings", type=int, default=2)
    ap.add_argument("--model", default=CTC_ZH)
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch

    torch.set_num_threads(16)
    if not torch.cuda.is_available():   # CPU only (CUDA_VISIBLE_DEVICES=): transformers probes the accelerator on import
        torch.accelerator.current_accelerator = lambda *a_, **k_: None
    sc = Scorer(a.model, "zh", device="cpu")
    rng = random.Random(a.seed)
    split = json.load(open(DATA / "meetings.json"))
    ms = sorted(m for m, s in split.items() if s == "train")[: a.meetings]
    sel = []
    for m in ms:
        us = [u for u in map(json.loads, open(DATA / "zh_mix_v8" / "utts" / f"{m}.jsonl", encoding="utf-8")) if u["placed"]]
        x, sr = sf.read(str(DATA / "session_audio" / "zh_mix_v8" / f"{m}.wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        k = 0
        for u in us:
            d = u["zh_end"] - u["zh_begin"]
            t = ctc_text_zh("", u["zh"], sc.vocab)
            if u["hes"] or u.get("flags") or not 1.5 <= d <= 8.0 or not 0.9 <= u.get("tempo", 1.0) <= 1.2 or len(t) < 6:
                continue
            if any(v is not u and v["zh_begin"] < u["zh_end"] + 0.3 and v["zh_end"] > u["zh_begin"] - 0.3 for v in us):
                continue
            w = x[max(0, round((u["zh_begin"] - 0.1) * sr)): round((u["zh_end"] + 0.1) * sr)]
            if sr != 16000:
                import torchaudio

                w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
            pcm = base64.b64encode((np.clip(w, -1, 1) * 32767).astype(np.int16).tobytes()).decode()
            sel.append({"id": u["record_id"], "meeting": m, "text": u["zh"], "chars": t, "pcm16": pcm, "dur": round(d, 2)})
            k += 1
            if k >= a.n:
                break
    voc = sorted(ch for ch in sc.vocab if len(ch) == 1 and "一" <= ch <= "鿿")
    rows = []
    for i, s in enumerate(sel):
        others = [o for o in sel if o["id"] != s["id"]]
        wr = min(others, key=lambda o: (abs(len(o["chars"]) - len(s["chars"])), o["id"]))
        c = list(s["chars"])
        for j in rng.sample(range(len(c)), max(1, round(0.3 * len(c)))):
            c[j] = rng.choice(voc)
        tg = {"correct": s["text"], "wrong": wr["text"], "omission": s["chars"][: len(s["chars"]) // 2],
              "insertion": s["text"] + wr["chars"][: len(wr["chars"]) // 2], "sub30": "".join(c)}
        nll = dict(zip(tg, sc.score([{"pcm16": s["pcm16"], "prefix": "", "text": v} for v in tg.values()])))
        w = np.frombuffer(base64.b64decode(s["pcm16"]), dtype=np.int16).astype(np.float32) / 32768.0
        with torch.no_grad():
            xx = sc.proc(w, sampling_rate=16000, return_tensors="pt").input_values
            lp = torch.log_softmax(sc.model(xx).logits.float(), -1)[0]
        hyp = ctc_text_zh("", sc.greedy(lp), sc.vocab)
        rows.append({"id": s["id"], "dur": s["dur"], "n_chars": len(s["chars"]), "text": s["text"], "best_path": hyp,
                     "cer": round(cer(s["chars"], hyp), 4), "nll": {k: round(v, 4) for k, v in nll.items()}})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    var = ("wrong", "omission", "insertion", "sub30")
    summ = {"model": a.model, "utterances": len(rows), "meetings": ms,
            "nll_mean": {k: round(float(np.mean([r["nll"][k] for r in rows])), 4) for k in ("correct",) + var},
            "p_correct_lower": {k: round(float(np.mean([r["nll"]["correct"] < r["nll"][k] for r in rows])), 4) for k in var},
            "cer_best_path_mean": round(float(np.mean([r["cer"] for r in rows])), 4), "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
