"""Robust wave 15 CV-LA (reports/mist_w3po_v1/robust_wave15_cvla_prereg.zh-CN.md): the enhanced view v1, causally.

For every S1 dev item (eval_outputs/mist_w3po_v1/wave2_S1/data/items_<dir>.jsonl, the audio exactly as
si_thinker_stream reads it: channel mean, [0, dur_ms * 16)) and every non-final LA2 read t = 1000, 2000, ... < dur_ms
(--windows 1, no snap), v1_t = DeepFilterNet3(x[: t * 16]): only the prefix heard so far goes in, the model state is
reset for every call (init_df defaults, enhance(pad=True), no attenuation limit; 16 kHz -> 48 kHz -> 16 kHz with
df.io.resample, then cut / zero-padded to t * 16 samples). The final read needs no v1 (CV-LA commits v0's whole draft).
Per item one float32 .npy = the v1 prefixes of its reads concatenated in read order; index.jsonl holds id, file, reads,
sha1 of the input samples and the per-read CPU time (one thread per worker process).
Run with the xtranslator venv (deepfilternet 0.5.6), CPU only:
    CUDA_VISIBLE_DEVICES= <xtranslator python> -I experiments/mist_w3po_v1/voicecond/wave15_enh_views.py --direction en2zh \
        --out eval_outputs/mist_w3po_v1/wave15/views/en2zh --workers 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from multiprocessing import get_context
from pathlib import Path

REPO = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
MODEL_DIR = REPO / "pretrained_models/DeepFilterNet3"
WINDOW = 1000
_M = {}


def _stub_torchaudio_backend():
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # import stub only (removed from torchaudio; enhance() does not use it)
        pass

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def read_audio(wav: str, dur_ms: int):
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    assert sr == 16000, wav
    a = x.mean(axis=1)[0: dur_ms * 16]
    return np.ascontiguousarray(a), len(a) // 16


def reads_of(dur_ms: int) -> list[int]:
    return [t for t in range(WINDOW, dur_ms, WINDOW)]   # t < dur_ms: the non-final reads


def _init():
    _stub_torchaudio_backend()
    import torch
    torch.set_num_threads(1)
    from df.enhance import enhance, init_df
    from df.io import resample
    model, st, _ = init_df(str(MODEL_DIR), log_level="ERROR")
    _M.update(model=model, st=st, enhance=enhance, resample=resample, torch=torch)


def enh16(a):
    import numpy as np
    torch = _M["torch"]
    with torch.no_grad():
        t48 = _M["resample"](torch.from_numpy(a)[None], 16000, 48000)
        y = _M["enhance"](_M["model"], _M["st"], t48, pad=True)
        y16 = _M["resample"](y, 48000, 16000)[0].numpy().astype(np.float32)
    out = np.zeros(len(a), np.float32)
    n = min(len(a), len(y16))
    out[:n] = y16[:n]
    return out


def work(job):
    import numpy as np
    k, it, out_dir = job
    a, dur = read_audio(it["audio"], int(it["dur_ms"]))
    reads = reads_of(dur)
    parts, secs = [], []
    for t in reads:
        t0 = time.perf_counter()
        parts.append(enh16(a[: t * 16]))
        secs.append(round(time.perf_counter() - t0, 5))
    f = f"{k:05d}.npy"
    np.save(Path(out_dir) / f, np.concatenate(parts) if parts else np.zeros(0, np.float32))
    return {"id": it["id"], "file": f, "dur_ms": dur, "reads": reads, "sha1": hashlib.sha1(a.tobytes()).hexdigest(),
            "secs": secs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    # S1_DATA (robust final test, 2026-11-20): another data root with the same items_<dir>.jsonl layout; default unchanged
    data = Path(__import__("os").environ.get("S1_DATA", str(REPO / "eval_outputs/mist_w3po_v1/wave2_S1/data")))
    items = [json.loads(l) for l in open(data / f"items_{a.direction}.jsonl",
                                         encoding="utf-8")]
    if a.limit:
        items = items[: a.limit]
    out.mkdir(parents=True)
    jobs = [(k, it, str(out)) for k, it in enumerate(items)]
    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for n, r in enumerate(pool.imap(work, jobs, chunksize=4), 1):
            rows.append(r)
            if n % 200 == 0:
                print(f"{a.direction}: {n}/{len(jobs)} items, {time.time() - t0:.0f}s", flush=True)
    with open(out / "index.jsonl", "x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    secs = [s for r in rows for s in r["secs"]]
    pre = [t / 1000 for r in rows for t in r["reads"]]
    import numpy as np
    summ = {"direction": a.direction, "items": len(rows), "reads": len(secs), "model": str(MODEL_DIR),
            "per_read_sec_mean": round(float(np.mean(secs)), 4), "per_read_sec_median": round(float(np.median(secs)), 4),
            "per_read_sec_p95": round(float(np.percentile(secs, 95)), 4),
            "rtf_prefix": round(float(np.sum(secs) / np.sum(pre)), 5), "wall_s": round(time.time() - t0, 1)}
    (out / "DONE.json").write_text(json.dumps(summ), encoding="utf-8")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
