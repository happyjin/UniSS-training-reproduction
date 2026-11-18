"""Idea 2: index the unpacked open corpora (prereg reports/mist_w3po_v1/voicecond_idea2_prereg.zh-CN.md §0, §1.1, §1.2).

    python -m experiments.mist_w3po_v1.voicecond.idea2_index_sources speakers ls_train-clean-100   (ls_<subset> | aishell3 | cv_zh)
    python -m experiments.mist_w3po_v1.voicecond.idea2_index_sources backgrounds                   (needs bench/HELDOUT_BACKGROUNDS.txt)

Writes data/open_voicecond_v1/idea2/index/speakers_<tag>.jsonl (one line per speaker: spk, lang, corpus, subset, split,
utts [[abs path, dur s], ...]) and backgrounds.jsonl (rel, path, kind music|noise|crowd|sfx, dur, split). Never overwrites.
Splits: LibriSpeech by subset; AISHELL-3 train speakers -> train, test speakers sorted by sha1(id): first half dev, second
half test; Common Voice zh-CN: int(sha1(client_id), 16) % 10 (0-7 train, 8 dev, 9 test) after the CoVoST 2 test-speaker
exclusion and the per-clip SNR >= 20 dB / per-speaker >= 30 s filter; backgrounds: int(sha1(rel), 16) % 10.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
I2 = REPO / "data/open_voicecond_v1/idea2"
SRC = I2 / "src"
IDX = I2 / "index"
HELDOUT = REPO / "data/open_voicecond_v1/bench/HELDOUT_BACKGROUNDS.txt"   # + HELDOUT_BACKGROUNDS_MUSAN.txt and the sha1 rule
CROWD = {"Crowd", "Cheering", "Applause", "Chatter", "Clapping"}
SFX = {"Explosion", "Gunshot_and_gunfire", "Whoosh_and_swoosh_and_swish", "Boom", "Fireworks", "Crack"}
BAD = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
       "Conversation", "Speech_synthesizer", "Singing", "Male_singing", "Female_singing", "Music"}


def h10(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % 10


def split_of(s: str) -> str:
    k = h10(s)
    return "train" if k < 8 else ("dev" if k == 8 else "test")


def dur_of(p):
    try:
        i = sf.info(str(p))
        return round(i.frames / i.samplerate, 3)
    except Exception:  # noqa: BLE001
        return 0.0


def snr_db(p):
    """energy-percentile SNR: P90 - P10 of 25 ms frame energies (dB)"""
    try:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001
        return -1.0, 0.0
    x = x.mean(1)
    n = int(0.025 * sr)
    if len(x) < 4 * n:
        return -1.0, len(x) / sr
    f = x[: len(x) // n * n].reshape(-1, n)
    e = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    return float(np.percentile(e, 90) - np.percentile(e, 10)), len(x) / sr


def write_jsonl(path: Path, rows):
    if path.exists():
        sys.exit(f"refuse to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with open(tmp, "x") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)
    print(f"wrote {path} ({len(rows)} rows)")


def speakers(tag: str):
    rows = []
    if tag.startswith("ls_"):
        sub = tag[3:]
        split = {"dev-clean": "dev", "test-clean": "test"}.get(sub, "train")
        root = SRC / "librispeech/LibriSpeech" / sub
        files = sorted(root.glob("*/*/*.flac"))
        with ProcessPoolExecutor(48) as ex:
            durs = list(ex.map(dur_of, files, chunksize=256))
        by = {}
        for p, d in zip(files, durs):
            if d > 0:
                by.setdefault(p.parent.parent.name, []).append([str(p), d])
        for s, u in sorted(by.items()):
            rows.append({"spk": f"ls:{s}", "lang": "en", "corpus": "librispeech", "subset": sub, "split": split, "utts": u})
    elif tag == "aishell3":
        root = SRC / "aishell3"
        files = sorted(root.glob("**/wav/SSB*/*.wav"))
        with ProcessPoolExecutor(48) as ex:
            durs = list(ex.map(dur_of, files, chunksize=256))
        by = {}
        for p, d in zip(files, durs):
            if d > 0:
                by.setdefault((p.parent.parent.parent.name, p.parent.name), []).append([str(p), d])
        test_spk = sorted({s for (sub, s) in by if sub == "test"}, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
        half = set(test_spk[: len(test_spk) // 2])
        for (sub, s), u in sorted(by.items()):
            split = "train" if sub == "train" else ("dev" if s in half else "test")
            rows.append({"spk": f"a3:{s}", "lang": "zh", "corpus": "aishell3", "subset": sub, "split": split, "utts": u})
    elif tag == "cv_zh":
        tdir = I2 / "cv_zh/asr/common_voice/transcript/zh-CN"
        val = list(csv.DictReader(open(tdir / "validated.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE))
        cov = {l.split("\t")[0] for l in open(I2 / "covost_meta/x/covost_v2.zh-CN_en.tsv")
               if l.rstrip("\n").split("\t")[-1] in ("test", "test_dup", "test_covost")}
        bad = {r["client_id"] for r in val if r["path"] in cov}
        vset = {r["path"]: r["client_id"] for r in val}
        files = {p.name: p for p in (SRC / "cv_zh").glob("*/*.mp3")}
        keep = [(n, c) for n, c in vset.items() if n in files and c not in bad and n not in cov]
        with ProcessPoolExecutor(48) as ex:
            res = list(ex.map(snr_db, [files[n] for n, _ in keep], chunksize=64))
        by, nclip, nsnr = {}, 0, 0
        for (n, c), (snr, d) in zip(keep, res):
            nclip += 1
            if snr >= 20 and d > 0:
                nsnr += 1
                by.setdefault(c, []).append([str(files[n]), round(d, 3), round(snr, 1)])
        for c, u in sorted(by.items()):
            u = sorted(u, key=lambda x: hashlib.sha1((c + x[0]).encode()).hexdigest())[:80]
            if sum(x[1] for x in u) < 30:
                continue
            rows.append({"spk": f"cv:{c[:16]}", "client_id": c, "lang": "zh", "corpus": "cv_zh", "subset": "train",
                         "split": split_of(c), "utts": [x[:2] for x in u]})
        print(f"cv_zh: excluded covost-test speakers {len(bad)}; clips considered {nclip}, snr>=20 {nsnr}; speakers {len(rows)}")
    else:
        sys.exit(f"unknown tag {tag}")
    write_jsonl(IDX / f"speakers_{tag}.jsonl", rows)
    from collections import Counter
    print(dict(Counter(r["split"] for r in rows)))


def backgrounds():
    if not HELDOUT.exists():
        sys.exit(f"{HELDOUT} not there yet: backgrounds are not drawn before it exists (prereg §0)")
    held = set()
    for l in open(HELDOUT):
        l = l.strip()
        if l and not l.startswith("#"):
            held.add(l.rstrip("\n").split("\t")[-1].strip())
    hm = HELDOUT.parent / "HELDOUT_BACKGROUNDS_MUSAN.txt"
    if hm.exists():
        for l in open(hm):
            l = l.strip()
            if l and not l.startswith("#"):
                held.add(l.rstrip("\n").split("\t")[-1].strip())

    def bench_held(rel: str) -> bool:
        """OpenBroadcast-SI v1 rule (HELDOUT_BACKGROUNDS.txt header), applied also before the MUSAN list exists"""
        bp = rel.split("/", 1)[1]          # musan/musan/music/... -> musan/music/...; fsd50k/FSD50K.eval_audio/x.wav -> FSD50K.eval_audio/x.wav
        return bp in held or int(hashlib.sha1(("OpenBroadcast-SI-v1|" + bp).encode()).hexdigest(), 16) % 1000 < 150
    rows = []
    mroot = SRC / "musan/musan"
    vocals = {}
    for ann in (mroot / "music").glob("*/ANNOTATIONS"):
        for l in open(ann):
            t = l.split()
            if len(t) >= 3:
                vocals[t[0]] = t[2]
    cand = []
    for p in sorted((mroot / "music").glob("*/*.wav")):
        if p.parent.name in ("fma", "fma-western-art"):
            continue
        if vocals.get(p.stem, "Y") != "N":
            continue
        cand.append((p, "music"))
    for p in sorted((mroot / "noise").glob("*/*.wav")):
        cand.append((p, "noise"))
    gt = {}
    gzip = REPO / "data/open_voicecond_v1/raw/fsd50k/FSD50K.ground_truth.zip"
    import zipfile
    with zipfile.ZipFile(gzip) as z:
        for r in csv.DictReader(z.read("FSD50K.ground_truth/eval.csv").decode().splitlines()):
            gt[r["fname"]] = set(r["labels"].split(","))
    froot = SRC / "fsd50k/FSD50K.eval_audio"
    for fn, labs in sorted(gt.items()):
        if labs & BAD:
            continue
        kind = "crowd" if labs & CROWD else ("sfx" if labs & SFX else None)
        if kind:
            cand.append((froot / f"{fn}.wav", kind))
    nheld = 0
    for p, kind in cand:
        rel = str(p.relative_to(SRC))
        if bench_held(rel):
            nheld += 1
            continue
        if not p.exists():
            continue
        rows.append({"rel": rel, "path": str(p), "kind": kind, "split": split_of(rel)})
    with ProcessPoolExecutor(48) as ex:
        durs = list(ex.map(dur_of, [r["path"] for r in rows], chunksize=64))
    for r, d in zip(rows, durs):
        r["dur"] = d
    rows = [r for r in rows if r["dur"] > 0.2]
    print(f"backgrounds: candidates {len(cand)}, held out {nheld}, kept {len(rows)}")
    from collections import Counter
    print(Counter((r["kind"], r["split"]) for r in rows))
    write_jsonl(IDX / "backgrounds.jsonl", rows)


if __name__ == "__main__":
    if sys.argv[1] == "speakers":
        speakers(sys.argv[2])
    elif sys.argv[1] == "backgrounds":
        backgrounds()
