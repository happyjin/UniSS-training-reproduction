"""Robust wave 2, diagnosis S2 (reports/mist_w3po_v1/robust_wave2_diag_prereg.zh-CN.md): REFERENCE-turn ("oracle")
diarizer posteriors for the OpenBroadcast-SI DEV noisy condition (id2-dev), and the C2 gap-0.6 online windows rebuilt
on them. Dev only; the test benchmark is never read.

Recipe = the project's oracle posteriors (experiments/uniss_qwen3_omni_v1/data_clean/build_ami_gstar_si_rows.py D3/D4,
also build_ami_gstar_stream_rows.py, i.e. data/ami_multispk_si_v1/e2e_gstar_v1/post/oracle_*):
  D3  reference speaker activity on the 80 ms grid: every utterance of bench_dev/id2/<dir>/utts/<m>.jsonl sets
      act[speaker][int(begin / 0.08) : ceil(end / 0.08)] = 1
  D4  reference speaker -> slot by Hungarian matching of frame overlap with the meeting's Nemotron lowlat posteriors
      (> 0.5) on the whole session (build_ami_gstar_si_rows.slot_map, imported unchanged); the oracle posterior puts each
      speaker's 0/1 activity on its slot's channel. Same shape / dtype / keys as the Nemotron post.pt (T x 8, float16).

Steps (each refuses to overwrite its outputs; CPU only):
  post --dir D     data/open_voicecond_v1/bench_dev/id2/<D>/diar/oracle_lowlat/<m>.post.pt + slot_map.json + stats.json
  windows --dir D  bench_dev_build.windows UNCHANGED except the posterior dir (diar/oracle_lowlat instead of
                   diar/nemotron_lowlat) and the output dirs eval_outputs/mist_w3po_v1/data/
                   windows_obsidev_id2or_<D>_c2g06_v1/windows_test_online_<D>.jsonl + windows_obsidev_id2or_<D>_c2g06_chunk_v1/
                   (C2 gap 0.6 online windows, specclose chunk-protocol windows with judge off)

    python -m experiments.mist_w3po_v1.voicecond.s2_oracle_build post|windows --dir zh2en|en2zh
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import bench_build as B  # noqa: E402
from experiments.mist_w3po_v1.voicecond import bench_dev_build as D  # noqa: E402
from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import slot_map  # noqa: E402

COND = "id2"
DIAR_IN = "nemotron_lowlat"
DIAR_OR = "oracle_lowlat"
FRAME = 0.08


def win_dirs(d: str) -> tuple[Path, Path]:
    return D.DD / f"windows_obsidev_id2or_{d}_c2g06_v1", D.DD / f"windows_obsidev_id2or_{d}_c2g06_chunk_v1"


def post(d: str) -> None:
    C = D.DEV / COND / d
    out = C / "diar" / DIAR_OR
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    meetings = json.load(open(C / "meetings.json"))
    smap, st = {}, {}
    for m in sorted(meetings):
        src = torch.load(C / "diar" / DIAR_IN / f"{m}.post.pt", weights_only=False)
        p = src["post80"].float().numpy()
        T = len(p)
        act = defaultdict(lambda: np.zeros(T, dtype=bool))
        us = [json.loads(l) for l in open(C / "utts" / f"{m}.jsonl", encoding="utf-8")]
        for u in us:   # D3
            act[u["speaker_id"]][int(u["begin_s"] / FRAME): int(np.ceil(u["end_s"] / FRAME))] = True
        act = dict(act)
        mp = slot_map(act, p)   # D4
        assert len(set(mp.values())) == len(mp), (m, mp)   # K <= 4 speakers: one slot each
        o = np.zeros_like(p, dtype=np.float16)
        for s, k in mp.items():
            o[:, k - 1] = np.maximum(o[:, k - 1], act[s][:T])
        row = {k: v for k, v in src.items() if k != "post80"}
        row.update(post80=torch.from_numpy(o), model="oracle-reference-turns (S2; slot map vs Nemotron lowlat)",
                   source_diar=DIAR_IN)
        f = out / f"{m}.post.pt"
        with open(f, "xb") as fh:
            torch.save(row, fh)
        nem, ora = p > 0.5, o > 0.5
        smap[m] = mp
        st[m] = {"speakers": len(mp), "frames": T, "oracle_speech_frames": int(ora.any(1).sum()),
                 "nemotron_speech_frames": int(nem.any(1).sum()),
                 "speech_frame_agree": round(float((ora.any(1) == nem.any(1)).mean()), 4),
                 "slot_frame_agree_on_speech": round(float((ora[ora.any(1)] == nem[ora.any(1)]).all(1).mean()), 4)}
    with B.x_open(out / "slot_map.json") as fh:
        json.dump(smap, fh, indent=0)
    with B.x_open(out / "stats.json") as fh:
        json.dump(st, fh, indent=1)
    agg = {k: round(float(np.mean([v[k] for v in st.values()])), 4) for k in ("speech_frame_agree", "slot_frame_agree_on_speech")}
    print(f"{COND} {d}: {len(st)} oracle posteriors -> {out}; mean {json.dumps(agg)}", flush=True)


def windows(d: str) -> None:
    """bench_dev_build.windows with only the posterior dir and the output dirs changed."""
    from experiments.mist_w3po_v1.data.build_windows_zh2en import LOOKAHEAD, online_windows
    C = D.DEV / COND / d
    out, chunk = win_dirs(d)
    wf = out / f"windows_test_online_{d}.jsonl"
    if wf.exists() or chunk.exists() or out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {wf} / {chunk}")
    for sub in ("audio", "post"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meetings = json.load(open(C / "meetings.json"))
    n = 0
    with B.x_open(wf) as fh:
        for m in sorted(meetings):
            p = torch.load(C / "diar" / DIAR_OR / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
            x, sr = sf.read(str(D.sess_path(COND, d, m)), dtype="float32", always_2d=True)
            x = x.mean(1)
            sess_s = len(x) / sr
            for k, (st, en, j) in enumerate(online_windows(p, 0.6)):
                ws, we = max(0.0, st * FRAME - 0.2), min(sess_s, en * FRAME + 0.3)
                close = min(sess_s, j * FRAME + LOOKAHEAD) if j < len(p) else sess_s
                wid = f"{m}_on_{k:04d}"
                wav, pp = out / "audio" / f"{wid}.wav", out / "post" / f"{wid}.npy"
                if wav.exists() or pp.exists():
                    raise SystemExit(f"拒绝覆盖已有输出: {wav}")
                sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
                f0, nf = round(ws / FRAME), int(np.ceil((we - ws) / FRAME))
                s = p[f0: f0 + nf]
                if len(s) < nf:
                    s = np.concatenate([s, np.zeros((nf - len(s), p.shape[1]), dtype=s.dtype)])
                np.save(pp, s.astype(np.float16))
                fh.write(json.dumps({"id": wid, "meeting": m, "split": "dev", "wtype": "online", "dir": d,
                                     "start_s": round(ws, 3), "dur_ms": round((we - ws) * 1000),
                                     "audio": str(wav.resolve()), "post": {"lowlat": str(pp.resolve())},
                                     "open_detect_ms": round((st * FRAME + LOOKAHEAD - ws) * 1000),
                                     "close_detect_ms": round((close - ws) * 1000), "utts": [],
                                     "strict_review": False}) + "\n")
                n += 1
    print(f"{COND}-oracle {d}: {n} C2 windows -> {wf}", flush=True)
    from experiments.mist_w3po_v1.data import specclose as S

    def load_meeting(m: str):
        p = torch.load(C / "diar" / DIAR_OR / f"{m}.post.pt", weights_only=False)["post80"].float().numpy()
        x, sr = sf.read(str(D.sess_path(COND, d, m)), dtype="float32", always_2d=True)
        return p, x.mean(1), sr

    S.DATA, S.DIAR, S.MIX = C, DIAR_OR, D.DIRS[d][2]
    S.load_meeting = load_meeting
    S.cmd_build(argparse.Namespace(judge="off", base_win=str(wf), out=str(chunk)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["post", "windows"])
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    a = ap.parse_args()
    {"post": post, "windows": windows}[a.step](a.dir)


if __name__ == "__main__":
    main()
