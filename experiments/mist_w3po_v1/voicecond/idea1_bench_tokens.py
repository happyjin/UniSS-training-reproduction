"""Idea 1 M3 (reports/mist_w3po_v1/voicecond_idea1_prereg.zh-CN.md v4.1/v4.2): voice-token sidecars of the frozen
encoders Z (ZNE-e2e) and C (denoising control) for every talker item of OpenBroadcast-SI v1 baseline B runs
speech/obsi_b_<cond>_<dir>__t1c (prompt = the item's voice_audio in talker_in_all/voice_{causal,fallback}).

vocals  (conda_envs/uniss-eval python, CPU pool) HDemucs vocals (game_muffle_analysis.demucs) of every prompt ->
        data/open_voicecond_v1/idea1/bench_vocals/<cond>_<dir>/<id>.wav (existing files kept)
    python -m experiments.mist_w3po_v1.voicecond.idea1_bench_tokens vocals --procs 40
    python -m experiments.mist_w3po_v1.voicecond.idea1_bench_tokens vocals_gpu        (same recipe on cuda:0, one by one)
tokens  (venvs/vllm_omni_030 python, GPU) bh = y - vhat (zero-padded); Z: Q(2 phi_Z(y) - phi_Z(y + (sqrt2-1) bh)),
        C: Q(phi_C(y)); also records B = Q(phi_0(y)), V = Q(phi_0(vhat)) and the code errors against phi_0 of the clean
        twin's prompt with the same item id (only if its duration differs by <= 0.1 s) ->
        eval_outputs/mist_w3po_v1/voicecond_idea1/m3/sidecar_{z,c}_<cond>_<dir>.json ({id: [32 ids]}) and
        m3/codes_<cond>_<dir>.jsonl. Refuses existing outputs.
    python -m experiments.mist_w3po_v1.voicecond.idea1_bench_tokens tokens --zne <pt> --ctrl <pt> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
VOC = REPO / "data/open_voicecond_v1/idea1/bench_vocals"
M3 = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea1/m3"
CONDS = ("id2", "ood1", "clean")
DIRS = ("zh2en", "en2zh")
TWIN = {"id2": ("obz", "obe"), "ood1": ("ooz", "ooe"), "clean": ("ocz", "oce")}


def items(cond, d):
    rows = [json.loads(l) for l in open(SPD / f"obsi_b_{cond}_{d}__t1c/talker_in/items.jsonl", encoding="utf-8")]
    out = {}
    for r in rows:
        p = Path(r["voice_audio"])
        out[r["id"]] = p if p.is_absolute() else REPO / p
    return out


def sep(job):
    src, dst = job
    import soundfile as sf
    from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import demucs, read16
    if Path(dst).exists():
        return 0
    x = read16(src)
    _, v = demucs(x)
    tmp = Path(str(dst) + ".partial.wav")
    sf.write(str(tmp), v, 16000, subtype="FLOAT")
    tmp.rename(dst)
    return 1


def vocals(a) -> None:
    jobs = []
    for c in CONDS:
        for d in DIRS:
            od = VOC / f"{c}_{d}"
            od.mkdir(parents=True, exist_ok=True)
            jobs += [(str(p), str(od / f"{i}.wav")) for i, p in items(c, d).items()]
    with mp.get_context("fork").Pool(a.procs) as pool:
        n = sum(pool.map(sep, jobs, chunksize=4))
    print(json.dumps({"prompts": len(jobs), "separated_now": n}))


def vocals_gpu(a) -> None:
    """the same recipe as game_muffle_analysis.demucs (exact length, no padding), one prompt at a time on cuda:0;
    with --items / --out: the prompts of that items file into <out>/<id>.wav instead of the benchmark set"""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from torchaudio.models import hdemucs_high
    from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import CK, read16
    h = hdemucs_high(sources=["drums", "bass", "other", "vocals"])
    h.load_state_dict(torch.load(CK / "hdemucs_high_trained.pt", map_location="cpu"))
    h = h.to("cuda:0").eval()
    n = done = 0
    if getattr(a, "items", None):
        jobs = [(Path(a.out), {r["id"]: (Path(r["voice_audio"]) if Path(r["voice_audio"]).is_absolute() else REPO / r["voice_audio"])
                               for r in map(json.loads, open(a.items, encoding="utf-8"))})]
    else:
        jobs = [(VOC / f"{c}_{d}", items(c, d)) for c in CONDS for d in DIRS]
    for od, its in jobs:
            od.mkdir(parents=True, exist_ok=True)
            for iid, p in its.items():
                n += 1
                dst = od / f"{iid}.wav"
                if dst.exists():
                    continue
                x = torch.from_numpy(read16(p)).to("cuda:0")
                with torch.no_grad():
                    y = AF.resample(x, 16000, 44100)
                    y = torch.stack([y, y])[None]
                    ref = y.mean(1)
                    m, s = ref.mean(), ref.std() + 1e-8
                    st = h((y - m) / s)[0] * s + m
                    voc = AF.resample(st[3].mean(0), 44100, 16000)
                tmp = Path(str(dst) + ".partial.wav")
                sf.write(str(tmp), voc.cpu().numpy().astype(np.float32), 16000, subtype="FLOAT")
                tmp.rename(dst)
                done += 1
    print(json.dumps({"prompts": n, "separated_now": done}))


def tokens(a) -> None:
    sys.path.insert(0, str(REPO))
    import soundfile as sf
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    M3.mkdir(parents=True, exist_ok=True)
    P0 = BiCodecPhi(a.device)
    PZ = load_encoder(BiCodecPhi(a.device), a.zne)
    PC = load_encoder(BiCodecPhi(a.device), a.ctrl)
    sq = math.sqrt(2.0) - 1

    def rd(p):
        x, sr = sf.read(str(p), dtype="float64", always_2d=True)
        assert sr == 16000, p
        return x[:, 0]

    clean_u = {}
    summ = {}
    for c in ("clean", "id2", "ood1"):
        for d in DIRS:
            outs = {k: M3 / f"sidecar_{k}_{c}_{d}.json" for k in ("z", "c")}
            codes = M3 / f"codes_{c}_{d}.jsonl"
            for p in list(outs.values()) + [codes]:
                if p.exists():
                    raise SystemExit(f"拒绝覆盖已有输出: {p}")
            side = {"z": {}, "c": {}}
            rows = []
            for iid, p in items(c, d).items():
                y = rd(p)
                v = rd(VOC / f"{c}_{d}" / f"{iid}.wav")
                bh = y.copy()
                m = min(len(y), len(v))
                bh[:m] = y[:m] - v[:m]
                u0y, u0v = P0.phi_wave([y, v])
                z1, z2 = PZ.phi_wave([y, y + sq * bh])
                uc = PC.phi_wave(y)
                uz = 2 * z1 - z2
                side["z"][iid] = [int(t) for t in PZ.tokens(uz)]
                side["c"][iid] = [int(t) for t in PC.tokens(uc)]
                key = iid[3:]                                   # strip the condition prefix (obz / ooz / ocz ...)
                rec = {"id": iid, "dur": round(len(y) / 16000, 3),
                       "ham_z_vs_b": int((np.array(side["z"][iid]) != P0.tokens(u0y)).sum()),
                       "ham_c_vs_b": int((np.array(side["c"][iid]) != P0.tokens(u0y)).sum())}
                if c == "clean":
                    clean_u[(d, key)] = (u0y, rec["dur"])
                else:
                    t = clean_u.get((d, key))
                    if t is not None and abs(t[1] - rec["dur"]) <= 0.1:
                        tgt = t[0]
                        rec.update({f"err_{k}": float(np.linalg.norm(u - tgt))
                                    for k, u in (("b", u0y), ("v", u0v), ("z", uz), ("c", uc))})
                if c == "clean":
                    rec.update({"clean_shift_z": float(np.linalg.norm(uz - u0y)), "clean_shift_c": float(np.linalg.norm(uc - u0y))})
                rows.append(rec)
            for k, p in outs.items():
                with open(p, "x", encoding="utf-8") as fh:
                    json.dump(side[k], fh)
            with open(codes, "x", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            s = {"items": len(rows)}
            if c != "clean":
                mm = [r for r in rows if "err_b" in r]
                s["matched"] = len(mm)
                s.update({f"err_{k}_median": float(np.median([r[f"err_{k}"] for r in mm])) if mm else None
                          for k in ("b", "v", "z", "c")})
            else:
                s.update({k: float(np.median([r[k] for r in rows])) for k in ("clean_shift_z", "clean_shift_c")})
            summ[f"{c}_{d}"] = s
            print(json.dumps({f"{c}_{d}": s}), flush=True)
    (M3 / "codes_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vocals")
    v.add_argument("--procs", type=int, default=40)
    g = sub.add_parser("vocals_gpu")
    g.add_argument("--items", default="")
    g.add_argument("--out", default="")
    t = sub.add_parser("tokens")
    t.add_argument("--zne", required=True)
    t.add_argument("--ctrl", required=True)
    t.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"vocals": vocals, "vocals_gpu": vocals_gpu, "tokens": tokens}[a.cmd](a)


if __name__ == "__main__":
    main()
