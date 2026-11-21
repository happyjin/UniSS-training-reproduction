"""voice_onset_v1 P3: posterior-MBR AbS search (reports/voice_onset_v1/P1_P3_prereg.zh-CN.md section 4). Identical to
test_abs_search.py (budget, FSQ dim queue, flips, start = the V1 tokens of the root's prompt) except the utility target:
U(s) = cos(TitaNet(R(s)), m) with m = the trained Onset posterior mean (targets_<d>.jsonl) instead of TitaNet(prompt).
    PYTHONPATH=.:.tmp/b_abs_pyshim venvs/nemo_diar/bin/python -m experiments.voice_onset_v1.p3_search --tokens-root <root>/tokens \
        --targets <root> --dirs en2zh zh2en --shard 0 --num-shards 4 --out <dir> [--device cuda:0]
Refuses existing output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import a_sv_common as C  # noqa: E402,F401

BUDGET_ROUNDS, PER_ROUND = 4, 16
KEEP = {"rsc", "rsn", "obclean", "obid2", "obood1"}
BASIS = 4 ** np.arange(6)


def ids_to_levels(ids) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    return ((ids[:, None] // BASIS[None]) % 4) - 2          # (32, 6) in {-2..1}


def levels_to_ids(L) -> np.ndarray:
    return ((np.asarray(L, dtype=np.int64) + 2) * BASIS).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=list(C.DIRS))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows of the shard per direction")
    ap.add_argument("--tokens-root", required=True)
    ap.add_argument("--targets", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    TT = Path(a.tokens_root)
    for d in a.dirs:
        if (out / f"search_{d}_{a.shard}.jsonl").exists():
            sys.exit(f"拒绝覆盖已有输出: {out / f'search_{d}_{a.shard}.jsonl'}")
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from experiments.mist_w3po_v1.voicecond.bicodec_phi import BiCodecPhi
    from experiments.mist_w3po_v1.voicecond.idea1_spksim import NEMO
    from experiments.mist_w3po_v1.voicecond.idea1_tokens import load_encoder
    from experiments.mist_w3po_v1.voicecond.idea1_zne import read_mono64
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    sync = (lambda: torch.cuda.synchronize(dev)) if cuda else (lambda: None)
    P0 = BiCodecPhi(a.device)
    Pv = load_encoder(BiCodecPhi(a.device), str(C.V1))
    P0.model.eval(); Pv.model.eval()
    tn = EncDecSpeakerLabelModel.restore_from(NEMO, map_location=a.device).eval()
    # sanity: level <-> id mapping equals the FSQ layer
    rng = np.random.default_rng(0)
    u = rng.normal(size=(4, 32, 6)).astype(np.float32)
    for k in range(4):
        L = np.round(P0.bounded(u[k])).astype(np.int64)
        assert (levels_to_ids(L) == P0.tokens(u[k])).all(), "level/id mapping mismatch"

    @torch.no_grad()
    def tn_emb(x: torch.Tensor) -> torch.Tensor:        # (B, T) float32 on device -> (B, 192) L2-normalised
        _, e = tn.forward(input_signal=x, input_signal_length=torch.full((x.shape[0],), x.shape[1], device=dev))
        e = e.float()
        return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    @torch.no_grad()
    def resynth(G: np.ndarray, sem: torch.Tensor) -> torch.Tensor:   # (B, 32) ids -> (B, T) float32
        g = torch.as_tensor(G, dtype=torch.long, device=dev)[:, None, :]
        w = P0.model.detokenize(sem.expand(g.shape[0], -1), g)
        return w.reshape(g.shape[0], -1).float()

    @torch.no_grad()
    def dvec(ids) -> np.ndarray:     # (1, 1, 32) as BiCodecTokenizer.detokenize / d1ts_tokens
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=dev).reshape(1, 1, 32)
        return P0.se.detokenize(t).float().cpu().numpy().astype(np.float64).reshape(-1)

    def dcos(x, y):
        dx, dy = dvec(x), dvec(y)
        return float(dx @ dy / (np.linalg.norm(dx) * np.linalg.norm(dy) + 1e-9))

    for d in a.dirs:
        rows = [json.loads(l) for l in open(TT / f"tokens_v1_{d}.jsonl", encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r["id"].split("__")[0] in KEEP]
        tgt = {r["key"]: r["m"] for r in map(json.loads, open(Path(a.targets) / f"targets_{d}.jsonl", encoding="utf-8"))}
        rows = [r for r in rows if r["key"] in tgt]
        mine = [r for n, r in enumerate(rows) if n % a.num_shards == a.shard]
        if a.limit:
            mine = mine[: a.limit]
        out.mkdir(parents=True, exist_ok=True)
        t_all = time.time()
        with open(out / f"search_{d}_{a.shard}.jsonl", "x", encoding="utf-8") as fh:
            for n, r in enumerate(mine):
                p = TT / "prompts" / d / f"{r['key']}.wav"
                assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
                noisy = False   # no oracle on the test sets
                # (B and V1 compute these anyway; timed separately, not part of the added latency)
                sync(); t0 = time.perf_counter()
                with torch.no_grad():
                    gB, sem = P0.tok.tokenize(str(p))
                sync(); t1 = time.perf_counter()
                ids_B = gB.reshape(-1)[:32].long().cpu().numpy()
                sem = sem.reshape(1, -1).long().to(dev)
                b = Pv.bounded(Pv.phi_wave(read_mono64(p)))                     # (32, 6)
                sync(); t2 = time.perf_counter()
                # ---- added latency starts here: TitaNet reference + 4 rounds x 16 evaluations
                x, sr = sf.read(str(p), dtype="float32", always_2d=True)
                assert sr == C.SR, p
                ref = torch.nn.functional.normalize(torch.tensor(tgt[r["key"]], dtype=torch.float32, device=dev), dim=0)
                s0 = np.asarray(r["ids"], dtype=np.int64)
                L0 = ids_to_levels(s0)
                lf, bf = L0.reshape(-1), b.reshape(-1)
                alt = np.where(bf > lf, lf + 1, lf - 1)
                alt = np.where(alt > 1, lf - 1, np.where(alt < -2, lf + 1, alt))
                marg = 0.5 - np.abs(bf - lf)
                Q = list(np.lexsort((np.arange(192), marg)))                 # ascending margin, ties by index
                cur = lf.copy()
                ucur = float(tn_emb(resynth(levels_to_ids(cur.reshape(32, 6))[None], sem))[0] @ ref)
                u_start, flips, n_acc, n_eval = ucur, [], 0, 0
                for _ in range(BUDGET_ROUNDS):
                    singles = [Q.pop(0) for _ in range(PER_ROUND - 1)]
                    cands = []
                    for i in singles:
                        c = cur.copy(); c[i] = alt[i]; cands.append(c)
                    G = np.stack([levels_to_ids(c.reshape(32, 6)) for c in cands])
                    U = (tn_emb(resynth(G, sem)) @ ref).cpu().numpy()
                    imp = [i for i, uu in zip(singles, U) if uu > ucur]
                    if len(imp) >= 2:
                        c = cur.copy(); c[imp] = alt[imp]; tag = ("comb", imp)
                    else:
                        j = Q.pop(0); c = cur.copy(); c[j] = alt[j]; tag = ("single", [j])
                    u16 = float(tn_emb(resynth(levels_to_ids(c.reshape(32, 6))[None], sem))[0] @ ref)
                    n_eval += PER_ROUND
                    allU = list(U) + [u16]
                    allF = [[i] for i in singles] + [tag[1]]
                    k = int(np.argmax(allU))
                    if allU[k] > ucur:
                        for i in allF[k]:
                            cur[i] = alt[i]
                        flips += [int(i) for i in allF[k]]
                        ucur = float(allU[k]); n_acc += 1
                sync(); t3 = time.perf_counter()
                ids_abs = levels_to_ids(cur.reshape(32, 6))
                rec = {"id": r["id"], "key": r["key"], "sha1": r["sha1"], "ids": [int(v) for v in ids_abs],
                       "ids_v1": [int(v) for v in s0], "ids_B": [int(v) for v in ids_B], "u_start": u_start,
                       "u_best": ucur, "n_accept": n_acc, "flips": flips, "n_eval": n_eval,
                       "t_tok": t1 - t0, "t_v1": t2 - t1, "t_search": t3 - t2, "n_sem": int(sem.shape[1])}
                if noisy:
                    so = np.asarray(orc[r["id"]], dtype=np.int64)
                    rec.update(dcos_abs=dcos(ids_abs, so), dcos_v1=dcos(s0, so), dcos_B=dcos(ids_B, so),
                               ham_abs=int((ids_abs != so).sum()), ham_v1=int((s0 != so).sum()))
                    G = np.stack([s0, ids_abs, so, ids_B])
                    W = resynth(G, sem).cpu().numpy()
                    for tag_, w in zip(("v1", "abs", "oracle", "B"), W):
                        with open(rdir / f"{r['key']}__{tag_}.wav", "xb") as wf:
                            sf.write(wf, w, C.SR, format="WAV", subtype="FLOAT")
                fh.write(json.dumps(rec) + "\n")
                if (n + 1) % 50 == 0:
                    fh.flush()
                    print(f"[p3_search] {d} shard {a.shard} {n + 1}/{len(mine)} {time.time() - t_all:.0f}s", flush=True)
        print(f"[p3_search] {d} shard {a.shard} done {len(mine)} rows {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
