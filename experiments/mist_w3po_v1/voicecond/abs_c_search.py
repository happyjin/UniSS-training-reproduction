"""Robust B', AbS-C search (reports/mist_w3po_v1/robust_abs_c_prereg.zh-CN.md section 2; training-free).
Identical to b_abs_search.py (start s0 = V1 table tokens, FSQ dim queue by boundary margin, 4 rounds x 16 candidate
evaluations, 15 single flips + 1 combined / fill-in candidate, utility U(s) = cos(TitaNet(R(s)), TitaNet(noisy prompt)),
R(s) = BiCodec detokenize [global tokens s, the prompt's own semantic tokens]) plus ONE admissibility condition:
candidate c is admissible iff SQUIM-objective STOI(R(c)) >= STOI(R(s0)) (same R batch as U; s0 always admissible).
The "improving" set (for the combined candidate) and the per-round acceptance only use admissible candidates.
SQUIM PESQ is recorded only. Not timed / not part of the search: STOI / PESQ of the AbS tokens' resynthesis
(b_abs/tokens/tokens_abs_<d>.jsonl) for every row. ECAPA2 is NOT used here.
Runs in venvs/nemo_diar with PYTHONPATH=.:.tmp/b_abs_pyshim (nothing installed). Per (direction, shard k of rows mod n):
  <out>/search_<d>_<k>.jsonl  {id, key, sha1, ids (AbS-C), ids_v1, ids_abs, u_start, u_best, n_accept, flips, n_eval,
                               n_rejected, stoi_v1, pesq_v1, stoi_best, pesq_best, stoi_abs, pesq_abs, t_tok, t_v1,
                               t_search, (noisy rows) dcos_absc / dcos_v1 vs oracle tokens, ham_absc_oracle, ham_absc_abs}
  <out>/resynth/<d>/<key>__{v1,absc}.wav  (noisy rows; noisy semantic tokens + those global tokens; float wav)
    PYTHONPATH=.:.tmp/b_abs_pyshim venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.abs_c_search \
        --dirs en2zh zh2en --shard 0 --num-shards 4 --out <dir> [--device cuda:0] [--limit N]
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
from experiments.mist_w3po_v1.voicecond import a_sv_common as C  # noqa: E402

BUDGET_ROUNDS, PER_ROUND = 4, 16
BASIS = 4 ** np.arange(6)
ABS_TOK = REPO / "eval_outputs/mist_w3po_v1/b_abs/tokens"


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
    a = ap.parse_args()
    out = Path(a.out)
    for d in a.dirs:
        if (out / f"search_{d}_{a.shard}.jsonl").exists():
            sys.exit(f"拒绝覆盖已有输出: {out / f'search_{d}_{a.shard}.jsonl'}")
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from torchaudio.models import squim_objective_base
    from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import CK
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
    sq = squim_objective_base()          # same weights / loader as the SQUIM PESQ metric (game_muffle_analysis.models)
    sq.load_state_dict(torch.load(CK / "squim_objective_dns2020.pth", map_location="cpu"))
    sq = sq.to(dev).eval()
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
    def qual(x: torch.Tensor):                          # (B, T) -> STOI (B,), PESQ (B,) numpy (SQUIM objective)
        st, pq, _ = sq(x[:, : C.SR * 20].float())
        return st.float().cpu().numpy(), pq.float().cpu().numpy()

    @torch.no_grad()
    def resynth(G: np.ndarray, sem: torch.Tensor) -> torch.Tensor:   # (B, 32) ids -> (B, T) float32
        g = torch.as_tensor(G, dtype=torch.long, device=dev)[:, None, :]
        w = P0.model.detokenize(sem.expand(g.shape[0], -1), g)
        return w.reshape(g.shape[0], -1).float()

    @torch.no_grad()
    def dvec(ids) -> np.ndarray:
        t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=dev).reshape(1, 1, 32)
        return P0.se.detokenize(t).float().cpu().numpy().astype(np.float64).reshape(-1)

    def dcos(x, y):
        dx, dy = dvec(x), dvec(y)
        return float(dx @ dy / (np.linalg.norm(dx) * np.linalg.norm(dy) + 1e-9))

    for d in a.dirs:
        rows = C.dev_rows(d)
        orc = {r["id"]: r["ids"] for r in map(json.loads, open(C.ORT / f"tokens_oracle_{d}.jsonl", encoding="utf-8"))}
        abst = {r["id"]: r["ids"] for r in map(json.loads, open(ABS_TOK / f"tokens_abs_{d}.jsonl", encoding="utf-8"))}
        mine = [r for n, r in enumerate(rows) if n % a.num_shards == a.shard]
        if a.limit:
            mine = mine[: a.limit]
        rdir = out / "resynth" / d
        rdir.mkdir(parents=True, exist_ok=True)
        t_all = time.time()
        with open(out / f"search_{d}_{a.shard}.jsonl", "x", encoding="utf-8") as fh:
            for n, r in enumerate(mine):
                p = C.dev_prompt(d, r["key"])
                assert hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
                noisy = "_noisy__" in r["id"]
                sync(); t0 = time.perf_counter()
                with torch.no_grad():
                    gB, sem = P0.tok.tokenize(str(p))
                sync(); t1 = time.perf_counter()
                sem = sem.reshape(1, -1).long().to(dev)
                b = Pv.bounded(Pv.phi_wave(read_mono64(p)))                     # (32, 6)
                sync(); t2 = time.perf_counter()
                # ---- added latency starts here: TitaNet reference + 4 rounds x 16 evaluations (+ SQUIM)
                x, sr = sf.read(str(p), dtype="float32", always_2d=True)
                assert sr == C.SR, p
                ref = tn_emb(torch.from_numpy(x.mean(1))[None].to(dev))[0]
                s0 = np.asarray(r["ids"], dtype=np.int64)
                L0 = ids_to_levels(s0)
                lf, bf = L0.reshape(-1), b.reshape(-1)
                alt = np.where(bf > lf, lf + 1, lf - 1)
                alt = np.where(alt > 1, lf - 1, np.where(alt < -2, lf + 1, alt))
                marg = 0.5 - np.abs(bf - lf)
                Q = list(np.lexsort((np.arange(192), marg)))                 # ascending margin, ties by index
                cur = lf.copy()
                W0 = resynth(levels_to_ids(cur.reshape(32, 6))[None], sem)
                ucur = float(tn_emb(W0)[0] @ ref)
                st0, pq0 = qual(W0)
                tau = float(st0[0])                                           # STOI of V1's resynthesis
                stcur, pqcur = tau, float(pq0[0])
                u_start, flips, n_acc, n_eval, n_rej = ucur, [], 0, 0, 0
                for _ in range(BUDGET_ROUNDS):
                    singles = [Q.pop(0) for _ in range(PER_ROUND - 1)]
                    cands = []
                    for i in singles:
                        c = cur.copy(); c[i] = alt[i]; cands.append(c)
                    G = np.stack([levels_to_ids(c.reshape(32, 6)) for c in cands])
                    W = resynth(G, sem)
                    U = (tn_emb(W) @ ref).cpu().numpy()
                    ST, PQ = qual(W)
                    adm = ST >= tau
                    imp = [i for i, uu, ok in zip(singles, U, adm) if ok and uu > ucur]
                    if len(imp) >= 2:
                        c = cur.copy(); c[imp] = alt[imp]; tag = ("comb", imp)
                    else:
                        j = Q.pop(0); c = cur.copy(); c[j] = alt[j]; tag = ("single", [j])
                    W16 = resynth(levels_to_ids(c.reshape(32, 6))[None], sem)
                    u16 = float(tn_emb(W16)[0] @ ref)
                    st16, pq16 = qual(W16)
                    n_eval += PER_ROUND
                    allU = list(U) + [u16]
                    allA = list(adm) + [bool(st16[0] >= tau)]
                    allS = list(ST) + [float(st16[0])]
                    allP = list(PQ) + [float(pq16[0])]
                    allF = [[i] for i in singles] + [tag[1]]
                    n_rej += int(sum(not ok for ok in allA))
                    masked = [uu if ok else -np.inf for uu, ok in zip(allU, allA)]
                    k = int(np.argmax(masked))
                    if masked[k] > ucur:
                        for i in allF[k]:
                            cur[i] = alt[i]
                        flips += [int(i) for i in allF[k]]
                        ucur = float(allU[k]); n_acc += 1
                        stcur, pqcur = float(allS[k]), float(allP[k])
                sync(); t3 = time.perf_counter()
                ids_c = levels_to_ids(cur.reshape(32, 6))
                ids_a = np.asarray(abst[r["id"]], dtype=np.int64)
                sta, pqa = qual(resynth(ids_a[None], sem))
                rec = {"id": r["id"], "key": r["key"], "sha1": r["sha1"], "ids": [int(v) for v in ids_c],
                       "ids_v1": [int(v) for v in s0], "ids_abs": [int(v) for v in ids_a], "u_start": u_start,
                       "u_best": ucur, "n_accept": n_acc, "flips": flips, "n_eval": n_eval, "n_rejected": n_rej,
                       "stoi_v1": tau, "pesq_v1": float(pq0[0]), "stoi_best": stcur, "pesq_best": pqcur,
                       "stoi_abs": float(sta[0]), "pesq_abs": float(pqa[0]), "ham_absc_abs": int((ids_c != ids_a).sum()),
                       "t_tok": t1 - t0, "t_v1": t2 - t1, "t_search": t3 - t2, "n_sem": int(sem.shape[1])}
                if noisy:
                    so = np.asarray(orc[r["id"]], dtype=np.int64)
                    rec.update(dcos_absc=dcos(ids_c, so), dcos_v1=dcos(s0, so), ham_absc_oracle=int((ids_c != so).sum()))
                    Wn = resynth(np.stack([s0, ids_c]), sem).cpu().numpy()
                    for tag_, w in zip(("v1", "absc"), Wn):
                        with open(rdir / f"{r['key']}__{tag_}.wav", "xb") as wf:
                            sf.write(wf, w, C.SR, format="WAV", subtype="FLOAT")
                fh.write(json.dumps(rec) + "\n")
                if (n + 1) % 50 == 0:
                    fh.flush()
                    print(f"[absc] {d} shard {a.shard} {n + 1}/{len(mine)} {time.time() - t_all:.0f}s", flush=True)
        print(f"[absc] {d} shard {a.shard} done {len(mine)} rows {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
