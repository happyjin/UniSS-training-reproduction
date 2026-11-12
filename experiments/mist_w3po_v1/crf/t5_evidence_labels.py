"""T5: evidence-time labels (pre-registration reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T5).

SimulS2ST-Omni-style trajectories (arXiv 2607.19810 §3.2) built the way the single-speaker T1c data was built
(experiments/uniss_qwen3_omni_v1/data_clean/omni_align.py, timing_rules.py, simul_segments.py): source unit times from a
forced aligner, OmniAlign word links, support = latest end of the linked source units, backward fill, tau = running max
of support + delta (200 ms), no 1 s lag.

  --align   (venvs/align_omni_v1, CPU fp32, multiprocessing) per utterance: the CJK characters of its zh text with the
            FunASR times of data/ami_multispk_si_v1/zh_fa_v6 (meeting clock: aligned on the mix, so trim / time_stretch
            / zh_begin are already applied), jieba words (omni_align.source_units), the English words of its ref
            (realsi_align.EN_WORD), OmniAlign links (layer 7, both softmaxes > 1e-3, word strength = max over subword
            pairs, as realsi_align.cmd_links), NIR (build_si_train_index.nir); for dev also the mBERT layer-8 mutual
            argmax links (reports/gstar_si_v6/deadline_audit.Aligner) for the audit.
  --build   (qwen3omni env) the training sched directory: per train window (windows_train_ref), chunks (punctuation
            chunks, < 3 words merged forward unless sentence-final, the utterance's last chunk kept), support / tau /
            label read e, aligned contiguous source spans for the time tokens, S_prop deadline u (letter-proportional
            end + 1 s, running max per speaker in kap order); windows with an utterance of NIR > 0.3 dropped; records
            by build_piece_sched.sched_record (bounds free), checked by build_piece_sched.check_record and the
            trainer's set_reach; summary.json for the trainer. The lattice mode is latent_write_crf "t5"
            (h = min(R - 1, max(u, e))).
  --audit   the dev5 label audit (GO lines of the pre-registration) on B-2's dev5 targets (same windows, members,
            reads, online clamps, unit LAAL code).

    <align_omni_v1 python> -m experiments.mist_w3po_v1.crf.t5_evidence_labels --align --split train --out <dir>/train
    <align_omni_v1 python> -m experiments.mist_w3po_v1.crf.t5_evidence_labels --align --split dev5 --out <dir>/dev5
    python -m experiments.mist_w3po_v1.crf.t5_evidence_labels --audit --align-dir <dir>/dev5 --out <new dir>
    python -m experiments.mist_w3po_v1.crf.t5_evidence_labels --build --align-dir <dir>/train --grid 500 --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path("data/ami_multispk_si_v1")
MIX_UTTS = DATA / "zh_mix_v8" / "utts"
FA = DATA / "zh_fa_v6"
TRAIN_WIN = DATA / "e2e_gstar_zh2en_v1" / "windows_train_ref.jsonl"
DEV5_TARGETS = Path("eval_outputs/mist_w3po_v1/crf/b2_earliest_write_dev5/cpu/targets.jsonl")
DEV5_WIN = Path("eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl")
OMNI = "/opt/dlami/nvme/jasonleeeli/tools/OmniAlign/OmniAlign"
MBERT = "/opt/dlami/nvme/jasonleeeli/cache/hf_v6/models--google-bert--bert-base-multilingual-cased"
CJK = re.compile(r"[一-鿿]")
EN_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
DELTA_MS = 200
MIN_WORDS = 3
SENT_END = tuple(".?!。？！")
NIR_MAX = 0.3
LAYER, THR = 7, 1e-3
LAG_MS = 1000
SEED = 20261111


# ------------------------------------------------------------------------------------------------ align (CPU)
def nir(links: list) -> float:
    """build_si_train_index.nir: inversions of the source indices of the links sorted by (target, source) / pairs."""
    seq = [s for _, s in sorted((t, s) for t, s, *_ in links)]
    n = len(seq)
    if n < 2:
        return 0.0
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])
    return inv / (n * (n - 1) / 2)


def utt_units(zh: str, fa_chars: list) -> tuple[list[str], list[list[float]], str]:
    """(jieba words, [start_s, end_s] per word on the meeting clock, status)."""
    import jieba

    chars = CJK.findall(zh or "")
    if not chars or not fa_chars:
        return [], [], "no_chars"
    if len(chars) != len(fa_chars):
        return [], [], "mismatch"
    words, times, pos = [], [], 0
    for w in jieba.cut("".join(chars)):
        a, b = pos, pos + len(w) - 1
        words.append(w)
        times.append([float(fa_chars[a][0]), float(fa_chars[b][1])])
        pos += len(w)
    return words, times, "ok"


_M = {}


def _init_worker(dev: bool, threads: int):
    import jieba
    import torch
    import transformers

    torch.set_num_threads(threads)
    jieba.setLogLevel(60)
    _M["tok"] = transformers.AutoTokenizer.from_pretrained(OMNI, trust_remote_code=True)
    _M["model"] = transformers.AutoModel.from_pretrained(OMNI, trust_remote_code=True, torch_dtype=torch.float32).eval()
    if dev:
        import sys

        sys.path.insert(0, "reports/gstar_si_v6")
        import deadline_audit as DAU

        DAU.MBERT = MBERT
        _M["mbert"] = DAU.Aligner()


def _align_job(items: list[dict]) -> list[dict]:
    import torch

    from experiments.uniss_qwen3_omni_v1.data_clean.omni_align import encode, hidden

    tok, model = _M["tok"], _M["model"]
    out = []
    for it in items:
        words, times, st = utt_units(it["zh"], it["fa"])
        en = EN_WORD.findall(it["ref"])
        rec = {"record_id": it["record_id"], "meeting": it["meeting"], "zh_begin": it["zh_begin"],
               "zh_end": it["zh_end"], "status": st, "units": words, "unit_times": times, "en_words": en, "links": []}
        if st == "ok" and en:
            (sid, sown), (tid, town) = encode(tok, words), encode(tok, en)
            a, b = hidden(model, tok, [sid], LAYER, torch.device("cpu"))[0], \
                hidden(model, tok, [tid], LAYER, torch.device("cpu"))[0]
            sim = a @ b.T
            s1, s2 = torch.softmax(sim, dim=-1), torch.softmax(sim, dim=-2)
            strength = torch.minimum(s1, s2)
            best = {}
            for i, j in torch.nonzero((s1 > THR) & (s2 > THR), as_tuple=False).tolist():
                if i < len(sown) and j < len(town):
                    k = (town[j], sown[i])
                    best[k] = max(best.get(k, 0.0), float(strength[i, j]))
            rec["links"] = [[t, u, round(c, 4)] for (t, u), c in sorted(best.items())]
            if not rec["links"]:
                rec["status"] = "no_links"
            if "mbert" in _M:
                rec["mbert_links"] = [[j, i] for i, j in _M["mbert"].align(words, en)]
        rec["nir"] = nir(rec["links"])
        out.append(rec)
    return out


def load_mix(meetings) -> tuple[dict, dict]:
    mix, fa = {}, {}
    for m in meetings:
        for l in open(MIX_UTTS / f"{m}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r.get("placed"):
                mix[r["record_id"]] = r
        p = FA / f"{m}.jsonl"
        if p.exists():
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                fa[r["record_id"]] = r["chars"]
    return mix, fa


def utterances(split: str, targets: str = "") -> list[dict]:
    """[{record_id, meeting, ref}] (unique record ids) of the train windows or of B-2's dev5 targets."""
    seen, out = set(), []
    if split == "train":
        src = (json.loads(l) for l in open(TRAIN_WIN, encoding="utf-8"))
        for w in src:
            assert w["split"] == "train"
            for u in w["utts"]:
                if u["record_id"] not in seen and not u["hesitation"] and u["ref"].strip():
                    seen.add(u["record_id"])
                    out.append({"record_id": u["record_id"], "meeting": w["meeting"], "ref": u["ref"]})
    else:
        for l in open(targets or DEV5_TARGETS, encoding="utf-8"):
            rec = json.loads(l)
            for u in rec["members"]:
                if u["record_id"] not in seen and not u["hesitation"] and u["ref"].strip():
                    seen.add(u["record_id"])
                    out.append({"record_id": u["record_id"], "meeting": rec["meeting"], "ref": u["ref"]})
    return out


def run_align(a) -> None:
    import multiprocessing as mp

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    U = utterances(a.split, a.targets)
    if a.limit:
        U = U[: a.limit]
    mix, fa = load_mix(sorted({u["meeting"] for u in U}))
    items = []
    for u in U:
        mr = mix[u["record_id"]]
        assert mr.get("placed"), u["record_id"]
        items.append({**u, "zh": mr["zh"], "fa": fa.get(u["record_id"]), "zh_begin": mr["zh_begin"],
                      "zh_end": mr["zh_end"]})
    out.mkdir(parents=True)
    jobs = [items[i: i + 64] for i in range(0, len(items), 64)]
    st, nirs = Counter(), []
    with open(out / "align.jsonl", "x", encoding="utf-8") as fh, \
            mp.get_context("spawn").Pool(a.workers, _init_worker, (a.split != "train", a.threads)) as pool:
        for k, res in enumerate(pool.imap(_align_job, jobs)):
            for r in res:
                st[r["status"]] += 1
                if r["status"] == "ok":
                    nirs.append(r["nir"])
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if k % 100 == 0:
                print(f"{k + 1}/{len(jobs)} batches {time.time() - t0:.0f}s", flush=True)
    x = np.asarray(nirs)
    summ = {"split": a.split, "utterances": len(items), "status": dict(st),
            "nir": {"mean": float(x.mean()), "p50": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                    "gt_0.3": int((x > NIR_MAX).sum())} if len(x) else {},
            "elapsed_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ), flush=True)
    print(f"T5_ALIGN_DONE {out}", flush=True)


# ------------------------------------------------------------------------------------------------ labels
def merge_chunks(ref: str) -> list[str]:
    """punctuation chunks (build_ami_ms_rows.chunks_of), whitespace-normalised as B-2's pieces; a chunk of < 3 words not
    ending a sentence is merged into the next one; the utterance's last chunk stays."""
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import chunks_of

    cs = [" ".join(c.strip().split()) for c in chunks_of(ref)]
    cs = [c for c in cs if c]
    out, cur = [], ""
    for i, c in enumerate(cs):
        cur = f"{cur} {c}".strip() if cur else c
        if i == len(cs) - 1 or len(cur.split()) >= MIN_WORDS or cur.endswith(SENT_END):
            out.append(cur)
            cur = ""
    assert " ".join(out).split() == " ".join(cs).split()
    return out


def pieces4_chunks(ref: str) -> list[str]:
    """--granularity pieces4 (NOT the registered setting; only if the pre-registration is amended): B-2's pieces, the
    punctuation chunks cut greedily into <= 4-word groups (precheck_b2_earliest_write.utt_pieces' texts)."""
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import chunks_of

    out = []
    for c in chunks_of(ref):
        w = c.strip().split()
        out += [" ".join(w[i: i + 4]) for i in range(0, len(w), 4)]
    return [x for x in out if x]


def split_chunks(ref: str, gran: str) -> list[str]:
    return merge_chunks(ref) if gran == "merge3" else pieces4_chunks(ref)


def coalesce_same_read(x: tuple, tau_tok: dict, t_ms: list[int]) -> tuple:
    """--granularity pieces4rc (reports/mist_w3po_v1/thinker_rc5_prereg_v1.zh-CN.md §1): one utterance's pieces4 pieces
    with the same label read e (the first read at or after the tau of the piece's last word) that are adjacent become
    one piece (texts joined by a space, spans concatenated); a piece without an EN_WORD token joins the next piece (at
    the utterance's end: the previous one). The label read of every word is unchanged by construction (the merged
    piece's last word is a merged piece's last word with the same e). An utterance without any EN_WORD token is kept."""
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B

    mi, u, cs, spans, sup, raw, mb = x
    keys = [B.first_read(t_ms, tau_tok[(mi, sp[-1])]) if sp else None for sp in spans]
    if all(k is None for k in keys):
        return x
    units, pend = [], []
    for i, k in enumerate(keys):
        pend.append(i)
        if k is not None:
            units.append([pend, k])
            pend = []
    if pend:
        units[-1][0] += pend
    merged = []
    for idx, k in units:
        if merged and merged[-1][1] == k:
            merged[-1][0] += idx
        else:
            merged.append([list(idx), k])
    cs2 = [" ".join(cs[i] for i in idx) for idx, _ in merged]
    sp2 = [[j for i in idx for j in spans[i]] for idx, _ in merged]
    return mi, u, cs2, sp2, sup, raw, mb


def word_support(al: dict | None, end_ms_abs: float) -> tuple[list[float], list[float | None]]:
    """(filled support per EN_WORD token, raw OmniAlign evidence per token or None) in absolute seconds; no alignment ->
    every token = the utterance end (zh_end)."""
    if al is None or al["status"] != "ok":
        n = len(al["en_words"]) if al else 0
        return [end_ms_abs] * n, [None] * n
    ends = [t[1] for t in al["unit_times"]]
    raw = [None] * len(al["en_words"])
    for t, u, _c in al["links"]:
        raw[t] = max(raw[t] if raw[t] is not None else -1e9, ends[u])
    sup, nxt = [0.0] * len(raw), end_ms_abs
    for j in range(len(raw) - 1, -1, -1):
        sup[j] = raw[j] if raw[j] is not None else nxt
        nxt = sup[j]
    return sup, raw


def mbert_evidence(al: dict | None) -> list[float | None]:
    if al is None or al["status"] not in ("ok", "no_links") or "mbert_links" not in al:
        return [None] * (len(al["en_words"]) if al else 0)
    ends = [t[1] for t in al["unit_times"]]
    ev = [None] * len(al["en_words"])
    for t, u in al["mbert_links"]:
        ev[t] = max(ev[t] if ev[t] is not None else -1e9, ends[u])
    return ev


def window_chunks(members: list[dict], start_s: float, dur_ms: int, t_ms: list[int], slot_of: dict, align: dict,
                  gran: str = "merge3") -> list[dict]:
    """T5 chunks of one window (members: record_id, speaker_id, begin_s, end_s (window-relative), ref, hesitation):
    per chunk text / n_words / utt (member index) / k / slot / src_start_ms / src_end_ms (aligned, contiguous) /
    tau_ms / e (label read) / prop_end_ms / u (S_prop read, before the per-speaker running max) / evidence_ms (OmniAlign,
    raw) / mbert_ms (dev)."""
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import zh_len

    per_utt = []
    for mi, u in enumerate(members):
        if u.get("hesitation") or not u["ref"].strip():
            continue
        al = align.get(u["record_id"])
        zh_end_abs = al["zh_end"] if al else start_s + u["end_s"]
        sup, raw = word_support(al, zh_end_abs)
        mb = mbert_evidence(al)
        toks = EN_WORD.findall(u["ref"])
        if al is not None:
            assert al["en_words"] == toks, u["record_id"]
        cs = split_chunks(u["ref"], gran)
        spans, k = [], 0
        for c in cs:
            n = sum(len(EN_WORD.findall(w)) for w in c.split())
            spans.append(list(range(k, k + n)))
            k += n
        assert k == len(toks), u["record_id"]
        per_utt.append((mi, u, cs, spans, sup, raw, mb))
    # tau: running max per speaker, utterances in time order, tokens in order
    tau_tok = {}
    by_spk = defaultdict(list)
    for x in per_utt:
        by_spk[x[1]["speaker_id"]].append(x)
    for xs in by_spk.values():
        run = -1e18
        for mi, u, cs, spans, sup, raw, mb in sorted(xs, key=lambda x: (x[1]["begin_s"], x[0])):
            for j, s in enumerate(sup):
                run = max(run, (s - start_s) * 1000 + DELTA_MS)
                tau_tok[(mi, j)] = run
    if gran == "pieces4rc":
        per_utt = [coalesce_same_read(x, tau_tok, t_ms) for x in per_utt]
    out = []
    for mi, u, cs, spans, sup, raw, mb in per_utt:
        slot = int(slot_of[u["speaker_id"]])
        b0, b1 = u["begin_s"], u["end_s"]
        total = sum(zh_len(c) for c in cs)
        acc, prev_prop = 0, b0
        prev_end = b0 * 1000
        for k, (c, sp) in enumerate(zip(cs, spans)):
            acc += zh_len(c)
            prop = max(b0 + acc / total * (b1 - b0), prev_prop)
            prev_prop = prop
            if sp:
                end = max((sup[j] - start_s) * 1000 for j in sp)
                tau = tau_tok[(mi, sp[-1])]
            else:     # no EN_WORD token in the chunk: the next token's support (else the utterance end)
                nxt = next((j for sp2 in spans[k + 1:] for j in sp2), None)
                end = (sup[nxt] - start_s) * 1000 if nxt is not None else (zh_end_of(u, start_s, align)) * 1000
                tau = max(tau_tok.get((mi, nxt), -1e18), end + DELTA_MS) if nxt is not None else end + DELTA_MS
            end = min(max(end, prev_end), dur_ms)
            ev = [raw[j] for j in sp if raw[j] is not None]
            evm = [mb[j] for j in sp if mb[j] is not None]
            out.append({"text": c, "n_words": len(c.split()), "utt": mi, "k": k, "slot": slot,
                        "src_start_ms": int(round(prev_end)), "src_end_ms": int(round(end)), "tau_ms": float(tau),
                        "e": B.first_read(t_ms, tau), "prop_end_ms": int(round(prop * 1000)),
                        "u": B.first_read(t_ms, round(prop * 1000) + LAG_MS),
                        "evidence_ms": (max(ev) - start_s) * 1000 if ev else None,
                        "mbert_ms": (max(evm) - start_s) * 1000 if evm else None, "last": k == len(cs) - 1,
                        "n_chunks_utt": len(cs), "record_id": u["record_id"],
                        "nir": align[u["record_id"]]["nir"] if u["record_id"] in align else 0.0,
                        "aligned": bool(u["record_id"] in align and align[u["record_id"]]["status"] == "ok")})
            prev_end = end
    return out


def zh_end_of(u: dict, start_s: float, align: dict) -> float:
    al = align.get(u["record_id"])
    return (al["zh_end"] - start_s) if al else u["end_s"]


def kap_order(ch: list[dict]) -> list[int]:
    """build_piece_sched.sched_record's S order: (e, src_end, src_start, utt, k)."""
    return sorted(range(len(ch)), key=lambda j: (ch[j]["e"], ch[j]["src_end_ms"], ch[j]["src_start_ms"], ch[j]["utt"],
                                                  ch[j]["k"]))


def u_cummax(ch: list[dict]) -> list[int]:
    best, u = {}, [0] * len(ch)
    for j in kap_order(ch):
        v = max(ch[j]["u"], best.get(ch[j]["slot"], -1))
        best[ch[j]["slot"]] = v
        u[j] = v
    return u


def load_align(d: str) -> dict:
    return {r["record_id"]: r for r in map(json.loads, open(Path(d) / "align.jsonl", encoding="utf-8"))}


# ------------------------------------------------------------------------------------------------ dev5 audit
def run_audit(a) -> None:
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.mist_w3po_v1.evaluation import metric_headroom as M   # puts reports/gstar_si_v8 on sys.path
    from meeting_score import universe

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    align = load_align(a.align_dir)
    recs = [json.loads(l) for l in open(a.targets or DEV5_TARGETS, encoding="utf-8")]
    W, smap = B.load_dev(a.win or str(DEV5_WIN))
    ev = {}
    st = Counter()
    early, early_omni, no_ev, rows = 0, 0, 0, []
    for rec in recs:
        ch = window_chunks(rec["members"], rec["start_s"], rec["dur_ms"], rec["t_ms"], smap[rec["meeting"]], align,
                           a.granularity)
        R = len(rec["t_ms"])
        T = rec["t_ms"]
        evs = []
        for c in ch:
            t = T[c["e"]]
            evs.append({"utt": c["utt"], "n_words": c["n_words"], "read": c["e"], "final": c["e"] == R - 1, "t": t,
                        "c": B.online(t, rec)})
            st["chunks"] += 1
            st["words"] += c["n_words"]
            st["final_words"] += c["n_words"] * (c["e"] == R - 1)
            st["e_gt_u"] += c["e"] > c["u"]
            if c["mbert_ms"] is None:
                no_ev += 1
            else:
                early += t < c["mbert_ms"]
            if c["evidence_ms"] is not None:
                early_omni += t < c["evidence_ms"]
                st["omni_ev_chunks"] += 1
            rows.append({"id": rec["id"], **{k: c[k] for k in ("text", "n_words", "e", "u", "tau_ms", "prop_end_ms",
                                                              "evidence_ms", "mbert_ms", "nir", "aligned")}})
        ev[rec["id"]] = {"T5": evs, "S_prop": B.schedule(rec, "S_prop")}
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    sc = M.scored_utts(W, RW)
    U = universe(M.DR, {w["meeting"] for w in W})
    lat, reg = {"T5": [], "S_prop": []}, []
    for m in sorted(U):
        units = M.assign(m, [], sc, U[m])["units"]
        per = {}
        for nm in ("T5", "S_prop"):
            words_of = defaultdict(list)
            for rec in recs:
                if rec["meeting"] != m:
                    continue
                for e in ev[rec["id"]][nm]:
                    rid = rec["members"][e["utt"]]["record_id"]
                    words_of[rid].append((e["n_words"], rec["start_s"] + e["c"] / 1000))
            per[nm] = B.unit_laal(units, words_of)
        for uid in sorted(per["S_prop"]["laal"]):
            if uid not in per["T5"]["laal"]:
                continue
            reg.append(per["S_prop"]["laal"][uid][1])
            for nm in ("T5", "S_prop"):
                lat[nm].append(per[nm]["laal"][uid][0])
    reg = np.asarray(reg)
    tab = {nm: {"all": float(np.mean(x)), "ov": float(np.mean(np.asarray(x)[reg == "ov"])),
                "1spk": float(np.mean(np.asarray(x)[reg == "1spk"])), "n": len(x)} for nm, x in lat.items()}
    # forced-alignment spot check: 50 random dev (validation) placed utterances with char times
    dev_meetings = sorted({w["meeting"] for w in W})
    allval = sorted({json.loads(l)["meeting"] for l in open(
        a.win or "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl", encoding="utf-8")})
    mix, fa = load_mix(allval)
    cand = sorted(r for r in mix if r in fa and fa[r] and "validation" in r)
    rng = random.Random(SEED)
    pick = rng.sample(cand, 50)
    errs = [abs(fa[r][0][0] - mix[r]["zh_begin"]) for r in pick] + [abs(fa[r][-1][1] - mix[r]["zh_end"]) for r in pick]
    errs_all = [abs(fa[r][0][0] - mix[r]["zh_begin"]) for r in cand] + [abs(fa[r][-1][1] - mix[r]["zh_end"])
                                                                        for r in cand]
    nirs = np.asarray([align[r]["nir"] for r in align if align[r]["status"] == "ok"])
    res = {
        "laal": tab,
        "go1_laal_le_sprop": tab["T5"]["all"] <= tab["S_prop"]["all"] if a.win else
        tab["T5"]["all"] <= 2.179 and abs(tab["S_prop"]["all"] - 2.179) < 5e-4,
        "final_word_share": st["final_words"] / st["words"],
        "go2_final_share_le_0.30": st["final_words"] / st["words"] <= 0.30,
        "early_vs_mbert_share": early / max(1, st["chunks"] - no_ev), "chunks_without_mbert_link": no_ev,
        "early_vs_omni_share": early_omni / max(1, st["omni_ev_chunks"]),
        "go3_early_le_0.05": early / max(1, st["chunks"] - no_ev) <= 0.05,
        "spot_median_boundary_err_s": float(np.median(errs)),
        "spot_median_start_s": float(np.median(errs[:50])), "spot_median_end_s": float(np.median(errs[50:])),
        "all_dev_median_boundary_err_s": float(np.median(errs_all)), "all_dev_utts": len(cand),
        "go4_spot_le_0.1": float(np.median(errs)) <= 0.1,
        "stats": dict(st), "e_gt_u_share": st["e_gt_u"] / st["chunks"],
        "align_status": dict(Counter(r["status"] for r in align.values())),
        "nir_dev5": {"mean": float(nirs.mean()), "p90": float(np.percentile(nirs, 90)),
                     "gt_0.3_share": float((nirs > NIR_MAX).mean())},
        "chunk_words_mean": st["words"] / st["chunks"], "spot_ids": pick, "dev_meetings": dev_meetings}
    res["GO"] = bool(res["go1_laal_le_sprop"] and res["go2_final_share_le_0.30"] and res["go3_early_le_0.05"]
                     and res["go4_spot_le_0.1"])
    out.mkdir(parents=True)
    with open(out / "chunks.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(out / "results.json", "x", encoding="utf-8") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("spot_ids",)}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ train sched build
def _build_job(args):
    k, ws, align, smap, out, grid, gran = args
    from experiments.mist_w3po_v1.crf import build_piece_sched as BPS
    from experiments.mist_w3po_v1.crf import precheck_b2_earliest_write as B
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import q

    tok = tokenizer()
    st, bad = Counter(), []
    with open(Path(out) / f"shard_{k:03d}.jsonl", "x", encoding="utf-8") as fh:
        for w in ws:
            t_ms = B.read_times(w["dur_ms"], grid)
            ch = window_chunks(w["utts"], w["start_s"], w["dur_ms"], t_ms, smap[w["meeting"]], align, gran)
            if any(c["nir"] > NIR_MAX for c in ch):
                st["dropped_nir"] += 1
                continue
            if not ch:
                st["empty"] += 1
                continue
            u = u_cummax(ch)
            P, S_old = [], []
            for c in ch:
                c["piece"] = (f"<|spk{c['slot']}|><|t{q(c['src_start_ms'] / 1000)}|>{c['text']}"
                              f"<|t{q(c['src_end_ms'] / 1000)}|>")
                ids = tok(c["piece"], add_special_tokens=False)["input_ids"]
                c["tok0"], c["tok1"] = len(S_old), len(S_old) + len(ids)
                S_old += ids
                P.append(c)
            tg = {"pieces": P, "S_ids": S_old, "t_ms": t_ms}
            extra = {"label_src": ["t5"] * len(P), "tau_ms": [c["tau_ms"] for c in P],
                     "prop_end_ms": [c["prop_end_ms"] for c in P], "evidence_ms": [c["evidence_ms"] for c in P],
                     "nir": [c["nir"] for c in P], "aligned": [c["aligned"] for c in P]}
            rec = BPS.sched_record(w, tg, [c["e"] for c in P], u, extra, tok, "free")
            chk = BPS.check_record(rec)
            if not chk["ok"]:
                bad.append(w["id"])
            st["windows"] += 1
            st["chunks"] += len(P)
            st["words"] += sum(c["n_words"] for c in P)
            st["final_words"] += sum(c["n_words"] for c in P if c["e"] == len(t_ms) - 1)
            st["e_gt_u"] += sum(c["e"] > uu for c, uu in zip(P, u))
            st["unaligned_chunks"] += sum(not c["aligned"] for c in P)
            st["reads"] += len(t_ms)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return dict(st), bad


def _feas(rec: dict) -> tuple:
    import torch

    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    torch.set_num_threads(1)
    w = T.make_window(0, rec, True)
    T.set_reach([w], 128)
    return rec["id"], bool(w.feasible)


def run_build(a) -> None:
    import multiprocessing as mp

    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    t0 = time.time()
    align = load_align(a.align_dir)
    smap = json.loads((DATA / "e2e_gstar_zh2en_v1" / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    W = [json.loads(l) for l in open(TRAIN_WIN, encoding="utf-8")]
    W.sort(key=lambda w: w["id"])
    if a.limit:
        W = W[: a.limit]
    assert all(w["split"] == "train" for w in W)
    out.mkdir(parents=True)
    shards = [W[i: i + 500] for i in range(0, len(W), 500)]
    jobs = []
    for k, sh in enumerate(shards):
        rids = {u["record_id"] for w in sh for u in w["utts"]}
        jobs.append((k, sh, {r: align[r] for r in rids if r in align}, {m: smap[m] for m in {w["meeting"] for w in sh}},
                     str(out), a.grid, a.granularity))
    st, bad = Counter(), []
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        for i, (s, b) in enumerate(pool.imap_unordered(_build_job, jobs)):
            st.update(s)
            bad += b
            print(f"shard {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    wins = T.load_windows(str(out), True)
    plan = T.plan_steps([w.R for w in wins], 20261109, 3000, 64)
    used = sorted({i for s in plan for i in s})
    with mp.get_context("spawn").Pool(a.workers) as pool:
        feas = pool.map(_feas, [wins[i].rec for i in used], chunksize=32)
    infeasible = [i for i, f in feas if not f]
    summ = {"windows": len(wins), "stats": dict(st), "plan_windows": len(used), "n_infeasible": len(infeasible),
            "plan_windows_infeasible": infeasible[:20], "label_path_check_bad": bad[:20], "n_bad": len(bad),
            "final_word_share": st["final_words"] / max(1, st["words"]),
            "L_ref_mean_lat_heur_windows": float(np.mean([w.lat_heur for w in wins])),
            "lat_heur_plan_windows": float(np.mean([wins[i].lat_heur for i in used])), "grid_ms": a.grid,
            "args": vars(a), "elapsed_s": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)
    if infeasible or bad:
        raise SystemExit("T5_SCHED_CHECK_FAILED")
    print(f"T5_SCHED_DONE {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--split", choices=["train", "dev5"], default="train")
    ap.add_argument("--targets", default="", help="dev: B-2-style targets.jsonl (default: B-2's dev5 targets)")
    ap.add_argument("--win", default="", help="audit: the dev windows of --targets (default dev5; given: GO line 1 "
                    "against S_prop recomputed on them, spot check drawn from their meetings)")
    ap.add_argument("--align-dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=500)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--granularity", choices=["merge3", "pieces4", "pieces4rc"], default="merge3",
                    help="merge3 = the registered T5 chunks; pieces4 = B-2's <= 4-word pieces (only after an amendment); "
                         "pieces4rc = pieces4 with adjacent same-label-read pieces of an utterance coalesced (RC5)")
    a = ap.parse_args()
    if a.align:
        return run_align(a)
    if a.audit:
        return run_audit(a)
    if a.build:
        return run_build(a)
    raise SystemExit("--align, --audit or --build")


if __name__ == "__main__":
    main()
