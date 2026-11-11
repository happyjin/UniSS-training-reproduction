"""B-2 precheck: the earliest writable read given by the Thinker's own likelihood.

Pre-registration: reports/mist_w3po_v1/thinker_latency_b2_prereg_v1.zh-CN.md (commit 7d3990e8, before any result).
Results: reports/mist_w3po_v1/thinker_latency_b2_results_v1.zh-CN.md.
Dev only (the 5 meetings of windows_zh2en_c2g06_v1_dev5), one setting, no training, no sweep.

Windows and references (CPU)
  Each dev5 online window (C2, gap 0.6) gets its reference targets by the S0-3 rule of data/build_windows_zh2en.py
  (the windows_train_diar rule; talk_utts / members below mirror its lines):
    - the overlapping placed, non-hesitation, non-removed zh_mix_v8 utterances must all lie >= 0.9 inside the window
      (else the window is skipped: "straddle");
    - at least one must be present (else "no_speech");
    - none may fail the §11.5 gate (else "gate");
    - a speaker without a session slot also skips the window ("no_slot"; none expected).
  References are en_norm_v1. Speaker slots are the session map e2e_gstar_zh2en_v1/slot_map.json["lowlat"], the slots
  of sched_v1 (v2' training) and of build_s0_rows without --winmap.
  Punctuation chunks come from build_ami_ms_rows.utt_chunks, with no word times, as v8 / sched_v1 / build_s0_rows: the
  source end of a chunk is the character-proportional time inside its utterance.
  Pieces: each punctuation chunk is cut into pieces of <= 4 words, greedily from its start (as B-1).
    - A piece's source end is the same character-proportional time, so the last piece of a chunk ends exactly where
      utt_chunks ends the chunk (asserted).
    - Each piece is a closed G-STAR segment <|spk k|><|t a|>text<|t b|> (a, b = its source start / end; q() of
      build_ami_gstar_si_rows).
  Canonical order kappa = S_prop's label order: (min(source end + 1.0 s, window end), source start), stable over
  (utterance, piece). This is the sort key of sched_v1's S (build_sched_windows.v8_chunks).

Schedules (window-relative write times, then the online clamp of overlap_score / speech_latency_lit:
  c = max(t, open_detect_ms); a write at the final read gets c = max(c, close_detect_ms))
  S_cur   punctuation chunks; the first read of the 1.0 s grid at or after source end + 1.0 s, else the final read
          (v8's labels = the A' label path)
  S_prop  pieces; the first read of the 0.5 s grid at or after source end + 1.0 s, else the final read (B-1)
  S_lik   pieces; e_j below
  Reads: read_times(dur, period) = min(k * period, dur), k = 1.. (online_decode's schedule; the last = the final read).

Scoring (GPU) uses the A' Thinker exactly as its dev decode built it:
  - v8 merged loaded as Engine.from_merged does (load_engine: the same weights, but an explicit 2-card layer map,
    so the audio tower sits on cuda:0 with the speaker projector) + decode_crf_ckpt.apply_ckpt(v2'
    crf_v2p_20261110T075131Z step_03000).
  - Reads every 0.5 s, plus the final read. Each read is ONE teacher-forced forward of
      prompt(audio <= t_r, ctx "") + S (all pieces in kappa order)
    with the speaker posteriors zeroed after t_r - 1.04 s.
  - At the final read the posteriors are cut at online_decode_finalpost.final_cut_ms(t, close_detect). This is the A'
    decode's --final-read-post close_detect, and v2' was trained with --final-read-full-post 1.
  - This is the engine path of precheck_serial_v2.score_reqs_v2: left padding, and <wait> barred before the softmax at
    the final read (Engine.read; v2' pass 1 --final-bar-wait 1).
  s_j(r) = the mean log-prob of piece j's tokens (speaker, start time, text, end time).

Earliest writable read (Hibiki-style largest jump, no threshold)
  e_j = argmax_r [s_j(r) - s_j(r-1)], ties -> the earlier read.
  Then the cumulative max over each speaker's pieces in kappa order (monotone), then the online clamp.
  The pre-registration says "the term before the first read is -inf". Read literally (s_j(r_0 - 1) = -inf), the first
  read's jump is +inf, so every e_j would be the first read whatever the model says: the rule would be degenerate. The
  closest faithful reading, used here (FIRST_JUMP), gives the first read's JUMP the value -inf: the first read is never
  the argmax, unless it is the only read. This is logged in the results.

Label-path text LAAL (per scored reference unit)
  - Units are metric_headroom.assign's: utterances of a speaker < 1 s apart merged; ov / 1spk by the scorer's rule.
  - The output words are the reference words of the unit's utterances (each word in its own utterance's unit).
  - Word delay = absolute online write time - unit start; LAAL = speech_latency_lit.laal.
  - The mean is taken over the "complete" units (every utterance a member of a kept window). The same units and the
    same words are used in all three schedules, so BLEU is unchanged.

GO (retrain with likelihood labels) iff both hold:
  (1) mean LAAL(S_lik) <= mean LAAL(S_prop) over the complete units (all);
  (2) in utterances with >= 2 pieces, >= 30 % of the non-last pieces have e_j (after the cumulative max and the
      clamp) earlier than the utterance end.

Stages (the root is eval_outputs/mist_w3po_v1/crf/b2_earliest_write_dev5; every output refuses to overwrite):
    python -m experiments.mist_w3po_v1.crf.precheck_b2_earliest_write --cpu-stage --root <new root>
    python -m experiments.mist_w3po_v1.crf.precheck_b2_earliest_write --score --root <root> --shard k/n \\
        [--batch 16 --tok-budget 12000 --mem-cap-gib 56 --limit n]     (CUDA_VISIBLE_DEVICES = 2 cards)
    python -m experiments.mist_w3po_v1.crf.precheck_b2_earliest_write --analyse --root <root>
    PYTHONPATH=.:third_party/Megatron-LM python -m experiments.mist_w3po_v1.crf.test_precheck_b2_earliest_write
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WIN = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
CKPT = "eval_outputs/mist_w3po_v1/crf/train/crf_v2p_20261110T075131Z/ckpt/step_03000"
THINKER = "eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged"
DATA = "data/ami_multispk_si_v1"
MIX = "zh_mix_v8"
REF = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1"
EN_NORM = "data/ami_multispk_si_v1/en_norm_v1.jsonl"
RECHECK = "data/ami_multispk_si_v1/zh_revoice_v1/talker"
INSIDE = 0.9
PIECE = 4
LAG_MS = 1000
GRID_CUR = 1000
GRID_LIK = 500
GO_SHARE = 0.30
FIRST_JUMP = -math.inf
WAIT_ID = 151649
SCHEDS = ("S_cur", "S_prop", "S_lik")


# ------------------------------------------------------------------------------------------------ windows (CPU)
def talk_utts(m: str, data: str = DATA, mix: str = MIX, en_norm: dict | None = None, asr: dict | None = None) -> list:
    """The `talk` list of build_windows_zh2en.main for meeting m (its lines verbatim): placed, non-hesitation,
    non-removed utterances, with "ref" (en_norm) and "_bad" (the §11.5 gate, or no_reference)."""
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix import bad_reason, cer_of, removed

    asr = asr or {}
    en_norm = en_norm or {}
    talk = []
    for u in map(json.loads, open(Path(data) / mix / "utts" / f"{m}.jsonl", encoding="utf-8")):
        if u["hes"] or removed(u) or not u["placed"]:
            continue
        if u["record_id"] in asr:
            u["cer2"] = round(cer_of(asr[u["record_id"]], u["zh"]), 4)
        elif not asr:   # recheck ASR absent on this server: the v8 build passed every non-removed utterance
            u["cer2"] = 0.0
        u["ref"] = en_norm.get(u["record_id"], "")
        u["_bad"] = bad_reason(u) or (None if u["ref"].strip() else "no_reference")
        talk.append(u)
    return talk


def members(w: dict, talk: list, inside: float = INSIDE) -> tuple[list | None, str | None]:
    """build_windows_zh2en's train-diar membership of window w: (members in the reference-window format, None) or
    (None, skip reason). The window spans [start_s, start_s + dur_ms / 1000]."""
    ws = w["start_s"]
    we = ws + w["dur_ms"] / 1000.0
    ins = []
    for u in talk:
        ov = min(we, u["zh_end"]) - max(ws, u["zh_begin"])
        if ov <= 0:
            continue
        if ov < inside * (u["zh_end"] - u["zh_begin"]):
            return None, "straddle"
        ins.append(u)
    if not ins:
        return None, "no_speech"
    if any(u["_bad"] for u in ins):
        return None, "gate"
    return [{"record_id": u["record_id"], "speaker_id": u["speaker_id"],
             "begin_s": round(max(0.0, u["zh_begin"] - ws), 3), "end_s": round(min(we, u["zh_end"]) - ws, 3),
             "ref": u["ref"], "hesitation": False}
            for u in sorted(ins, key=lambda u: (u["zh_begin"], u["zh_end"]))], None


def utt_pieces(u: dict, t0: float, piece: int = PIECE, lag_ms: int = LAG_MS) -> tuple[list, list]:
    """(punctuation chunks = utt_chunks, pieces of <= piece words) of one utterance u = {"begin_time", "end_time" (s,
    absolute), "zh" (the English reference here)}; times in ms relative to t0 (the window start), as utt_chunks."""
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import chunks_of, utt_chunks, zh_len

    ch = utt_chunks(u, [], lag_ms, t0)
    cs = chunks_of(u["zh"])
    assert len(cs) == len(ch)
    total = sum(zh_len(c) for c in cs)
    b0, b1 = u["begin_time"], u["end_time"]
    out, acc, prev_end = [], 0, b0
    for ci, c in enumerate(cs):
        words = c.strip().split()
        groups = [words[i: i + piece] for i in range(0, len(words), piece)]
        for gi, g in enumerate(groups):
            acc += sum(zh_len(x) for x in g)
            end = max(b0 + acc / total * (b1 - b0), prev_end)
            out.append({"chunk": ci, "text": " ".join(g), "n_words": len(g),
                        "src_start_ms": round((prev_end - t0) * 1000), "src_end_ms": round((end - t0) * 1000)})
            prev_end = end
        assert out[-1]["src_end_ms"] == ch[ci]["src_end_ms"], (u, ci)
    for c in ch:
        c["text"] = c["text"].strip()
        c["n_words"] = len(c["text"].split())
    assert sum(c["n_words"] for c in ch) == sum(p["n_words"] for p in out)
    return ch, out


def build_window(w: dict, mem: list, slot_of: dict, tok=None) -> dict:
    """Targets of one kept window: chunks (S_cur), pieces in kappa order with token spans, S / S_ids."""
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import q

    dur, ws = w["dur_ms"], w["start_s"]
    chunks, pieces = [], []
    for mi, u in enumerate(mem):
        if u["hesitation"] or not u["ref"].strip():
            continue
        uu = {"begin_time": ws + u["begin_s"], "end_time": ws + u["end_s"], "zh": u["ref"]}
        ch, pc = utt_pieces(uu, ws)
        slot = int(slot_of[u["speaker_id"]])
        for c in ch:
            chunks.append({"utt": mi, "slot": slot, "text": c["text"], "n_words": c["n_words"],
                           "src_start_ms": c["src_start_ms"], "src_end_ms": c["src_end_ms"],
                           "commit_ms": min(c["commit_ms"], dur)})
        for k, p in enumerate(pc):
            p.update({"utt": mi, "slot": slot, "k": k, "last": k == len(pc) - 1,
                      "commit_ms": min(p["src_end_ms"] + LAG_MS, dur)})
            p["piece"] = f"<|spk{slot}|><|t{q(p['src_start_ms'] / 1000)}|>{p['text']}<|t{q(p['src_end_ms'] / 1000)}|>"
            pieces.append(p)
    order = sorted(range(len(pieces)), key=lambda i: (pieces[i]["commit_ms"], pieces[i]["src_start_ms"]))
    pieces = [pieces[i] for i in order]
    rec = {"id": w["id"], "meeting": w["meeting"], "start_s": ws, "dur_ms": dur, "audio": w["audio"],
           "post": w["post"]["lowlat"], "open_detect_ms": w["open_detect_ms"], "close_detect_ms": w["close_detect_ms"],
           "members": mem, "chunks": chunks, "pieces": pieces,
           "t_cur": read_times(dur, GRID_CUR), "t_ms": read_times(dur, GRID_LIK)}
    rec["S"] = "".join(p["piece"] for p in pieces)
    if tok is not None:
        ids = []
        for p in pieces:
            pi = tok(p["piece"], add_special_tokens=False)["input_ids"]
            p["tok0"], p["tok1"] = len(ids), len(ids) + len(pi)
            ids += pi
        rec["S_ids"] = ids
        rec["joint_tok_equal"] = (tok(rec["S"], add_special_tokens=False)["input_ids"] if rec["S"] else []) == ids
    return rec


def read_times(dur: int, period: int) -> list[int]:
    """online_decode's read schedule: min(k * period, dur), k = 1..ceil(dur / period); the last read is final."""
    n = max(1, -(-dur // period))
    return [min((r + 1) * period, dur) for r in range(n)]


def first_read(t_ms: list[int], x: float) -> int:
    """index of the first read with t >= x, else the final read (= build_sched_windows.first_read_at)."""
    return min(bisect.bisect_left(t_ms, x), len(t_ms) - 1)


def online(t: int, w: dict) -> int:
    """The online write time of a write at read time t (window-relative ms): >= open detect; a final-read write
    >= close detect (overlap_score / speech_latency_lit.online)."""
    c = max(t, w["open_detect_ms"])
    if t >= w["dur_ms"]:
        c = max(c, w["close_detect_ms"])
    return c


# ------------------------------------------------------------------------------------------------ schedules
def earliest(s: np.ndarray, first_jump: float = FIRST_JUMP) -> np.ndarray:
    """s [R, J] (s_j(r) per read r and piece j) -> the read index of the largest jump s_j(r) - s_j(r-1) per piece; the
    first read's jump is first_jump (-inf: never chosen unless R = 1); ties -> the earlier read."""
    R, J = s.shape
    jump = np.full((R, J), first_jump, np.float64)
    if R > 1:
        jump[1:] = s[1:] - s[:-1]
    return jump.argmax(0)


def cummax_by_speaker(e, slots) -> list[int]:
    """running max of e within each speaker, in the given (kappa) order."""
    best, out = {}, []
    for ej, k in zip(e, slots):
        v = max(int(ej), best.get(k, -1))
        best[k] = v
        out.append(v)
    return out


def schedule(rec: dict, name: str, e_lik: list | None = None) -> list[dict]:
    """Write events of one window under a schedule: [{"utt", "n_words", "read", "final", "t", "c"}] per chunk / piece
    (t = read time, c = online write time, both window-relative ms)."""
    if name == "S_cur":
        T, items = rec["t_cur"], rec["chunks"]
        idx = [first_read(T, it["src_end_ms"] + LAG_MS) for it in items]
    elif name == "S_prop":
        T, items = rec["t_ms"], rec["pieces"]
        idx = [first_read(T, it["src_end_ms"] + LAG_MS) for it in items]
    elif name == "S_lik":
        T, items = rec["t_ms"], rec["pieces"]
        idx = cummax_by_speaker(e_lik, [p["slot"] for p in items])
    else:
        raise ValueError(name)
    out = []
    for it, i in zip(items, idx):
        out.append({"utt": it["utt"], "n_words": it["n_words"], "read": int(i), "final": i == len(T) - 1,
                    "t": T[i], "c": online(T[i], rec)})
    return out


def piece_scores(tok_lp: np.ndarray, pieces: list) -> np.ndarray:
    """tok_lp [R, N] -> s [R, J]: mean token log-prob of each piece (tokens tok0..tok1)."""
    return np.stack([tok_lp[:, p["tok0"]: p["tok1"]].mean(1) for p in pieces], 1)


# ------------------------------------------------------------------------------------------------ LAAL
def unit_laal(units: dict, words_of: dict) -> dict:
    """units: metric_headroom.assign(...)["units"] of one meeting; words_of: record id -> [(n_words, absolute write
    time s)]. Returns {unit id: (laal, reg)} for the complete units (every utterance present), and the incomplete ids."""
    from experiments.mist_w3po_v1.evaluation.speech_latency_lit import laal

    res, missing = {}, []
    for uid, u in units.items():
        if not all(r in words_of for r in u["utts"]):
            missing.append(uid)
            continue
        d = [c - u["a"] for r in u["utts"] for n, c in words_of[r] for _ in range(n)]
        res[uid] = (laal(d, u), u["reg"])
    return {"laal": res, "missing": missing}


# ------------------------------------------------------------------------------------------------ CPU stage
def load_dev(win: str = WIN) -> tuple[list, dict]:
    W = [json.loads(l) for l in open(win, encoding="utf-8")]
    assert all(w["split"] == "validation" for w in W), "dev only"
    smap = json.loads((Path(REF) / "slot_map.json").read_text(encoding="utf-8"))["lowlat"]
    return W, smap


def run_cpu_stage(args) -> None:
    from experiments.mist_w3po_v1.crf.build_sched_windows import tokenizer

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {root}")
    t0 = time.time()
    W, smap = load_dev(args.win)
    en = {r["record_id"]: r["en"] for r in map(json.loads, open(EN_NORM, encoding="utf-8"))}
    asr = {}
    for f in sorted(Path(RECHECK).glob("asr_s*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            asr[r["id"]] = r.get("asr_text", "")
    tok = tokenizer()
    talk = {m: talk_utts(m, en_norm=en, asr=asr) for m in sorted({w["meeting"] for w in W})}
    recs, skip, seen = [], Counter(), Counter()
    for w in W:
        mem, why = members(w, talk[w["meeting"]])
        if why is None and any(u["speaker_id"] not in smap.get(w["meeting"], {}) for u in mem):
            why = "no_slot"
        if why is None:
            rec = build_window(w, mem, smap[w["meeting"]], tok)
            if not rec["pieces"]:
                why = "empty_target"
        if why is not None:
            skip[why] += 1
            continue
        for u in mem:
            seen[u["record_id"]] += 1
        recs.append(rec)
    assert all(v == 1 for v in seen.values()), "an utterance in two kept windows"
    root.mkdir(parents=True)
    (root / "cpu").mkdir()
    with open(root / "cpu" / "targets.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    summ = {"windows": len(W), "kept": len(recs), "skipped": dict(skip), "skipped_total": sum(skip.values()),
            "recheck_asr": len(asr), "members": int(sum(seen.values())),
            "chunks": sum(len(r["chunks"]) for r in recs), "pieces": sum(len(r["pieces"]) for r in recs),
            "words": sum(p["n_words"] for r in recs for p in r["pieces"]),
            "reads_0p5": sum(len(r["t_ms"]) for r in recs), "reads_1p0": sum(len(r["t_cur"]) for r in recs),
            "S_tokens": sum(len(r["S_ids"]) for r in recs),
            "joint_tok_mismatch": sum(not r["joint_tok_equal"] for r in recs),
            "kept_audio_s": round(sum(r["dur_ms"] for r in recs) / 1000, 1),
            "all_audio_s": round(sum(w["dur_ms"] for w in W) / 1000, 1),
            "args": vars(args), "cpu_s": round(time.time() - t0, 1)}
    (root / "cpu" / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ GPU stage
def parse_shard(s: str) -> tuple[int, int]:
    k, n = (int(x) for x in s.split("/"))
    assert 0 <= k < n, s
    return k, n


def est_len(rec: dict, r: int) -> int:
    """prompt tokens of read r (~100 + 13 audio tokens per s + 1 speaker token per 5) + S."""
    n = math.ceil(13 * rec["t_ms"][r] / 1000) + 1
    return 100 + n + math.ceil(n / 5) + len(rec["S_ids"])


def shard_of(recs: list, k: int, n: int) -> list:
    """longest-processing-time split of the windows by their summed read lengths (deterministic)."""
    cost = [sum(est_len(r, i) for i in range(len(r["t_ms"]))) for r in recs]
    load, part = [0] * n, [[] for _ in range(n)]
    for i in sorted(range(len(recs)), key=lambda i: (-cost[i], recs[i]["id"])):
        j = min(range(n), key=lambda j: (load[j], j))
        part[j].append(i)
        load[j] += cost[i]
    return [recs[i] for i in sorted(part[k])]


def layer_map(n_cards: int, n_layers: int = 48) -> dict:
    """Explicit device map over the visible cards: cuda:0 holds the audio tower, the vision tower, the embeddings, the
    rotary embedding and the first 1/n of the text layers; the last card the final norm and lm_head; talker and
    code2wav on CPU (dropped after loading, as train_crf_sft.load_thinker)."""
    dm = {"thinker.audio_tower": 0, "thinker.visual": 0, "thinker.model.embed_tokens": 0, "thinker.model.rotary_emb": 0,
          "thinker.model.norm": n_cards - 1, "thinker.lm_head": n_cards - 1, "talker": "cpu", "code2wav": "cpu"}
    for i in range(n_layers):
        dm[f"thinker.model.layers.{i}"] = min(n_cards - 1, i * n_cards // n_layers)
    return dm


def load_engine(path: str, direction: str = "zh2en"):
    """Engine.from_merged's body with layer_map instead of device_map "auto". With "auto" over 2 cards the audio tower
    landed on cuda:1 while Engine.audio_embeds builds the speaker stream on cuda:0 (torch.cat device error, first
    launch 2026-11-11 06:44 UTC); the weights, dtypes, sd_proj and Prompter are those of Engine.from_merged.
    Engine.dev is cuda:0 explicitly (Engine.from_merged: torch.device("cuda"), the current device)."""
    import gc
    import os

    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration

    from experiments.mist_w3po_v1.core.engine import Engine, Prompter
    from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import N_SLOT, SDProjector

    torch.set_num_threads(int(os.environ.get("GSTAR_THREADS", max(4, (os.cpu_count() or 32) // 8))))
    omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                                device_map=layer_map(torch.cuda.device_count()))
    thinker = omni.thinker.eval()
    del omni
    gc.collect()
    torch.cuda.empty_cache()
    # loading leaves the CURRENT device at the last card, so Engine's torch.device("cuda") (speaker projector, inputs)
    # would resolve to it while the audio tower sits on cuda:0 (second launch, 06:48 UTC): pin both to cuda:0
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    sd = SDProjector(N_SLOT, thinker.config.text_config.hidden_size)
    sd.load_state_dict(torch.load(Path(path) / "sd_proj.pt"))
    sd = sd.to(dev, torch.bfloat16).eval()
    return Engine(thinker, sd, Prompter(direction), dev)


def make_item(rec: dict, r: int, wav, post):
    """ReadItem of read r with the whole S as the committed sequence; the final read's posteriors are cut at
    online_decode_finalpost.final_cut_ms(t, close_detect) (full=True: audio_embeds adds no second mask)."""
    from experiments.mist_w3po_v1.core.engine import ReadItem
    from experiments.mist_w3po_v1.evaluation.online_decode_finalpost import final_cut_ms

    t = rec["t_ms"][r]
    if r == len(rec["t_ms"]) - 1:
        p = post.copy()
        p[final_cut_ms(t, rec["close_detect_ms"]) // 80:] = 0.0
        return ReadItem(wav=wav, post=p, t_ms=t, final=True, committed=list(rec["S_ids"]), ctx="",
                        akey=f"{rec['id']}|{t}|fp", full=True)
    return ReadItem(wav=wav, post=post, t_ms=t, final=False, committed=list(rec["S_ids"]), ctx="",
                    akey=f"{rec['id']}|{t}")


def score_batch(eng, part: list, wavs: dict) -> list[np.ndarray]:
    """Teacher-forced token log-probs of S for a batch of (rec, r): the forward of precheck_serial_v2.score_reqs_v2
    (Engine.audio_embeds, Prompter.ids with ctx "", left padding + mask), <wait> barred before the softmax at the
    final read; log p is taken per row as logit - logsumexp (float32 logits, as Engine.logits)."""
    import torch

    from experiments.mist_w3po_v1.core.engine import AUDIO_PAD
    from experiments.mist_w3po_v1.rl.rollout import load_window

    its = []
    for rec, r in part:
        if rec["id"] not in wavs:
            wavs[rec["id"]] = load_window({"audio": rec["audio"], "post": {"lowlat": rec["post"]}})
        its.append(make_item(rec, r, *wavs[rec["id"]]))
    ms = [len(rec["S_ids"]) for rec, _ in part]
    out = []
    with torch.no_grad():
        ae = eng.audio_embeds(its, eng.emb_layer.weight.dtype)
        prompts = [eng.pr.ids(n, it.ctx, it.committed) for it, (n, _) in zip(its, ae)]
        B, L, M = len(its), max(len(p) for p in prompts), max(ms)
        ids = torch.full((B, L), eng.tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.as_tensor(p)
            mask[i, L - len(p):] = 1
        ids, mask = ids.to(eng.dev), mask.to(eng.dev)
        emb = eng.emb_layer(ids)
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), torch.cat([e for _, e in ae], 0))
        h = eng.thinker.model(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                              return_dict=True).last_hidden_state[:, L - M - 1: L - 1]
        for i, ((rec, r), it) in enumerate(zip(part, its)):
            m = ms[i]
            lg = eng.logits(h[i, M - m:]).to(eng.dev)                       # [m, V] float32
            if it.final:
                lg[:, WAIT_ID] = -1e9                                      # Engine.read line 246
            tgt = torch.as_tensor(rec["S_ids"], dtype=torch.long, device=lg.device)
            lp = lg.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(lg, -1)
            out.append(lp.double().cpu().numpy())
            del lg
        del h, emb
    return out


def batches(reqs: list, batch: int, budget: int) -> list[list]:
    """requests sorted by estimated length, packed while (rows x longest) <= budget and rows <= batch."""
    reqs = sorted(reqs, key=lambda x: (est_len(*x), x[0]["id"], x[1]))
    out, cur, mx = [], [], 0
    for q in reqs:
        L = est_len(*q)
        if cur and (len(cur) >= batch or (len(cur) + 1) * max(mx, L) > budget):
            out.append(cur)
            cur, mx = [], 0
        cur.append(q)
        mx = max(mx, L)
    if cur:
        out.append(cur)
    return out


def run_score(args) -> None:
    import torch

    from experiments.mist_w3po_v1.crf.decode_crf_ckpt import apply_ckpt

    root = Path(args.root)
    k, n = parse_shard(args.shard)
    out = root / (f"score_{k}of{n}" + (f"_limit{args.limit}" if args.limit else ""))
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    recs = [json.loads(l) for l in open(root / "cpu" / "targets.jsonl", encoding="utf-8")]
    recs = shard_of(recs, k, n)
    if args.limit:
        recs = recs[: args.limit]
    out.mkdir(parents=True)
    t_start = time.time()
    if args.mem_cap_gib > 0:
        for i in range(torch.cuda.device_count()):
            tot = torch.cuda.get_device_properties(i).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib * 2**30 / tot), i)
    eng = load_engine(args.thinker, "zh2en")
    info = apply_ckpt(eng, args.ckpt)
    print("device map: " + json.dumps(sorted(Counter(str(p.device) for p in eng.thinker.parameters()).items()))
          + f" audio_tower {next(eng.thinker.audio_tower.parameters()).device} sd_proj "
          f"{next(eng.sd_proj.parameters()).device} engine {eng.dev} current cuda:{torch.cuda.current_device()}",
          flush=True)
    print("crf ckpt applied " + json.dumps(info), flush=True)
    load_s = time.time() - t_start
    bad = [r["id"] for r in recs if eng.tok(r["S"], add_special_tokens=False)["input_ids"] != r["S_ids"]
           and r.get("joint_tok_equal", True)]
    if bad:
        raise SystemExit(f"tokenizer mismatch between the targets and the engine: {bad[:5]}")
    reqs = [(rec, r) for rec in recs for r in range(len(rec["t_ms"]))]
    bl = batches(reqs, args.batch, args.tok_budget)
    print(f"shard {k}/{n}: {len(recs)} windows, {len(reqs)} reads, {len(bl)} batches", flush=True)
    res = {rec["id"]: [None] * len(rec["t_ms"]) for rec in recs}
    left = {rec["id"]: len(rec["t_ms"]) for rec in recs}
    by_id = {rec["id"]: rec for rec in recs}
    wavs = {}
    t1 = time.time()
    with open(out / "scores.jsonl", "x", encoding="utf-8") as fh:
        for b, part in enumerate(bl):
            for (rec, r), lp in zip(part, score_batch(eng, part, wavs)):
                res[rec["id"]][r] = lp
                left[rec["id"]] -= 1
                if left[rec["id"]] == 0:
                    tl = np.stack(res[rec["id"]])
                    s = piece_scores(tl, by_id[rec["id"]]["pieces"])
                    fh.write(json.dumps({"id": rec["id"], "t_ms": rec["t_ms"],
                                         "tok_lp": np.round(tl, 5).tolist(), "s": np.round(s, 6).tolist()}) + "\n")
                    res[rec["id"]] = None
                    wavs.pop(rec["id"], None)
            if (b + 1) % 20 == 0:
                fh.flush()
                print(f"{b + 1}/{len(bl)} batches, {time.time() - t1:.0f}s", flush=True)
    assert all(v == 0 for v in left.values())
    peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]
    st = {"windows": len(recs), "reads": len(reqs), "batches": len(bl), "load_s": round(load_s, 1),
          "scoring_s": round(time.time() - t1, 1), "wall_s": round(time.time() - t_start, 1),
          "cards": torch.cuda.device_count(), "peak_alloc_gib": peak, "ckpt": info, "args": vars(args)}
    (out / "done.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(f"B2_SCORED {out} {json.dumps(st)}", flush=True)


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def q(x) -> dict | None:
    x = np.asarray(x, np.float64)
    if not len(x):
        return None
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "p10": round(float(np.percentile(x, 10)), 3),
            "p50": round(float(np.median(x)), 3), "p90": round(float(np.percentile(x, 90)), 3)}


def analyse_windows(recs: list, scores: dict) -> dict:
    """Per window: the three schedules' write events and the S_lik per-piece details."""
    ev, pieces = {}, []
    for rec in recs:
        s = np.asarray(scores[rec["id"]]["s"], np.float64)
        assert s.shape == (len(rec["t_ms"]), len(rec["pieces"])), rec["id"]
        assert scores[rec["id"]]["t_ms"] == rec["t_ms"]
        e_raw = earliest(s)
        e_lik = cummax_by_speaker(e_raw, [p["slot"] for p in rec["pieces"]])
        ev[rec["id"]] = {nm: schedule(rec, nm, e_raw if nm == "S_lik" else None) for nm in SCHEDS}
        prop = ev[rec["id"]]["S_prop"]
        for j, (p, a) in enumerate(zip(rec["pieces"], ev[rec["id"]]["S_lik"])):
            u = rec["members"][p["utt"]]
            npc = sum(1 for x in rec["pieces"] if x["utt"] == p["utt"])
            pieces.append({"id": rec["id"], "j": j, "utt": u["record_id"], "n_pieces_utt": npc, "last": p["last"],
                           "n_words": p["n_words"], "src_end_ms": p["src_end_ms"], "utt_end_ms": round(u["end_s"] * 1000),
                           "e_raw": int(e_raw[j]), "t_raw": rec["t_ms"][int(e_raw[j])], "e": int(e_lik[j]),
                           "t": a["t"], "c": a["c"], "final": a["final"], "c_prop": prop[j]["c"],
                           "final_prop": prop[j]["final"], "R": len(rec["t_ms"])})
    return {"events": ev, "pieces": pieces}


def run_analyse(args) -> None:
    from meeting_score import universe

    from experiments.mist_w3po_v1.evaluation import metric_headroom as M

    root = Path(args.root)
    out = root / "analysis"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    recs = [json.loads(l) for l in open(root / "cpu" / "targets.jsonl", encoding="utf-8")]
    cpu = json.loads((root / "cpu" / "summary.json").read_text())
    scores, done = {}, []
    for d in sorted(root.glob("score_*of*")):
        if "_limit" in d.name or not (d / "done.json").exists():
            continue
        done.append(json.loads((d / "done.json").read_text()))
        for l in open(d / "scores.jsonl", encoding="utf-8"):
            r = json.loads(l)
            assert r["id"] not in scores, r["id"]
            scores[r["id"]] = r
    miss = [r["id"] for r in recs if r["id"] not in scores]
    if miss:
        raise SystemExit(f"{len(miss)} windows without scores, e.g. {miss[:5]}")
    A = analyse_windows(recs, scores)
    ev, P = A["events"], A["pieces"]
    # label-path LAAL per scored unit
    W, _ = load_dev(args.win)
    RW = [json.loads(l) for l in open(M.REPO / M.REF_WIN, encoding="utf-8")]
    sc = M.scored_utts(W, RW)
    U = universe(M.DR, {w["meeting"] for w in W})
    lat = {nm: [] for nm in SCHEDS}
    reg = []
    n_units, n_missing, unit_rows, complete = 0, 0, [], set()
    for m in sorted(U):
        units = M.assign(m, [], sc, U[m])["units"]
        per = {}
        for nm in SCHEDS:
            words_of = defaultdict(list)
            for rec in recs:
                if rec["meeting"] != m:
                    continue
                for e in ev[rec["id"]][nm]:
                    rid = rec["members"][e["utt"]]["record_id"]
                    words_of[rid].append((e["n_words"], rec["start_s"] + e["c"] / 1000))
            per[nm] = unit_laal(units, words_of)
        n_units += len(units)
        n_missing += len(per["S_cur"]["missing"])
        assert per["S_cur"]["missing"] == per["S_prop"]["missing"] == per["S_lik"]["missing"]
        for uid in sorted(per["S_cur"]["laal"]):
            complete.update(units[uid]["utts"])
            reg.append(per["S_cur"]["laal"][uid][1])
            row = {"meeting": m, "unit": uid, "reg": reg[-1], "dur_s": round(units[uid]["b"] - units[uid]["a"], 3),
                   "words": len(units[uid]["ref"].split())}
            for nm in SCHEDS:
                lat[nm].append(per[nm]["laal"][uid][0])
                row[nm] = round(per[nm]["laal"][uid][0], 4)
            unit_rows.append(row)
    reg = np.asarray(reg)
    laal_tab = {}
    for nm in SCHEDS:
        x = np.asarray(lat[nm])
        laal_tab[nm] = {"all": q(x), "ov": q(x[reg == "ov"]), "1spk": q(x[reg == "1spk"])}
    # in-utterance share (GO 2): utterances with >= 2 pieces, non-last pieces, label write c < utterance end
    nl = [p for p in P if p["n_pieces_utt"] >= 2 and not p["last"]]
    share = float(np.mean([p["c"] < p["utt_end_ms"] for p in nl])) if nl else 0.0
    share_raw = float(np.mean([p["t_raw"] < p["utt_end_ms"] for p in nl])) if nl else 0.0
    share_prop = float(np.mean([p["c_prop"] < p["utt_end_ms"] for p in nl])) if nl else 0.0
    go1 = laal_tab["S_lik"]["all"]["mean"] <= laal_tab["S_prop"]["all"]["mean"]
    go2 = share >= GO_SHARE
    # implied lag: label write - proportional source end (s), last vs non-last pieces of the utterance
    lag = {}
    for name, sel in (("all", lambda p: True), ("last", lambda p: p["last"]), ("nonlast", lambda p: not p["last"])):
        ps = [p for p in P if sel(p)]
        lag[name] = {"S_lik_label": q([(p["c"] - p["src_end_ms"]) / 1000 for p in ps]),
                     "S_lik_raw_argmax_read": q([(p["t_raw"] - p["src_end_ms"]) / 1000 for p in ps]),
                     "S_prop_label": q([(p["c_prop"] - p["src_end_ms"]) / 1000 for p in ps])}
    # final-read word shares: all words of the kept windows, and the words of the complete scored units
    fr = {}
    for nm in SCHEDS:
        tot = fin = tot_u = fin_u = 0
        for rec in recs:
            for e in ev[rec["id"]][nm]:
                tot += e["n_words"]
                fin += e["n_words"] * e["final"]
                if rec["members"][e["utt"]]["record_id"] in complete:
                    tot_u += e["n_words"]
                    fin_u += e["n_words"] * e["final"]
        fr[nm] = {"all_words": round(fin / tot, 4), "unit_words": round(fin_u / max(tot_u, 1), 4),
                  "words": tot, "unit_words_n": tot_u}
    gpu = {"shards": len(done), "wall_s": [d["wall_s"] for d in done], "scoring_s": [d["scoring_s"] for d in done],
           "load_s": [d["load_s"] for d in done], "cards_per_shard": [d["cards"] for d in done],
           "gpu_card_hours": round(sum(d["wall_s"] * d["cards"] for d in done) / 3600, 3),
           "peak_alloc_gib": [d["peak_alloc_gib"] for d in done]}
    res = {"laal": laal_tab, "units": {"total": n_units, "complete": len(unit_rows), "incomplete": n_missing},
           "go": {"line1_lik_le_prop": bool(go1), "line2_share": round(share, 4), "line2_ok": bool(go2),
                  "line2_n_nonlast": len(nl), "line2_utts": len({p["utt"] for p in nl}), "GO": bool(go1 and go2)},
           "line2_supplement": {"raw_argmax_read_before_end": round(share_raw, 4),
                                "S_prop_before_end": round(share_prop, 4)},
           "implied_lag_s": lag, "final_read_word_share": fr,
           "S_lik_final_read_pieces": round(float(np.mean([p["final"] for p in P])), 4),
           "S_lik_first_read_pieces": round(float(np.mean([p["e"] == 0 for p in P])), 4),
           "windows": {"total": cpu["windows"], "kept": cpu["kept"], "skipped": cpu["skipped"]},
           "pieces": len(P), "gpu": gpu, "first_jump": "the first read's jump := -inf (FIRST_JUMP)",
           "args": vars(args)}
    out.mkdir(parents=True)
    with open(out / "results.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(out / "pieces.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in P)
    with open(out / "units.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in unit_rows)
    print(json.dumps(res, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-stage", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--root", required=True)
    ap.add_argument("--win", default=WIN)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--thinker", default=THINKER)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--mem-cap-gib", type=float, default=56.0, help="CUDA allocator cap per visible card (0: none)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tok-budget", type=int, default=12000, help="rows x longest estimated row, per batch")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first n windows of the shard (own output dir)")
    args = ap.parse_args()
    if args.cpu_stage:
        run_cpu_stage(args)
    if args.score:
        run_score(args)
    if args.analyse:
        run_analyse(args)


if __name__ == "__main__":
    main()
