"""Turn an SIDataset batch into what SIModel.forward reads.

offline rows: Plan 1's sequence exactly (e2_batch, light), the codes and the end
token supervised; each label carries its row's anchor weight.

si rows: the same sequence opened up into blocks, append-only, so inference
can extend one KV cache as the source arrives:

    <task_s2s> <balance> <lang> <start_global> 32 voice <end_global>
    for each block:  GLM tokens heard since the last block
                     <write> <task_s2t> <lang> <speed> <start_content> text <end_content>
                     <write> <task_tts> <lang> <speed> [<streaming_mode>] <start_semantic>
                     codes  then  <wait_read> (more to come) | <end_semantic> (the last block)

With block_speed (T1c) each block carries its own speed token: the source
time since the previous block was written over the block's own speech, the
token's meaning offline (source over target duration) taken block by block.
With one token a row, T1b paced each block by the source it covered and drew
a short translation of a long, hesitant source out over its length.

<streaming_mode> marks a block that is not the last: the one place the model
learns that this piece of speech continues. The last block's header is the
offline header unchanged. The codes and the closing token of every block are
supervised. Beside it, the row's offline sequence for the teacher (Plan 1's
model reading the whole source and the whole translation), with each code's
position in both, so the streaming model can be distilled towards what the
offline one would say at that code.
"""

from __future__ import annotations

import random
import zlib

import torch

from experiments.uniss_qwen3_omni_v1.training.e2_batch import (
    BALANCE, END_CONTENT, END_GLOBAL, END_SEMANTIC, GLM_0, GLOBAL_0, LANG3, LANG_TOKEN, MAX_CODES, SEMANTIC_0,
    START_CONTENT, START_GLOBAL, START_SEMANTIC, TASK_S2S, TASK_S2T, TASK_TTS, WRITE, speed_token, uniss_header,
    uniss_prefix)

WAIT_READ, STREAMING_MODE = 180395, 180406   # <|wait_read|>, <|streaming_mode|>: in UniSS's vocabulary, never trained
# lookahead (docs/si_bayes_belief_plan.zh-CN.md 2.7): text the Thinker has not committed yet, shown to the talker
# between these two reserved tokens after a block's text and never spoken; <|audio_placeholder_1/2|>, unused so far
LOOK_OPEN, LOOK_CLOSE = 180403, 180404


def anchor_weight(tgt: str, cmn_mult: float) -> float:
    return cmn_mult if tgt == "cmn" else 1.0


def _pad(seqs: list[list[int]], device):
    longest = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), longest), dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def offline_row(tgt: str, globals_: list[int], glm: list[int], text_ids: list[int], codes: list[int],
                speed: int) -> tuple[list[int], int, list[int]]:
    """Plan 1's sequence; returns (sequence, position predicting the first target, targets)."""
    ended = len(codes) <= MAX_CODES
    codes = codes[:MAX_CODES]
    prefix = uniss_prefix(tgt, globals_, glm, speed)
    head = uniss_header(tgt, speed)
    target = [SEMANTIC_0 + c for c in codes] + ([END_SEMANTIC] if ended else [])
    return prefix + text_ids + head + target, len(prefix) + len(text_ids) + len(head) - 1, target


def block_speeds(blocks: list[dict], n_codes: int) -> list[int]:
    """Per block: speed_token(source seconds since the previous write, the block's own codes)."""
    out, w_prev, c_prev = [], 0, 0
    for b in blocks:
        c = min(b["code_end"], n_codes)
        out.append(speed_token(max(b["write_ms"] - w_prev, 1) / 1000, max(c - c_prev, 1)))
        w_prev, c_prev = b["write_ms"], max(c, c_prev)
    return out


def lookahead_texts(blocks: list[dict], tgt: str, rng, k_max: int, p_noise: float, p_none: float) -> list[str]:
    """Per block, a few units of what comes after it (none after the last): the next 1..k_max English words or
    1..2k_max Chinese characters; with p_noise units drawn from elsewhere in the translation instead (a draft's
    wrong tail), with p_none nothing (the posterior held nothing back)."""
    zh = tgt == "cmn"
    split = (lambda x: [c for c in x if not c.isspace()]) if zh else (lambda x: x.split())
    glue = "" if zh else " "
    out = []
    for i, b in enumerate(blocks):
        rest = split("".join(x["text"] for x in blocks[i + 1:]) if zh else " ".join(x["text"] for x in blocks[i + 1:]))
        if b["final"] or not rest or rng.random() < p_none:
            out.append("")
            continue
        k = rng.randint(1, 2 * k_max if zh else k_max)
        if rng.random() < p_noise:
            pool = split(" ".join(x["text"] for x in blocks))
            out.append(glue.join(rng.choice(pool) for _ in range(k)))
        else:
            out.append(glue.join(rest[:k]))
    return out


def streaming_row(tgt: str, globals_: list[int], glm: list[int], blocks: list[dict], block_ids: list[list[int]],
                  codes: list[int], speed: int | list[int], look_ids: list[list[int]] | None = None):
    """Returns (sequence, [(position, label, code_index or -1 for a closing token)]). speed: one token, or one a block.
    look_ids: per block, lookahead text ids placed between LOOK_OPEN and LOOK_CLOSE after its text (empty: none)."""
    lang = LANG_TOKEN[tgt]
    seq = [TASK_S2S, BALANCE, lang, START_GLOBAL] + [GLOBAL_0 + int(g) for g in globals_] + [END_GLOBAL]
    labels, g_prev, c_prev = [], 0, 0
    speeds = speed if isinstance(speed, list) else [speed] * len(blocks)
    for j, (b, ids, speed) in enumerate(zip(blocks, block_ids, speeds)):
        seq += [GLM_0 + int(t) for t in glm[g_prev: b["glm_end"]]]
        seq += [WRITE, TASK_S2T, lang, speed, START_CONTENT] + ids + [END_CONTENT]
        if look_ids and look_ids[j]:
            seq += [LOOK_OPEN] + look_ids[j] + [LOOK_CLOSE]
        seq += [WRITE, TASK_TTS, lang, speed]
        seq += ([] if b["final"] else [STREAMING_MODE]) + [START_SEMANTIC]
        piece = codes[c_prev: b["code_end"]]
        for k, c in enumerate(piece):
            labels.append((len(seq) - 1 + k, SEMANTIC_0 + c, c_prev + k))
        seq += [SEMANTIC_0 + c for c in piece]
        closing = END_SEMANTIC if b["final"] else WAIT_READ
        labels.append((len(seq) - 1, closing, len(codes) if b["final"] else -1))
        seq.append(closing)
        g_prev, c_prev = b["glm_end"], b["code_end"]
    return seq, labels


def build_si_batch(batch: dict, tokenizer, *, device, anchor_cmn_mult: float = 2.0, block_speed: bool = False,
                   lookahead_max: int = 0, lookahead_noise: float = 0.3, lookahead_none: float = 0.2) -> dict:
    rows = len(batch["translation"])
    tgts = [LANG3[t] for t in batch["target_lang"]]
    t = lambda x, dtype=torch.long: torch.as_tensor(x, dtype=dtype, device=device)
    if batch["kind"] == "offline":
        replies = tokenizer(list(batch["translation"]), add_special_tokens=False)["input_ids"]
        seqs, lb, lp, ll, lw = [], [], [], [], []
        for i in range(rows):
            codes = batch["codes"][i].tolist()
            speed = speed_token(float(batch["source_seconds"][i]), len(codes))
            seq, start, target = offline_row(tgts[i], batch["globals"][i].tolist(), batch["source_glm"][i].tolist(),
                                             replies[i], codes, speed)
            seqs.append(seq)
            lb += [i] * len(target)
            lp += list(range(start, start + len(target)))
            ll += target
            lw += [anchor_weight(tgts[i], anchor_cmn_mult)] * len(target)
        ids, mask = _pad(seqs, device)
        return {"stream": t(0), "uniss_ids": ids, "uniss_mask": mask, "label_b": t(lb), "label_p": t(lp),
                "labels": t(ll), "label_w": t(lw, torch.float32)}

    seqs, tseqs, lb, lp, ll, closing = [], [], [], [], [], []
    kb, ks, kt = [], [], []   # KL pairs: row, index into the labels, teacher position
    for i in range(rows):
        codes = batch["codes"][i].tolist()[:MAX_CODES]
        speed = speed_token(float(batch["source_seconds"][i]), len(batch["codes"][i]))
        glm, glob = batch["source_glm"][i].tolist(), batch["globals"][i].tolist()
        blocks = batch["blocks"][i]
        block_ids = tokenizer([b["text"] for b in blocks], add_special_tokens=False)["input_ids"]
        look_ids = None
        if lookahead_max > 0:   # seeded by the row, so a resumed run draws the same lookahead
            rng = random.Random(zlib.crc32(batch["translation"][i].encode("utf-8")) ^ len(blocks))
            looks = lookahead_texts(blocks, tgts[i], rng, lookahead_max, lookahead_noise, lookahead_none)
            look_ids = [tokenizer(x, add_special_tokens=False)["input_ids"] if x else [] for x in looks]
        seq, labels = streaming_row(tgts[i], glob, glm, blocks, block_ids, codes,
                                    block_speeds(blocks, len(codes)) if block_speed else speed, look_ids)
        full = tokenizer(batch["translation"][i], add_special_tokens=False)["input_ids"]
        tseq, tstart, _ = offline_row(tgts[i], glob, glm, full, codes, speed)
        seqs.append(seq)
        tseqs.append(tseq)
        for pos, label, k in labels:
            if 0 <= k and tstart + k < len(tseq):   # a code, or the end token: the teacher predicts it at tstart + k
                kb.append(i)
                ks.append(len(ll))
                kt.append(tstart + k)
            lb.append(i)
            lp.append(pos)
            ll.append(label)
            closing.append(label in (WAIT_READ, END_SEMANTIC))
    ids, mask = _pad(seqs, device)
    tids, tmask = _pad(tseqs, device)
    return {"stream": t(1), "uniss_ids": ids, "uniss_mask": mask, "label_b": t(lb), "label_p": t(lp),
            "labels": t(ll), "closing": t(closing, torch.bool), "teacher_ids": tids, "teacher_mask": tmask,
            "kl_b": t(kb), "kl_s": t(ks), "kl_t": t(kt)}
