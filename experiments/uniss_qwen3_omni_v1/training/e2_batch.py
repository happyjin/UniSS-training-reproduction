"""Turn an E2Dataset batch into what E2Model.forward reads.

Two sequences per row.

The Thinker's: the inference prompt word for word (the instruction E1 and
P1 send, read from phase_d_e2e.py), then the translation, teacher-forced.
Prompts are left-padded so every translation starts at one column, and its
last-layer hidden states are read from there.

UniSS's: exactly the sequence P2 step zero ran on -- UniSS's Performance
prompt with the source speaker's 32 global tokens and the source's GLM-4
content tokens, the translation in the content slot (the Thinker's token
ids; the vocabularies agree), the TTS header UniSS writes next, then the
target's BiCodec semantic codes and the end token, which are the only
supervised positions. Built from ids directly; no text is re-tokenised.

Alongside, the positions E2's three added inputs land on: each translation
token (the Thinker's hidden state), each GLM-4 token (the source's prosody
at that moment) and <|start_semantic_token|> (the length to speak for).
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
LANG3 = {"en": "eng", "zh": "cmn", "eng": "eng", "cmn": "cmn"}
LANG_TOKEN = {"eng": 180373, "cmn": 180372}
GLOBAL_0, SEMANTIC_0, GLM_0 = 151665, 155761, 163953
TASK_S2S, BALANCE, START_GLOBAL, END_GLOBAL = 180379, 180400, 180387, 180391
# process_input's speed token for speed=1.0 is int((1.0 - 0.1) / 0.1) = 9: <|speed_9|>,
# the token UniSS also writes in its TTS header. The speed is source duration over
# target duration -- measured on UniSS itself, speed 6 spoke 1.14-1.18x the source,
# 9 spoke 0.93-0.95x, 12 spoke 0.75x -- so a training row must carry its own: fed
# speed 9 on every row, a thousand steps taught it that 9 means any length, and
# SLC at 20% fell from 0.94/0.79 to 0.64/0.58. Inference asks for 9.
WRITE, TASK_S2T, TASK_TTS, SPEED_0 = 180396, 180380, 180375, 180337
SPEED_9 = SPEED_0 + 9


def speed_token(source_seconds: float, target_codes: int) -> int:
    """process_input's speed index for this row's actual source/target ratio."""
    speed = source_seconds / max(target_codes / 50.0, 1e-3)
    return SPEED_0 + min(max(int((speed - 0.1) / 0.1 + 1e-6), 0), 34)
START_CONTENT, END_CONTENT, START_SEMANTIC, END_SEMANTIC = 180386, 180390, 180388, 180392
DURATION_DIM = 5
# Frames of BiCodec semantic code a sequence may carry: 30 s, UniSS's own cap.
MAX_CODES = 1500


def inference_instruction() -> dict:
    """phase_d_e2e.INSTRUCTION, read without importing its training stack."""
    tree = ast.parse((REPO / "experiments/uniss_qwen3_omni_v1/evaluation/phase_d_e2e.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "INSTRUCTION" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("INSTRUCTION not found in phase_d_e2e.py")


def uniss_prefix(tgt: str, globals_: list[int], glm: list[int], speed: int = SPEED_9) -> list[int]:
    """uniss.cli.prompt.process_input(..., "Performance", tgt, speed=1.0), as ids."""
    lang = LANG_TOKEN[tgt]
    return ([TASK_S2S, BALANCE, lang, START_GLOBAL] + [GLOBAL_0 + int(g) for g in globals_] + [END_GLOBAL]
            + [GLM_0 + int(t) for t in glm] + [WRITE, TASK_S2T, lang, speed, START_CONTENT])


def uniss_header(tgt: str, speed: int = SPEED_9) -> list[int]:
    """What UniSS writes after the translation, up to the first code."""
    return [END_CONTENT, WRITE, TASK_TTS, LANG_TOKEN[tgt], speed, START_SEMANTIC]


def duration_features(target_codes: int, source_seconds: float) -> list[float]:
    """The length to speak for, absolute and relative to the source."""
    source_codes = max(50.0 * source_seconds, 1.0)
    return [target_codes / 1000.0, float(np.log1p(target_codes)) / 7.0,
            target_codes / source_codes, float(np.log(target_codes / source_codes + 1e-3)), 1.0]


def encode_thinker(processor, instruction: dict, tgts: list[str], waveforms, translations) -> dict:
    """The Thinker's side of a batch: inference prompt, then the translation.

    CPU-only, so E2Dataset runs it in the dataloader workers; run here in the
    training process it held the GPUs idle for the length of the mel
    extraction every micro-batch.
    """
    tok = processor.tokenizer
    tok.padding_side = "left"
    conversations = [[{"role": "user", "content": [
        {"type": "audio", "audio": "x.wav"}, {"type": "text", "text": instruction[t]}]}] for t in tgts]
    templated = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    encoded = processor(text=list(templated), audio=list(waveforms), sampling_rate=16_000,
                        return_tensors="pt", padding=True)
    replies = tok(list(translations), add_special_tokens=False)["input_ids"]
    pad_id = tok.pad_token_id or 0
    prompt_ids, prompt_mask = encoded["input_ids"], encoded["attention_mask"]
    width, reply_width = prompt_ids.shape[1], max(len(r) for r in replies)
    thinker_ids = torch.full((len(replies), width + reply_width), pad_id, dtype=torch.long)
    thinker_mask = torch.zeros_like(thinker_ids)
    thinker_ids[:, :width], thinker_mask[:, :width] = prompt_ids, prompt_mask
    for i, ids in enumerate(replies):
        thinker_ids[i, width : width + len(ids)] = torch.as_tensor(ids)
        thinker_mask[i, width : width + len(ids)] = 1
    out = {"thinker_input_ids": thinker_ids, "thinker_attention_mask": thinker_mask,
           "reply_start": width, "replies": replies}
    for key in ("input_features", "feature_attention_mask"):
        if key in encoded:
            out[key] = encoded[key]
    return out


def build_e2_batch(batch: dict, processor, *, device, instruction: dict,
                   duration_dropout: float = 0.2, rng: random.Random | None = None, light: bool = False) -> dict:
    """light: UniSS's side only (no Thinker input, no added-input positions),
    for runs without the added inputs. The translation is tokenised here
    then -- the vocabularies agree, so the ids are the Thinker's."""
    rng = rng or random
    rows = len(batch["translation"])
    tgts = [LANG3[t] for t in batch["target_lang"]]

    if light:
        replies = processor.tokenizer(list(batch["translation"]), add_special_tokens=False)["input_ids"]
        seqs, label_pos, labels = [], [], []
        for i in range(rows):
            codes = batch["codes"][i].tolist()[:MAX_CODES]
            ended = len(batch["codes"][i]) <= MAX_CODES
            speed = speed_token(float(batch["source_seconds"][i]), len(batch["codes"][i]))
            prefix = uniss_prefix(tgts[i], batch["globals"][i].tolist(), batch["source_glm"][i].tolist(), speed)
            head = uniss_header(tgts[i], speed)
            target = [SEMANTIC_0 + c for c in codes] + ([END_SEMANTIC] if ended else [])
            start = len(prefix) + len(replies[i]) + len(head) - 1
            seqs.append(prefix + replies[i] + head + target)
            label_pos += [(i, p) for p in range(start, start + len(target))]
            labels += target
        longest = max(len(s) for s in seqs)
        uniss_ids = torch.zeros((rows, longest), dtype=torch.long)
        uniss_mask = torch.zeros((rows, longest), dtype=torch.long)
        for i, s in enumerate(seqs):
            uniss_ids[i, : len(s)] = torch.as_tensor(s)
            uniss_mask[i, : len(s)] = 1
        t = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
        return {"uniss_ids": uniss_ids.to(device), "uniss_mask": uniss_mask.to(device),
                "label_b": t([b for b, _ in label_pos]), "label_p": t([p for _, p in label_pos]), "labels": t(labels)}

    thinker = batch.get("thinker") or encode_thinker(processor, instruction, tgts, batch["waveforms"],
                                                     batch["translation"])
    replies, width = thinker["replies"], int(thinker["reply_start"])

    # UniSS: P2 step zero's sequence, the target codes supervised.
    seqs, label_pos, labels = [], [], []
    content_b, content_pos, content_j = [], [], []
    glm_b, glm_pos, glm_feat = [], [], []
    start_b, start_pos, start_feat = [], [], []
    for i in range(rows):
        codes = batch["codes"][i].tolist()[:MAX_CODES]
        ended = len(batch["codes"][i]) <= MAX_CODES
        glm = batch["source_glm"][i].tolist()
        speed = speed_token(float(batch["source_seconds"][i]), len(batch["codes"][i]))
        prefix = uniss_prefix(tgts[i], batch["globals"][i].tolist(), glm, speed)
        head = uniss_header(tgts[i], speed)
        target = [SEMANTIC_0 + c for c in codes] + ([END_SEMANTIC] if ended else [])
        seq = prefix + replies[i] + head + target
        start = len(prefix) + len(replies[i]) + len(head) - 1
        seqs.append(seq)
        # The logits at position p predict token p + 1.
        label_pos += [(i, p) for p in range(start, start + len(target))]
        labels += target
        content_b += [i] * len(replies[i])
        content_pos += list(range(len(prefix), len(prefix) + len(replies[i])))
        content_j += list(range(len(replies[i])))
        first_glm = 4 + 32 + 1
        glm_b += [i] * len(glm)
        glm_pos += list(range(first_glm, first_glm + len(glm)))
        glm_feat.append(batch["prosody"][i][: len(glm)])
        keep = rng.random() >= duration_dropout   # dropped: speak at a free length
        start_b.append(i)
        start_pos.append(start)
        start_feat.append(duration_features(len(codes), float(batch["source_seconds"][i])) if keep
                          else [0.0] * DURATION_DIM)
    longest = max(len(s) for s in seqs)
    uniss_ids = torch.zeros((rows, longest), dtype=torch.long)
    uniss_mask = torch.zeros((rows, longest), dtype=torch.long)
    for i, s in enumerate(seqs):
        uniss_ids[i, : len(s)] = torch.as_tensor(s)
        uniss_mask[i, : len(s)] = 1

    def t(x, dtype=torch.long):
        return torch.as_tensor(x, dtype=dtype, device=device)

    out = {
        "thinker_input_ids": thinker["thinker_input_ids"].to(device),
        "thinker_attention_mask": thinker["thinker_attention_mask"].to(device),
        "reply_start": t(width), "uniss_ids": uniss_ids.to(device), "uniss_mask": uniss_mask.to(device),
        "label_b": t([b for b, _ in label_pos]), "label_p": t([p for _, p in label_pos]), "labels": t(labels),
        "content_b": t(content_b), "content_p": t(content_pos), "content_j": t(content_j),
        "glm_b": t(glm_b), "glm_p": t(glm_pos), "glm_feat": torch.cat(glm_feat).to(device=device, dtype=torch.float32),
        "start_b": t(start_b), "start_p": t(start_pos), "start_feat": t(start_feat, torch.float32),
    }
    for key in ("input_features", "feature_attention_mask"):
        if key in thinker:
            out[key] = thinker[key].to(device)
    return out
