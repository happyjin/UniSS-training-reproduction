#!/usr/bin/env python3
"""Batch-5.0 t2s (text→speech) translation evaluation.

与 s2s 评测（``eval_batch4_autoregressive`` / ``eval_s2s_*``）的区别
=================================================================
* **输入模态**：t2s 的 thinker 输入是**源文本**（``messages`` 的 user 消息里
  的 ``Translate "..."``），**没有源音频**（``audios: []``）。因此
  ``use_speaker=False``、``sample.audio=None``——不提取参考说话人音色。
* **layout**：batch5.0 四个 branch 的 checkpoint config 均为
  ``use_upstream_talker_input=True`` + ``full_add_layout=False``，推理统一走
  upstream text-channel 路径，layout 仅由 env ``QWEN_OMNISPEAK_FULL_ADD_LAYOUT``
  区分：full_add → true（codec_mask×T_prompt 前缀）；其余三个 → false。
* **GT 参考**：t2s 数据里 ``target_codec_codes`` 已是预编码 ``[T,16]`` 声学码，
  直接 ``decode_codec`` 即得 GT 音频，无需从 target_audio 重新 encode。

每个样本产出
-----------
* thinker 自回归译文 ``pred_text``，与 ``target_text`` 对比（字符级相似度）；
* talker 自回归音频 ``*_ar.wav``（含 codec 帧数 / 时长 / EOS 触发 / rtf）；
* GT 参考音频 ``*_gt.wav``（``target_codec_codes`` 直接解码）；
* 可选 teacher-forced 音频 ``*_tf.wav``（GT codec 作输入，logits argmax 解码）。

用法
----
  # 评单个 branch（自动定位最新 checkpoint 与其 val_dataset.jsonl）
  python inference/eval_t2s.py --branch full_add --n 5 --device cuda:0

  # 指定 checkpoint / val 集
  python inference/eval_t2s.py --branch reply_add \
      --ckpt output/.../checkpoint-15000 \
      --val-jsonl output/.../val_dataset.jsonl --n 10

  # 一次评全部四个 branch（各自 ckpt/layout）
  python inference/eval_t2s.py --all-branches --n 5 --device cuda:0 --full-eval

输出：``exp_results/batch5.0_t2s/<branch>/audios/*.wav`` + ``summary_<branch>.json``。
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom", "qwen_omnispeak"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QWEN3_TTS_TOKENIZER", os.path.join(REPO, "models", "Qwen3-TTS-Tokenizer-12Hz"))

import numpy as np  # noqa: E402


# branch -> (default run_name 后缀, QWEN_OMNISPEAK_FULL_ADD_LAYOUT)
# run_name 规律: <branch>-3b-lr1e-5-bs<N>x2-t2s；bs 按当前分档设置。
BRANCHES = {
    "bridge_concat":         {"run": "bridge_concat-3b-lr1e-5-bs64x2-t2s",         "full_add": False},
    "full_add":              {"run": "full_add-3b-lr1e-5-bs32x2-t2s",              "full_add": True},
    "reply_add":             {"run": "reply_add-3b-lr1e-5-bs64x2-t2s",             "full_add": False},
    "bridge_concat_thinker": {"run": "bridge_concat_thinker-3b-lr1e-5-bs52x2-t2s", "full_add": False},
}
BATCH_DIR = os.path.join(REPO, "output", "qwen_omnispeak_output", "batch5.0")


# --------------------------------------------------------------------------- #
# checkpoint / val 定位
# --------------------------------------------------------------------------- #
def _latest_version_dir(run_dir: str):
    """Return the newest ``v*-<timestamp>`` subdir under a run dir."""
    if not os.path.isdir(run_dir):
        return None
    vers = [d for d in os.listdir(run_dir) if d.startswith("v") and os.path.isdir(os.path.join(run_dir, d))]
    if not vers:
        return None
    vers.sort()  # v0<v1<...; timestamp suffix keeps chronological order within same v
    return os.path.join(run_dir, vers[-1])


def _latest_checkpoint(version_dir: str):
    if not version_dir or not os.path.isdir(version_dir):
        return None
    cks = [d for d in os.listdir(version_dir)
           if d.startswith("checkpoint-") and os.path.isdir(os.path.join(version_dir, d))]
    if not cks:
        return None
    cks.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(version_dir, cks[-1])


def resolve_ckpt_and_val(branch: str):
    """Auto-locate the newest checkpoint and its val_dataset.jsonl for a branch."""
    run_dir = os.path.join(BATCH_DIR, BRANCHES[branch]["run"])
    vdir = _latest_version_dir(run_dir)
    ckpt = _latest_checkpoint(vdir)
    val = os.path.join(vdir, "val_dataset.jsonl") if vdir else None
    if (not val or not os.path.isfile(val)) and os.path.isdir(run_dir):
        # fall back to any version dir that has a val_dataset.jsonl
        for d in sorted(os.listdir(run_dir), reverse=True):
            cand = os.path.join(run_dir, d, "val_dataset.jsonl")
            if os.path.isfile(cand):
                val = cand
                break
    return ckpt, val


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_t2s_samples(val_jsonl: str, n: int):
    """Parse t2s val samples. Each sample carries the training-time ``messages``
    (system=task_prompt, user='Translate "..."'), no source audio."""
    samples = []
    with open(val_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if not line.strip():
                continue
            d = json.loads(line)
            raw_msgs = d.get("messages", []) or []
            msgs = []
            for m in raw_msgs:
                msgs.append(json.loads(m) if isinstance(m, str) else m)
            samples.append({
                "id": str(d.get("sample_id") or d.get("id") or f"sample_{i}"),
                "messages": msgs,
                "task_prompt": d.get("task_prompt", ""),
                "target_text": d.get("target_text", "") or "",
                "target_codec_codes": d.get("target_codec_codes"),
                "source_lang": d.get("source_lang"),
                "target_lang": d.get("target_lang", "zh") or "zh",
            })
    return samples


def prepare_t2s_inputs(processor, messages, device):
    """Build thinker inputs from the training-time messages (text-only).

    CRITICAL: the val ``messages`` include the ``assistant`` turn (= GT
    translation). We MUST drop it and only feed ``system + user`` with
    ``add_generation_prompt=True``, otherwise the thinker would just copy the
    ground-truth answer (label leakage) instead of translating.
    """
    from inference.common import move_to_device, get_tokenizer
    prompt_msgs = _normalize_processor_messages(
        [m for m in messages if m.get("role") != "assistant"])
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
    else:
        text = get_tokenizer(processor).apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)
    encoded = processor(text=[text], return_tensors="pt", padding=True)
    return move_to_device(dict(encoded), device)


def _normalize_processor_messages(messages):
    """Convert dataset string content to Qwen-Omni text content blocks."""
    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content", "")
        if isinstance(content, str):
            item["content"] = [{"type": "text", "text": content}]
        normalized.append(item)
    return normalized


def _message_text(message):
    """Extract plain text from either dataset strings or processor blocks."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def _apply_chat_template(processor, messages, *, add_generation_prompt: bool):
    """Apply the same ChatML conversation shape used by the training sample."""
    from inference.common import get_tokenizer

    owner = processor if hasattr(processor, "apply_chat_template") else get_tokenizer(processor)
    return owner.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False)


def prepare_teacher_forced_thinker_inputs(processor, messages, device):
    """Build the training-time thinker sequence and its assistant-only labels.

    The prompt-only sequence (system + user + assistant generation marker) must
    be an exact prefix of the full conversation (system + user + GT assistant).
    Its length is also the ``t_prompt`` used by the full-add codec-mask prefix.
    """
    import torch
    from inference.common import move_to_device

    prompt_msgs = _normalize_processor_messages(
        [m for m in messages if m.get("role") != "assistant"])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        raise ValueError("teacher forcing requires a GT assistant message")

    prompt_text = _apply_chat_template(processor, prompt_msgs, add_generation_prompt=True)
    # The registered ms-swift training template has suffix '<|im_end|>' (without
    # the newline that Qwen2.5-OmniProcessor adds for a completed conversation).
    # Build this tail explicitly so the TF thinker labels are token-identical to
    # training rather than supervising one extra trailing newline token.
    full_text = prompt_text + _message_text(assistant_msgs[0]) + "<|im_end|>"
    prompt = processor(text=[prompt_text], return_tensors="pt", padding=True)
    full = processor(text=[full_text], return_tensors="pt", padding=True)

    prompt_ids = prompt["input_ids"]
    full_ids = full["input_ids"]
    prompt_len = int(prompt.get("attention_mask", torch.ones_like(prompt_ids))[0].sum().item())
    full_len = int(full.get("attention_mask", torch.ones_like(full_ids))[0].sum().item())
    if full_len <= prompt_len or not torch.equal(full_ids[0, :prompt_len], prompt_ids[0, :prompt_len]):
        raise ValueError(
            "GT conversation is not prompt-prefix compatible with training ChatML; "
            f"prompt_len={prompt_len}, full_len={full_len}")

    thinker_labels = torch.full_like(full_ids, -100)
    thinker_labels[:, prompt_len:full_len] = full_ids[:, prompt_len:full_len]
    return move_to_device(dict(full), device), thinker_labels.to(device), prompt_len


def prepare_predicted_text_thinker_inputs(inputs, token_ids, device):
    """Legacy Batch-4 TF condition: AR thinker reply + GT codec."""
    import torch
    from inference.common import split_thinker_kwargs

    input_ids, attention_mask, thinker_kwargs = split_thinker_kwargs(inputs)
    if not token_ids:
        raise ValueError("pred-text teacher forcing requires non-empty autoregressive text tokens")
    reply_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    thinker_ids = torch.cat([input_ids, reply_ids], dim=1)
    thinker_mask = torch.cat([attention_mask, torch.ones_like(reply_ids)], dim=1)
    thinker_labels = torch.full_like(thinker_ids, -100)
    thinker_labels[:, input_ids.size(1):] = reply_ids
    return {
        "input_ids": thinker_ids,
        "attention_mask": thinker_mask,
        **thinker_kwargs,
    }, thinker_labels, int(input_ids.size(1))


def teacher_forced_audio(
    model, processor, model_config, device, cfg, sample, prompt_inputs, pred_text_tokens,
    *, layout_full_add: bool, text_source: str, out_dir: str, sid: str,
    decode_audio: bool = True,
):
    """Run the training-style one-shot forward and decode argmax codec logits.

    ``text_source=target`` teacher-forces both thinker text and codec, matching
    Trainer validation. ``text_source=pred`` retains the older Batch-4 diagnostic
    where the thinker reply is autoregressive but the codec stream is forced.
    """
    import torch
    from codec_tokenizer import _build_training_tensors
    from inference.common import (
        _LANG_ALIASES, _codec_config, decode_codec, get_codec_tokenizer,
        save_audio, split_thinker_kwargs,
    )

    if text_source == "target":
        thinker_inputs, thinker_text_labels, t_prompt = prepare_teacher_forced_thinker_inputs(
            processor, sample["messages"], device)
    else:
        thinker_inputs, thinker_text_labels, t_prompt = prepare_predicted_text_thinker_inputs(
            prompt_inputs, pred_text_tokens, device)

    thinker_ids, thinker_mask, thinker_kwargs = split_thinker_kwargs(thinker_inputs)
    with torch.inference_mode():
        thinker_out = model.model.run_thinker(
            input_ids=thinker_ids,
            attention_mask=thinker_mask,
            output_text_embeds=True,
            **thinker_kwargs,
        )

    codec_cfg = _codec_config(model_config)
    lang_key = _LANG_ALIASES.get(
        str(sample.get("target_lang") or cfg.target_lang or "").strip().lower(),
        str(sample.get("target_lang") or cfg.target_lang or "").strip().lower(),
    )
    lang_id = codec_cfg["codec_language_id"].get(lang_key)
    if lang_id is not None:
        think_part = [
            codec_cfg["codec_think_id"], codec_cfg["codec_think_bos_id"],
            int(lang_id), codec_cfg["codec_think_eos_id"],
        ]
    else:
        think_part = [
            codec_cfg["codec_nothink_id"], codec_cfg["codec_think_bos_id"],
            codec_cfg["codec_think_eos_id"],
        ]
    codec_mask_id = int(getattr(model_config, "codec_mask_id", codec_cfg["codec_pad_id"]))
    prefix = ([codec_mask_id] * t_prompt if layout_full_add else []) + think_part
    prefix += [codec_cfg["codec_pad_id"], codec_cfg["codec_bos_id"]]

    gt_codes = torch.as_tensor(sample["target_codec_codes"], dtype=torch.long)
    expected_groups = codec_cfg["num_residual_groups"] + 1
    if gt_codes.ndim != 2:
        raise ValueError(f"target_codec_codes must be 2-D, got {tuple(gt_codes.shape)}")
    if gt_codes.shape[1] != expected_groups and gt_codes.shape[0] == expected_groups:
        gt_codes = gt_codes.transpose(0, 1).contiguous()
    if gt_codes.shape[1] != expected_groups:
        raise ValueError(
            f"expected [T,{expected_groups}] target_codec_codes, got {tuple(gt_codes.shape)}")
    fields = _build_training_tensors(
        gt_codes,
        control_prefix_ids=prefix,
        codec_eos_id=codec_cfg["codec_eos_id"],
        num_residual_groups=codec_cfg["num_residual_groups"],
        residual_vocab_size=codec_cfg["residual_vocab_size"],
    )
    codec_input_ids = fields["codec_input_ids"].to(device).unsqueeze(0)
    codec_labels = fields["codec_labels"].to(device).unsqueeze(0)
    residual_labels = fields["codec_residual_labels"].to(device).unsqueeze(0)

    with torch.inference_mode():
        out = model(
            thinker_text_embeds=thinker_out["text_embeds"],
            thinker_hidden_states=thinker_out["hidden_states"],
            thinker_attention_mask=thinker_mask,
            thinker_text_labels=thinker_text_labels,
            codec_input_ids=codec_input_ids,
            codec_labels=codec_labels,
            codec_residual_labels=residual_labels,
            codec_attention_mask=torch.ones_like(codec_input_ids),
            return_dict=True,
        )

    eos_id = codec_cfg["codec_eos_id"]
    main_pred_all = out.main_codec_logits.argmax(dim=-1)
    valid = codec_labels != -100
    keep = valid & (codec_labels != eos_id)
    main_pred = main_pred_all[keep].detach().cpu().numpy().astype(np.int64)
    if out.residual_codec_logits is None:
        residual_pred = np.zeros(
            (len(main_pred), codec_cfg["num_residual_groups"]), dtype=np.int64)
        residual_acc = None
    else:
        residual_pred_all = out.residual_codec_logits.argmax(dim=-1)
        residual_pred = residual_pred_all[keep].detach().cpu().numpy().astype(np.int64)
        residual_valid = residual_labels != -100
        residual_correct = int(
            (residual_pred_all[residual_valid] == residual_labels[residual_valid]).sum().item()
        ) if residual_valid.any() else 0
        residual_total = int(residual_valid.sum().item())
        residual_acc = float(
            residual_correct / residual_total
        ) if residual_total else None
    pred_codes = np.concatenate([main_pred[:, None], residual_pred], axis=1)
    tf_path = None
    tf_audio_ms = None
    if decode_audio:
        codec_tok = get_codec_tokenizer(tokenizer_id=cfg.codec_tokenizer, device_map="cpu")
        wav, sr = decode_codec(codec_tok, pred_codes)
        tf_path = os.path.join(out_dir, "audios", f"{sid}_tf.wav")
        save_audio(Path(tf_path), wav, sr)
        tf_audio_ms = round(1000.0 * len(wav) / sr, 1) if sr > 0 else None

    # Match the template/Swift ``token_acc`` stream: acoustic main frames only;
    # the final EOS target is reported separately below.
    main_acc = float(
        (main_pred_all[keep] == codec_labels[keep]).float().mean().item()
    ) if keep.any() else None
    main_correct = int((main_pred_all[keep] == codec_labels[keep]).sum().item())
    main_total = int(keep.sum().item())
    eos_mask = codec_labels == eos_id
    eos_correct = int(((main_pred_all == eos_id) & eos_mask).sum().item())
    return {
        "tf_audio": tf_path,
        "tf_text_source": text_source,
        "tf_codec_frames": int(len(main_pred)),
        "tf_audio_ms": tf_audio_ms,
        "tf_main_token_acc": round(main_acc, 4) if main_acc is not None else None,
        "tf_main_token_correct": main_correct,
        "tf_main_token_total": main_total,
        "tf_residual_token_acc": round(residual_acc, 4) if residual_acc is not None else None,
        "tf_residual_token_correct": residual_correct if out.residual_codec_logits is not None else 0,
        "tf_residual_token_total": residual_total if out.residual_codec_logits is not None else 0,
        "tf_eos_correct": eos_correct,
        "tf_eos_total": int(eos_mask.sum().item()),
    }


# --------------------------------------------------------------------------- #
# 译文质量（字符级相似度、长度比、BLEU）
# --------------------------------------------------------------------------- #
def text_metrics(pred: str, target: str):
    pred = (pred or "").strip()
    target = (target or "").strip()
    if not target:
        return {"char_similarity": None, "len_ratio": None}
    sim = difflib.SequenceMatcher(None, pred, target).ratio()
    len_ratio = (len(pred) / len(target)) if target else None
    return {"char_similarity": round(sim, 4),
            "len_ratio": round(len_ratio, 3) if len_ratio is not None else None}


def _bleu_tokens(text: str, target_lang: str):
    """Language-aware tokens for the mixed English/Chinese validation set."""
    text = (text or "").strip()
    lang = (target_lang or "").lower()
    if lang.startswith("zh"):
        # CJK characters are individual tokens; Latin spans and punctuation are
        # kept as tokens. This is the usual character-oriented Chinese BLEU.
        return re.findall(
            r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:['.\u2019-][A-Za-z0-9]+)*|[^\s]",
            text,
        )
    from nltk.tokenize import wordpunct_tokenize
    return wordpunct_tokenize(text)


def corpus_bleu_metrics(rows):
    """Return corpus BLEU×100 overall and per target language, without smoothing."""
    from nltk.translate.bleu_score import corpus_bleu

    usable = [r for r in rows if r.get("target_text") and r.get("pred_text") is not None]

    def score(group):
        refs = [[_bleu_tokens(r["target_text"], r.get("target_lang", ""))] for r in group]
        hyps = [_bleu_tokens(r["pred_text"], r.get("target_lang", "")) for r in group]
        return round(float(corpus_bleu(refs, hyps)) * 100.0, 4) if group else None

    langs = sorted({str(r.get("target_lang") or "unknown").lower() for r in usable})
    return {
        "bleu": score(usable),
        "bleu_by_target_lang": {
            lang: score([r for r in usable if str(r.get("target_lang") or "unknown").lower() == lang])
            for lang in langs
        },
        "bleu_tokenization": "English=NLTK wordpunct; Chinese=CJK-character; case-sensitive; corpus BLEU-4; no smoothing",
    }


def _direction(row):
    """en->zh / zh->en 方向标签。"""
    src = str(row.get("source_lang") or "?").lower()[:2]
    tgt = str(row.get("target_lang") or "?").lower()[:2]
    return f"{src}2{tgt}"


def direction_metrics(rows):
    """按翻译方向(en2zh / zh2en)分组汇总 BLEU/WER-CER/SIM/MOS。"""
    try:
        from inference import tts_metrics as TM
    except Exception:
        import tts_metrics as TM
    ok = [r for r in rows if "error" not in r]
    dirs = sorted({_direction(r) for r in ok})
    out = {}
    for d in dirs:
        g = [r for r in ok if _direction(r) == d]
        tgt_lang = "zh" if d.endswith("zh") else "en"
        # WER/CER：同方向目标语言一致
        asr_vals = [r["asr_metric"] for r in g if r.get("asr_metric") is not None]
        metric_name = "CER" if tgt_lang == "zh" else "WER"
        out[d] = {
            "n": len(g),
            "BLEU": TM.corpus_bleu([r.get("pred_text", "") for r in g],
                                   [r.get("target_text", "") for r in g], tgt_lang),
            f"{metric_name}": TM.safe_mean(asr_vals),
            "SIM": TM.safe_mean([r.get("sim") for r in g]),
            "MOS": TM.safe_mean([r.get("mos") for r in g]),
            "eos_rate": TM.safe_mean([1.0 if r.get("ar_eos_triggered") else 0.0
                                      for r in g if "ar_eos_triggered" in r]),
            "avg_ar_frames": TM.safe_mean([r.get("ar_codec_frames") for r in g]),
        }
    return {"by_direction": out}


# --------------------------------------------------------------------------- #
# 评测一个 branch
# --------------------------------------------------------------------------- #
def run_branch(branch: str, ckpt: str, val_jsonl: str, args):
    # 关键：按 branch 设置 layout env（必须在 load 模型/生成前）
    layout_full_add = BRANCHES[branch]["full_add"]
    os.environ["QWEN_OMNISPEAK_FULL_ADD_LAYOUT"] = "true" if layout_full_add else "false"
    # t2s 无源音频，强制关闭 speaker prompt
    os.environ["QWEN_OMNISPEAK_USE_SPEAKER_PROMPT"] = "false"

    from inference.common import (
        InferenceConfig, InferenceSample, load_model_and_processor,
        generate_text, generate_audio, decode_codec, save_audio, get_codec_tokenizer,
    )
    # TTS 指标模块（BLEU/WER-CER/SIM/MOS）；--no-tts-metrics 可关闭
    TM = None
    if not args.no_tts_metrics:
        try:
            from inference import tts_metrics as TM
        except Exception:
            import tts_metrics as TM

    out_dir = os.path.join(args.out_dir, branch)
    os.makedirs(os.path.join(out_dir, "audios"), exist_ok=True)

    samples = load_t2s_samples(val_jsonl, args.n)
    print(f"\n=== [{branch}] layout_full_add={layout_full_add} ===")
    print(f"  ckpt: {ckpt}")
    print(f"  val : {val_jsonl}  ({len(samples)} samples)")

    cfg = InferenceConfig(
        model=ckpt, device=args.device, dtype="bf16", output_dir=out_dir,
        max_new_tokens=args.max_new_tokens, max_codec_frames=args.max_codec_frames,
        temperature=0.0, top_k=0, target_lang=args.target_lang,
        codec_tokenizer=os.environ["QWEN3_TTS_TOKENIZER"], codec_tokenizer_device="cpu",
    )
    model, processor, model_config, device, _ = load_model_and_processor(cfg)
    model.eval()
    # SIM/MOS 需要 GT 波形，因此只要开了 TTS 指标就必须解码 GT
    need_gt_wav = (TM is not None)
    codec_tok = get_codec_tokenizer(
        tokenizer_id=cfg.codec_tokenizer, device_map="cpu") \
        if (not args.metrics_only or need_gt_wav) else None

    results = []
    for i, s in enumerate(samples):
        sid = f"{branch}_{i}"
        tgt_lang = s.get("target_lang") or args.target_lang
        print(f"\n--- [{branch}] sample {i} (id={s['id']}) ---")
        print(f"    TARGET_TEXT: {s['target_text'][:100]}")
        row = {"idx": i, "id": s["id"], "target_text": s["target_text"],
               "source_lang": s.get("source_lang"), "target_lang": tgt_lang}
        try:
            inputs = prepare_t2s_inputs(processor, s["messages"], device)

            # 1) thinker 自回归译文
            t_sample = InferenceSample(sid, "", None, "t2s", "text", tgt_lang)
            t0 = time.perf_counter()
            t_res = generate_text(model, processor, inputs, t_sample, cfg, stream=False)
            row["pred_text"] = t_res.text.strip()
            row["text_tokens"] = t_res.metrics.generated_text_tokens
            row["text_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            row.update(text_metrics(row["pred_text"], s["target_text"]))
            print(f"    PRED_TEXT:   {row['pred_text'][:100]}")
            print(f"    char_sim={row['char_similarity']} len_ratio={row['len_ratio']}")

            # 2) teacher forcing：GT codec 作输入，logits argmax 解码。
            #    target text 口径与 Trainer validation 完全一致；pred text
            #    口径复现 Batch-4 的“talker-only TF”诊断。
            if args.teacher_forced or args.full_eval:
                tf = teacher_forced_audio(
                    model, processor, model_config, device, cfg, s, inputs, t_res.token_ids,
                    layout_full_add=layout_full_add,
                    text_source=args.tf_text_source,
                    out_dir=out_dir,
                    sid=sid,
                    decode_audio=not args.metrics_only,
                )
                row.update(tf)
                print(
                    f"    -> TF({args.tf_text_source} text): {tf['tf_codec_frames']} frames, "
                    f"main_acc={tf['tf_main_token_acc']}, "
                    f"resid_acc={tf['tf_residual_token_acc']}, "
                    f"eos={tf['tf_eos_correct']}/{tf['tf_eos_total']}"
                )

            # BLEU（逐样本，翻译质量）
            row["bleu"] = TM.compute_bleu(row["pred_text"], s["target_text"], tgt_lang) if TM else None

            # 3) talker 自回归音频（传 thinker AR 译文 token 作为 reply 段）
            ar_wav = ar_sr = None
            if not args.teacher_forced or args.full_eval:
                a_sample = InferenceSample(sid, "", None, "t2s", "audio", tgt_lang)
                a_res = generate_audio(
                    model, processor, model_config, inputs, a_sample, cfg,
                    stream=False, realtime_schedule=False,
                    text_token_ids=t_res.token_ids, use_speaker=False,
                )
                ar_wav, ar_sr = a_res.audio, a_res.sample_rate
                ar_path = os.path.join(out_dir, "audios", f"{sid}_ar.wav")
                save_audio(Path(ar_path), ar_wav, ar_sr)
                frames = a_res.metrics.generated_codec_frames
                row["ar_audio"] = ar_path
                row["ar_codec_frames"] = frames
                row["ar_audio_ms"] = round(a_res.metrics.output_audio_duration_ms or 0, 1)
                row["ar_rtf"] = round(a_res.metrics.rtf or 0, 3)
                # EOS 触发 = 没有跑满上限
                row["ar_eos_triggered"] = bool(frames < args.max_codec_frames)
                print(f"    -> AR: {frames} frames, {row['ar_audio_ms']:.0f}ms, "
                      f"eos={row['ar_eos_triggered']}, rtf={row['ar_rtf']}")

            # 4) GT 参考音频（target_codec_codes 直接解码）
            gt_wav = gt_sr = None
            tc = s.get("target_codec_codes")
            if tc is not None:
                gt_codes = np.asarray(tc, dtype=np.int64)
                row["gt_codec_frames"] = int(gt_codes.shape[0])
                if (not args.metrics_only) or need_gt_wav:
                    gt_wav, gt_sr = decode_codec(codec_tok, gt_codes)
                    if not args.metrics_only:
                        gt_path = os.path.join(out_dir, "audios", f"{sid}_gt.wav")
                        save_audio(Path(gt_path), gt_wav, gt_sr)
                        row["gt_audio"] = gt_path
                    row["gt_audio_ms"] = round(1000.0 * len(gt_wav) / gt_sr, 1) if gt_sr > 0 else None
                    print(f"    -> GT: {row['gt_codec_frames']} frames, {row.get('gt_audio_ms')}ms")

            # 5) TTS 音频指标：WER/CER（ASR 转写 AR 音频）、SIM、MOS
            if TM and ar_wav is not None:
                try:
                    asr_text = TM.asr_transcribe(ar_wav, ar_sr, tgt_lang, args.asr_model, args.device)
                    row["asr_text"] = asr_text
                    mname, mval = TM.compute_wer_cer(asr_text, s["target_text"], tgt_lang)
                    row["asr_metric_name"] = mname
                    row["asr_metric"] = mval
                    print(f"    -> ASR({tgt_lang}): {mname}={mval}  '{asr_text[:60]}'")
                except Exception as _e:
                    row["asr_error"] = str(_e)
                if gt_wav is not None:
                    try:
                        row["sim"] = TM.compute_sim(ar_wav, ar_sr, gt_wav, gt_sr, args.device)
                        row["mos"] = TM.compute_mos(ar_wav, ar_sr, gt_wav, gt_sr, args.device)
                        print(f"    -> SIM={row['sim']}  MOS={row['mos']}")
                    except Exception as _e:
                        row["sim_mos_error"] = str(_e)
        except Exception as e:
            import traceback
            traceback.print_exc()
            row["error"] = str(e)
        results.append(row)

    # 汇总统计
    ok = [r for r in results if "error" not in r]
    main_correct = sum(r.get("tf_main_token_correct", 0) for r in ok)
    main_total = sum(r.get("tf_main_token_total", 0) for r in ok)
    residual_correct = sum(r.get("tf_residual_token_correct", 0) for r in ok)
    residual_total = sum(r.get("tf_residual_token_total", 0) for r in ok)
    summary = {
        "branch": branch,
        "layout_full_add": layout_full_add,
        "ckpt": ckpt,
        "val_jsonl": val_jsonl,
        "n": len(results),
        "n_ok": len(ok),
        "avg_char_similarity": round(float(np.mean([r["char_similarity"] for r in ok
                                                    if r.get("char_similarity") is not None])), 4) if ok else None,
        "avg_ar_codec_frames": (
            round(float(np.mean([r["ar_codec_frames"] for r in ok if "ar_codec_frames" in r])), 1)
            if any("ar_codec_frames" in r for r in ok) else None
        ),
        "eos_trigger_rate": (
            round(float(np.mean([1.0 if r.get("ar_eos_triggered") else 0.0
                                 for r in ok if "ar_eos_triggered" in r])), 3)
            if any("ar_eos_triggered" in r for r in ok) else None
        ),
        "avg_tf_main_token_acc": (
            round(float(np.mean([r["tf_main_token_acc"] for r in ok
                                 if r.get("tf_main_token_acc") is not None])), 4)
            if any(r.get("tf_main_token_acc") is not None for r in ok) else None
        ),
        "avg_tf_residual_token_acc": (
            round(float(np.mean([r["tf_residual_token_acc"] for r in ok
                                 if r.get("tf_residual_token_acc") is not None])), 4)
            if any(r.get("tf_residual_token_acc") is not None for r in ok) else None
        ),
        "tf_main_token_acc_micro": round(main_correct / main_total, 6) if main_total else None,
        "tf_main_token_correct": main_correct,
        "tf_main_token_total": main_total,
        "tf_residual_token_acc_micro": (
            round(residual_correct / residual_total, 6) if residual_total else None
        ),
        "tf_residual_token_correct": residual_correct,
        "tf_residual_token_total": residual_total,
        "results": results,
    }
    summary.update(corpus_bleu_metrics(ok))
    summary.update(direction_metrics(ok))
    summary_path = os.path.join(args.out_dir, f"summary_{branch}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[{branch}] summary -> {summary_path}")
    print(f"  ok={len(ok)}/{len(results)}  avg_char_sim={summary['avg_char_similarity']}  "
          f"avg_ar_frames={summary['avg_ar_codec_frames']}  eos_rate={summary['eos_trigger_rate']}  "
          f"BLEU={summary['bleu']}  tf_main_micro={summary['tf_main_token_acc_micro']}  "
          f"tf_resid_micro={summary['tf_residual_token_acc_micro']}")
    # 按方向的核心指标表
    for d, m in summary.get("by_direction", {}).items():
        key = "CER" if d.endswith("zh") else "WER"
        print(f"  [{d}] n={m['n']} BLEU={m['BLEU']} {key}={m.get(key)} "
              f"SIM={m['SIM']} MOS={m['MOS']} eos={m['eos_rate']}")

    # 释放显存，便于 --all-branches 连续评测
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="t2s (text->speech) translation evaluation")
    p.add_argument("--branch", choices=list(BRANCHES.keys()),
                   help="which branch to evaluate (auto-locates ckpt & val)")
    p.add_argument("--all-branches", action="store_true", help="evaluate all four branches")
    p.add_argument("--ckpt", help="override checkpoint dir (single-branch only)")
    p.add_argument("--val-jsonl", help="override val_dataset.jsonl (single-branch only)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=os.path.join(REPO, "exp_results", "batch5.0_t2s"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-codec-frames", type=int, default=384)
    p.add_argument("--target-lang", default="zh")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--teacher-forced", action="store_true",
                      help="write GT + teacher-forced audio (skip AR audio)")
    mode.add_argument("--full-eval", action="store_true",
                      help="write all three audios: GT + teacher-forced + autoregressive")
    p.add_argument(
        "--metrics-only", action="store_true",
        help="skip TF/GT waveform decoding and writing; use with --teacher-forced for fast metrics",
    )
    p.add_argument(
        "--tf-text-source", choices=("target", "pred"), default="target",
        help=("thinker text used by teacher-forced audio: target exactly matches Trainer val; "
              "pred reproduces the older Batch-4 talker-only TF diagnostic"),
    )
    p.add_argument("--asr-model", default="openai/whisper-large-v3",
                   help="Whisper ASR model for WER/CER (default whisper-large-v3)")
    p.add_argument("--no-tts-metrics", action="store_true",
                   help="disable WER/CER/SIM/MOS (only text char_sim + BLEU + frames)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.branch and not args.all_branches:
        raise SystemExit("Specify --branch <name> or --all-branches")

    branches = list(BRANCHES.keys()) if args.all_branches else [args.branch]
    all_summaries = {}
    for br in branches:
        if args.all_branches:
            ckpt, val = resolve_ckpt_and_val(br)
        else:
            ckpt = args.ckpt or resolve_ckpt_and_val(br)[0]
            val = args.val_jsonl or resolve_ckpt_and_val(br)[1]
        if not ckpt or not os.path.isdir(ckpt):
            print(f"[SKIP] {br}: checkpoint not found ({ckpt})")
            continue
        if not val or not os.path.isfile(val):
            print(f"[SKIP] {br}: val_dataset.jsonl not found ({val})")
            continue
        all_summaries[br] = run_branch(br, ckpt, val, args)

    # 跨 branch 对比表
    if all_summaries:
        print("\n================= t2s branch 对比 =================")
        print(f"{'branch':<24} {'char_sim':>9} {'ar_frames':>10} {'eos_rate':>9} "
              f"{'tf_main':>9} {'tf_resid':>9}")
        for br, s in all_summaries.items():
            print(f"{br:<24} {str(s['avg_char_similarity']):>9} "
                  f"{str(s['avg_ar_codec_frames']):>10} {str(s['eos_trigger_rate']):>9} "
                  f"{str(s['avg_tf_main_token_acc']):>9} {str(s['avg_tf_residual_token_acc']):>9}")
        # A single-branch process must not overwrite a shared summary_all.json;
        # parallel per-GPU evaluation otherwise races and leaves only one branch.
        if args.all_branches:
            combined = os.path.join(args.out_dir, "summary_all.json")
            with open(combined, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, ensure_ascii=False, indent=2)
            print(f"\ncombined summary -> {combined}")


if __name__ == "__main__":
    main()
