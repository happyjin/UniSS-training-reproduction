#!/usr/bin/env python3
"""Batch-4.3 s2s eval — BRIDGE-ADD (bridge-concat) branch.

talker 输入模式 = bridge-concat（与 `feature/talker-bridge-concat` 训练对齐）：
  thinker hidden+embed 的**全部位置**经 DualRoute bridge 投影成 sem_block，与 codec_emb
  **时间轴拼接** cat([sem_block, codec_emb])，codec 通过 block-causal attention 读语义。
  **不使用** upstream 的逐位相加 / codec_mask 前缀 / FULL_ADD_LAYOUT。
  control_prefix = build_control_prefix()（think/lang + pad + bos，无 codec_mask）。

⚠️ 必须在 `feature/talker-bridge-concat` 分支代码环境下运行（本脚本 import 该分支的
   inference.common，其 generate_audio / build_control_prefix / 模型 forward 与 upstream 分支不同）。
   bridge-concat 4.3 仍在训练；本脚本先建好，checkpoint 就绪后即可评。

用法（在 h100-2 bridge-concat 仓库根目录）：
  python inference/eval_s2s_bridge_add.py \
      --ckpt /path/to/bridge/checkpoint-XXXX_clean \
      --val-jsonl /path/to/val_dataset.jsonl \
      --n 5 --full-eval --device cuda:0 --out-dir exp_results/batch4.3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom", "qwen_omnispeak"))

from task_prompts import s2s_task_prompt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QWEN3_TTS_TOKENIZER", os.path.join(REPO, "models", "Qwen3-TTS-Tokenizer-12Hz"))
os.environ.setdefault("QWEN3_TTS_SPEAKER_ENCODER", os.path.join(REPO, "project", "Qwen3-TTS-12Hz-1.7B-Base"))

TAG = "bridge_add"


def load_samples(val_path, n):
    samples = []
    with open(val_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            audios = d.get("audios")
            if isinstance(audios, str):
                try:
                    audios = json.loads(audios.replace("'", '"'))
                except Exception:
                    audios = [audios]
            target_lang = d.get("target_lang", "zh")
            samples.append({
                "id": d.get("id", f"sample_{i}"),
                "audio": audios[0] if audios else None,
                "task_prompt": s2s_task_prompt(target_lang),
                "target_text": d.get("target_text", ""),
                "target_audio": d.get("target_audio", ""),
                "target_lang": target_lang,
            })
    return samples


def bridge_teacher_forced(model, model_config, device, cfg, s, t_res, out_dir, sid):
    """Bridge teacher forcing: thinker(all positions) -> bridge; feed GT codec, decode pred logits.

    bridge-concat 不切 reply 段：thinker 用 [源音频 prompt + 翻译 reply] 全部位置进 bridge
    （与训练一致，见 branch_comparison.md §1）。codec 用 build_control_prefix + GT target_audio。
    """
    import torch
    import numpy as np
    from inference.common import (prepare_inputs, split_thinker_kwargs, _codec_config,
                                  get_codec_tokenizer, decode_codec, save_audio,
                                  build_control_prefix)

    inputs = s["_inputs"]
    input_ids, attention_mask, thinker_kwargs = split_thinker_kwargs(inputs)

    # thinker: 源音频 + 翻译文本（全部位置进 bridge，bridge-concat 不切 reply）
    reply_ids = torch.tensor([t_res.token_ids], dtype=torch.long, device=device)
    thinker_in_ids = torch.cat([input_ids, reply_ids], dim=1)
    thinker_in_mask = torch.cat([attention_mask, torch.ones_like(reply_ids)], dim=1)
    with torch.inference_mode():
        thinker_out = model.model.run_thinker(
            input_ids=thinker_in_ids, attention_mask=thinker_in_mask,
            output_text_embeds=True, **thinker_kwargs)
        thinker_hidden = thinker_out["hidden_states"]
        thinker_text_embeds = thinker_out["text_embeds"]

    codec_tok = get_codec_tokenizer(tokenizer_id=cfg.codec_tokenizer, device_map="cpu")
    codec_cfg = _codec_config(model_config)
    eos_id = codec_cfg["codec_eos_id"]
    prefix = build_control_prefix(model_config, s.get("target_lang") or cfg.target_lang)
    cf = codec_tok.encode_target_audio(
        s["target_audio"], control_prefix_ids=prefix,
        codec_eos_id=eos_id,
        num_residual_groups=codec_cfg["num_residual_groups"],
        residual_vocab_size=codec_cfg["residual_vocab_size"])
    codec_input_ids = cf["codec_input_ids"].to(device).long().unsqueeze(0)
    codec_labels = cf["codec_labels"].to(device).long().unsqueeze(0)
    codec_residual_labels = cf["codec_residual_labels"].to(device).long().unsqueeze(0)
    codec_attention_mask = torch.ones_like(codec_input_ids)

    with torch.inference_mode():
        out = model(
            thinker_text_embeds=thinker_text_embeds, thinker_hidden_states=thinker_hidden,
            thinker_attention_mask=thinker_in_mask,
            codec_input_ids=codec_input_ids, codec_labels=codec_labels,
            codec_residual_labels=codec_residual_labels, codec_attention_mask=codec_attention_mask,
            return_dict=True)

    preds = out.main_codec_logits.argmax(dim=-1)
    resid_logits = out.residual_codec_logits
    valid = (codec_labels != -100)
    keep = valid & (codec_labels != eos_id)
    main_pred = preds[keep].flatten().cpu().numpy().astype(np.int64)
    if resid_logits is not None:
        resid_pred = resid_logits.argmax(dim=-1)[keep].cpu().numpy().astype(np.int64)
    else:
        resid_pred = np.zeros((len(main_pred), codec_cfg["num_residual_groups"]), dtype=np.int64)
    codes_pred = np.concatenate([main_pred[:, None], resid_pred], axis=1)
    wav, sr = decode_codec(codec_tok, codes_pred)
    tf_path = os.path.join(out_dir, "audios", f"{sid}_tf.wav")
    save_audio(Path(tf_path), wav, sr)

    gt_main = codec_labels[keep].flatten().cpu().numpy().astype(np.int64)
    gt_resid = codec_residual_labels[keep].cpu().numpy().astype(np.int64)
    if gt_resid.ndim == 1:
        gt_resid = gt_resid[:, None]
    gt_codes = np.concatenate([gt_main[:, None], gt_resid], axis=1)
    wav_gt, sr_gt = decode_codec(codec_tok, gt_codes)
    gt_path = os.path.join(out_dir, "audios", f"{sid}_gt.wav")
    save_audio(Path(gt_path), wav_gt, sr_gt)

    correct = ((preds == codec_labels) & valid).sum().float()
    token_acc = (correct / max(1, valid.sum())).item()
    print(f"    [TF] frames={len(main_pred)}, token_acc={token_acc:.4f}")
    return {"tf_audio": tf_path, "gt_audio": gt_path,
            "tf_codec_frames": int(len(main_pred)), "tf_token_acc": round(token_acc, 4)}


def parse_args():
    p = argparse.ArgumentParser(description="s2s eval — bridge_concat branch")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=os.path.join(REPO, "exp_results/batch4.3"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-codec-frames", type=int, default=512)
    p.add_argument("--target-lang", default="zh")
    p.add_argument("--full-eval", action="store_true")
    p.add_argument("--no-speaker", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from inference.common import (InferenceConfig, InferenceSample, load_model_and_processor,
                                  prepare_inputs, generate_text, generate_audio, save_audio)
    out_dir = os.path.join(args.out_dir, TAG)
    os.makedirs(os.path.join(out_dir, "audios"), exist_ok=True)
    samples = load_samples(args.val_jsonl, args.n)
    print(f"[bridge_add] Loaded {len(samples)} samples from {args.val_jsonl}")

    cfg = InferenceConfig(
        model=args.ckpt, device=args.device, dtype="bf16", output_dir=out_dir,
        max_new_tokens=args.max_new_tokens, max_codec_frames=args.max_codec_frames,
        temperature=0.0, top_k=0, target_lang=args.target_lang,
        codec_tokenizer=os.environ["QWEN3_TTS_TOKENIZER"], codec_tokenizer_device="cpu")
    model, processor, model_config, device, _ = load_model_and_processor(cfg)
    model.eval()

    results = []
    for i, s in enumerate(samples):
        sid = f"{TAG}_{i}"
        print(f"\n--- [{TAG}] sample {i} ---\n    TARGET_TEXT: {s['target_text'][:120]}")
        row = {"idx": i, "audio": s["audio"], "target_text": s["target_text"],
               "target_audio": s["target_audio"]}
        try:
            inputs = prepare_inputs(processor, s["task_prompt"], s["audio"], device)
            s["_inputs"] = inputs
            t_sample = InferenceSample(sid, s["task_prompt"], s["audio"], "s2s", "text", args.target_lang)
            t0 = time.perf_counter()
            t_res = generate_text(model, processor, inputs, t_sample, cfg, stream=False)
            row["pred_text"] = t_res.text.strip()
            row["text_tokens"] = t_res.metrics.generated_text_tokens
            row["text_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            print(f"    PRED_TEXT:   {row['pred_text'][:120]}")

            if args.full_eval:
                r = bridge_teacher_forced(model, model_config, device, cfg, s, t_res, out_dir, sid)
                row.update(r)
                a_sample = InferenceSample(sid, s["task_prompt"], s["audio"], "s2s", "audio", args.target_lang)
                a_res = generate_audio(model, processor, model_config, inputs, a_sample, cfg,
                                       stream=False, realtime_schedule=False)
                ar_path = os.path.join(out_dir, "audios", f"{sid}_ar.wav")
                save_audio(Path(ar_path), a_res.audio, a_res.sample_rate)
                row["ar_audio"] = ar_path
                row["ar_codec_frames"] = a_res.metrics.generated_codec_frames
                row["ar_audio_ms"] = round(a_res.metrics.output_audio_duration_ms or 0, 1)
                row["ar_rtf"] = round(a_res.metrics.rtf or 0, 3)
                print(f"    -> AR: {row['ar_codec_frames']} frames, {row['ar_audio_ms']:.0f}ms, saved {ar_path}")
            else:
                a_sample = InferenceSample(sid, s["task_prompt"], s["audio"], "s2s", "audio", args.target_lang)
                a_res = generate_audio(model, processor, model_config, inputs, a_sample, cfg,
                                       stream=False, realtime_schedule=False)
                ar_path = os.path.join(out_dir, "audios", f"{sid}_ar.wav")
                save_audio(Path(ar_path), a_res.audio, a_res.sample_rate)
                row["ar_audio"] = ar_path
                row["ar_codec_frames"] = a_res.metrics.generated_codec_frames
                print(f"    -> AR: {row['ar_codec_frames']} frames, saved {ar_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            row["error"] = str(e)
        results.append(row)

    summary_path = os.path.join(args.out_dir, f"summary_{TAG}.json")
    with open(summary_path, "w") as f:
        json.dump({TAG: results}, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")
    ok = [r for r in results if "error" not in r]
    print(f"  {TAG}: {len(ok)}/{len(results)} ok")


if __name__ == "__main__":
    main()
