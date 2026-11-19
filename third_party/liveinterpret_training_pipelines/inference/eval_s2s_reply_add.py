#!/usr/bin/env python3
"""Batch-4.3 s2s eval — REPLY-ADD branch.

talker 输入模式 = reply_add（与 `feature/talker-reply-add` / TALKER_METHOD=reply_add 训练对齐）：
  仅 reply 段的 thinker hidden+embed 经 adapter+text_projection 后与 codec 逐位相加，
  reply 之前填 tts_pad_embed。control_prefix **不含** codec_mask 前缀（只有 think/lang, (spk), pad, bos）。
  QWEN_OMNISPEAK_FULL_ADD_LAYOUT=false。

用法（默认 5 样本、full-eval=每样本 GT+TF+AR 三条音频）：
  python inference/eval_s2s_reply_add.py \
      --ckpt /path/to/reply_add/checkpoint-15805_clean \
      --val-jsonl /path/to/val_dataset.jsonl \
      --n 5 --full-eval --device cuda:0 --out-dir exp_results/batch4.3
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom", "qwen_omnispeak"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QWEN3_TTS_TOKENIZER", os.path.join(REPO, "models", "Qwen3-TTS-Tokenizer-12Hz"))
os.environ.setdefault("QWEN3_TTS_SPEAKER_ENCODER", os.path.join(REPO, "project", "Qwen3-TTS-12Hz-1.7B-Base"))

from inference.eval_batch4_autoregressive import load_samples, run_model  # noqa: E402

LAYOUT = "reply_add"
TAG = "reply_add"


def parse_args():
    p = argparse.ArgumentParser(description="s2s eval — reply_add branch")
    p.add_argument("--ckpt", required=True, help="reply_add checkpoint dir (clean, loadable as model)")
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=os.path.join(REPO, "exp_results/batch4.3"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-codec-frames", type=int, default=512)
    p.add_argument("--target-lang", default="zh")
    p.add_argument("--full-eval", action="store_true",
                   help="per sample: GT decode + teacher-forced + autoregressive (3 audios)")
    p.add_argument("--teacher-forced", action="store_true")
    p.add_argument("--no-speaker", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.val_jsonl, args.n)
    print(f"[reply_add] Loaded {len(samples)} eval samples from {args.val_jsonl}")
    results = run_model(TAG, args.ckpt, samples, args, layout=LAYOUT)
    summary_path = os.path.join(args.out_dir, f"summary_{TAG}.json")
    with open(summary_path, "w") as f:
        json.dump({TAG: results}, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")
    ok = [r for r in results if "error" not in r]
    print(f"  {TAG}: {len(ok)}/{len(results)} ok")


if __name__ == "__main__":
    main()
