#!/usr/bin/env python3
"""Stage 1 overfit probe: does the inferred talker supervision window hold?

The talker label placement in `src/train/collators/offline.py` is inferred, not
documented. The inference path never constructs labels, so no golden reference
exists for it. This probe converts that unobservable risk into a signal that
shows up in minutes.

Method: take a handful of utterances and train the talker on them repeatedly,
thinker frozen. If the supervision window is right, the talker can memorise them
and `audio_loss` collapses toward zero. If labels sit even one position off, the
task becomes predicting an unrelated token and the loss plateaus.

Stage 1 freezes the thinker, so `text_loss` is skipped entirely
(`modeling_omni_talker.py:475`) and the probe isolates the talker mask alone.

Also reports teacher-forced code accuracy, which is more interpretable than the
loss value: a correct window should reach near-100% on a memorised set.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

_CUSTOM_DIR = Path(__file__).resolve().parents[2]  # custom/：`import simuls2st_omni` 的根
if str(_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(_CUSTOM_DIR))
# 数据 / 模型的相对路径与源项目一样相对「数据根」解析；默认仓库根，可用 SIMULS2ST_ROOT 指到别处
REPO_ROOT = Path(os.environ.get("SIMULS2ST_ROOT") or _CUSTOM_DIR.parent)

MODEL_ROOT = Path(
    os.environ.get("SIMULS2ST_MODEL_ROOT", REPO_ROOT / "models" / "SimulS2ST-Omni")
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("overfit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "emilia" / "dualcodec",
    )
    parser.add_argument(
        "--model-path", type=str, default=str(MODEL_ROOT / "offline"),
        help="base weights; the talker codec head is re-initialised regardless",
    )
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--max-codes", type=int, default=120)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    from simuls2st_omni.train.collators.offline import OfflineTTSCollator
    from simuls2st_omni.train.datasets.tts import TTSCodesDataset
    from simuls2st_omni.train.model_init import load_base_model
    from simuls2st_omni.train.modeling_dual_head import IGNORE_INDEX
    from simuls2st_omni.train.modeling_omni_talker import OmniTalkerQwenOmni

    dataset = TTSCodesDataset(
        root=args.data_root,
        langs=["EN"],
        max_shards=1,
        max_samples=args.n_samples * 20,
        max_codes=args.max_codes,
    )
    indices = list(range(min(args.n_samples, len(dataset))))
    logger.info("probing on %d samples", len(indices))

    logger.info("loading base model from %s", args.model_path)
    base_model, processor, codec_tokens = load_base_model(
        args.model_path, dtype=torch.bfloat16, device_map=args.device
    )
    logger.info("codec tokens: %s", codec_tokens)

    model = OmniTalkerQwenOmni(base_model, freeze_thinker=True, freeze_talker=False)
    collator = OfflineTTSCollator(
        tokenizer=processor.tokenizer,
        codec_pad=codec_tokens["codec_pad"],
        codec_bos=codec_tokens["codec_bos"],
        codec_eos=codec_tokens["codec_eos"],
    )

    trainable = [p for p in model.talker.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("trainable talker params: %s", f"{n_trainable:,}")
    assert n_trainable > 0, "talker has no trainable parameters"
    assert not any(p.requires_grad for p in model.thinker.parameters()), (
        "thinker must be frozen for this probe"
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    device = torch.device(args.device)

    batches = [
        collator([dataset[i] for i in indices[s : s + args.batch_size]])
        for s in range(0, len(indices), args.batch_size)
    ]
    for batch in batches:
        expected = batch["reply_start_positions"] + 3
        talker_len = batch["talker_attention_mask"].sum(dim=1)
        assert (talker_len >= expected).all(), (
            f"talker sequence shorter than prefix+3: {talker_len} vs {expected}"
        )
        supervised = (batch["talker_labels"] != IGNORE_INDEX).sum(dim=1)
        assert (supervised > 0).all(), "some rows carry no talker supervision"
    logger.info(
        "collator sanity passed: %d batch(es), supervised positions per row = %s",
        len(batches),
        (batches[0]["talker_labels"] != IGNORE_INDEX).sum(dim=1).tolist(),
    )

    model.train()
    history: list[float] = []
    for step in range(1, args.steps + 1):
        batch = batches[(step - 1) % len(batches)]
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
            if k not in {"task_type", "ids"}
        }
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            talker_input_ids=inputs["talker_input_ids"],
            text_labels=None,  # thinker frozen; keeps the probe on the talker
            talker_labels=inputs["talker_labels"],
            talker_attention_mask=inputs["talker_attention_mask"],
            reply_start_positions=inputs["reply_start_positions"],
            reply_lengths=inputs["reply_lengths"],
            return_dict=True,
        )
        loss = outputs.audio_loss
        assert loss is not None, "audio_loss missing; talker_labels not consumed"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(float(loss.detach().float().cpu()))

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                logits = outputs.audio_logits[:, :-1, :]
                targets = inputs["talker_labels"][:, 1:]
                valid = targets != IGNORE_INDEX
                correct = (logits.argmax(dim=-1) == targets) & valid
                acc = float(correct.sum()) / max(int(valid.sum()), 1)
            logger.info(
                "step %4d  audio_loss %.4f  teacher-forced code acc %.3f",
                step,
                history[-1],
                acc,
            )

    first, last = history[0], min(history[-5:])
    logger.info("=" * 68)
    logger.info("initial loss %.4f -> best of final 5 steps %.4f", first, last)
    if last < 0.05:
        verdict = "PASS: talker memorised the set; supervision window is consistent"
    elif last < first * 0.3:
        verdict = "LIKELY OK: loss fell sharply; extend --steps to confirm"
    else:
        verdict = (
            "FAIL: loss did not collapse. Suspect the inferred label placement "
            "in collators/offline.py before training further."
        )
    logger.info(verdict)
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
