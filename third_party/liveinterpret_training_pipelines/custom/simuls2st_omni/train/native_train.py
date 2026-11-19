#!/usr/bin/env python3
"""Single training entry point for all three stages.

Stage differences live entirely in `src/train/stage_policy.py` and the YAML
config; this script only wires data, collator, sampler and trainer together.

    python scripts/train.py --config configs/stage1.yaml
    torchrun --nproc_per_node 8 scripts/train.py --config configs/stage1.yaml

Two structural constraints come from `QwenOmniTalkerTrainer`:

* It passes `batch_sampler=` to the DataLoader, so the sampler shards for DDP
  itself and must expose `__len__` for the trainer's step accounting.
* A batch carries a single `task_type` and label validity is judged per batch, so
  task mixing happens between batches, never inside one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

_CUSTOM_DIR = Path(__file__).resolve().parents[2]  # custom/：`import simuls2st_omni` 的根
if str(_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(_CUSTOM_DIR))
# 配置里的相对路径（models/、data/、outputs/）相对「数据根」解析：源项目是仓库根，
# 这里默认本仓库根，可用 SIMULS2ST_ROOT 指到数据所在目录（集群上即 /nvme/<前缀>）。
REPO_ROOT = Path(os.environ.get("SIMULS2ST_ROOT") or _CUSTOM_DIR.parent)

# --- CPU thread discipline (set before heavy imports) -----------------------
# Without these caps every spawned process (8 ranks x 8 dataloader workers plus
# the ranks themselves) builds its own default OMP pool, sized to all 192
# cores, and OMP spin-waits on it. 72 processes x 192 threads = ~13.8k threads
# contending for 192 cores; measured on this machine as load 270/192 with GPU
# util collapsing into 15-35% troughs. Each worker needs at most a few threads for
# tokenization and mel extraction, so 4 is plenty.
#
# Enforced ONLY via env vars: children inherit them at fork and size their
# pools at first use. Calling torch.set_num_threads inside a forked child
# instead deadlocks the OMP runtime (libgomp pool re-init post-fork), and
# OMP_WAIT_POLICY=PASSIVE changes the fork-inherited wait primitives -- both
# were measured to hang workers at 0% CPU inside the first fbank call.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("train")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert "stage" in config, "config must declare `stage`"
    return config


def resolve_path(value: str) -> str:
    """Interpret relative paths against the repo root, not the caller's cwd."""
    path = Path(value)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def build_tts_dataset(cfg: dict, defaults: dict | None = None, sample_seed: int = 0):
    """One TTSCodesDataset; `defaults` fills filters a source does not set."""
    from simuls2st_omni.train.datasets.tts import TTSCodesDataset

    merged = {**(defaults or {}), **cfg}
    return TTSCodesDataset(
        root=resolve_path(merged["root"]),
        langs=merged.get("langs"),
        max_shards=int(merged.get("max_shards", 0)),
        max_samples=int(merged.get("max_samples", 0)),
        max_hours=float(merged.get("max_hours", 0.0)),
        sample_seed=sample_seed,
        min_codes=int(merged.get("min_codes", 1)),
        max_codes=int(merged.get("max_codes", 1000)),
        min_dnsmos=float(merged.get("min_dnsmos", 0.0)),
    )


def concat_sources(task: str, parts: list, cfgs: list[dict]):
    from simuls2st_omni.train.datasets.concat import ConcatSourcesDataset

    dataset = ConcatSourcesDataset(
        parts,
        repeats=[int(c.get("repeat", 1)) for c in cfgs],
        names=[c.get("name", str(i)) for i, c in enumerate(cfgs)],
    )
    logger.info("%s sources: %s", task, dataset.describe())
    return dataset


def build_dataset(data_cfg: dict, processor=None, split: str = "train"):
    kind = data_cfg.get("kind", "tts")

    if kind == "tts":
        # `sources:` mixes several TTS roots (e.g. zh/en and ja/ko) with
        # per-source filters and `repeat`; top-level keys act as defaults.
        if "sources" in data_cfg:
            defaults = {k: v for k, v in data_cfg.items() if k != "sources"}
            cfgs = data_cfg["sources"]
            dataset = concat_sources(
                "tts", [build_tts_dataset(c, defaults) for c in cfgs], cfgs
            )
        else:
            dataset = build_tts_dataset(data_cfg)
        logger.info("%s dataset: %d samples", split, len(dataset))
        return dataset

    if kind == "multitask":
        return build_multitask_datasets(data_cfg, split=split)

    if kind == "streaming":
        from simuls2st_omni.train.datasets.streaming import StreamingS2STDataset

        assert processor is not None, "streaming dataset needs the tokenizer"
        dataset = StreamingS2STDataset(
            root=resolve_path(data_cfg["root"]),
            tokenizer=processor.tokenizer,
            si_root=resolve_path(data_cfg.get("si_root", "")),
            directions=data_cfg.get("directions"),
            latency_multipliers=data_cfg.get("latency_multipliers"),
            max_samples=int(data_cfg.get("max_samples", 0)),
            max_codes=int(data_cfg.get("max_codes", 1500)),
            seed=int(data_cfg.get("seed", 0)),
        )
        logger.info("%s dataset: %d samples", split, len(dataset))
        return dataset

    raise NotImplementedError(f"unknown dataset kind {kind!r}")


def build_multitask_datasets(data_cfg: dict, split: str = "train") -> dict:
    """One dataset per task family, keyed by the task name the sampler mixes on.

    Stage 2 draws a task per step and fills the batch from that task only, so
    each family stays a separate dataset and `ConcatDataset` would only obscure
    which rows belong to which task.
    """
    from simuls2st_omni.train.datasets.manifest import ManifestDataset

    sources = data_cfg["sources"]
    weights = data_cfg.get("weights", {})
    datasets: dict[str, Any] = {}
    # Weight overrides for pools that were split (see ASR substitution below).
    extra_weights: dict[str, float] = {}
    sample_seed = int(data_cfg.get("sample_seed", data_cfg.get("seed", 0)))

    def build_part(task: str, cfg: dict):
        """Returns (dataset, ASR-substitution view or None) for one source."""
        if task == "tts_unpaired":
            return build_tts_dataset(cfg, sample_seed=sample_seed), None
        if task == "s2s_translate":
            from simuls2st_omni.train.datasets.s2st import OfflineS2STDataset

            return OfflineS2STDataset(
                root=resolve_path(cfg["root"]),
                si_root=resolve_path(cfg["si_root"]),
                directions=cfg.get("directions"),
                max_samples=int(cfg.get("max_samples", 0)),
                max_hours=float(cfg.get("max_hours", 0.0)),
                sample_seed=sample_seed,
                max_codes=int(cfg.get("max_codes", 1500)),
            ), None
        roots = {k: resolve_path(v) for k, v in (cfg.get("roots") or {}).items()}
        main = ManifestDataset(
            manifest_dirs=[resolve_path(p) for p in cfg["manifests"]],
            corpus_roots=roots,
            task_type=cfg.get("task_type"),
            max_samples=int(cfg.get("max_samples", 0)),
            max_hours=float(cfg.get("max_hours", 0.0)),
            sample_seed=sample_seed,
            max_duration=float(cfg.get("max_duration", 30.0)),
            seed=sample_seed,
        )
        # ASR substitution is realised as a second view of the same S2TT
        # data with the task forced to ASR, registered as its own pool so
        # the sampler draws whole batches of it. Deciding per sample inside
        # `__getitem__` would mix two task types into one batch, which the
        # collator rejects: `compute_loss` reads a single scalar task.
        asr_prob = float(cfg.get("asr_prob", 0.0))
        if asr_prob <= 0:
            return main, None
        from simuls2st_omni.train.prompt_formats import TASK_S2TT_ASR

        parent_hours = float(cfg.get("max_hours", 0.0))
        asr_hours = float(cfg.get("asr_max_hours", 0.0))
        if asr_hours <= 0 and parent_hours > 0:
            asr_hours = parent_hours * asr_prob / max(1e-9, 1.0 - asr_prob)
        view = ManifestDataset(
            manifest_dirs=[resolve_path(p) for p in cfg["manifests"]],
            corpus_roots=roots,
            task_type=TASK_S2TT_ASR,
            max_samples=int(cfg.get("max_samples", 0)),
            max_hours=asr_hours,
            sample_seed=sample_seed + 1,
            max_duration=float(cfg.get("max_duration", 30.0)),
            require_source_text=True,
            seed=sample_seed + 1,
        )
        return main, view

    for task, weight in weights.items():
        if weight <= 0:
            continue
        cfg = sources.get(task)
        assert cfg, f"task {task!r} has weight {weight} but no source config"

        # `parts:` builds the task pool from several sources (e.g. zh/en and
        # ja/ko), each with its own caps; without it the task is one source.
        part_cfgs = cfg["parts"] if "parts" in cfg else [cfg]
        built_parts = [build_part(task, c) for c in part_cfgs]
        mains = [m for m, _ in built_parts]
        views = [(v, c) for (_, v), c in zip(built_parts, part_cfgs) if v is not None]
        datasets[task] = mains[0] if "parts" not in cfg else concat_sources(task, mains, part_cfgs)
        if views:
            asr_probs = {float(c["asr_prob"]) for _, c in views}
            assert len(asr_probs) == 1, f"{task}: parts must share one asr_prob, got {asr_probs}"
            asr_prob = asr_probs.pop()
            view = views[0][0] if "parts" not in cfg else concat_sources(
                f"{task}__asr", [v for v, _ in views], [c for _, c in views]
            )
            datasets[f"{task}__asr"] = view
            # Split the task's weight so the mix ratio is preserved: the
            # substituted share is `asr_prob` of the original weight.
            extra_weights[f"{task}__asr"] = weight * asr_prob
            extra_weights[task] = weight * (1.0 - asr_prob)
            logger.info(
                "%s dataset[%s__asr]: %d samples (asr_prob=%.2f)",
                split,
                task,
                len(view),
                asr_prob,
            )
        logger.info("%s dataset[%s]: %d samples", split, task, len(datasets[task]))

    assert datasets, "multitask config produced no dataset"
    # Record the effective weights so the sampler sees the split pools.
    effective = {t: float(weights[t]) for t in weights if float(weights[t]) > 0}
    effective.update(extra_weights)
    data_cfg["_effective_weights"] = {
        t: w for t, w in effective.items() if t in datasets and w > 0
    }
    return datasets


def build_collator(data_cfg: dict, processor, codec_tokens: dict, talker):
    kind = data_cfg.get("kind", "tts")
    if kind == "tts":
        from simuls2st_omni.train.collators.offline import OfflineTTSCollator

        return OfflineTTSCollator(
            tokenizer=processor.tokenizer,
            codec_pad=codec_tokens["codec_pad"],
            codec_bos=codec_tokens["codec_bos"],
            codec_eos=codec_tokens["codec_eos"],
        )
    if kind == "multitask":
        from simuls2st_omni.train.collators.multitask import MultiTaskCollator

        return MultiTaskCollator(
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            codec_pad=codec_tokens["codec_pad"],
            codec_bos=codec_tokens["codec_bos"],
            codec_eos=codec_tokens["codec_eos"],
        )
    if kind == "streaming":
        from simuls2st_omni.train.collators.streaming import StreamingCollator

        return StreamingCollator(
            codec_pad=codec_tokens["codec_pad"],
            codec_bos=codec_tokens["codec_bos"],
            codec_eos=codec_tokens["codec_eos"],
            codec_mask=codec_tokens["codec_mask"],
            text_bos=talker.text_bos_token,
            text_eos=talker.text_eos_token,
            text_pad=talker.text_pad_token,
            tokenizer=processor.tokenizer,
            feature_extractor=None,
        )
    raise NotImplementedError(f"no collator for dataset kind {kind!r}")


class MultiTaskConcatDataset(torch.utils.data.Dataset):
    """Flatten per-task datasets into one index space for the Trainer.

    `QwenOmniTalkerTrainer` hands a single dataset to the DataLoader while the
    task sampler yields batches of global indices, so the offsets that map a
    global index back to (task, local index) are recorded here and exposed as
    `task_indices` for the sampler to draw from.
    """

    def __init__(self, datasets: dict):
        self.tasks = sorted(datasets)
        self.datasets = datasets
        self.task_indices: dict[str, list[int]] = {}
        self._lookup: list[tuple[str, int]] = []
        for task in self.tasks:
            start = len(self._lookup)
            size = len(datasets[task])
            self._lookup.extend((task, i) for i in range(size))
            self.task_indices[task] = list(range(start, start + size))

    def __len__(self) -> int:
        return len(self._lookup)

    def __getitem__(self, index: int):
        task, local = self._lookup[index]
        return self.datasets[task][local]

    def cost(self, index: int) -> int:
        task, local = self._lookup[index]
        dataset = self.datasets[task]
        getter = getattr(dataset, "token_length", None) or dataset.code_length
        return getter(local)

    # The Stage 1 sampler calls `code_length`; keep the name working.
    code_length = cost


def build_multitask_sampler(dataset, data_cfg: dict, sampler_cfg: dict, seed: int):
    from simuls2st_omni.train.samplers_multitask import TaskMixBatchSampler

    weights = data_cfg.get("_effective_weights") or {
        t: float(w) for t, w in data_cfg["weights"].items() if float(w) > 0
    }
    weights = {t: float(w) for t, w in weights.items() if float(w) > 0}
    tasks = [t for t in weights if t in dataset.task_indices]
    assert tasks, "no configured task has a dataset"

    task_indices = {t: dataset.task_indices[t] for t in tasks}
    task_lengths = {
        t: [dataset.cost(i) for i in task_indices[t]] for t in tasks
    }
    return TaskMixBatchSampler(
        task_indices=task_indices,
        task_lengths=task_lengths,
        task_weights={t: weights[t] for t in tasks},
        token_budget=int(sampler_cfg.get("token_budget", 8192)),
        max_batch_size=int(sampler_cfg.get("max_batch_size", 32)),
        seed=seed,
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        steps_per_epoch=sampler_cfg.get("steps_per_epoch"),
    )


def build_held_out_sampler(dataset, sampler_cfg: dict, seed: int):
    """Fixed per-task batches over the held-out rows, same count on every rank."""
    from simuls2st_omni.train.samplers import ConcatBatchSampler, TokenBudgetBatchSampler

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    parts = []
    for task in dataset.tasks:
        indices = dataset.task_indices[task]
        parts.append(
            (
                indices,
                TokenBudgetBatchSampler(
                    lengths=[dataset.cost(i) for i in indices],
                    token_budget=int(sampler_cfg.get("token_budget", 8192)),
                    max_batch_size=int(sampler_cfg.get("max_batch_size", 32)),
                    shuffle=False,
                    seed=seed,
                    rank=rank,
                    world_size=world_size,
                ),
            )
        )
    return ConcatBatchSampler(parts)


def build_sampler(dataset, sampler_cfg: dict, seed: int, shuffle: bool, indices=None):
    from simuls2st_omni.train.samplers import TokenBudgetBatchSampler

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if indices is None:
        indices = range(len(dataset))
    # Bucket on talker length: prefix + pad + bos + codes + eos. The thinker side
    # scales with the transcript, which tracks code count closely enough.
    lengths = [dataset.code_length(i) for i in indices]

    return TokenBudgetBatchSampler(
        lengths=lengths,
        token_budget=int(sampler_cfg.get("token_budget", 8192)),
        max_batch_size=int(sampler_cfg.get("max_batch_size", 32)),
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )


def build_training_args(train_cfg: dict, output_dir: str):
    from transformers import TrainingArguments

    supported = {
        "num_train_epochs",
        "max_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_ratio",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "gradient_accumulation_steps",
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "logging_steps",
        "save_steps",
        "save_total_limit",
        "eval_steps",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "seed",
        "report_to",
        "ddp_find_unused_parameters",
        "ddp_timeout",
        "lr_scheduler_kwargs",
        "metric_for_best_model",
        "greater_is_better",
    }
    kwargs: dict[str, Any] = {k: v for k, v in train_cfg.items() if k in supported}

    # cosine_with_min_lr takes its floor through lr_scheduler_kwargs. The
    # argument is `min_lr_rate` (a fraction of peak lr); `min_lr_ratio` is a
    # different library's spelling and raises a TypeError only once the scheduler
    # is actually constructed, i.e. after model load.
    if "min_lr_ratio" in train_cfg:
        kwargs.setdefault("lr_scheduler_kwargs", {})
        kwargs["lr_scheduler_kwargs"]["min_lr_rate"] = float(train_cfg["min_lr_ratio"])

    # Fail here rather than minutes later inside Trainer.create_scheduler.
    scheduler_kwargs = kwargs.get("lr_scheduler_kwargs") or {}
    if scheduler_kwargs:
        import inspect

        from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType

        scheduler_type = SchedulerType(kwargs.get("lr_scheduler_type", "linear"))
        accepted = set(
            inspect.signature(TYPE_TO_SCHEDULER_FUNCTION[scheduler_type]).parameters
        )
        unknown = set(scheduler_kwargs) - accepted
        assert not unknown, (
            f"lr_scheduler_type={scheduler_type.value} does not accept {sorted(unknown)}; "
            f"accepted extras: {sorted(accepted - {'optimizer', 'num_warmup_steps', 'num_training_steps', 'last_epoch'})}"
        )

    kwargs.setdefault("report_to", [])
    kwargs.setdefault("dataloader_pin_memory", True)
    # The wrapper's forward returns a loss even when one head is unsupervised, so
    # some parameters legitimately receive no gradient in a given batch.
    kwargs.setdefault("ddp_find_unused_parameters", False)
    # batch_size is decided by the batch sampler; these keep the trainer quiet.
    kwargs["per_device_train_batch_size"] = 1
    kwargs["per_device_eval_batch_size"] = 1
    kwargs["output_dir"] = output_dir
    kwargs["logging_dir"] = str(Path(output_dir) / "logs")
    kwargs["save_safetensors"] = False
    kwargs["remove_unused_columns"] = False
    if "eval_steps" in kwargs:
        kwargs["eval_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build everything, report shapes and parameter counts, then exit",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0, help="override for smoke tests"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stage = config["stage"]
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    sampler_cfg = config.get("sampler", {})
    train_cfg = dict(config.get("training", {}))
    if args.max_steps:
        train_cfg["max_steps"] = args.max_steps
        train_cfg.pop("num_train_epochs", None)

    output_dir = resolve_path(train_cfg.pop("output_dir", f"outputs/{stage}"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seed = int(train_cfg.get("seed", 42))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = f"cuda:{local_rank}" if torch.cuda.is_available() else None

    logger.info("stage=%s output=%s", stage, output_dir)

    from simuls2st_omni.train.model_init import load_base_model
    from simuls2st_omni.train.modeling_omni_talker import QwenOmniTalkerTrainer
    from simuls2st_omni.train.stage_policy import (
        STAGE_BUILDERS,
        build_wrapped_model,
        describe_trainable,
    )

    assert stage in STAGE_BUILDERS, f"unknown stage {stage!r}"
    builder = STAGE_BUILDERS[stage]
    policy_kwargs = {}
    if stage == "stage2" and model_cfg.get("init_talker_from"):
        policy_kwargs["init_talker_from"] = resolve_path(model_cfg["init_talker_from"])
    if stage == "stage3":
        merge_from = model_cfg.get("merge_adapters_from")
        assert merge_from, f"{stage} requires model.merge_adapters_from"
        policy_kwargs["merge_adapters_from"] = resolve_path(merge_from)
    policy = builder(**policy_kwargs)

    base_model, processor, codec_tokens = load_base_model(
        resolve_path(model_cfg.get("path", "models/SimulS2ST-Omni/offline")),
        dtype=DTYPES[model_cfg.get("dtype", "bf16")],
        device_map=device_map,
    )
    logger.info("codec tokens: %s", codec_tokens)

    model = build_wrapped_model(base_model, policy)
    counts = describe_trainable(model)
    logger.info(
        "trainable %s / %s params (thinker %s, talker %s)",
        f"{counts['trainable']:,}",
        f"{counts['total']:,}",
        f"{counts['thinker_trainable']:,}",
        f"{counts['talker_trainable']:,}",
    )
    assert counts["trainable"] > 0, "no trainable parameters under this policy"

    built = build_dataset(data_cfg, processor=processor)
    collator = build_collator(data_cfg, processor, codec_tokens, model.talker)
    is_multitask = isinstance(built, dict)

    if is_multitask:
        # Task mixing owns batch composition, so there is no held-out slice to
        # carve here: an eval pass would have to mirror the mix to be
        # meaningful, and Stage 2's signal comes from the per-task train losses.
        dataset = MultiTaskConcatDataset(built)
        logger.info(
            "multitask dataset: %d samples across %s",
            len(dataset),
            {t: len(built[t]) for t in dataset.tasks},
        )
        train_sampler = build_multitask_sampler(dataset, data_cfg, sampler_cfg, seed)
        eval_sampler = None
        eval_indices: list[int] = []
        eval_dataset = None
        if data_cfg.get("_held_out"):
            eval_dataset = MultiTaskConcatDataset(data_cfg["_held_out"])
            eval_sampler = build_held_out_sampler(eval_dataset, sampler_cfg, seed)
            eval_indices = list(range(len(eval_dataset)))
            logger.info(
                "held-out eval: %s",
                {t: len(eval_dataset.datasets[t]) for t in eval_dataset.tasks},
            )
    else:
        dataset = built
        eval_dataset = dataset
        eval_n = int(data_cfg.get("eval_samples", 0))
        if eval_n and hasattr(dataset, "train_eval_split"):
            # Multi-source pools hold out a slice of every source and drop all
            # upsampled copies of it from training.
            train_indices, eval_indices = dataset.train_eval_split(eval_n)
        elif eval_n and eval_n < len(dataset):
            # Hold out the tail so the eval slice is disjoint from training.
            eval_indices = list(range(len(dataset) - eval_n, len(dataset)))
            train_indices = list(range(len(dataset) - eval_n))
        else:
            eval_indices, train_indices = [], list(range(len(dataset)))

        train_sampler = build_sampler(
            dataset, sampler_cfg, seed, shuffle=True, indices=train_indices
        )
        eval_sampler = (
            build_sampler(
                dataset, sampler_cfg, seed, shuffle=False, indices=eval_indices
            )
            if eval_indices
            else None
        )
    logger.info(
        "train batches/rank=%d  eval batches/rank=%s",
        len(train_sampler),
        len(eval_sampler) if eval_sampler else "n/a",
    )

    training_args = build_training_args(train_cfg, output_dir)
    if eval_sampler is None:
        # Task mixing owns batch composition, so there is no meaningful held-out
        # slice; turn evaluation off rather than letting Trainer reject the
        # eval_strategy/eval_dataset mismatch.
        training_args.eval_strategy = "no"

    trainer = QwenOmniTalkerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset if eval_indices else None,
        data_collator=collator,
        eval_collator=collator,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        processing_class=processor.tokenizer,
        use_thinker_lora=policy.thinker_lora is not None,
        use_talker_lora=policy.talker_lora is not None,
        max_seq_len=int(train_cfg.get("max_seq_len", 0)),
        max_talker_seq_len=int(train_cfg.get("max_talker_seq_len", 0)),
        max_talker_tokens_per_batch=int(train_cfg.get("max_talker_tokens_per_batch", 0)),
        max_thinker_tokens_per_batch=int(train_cfg.get("max_thinker_tokens_per_batch", 0)),
    )

    if args.dry_run:
        loader = trainer.get_train_dataloader()
        batch = next(iter(loader))
        shapes = {
            k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)
        }
        logger.info("first batch shapes: %s", json.dumps(shapes, indent=2))
        loss = trainer.compute_loss(model, {
            k: (v.to(model.talker.device) if torch.is_tensor(v) else v)
            for k, v in batch.items() if k != "ids"
        })
        logger.info("dry-run loss: %.4f", float(loss.detach().float().cpu()))
        logger.info("dry run OK")
        return

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    logger.info("saved final model to %s", output_dir)


if __name__ == "__main__":
    main()
