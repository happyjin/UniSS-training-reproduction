"""swift 接入层单测：采样计划 == 源采样器（逐 rank）、yaml -> swift 参数、延迟倍数抽取、配置覆盖。

不需要模型权重与 swift（需要 swift 的用例自动跳过）。完整的训练轨迹对拍见 tests/parity/run_parity.sh。
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from simuls2st_omni.make_smoke_fixture import STAGE1, STAGE2, STAGE3, build_manifests, build_s2st, build_tts


@pytest.fixture(scope='module')
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('simuls2st_data')
    rng = np.random.default_rng(0)
    build_tts(str(root / 'emilia' / 'dualcodec'), rng)
    build_manifests(str(root), rng)
    build_s2st(str(root), rng, 'offline', 10, streaming=False)
    build_s2st(str(root), rng, 'streaming', 14, streaming=True)
    return root


def _cfg(tmp_path, body: str, stage: str, **training):
    cfg = yaml.safe_load(body)
    cfg['model'] = {'path': 'model', 'dtype': 'fp32'}
    if stage == 'stage3':
        cfg['model']['merge_adapters_from'] = 'prev'
    cfg['training'] = {'seed': 42, **training}
    path = tmp_path / f'{stage}.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


@pytest.fixture
def root_env(monkeypatch, data_root):
    import simuls2st_omni.train.native_train as nt
    from simuls2st_omni.integration import plan as plan_mod
    monkeypatch.setattr(nt, 'REPO_ROOT', data_root)
    plan_mod._STAGES.clear()
    yield
    plan_mod._STAGES.clear()


def _source_rank_batches(monkeypatch, cfg, rank, world):
    """源 train.py 的同一段代码在 rank / world_size 下产出的 train / eval batch 序列。"""
    import simuls2st_omni.train.native_train as nt
    monkeypatch.setenv('RANK', str(rank))
    monkeypatch.setenv('WORLD_SIZE', str(world))
    data_cfg = dict(cfg['data'])
    built = nt.build_dataset(data_cfg, processor=type('P', (), {'tokenizer': None})())
    if isinstance(built, dict):
        ds = nt.MultiTaskConcatDataset(built)
        return list(nt.build_multitask_sampler(ds, data_cfg, cfg['sampler'], 42)), []
    n_eval = int(data_cfg.get('eval_samples', 0))
    train_idx = list(range(len(built) - n_eval)) if n_eval and n_eval < len(built) else list(range(len(built)))
    eval_idx = list(range(len(built) - n_eval, len(built))) if n_eval and n_eval < len(built) else []
    train = list(nt.build_sampler(built, cfg['sampler'], 42, shuffle=True, indices=train_idx))
    ev = list(nt.build_sampler(built, cfg['sampler'], 42, shuffle=False, indices=eval_idx)) if eval_idx else []
    return train, ev


@pytest.mark.parametrize('stage,body', [('stage1', STAGE1), ('stage2', STAGE2), ('stage3', STAGE3)])
@pytest.mark.parametrize('world', [1, 2, 3])
def test_plan_matches_source_sampler_per_rank(tmp_path, monkeypatch, root_env, stage, body, world):
    """swift 的 BatchSamplerShard 以 range(rank, N, world) 分片：rank r 第 s 步拿到 row[s*world+r]，
    必须等于源 sampler 在 rank r 的第 s 个 batch。"""
    from simuls2st_omni.integration.plan import build_stage
    from simuls2st_omni.integration.stage_config import load_stage_config
    cfg = load_stage_config(_cfg(tmp_path, body, stage))
    monkeypatch.setenv('WORLD_SIZE', str(world))
    plan = build_stage(cfg)
    assert len(plan.train_rows) % world == 0
    for rank in range(world):
        src_train, src_eval = _source_rank_batches(monkeypatch, cfg, rank, world)
        assert [list(map(int, b)) for b in plan.train_rows[rank::world]] == [list(map(int, b)) for b in src_train]
        assert [list(map(int, b)) for b in plan.eval_rows[rank::world]] == [list(map(int, b)) for b in src_eval]


def test_swift_args_follow_source_build_training_args(tmp_path, root_env):
    from simuls2st_omni.integration.stage_config import load_stage_config
    from simuls2st_omni.integration.swift_args import build_args, has_validation

    def as_dict(args):
        it = iter(args)
        return {k: next(it) for k in it}

    cfg = load_stage_config(_cfg(tmp_path, STAGE1, 'stage1', learning_rate=4e-4, min_lr_ratio=0.1,
                                 lr_scheduler_type='cosine_with_min_lr', eval_steps=5, bf16=True))
    a = as_dict(build_args(cfg, has_validation(cfg)))
    # swift 改过默认值的项必须钉回 HF / 源的取值
    assert a['--adam_beta2'] == '0.999' and a['--weight_decay'] == '0.0' and a['--logging_first_step'] == 'false'
    assert a['--optim'] == 'adamw_torch' and a['--gradient_checkpointing'] == 'false'
    assert a['--lr_scheduler_kwargs'] == '{"min_lr_rate": 0.1}'
    assert a['--per_device_train_batch_size'] == '1' and a['--train_dataloader_shuffle'] == 'false'
    assert a['--ddp_find_unused_parameters'] == 'false'          # Stage 1：thinker 冻结、talker 全部参与
    assert a['--val_dataset'] == 'simuls2st_omni_val' and a['--eval_strategy'] == 'steps'
    cfg2 = load_stage_config(_cfg(tmp_path, STAGE2, 'stage2', max_steps=12, eval_steps=5))
    a2 = as_dict(build_args(cfg2, has_validation(cfg2)))
    assert a2['--ddp_find_unused_parameters'] == 'true'          # 文本任务 batch 不经过 talker
    assert a2['--eval_strategy'] == 'no' and '--val_dataset' not in a2   # 源：多任务不评估
    assert a2['--lr_scheduler_type'] == 'linear'                 # yaml 未写时取 HF 默认（swift 默认 cosine）


def test_env_overrides(tmp_path, monkeypatch, root_env):
    from simuls2st_omni.integration.stage_config import load_stage_config
    from simuls2st_omni.integration.swift_args import training_kwargs
    path = _cfg(tmp_path, STAGE1, 'stage1', num_train_epochs=2)
    monkeypatch.setenv('MAX_STEPS', '7')
    kw = training_kwargs(load_stage_config(path))
    assert kw['max_steps'] == '7' and 'num_train_epochs' not in kw
    monkeypatch.setenv('SIMULS2ST_MERGE_ADAPTERS_FROM', '/x/stage2')
    assert load_stage_config(_cfg(tmp_path, STAGE3, 'stage3'))['model']['merge_adapters_from'] == '/x/stage2'


def test_stable_latency_draw_is_uniform_and_reproducible():
    from simuls2st_omni.integration.latency import stable_draw
    draws = [stable_draw(42, i) for i in range(20000)]
    assert draws == [stable_draw(42, i) for i in range(20000)]
    counts = np.bincount(np.minimum((np.array(draws) * 12).astype(int), 11), minlength=12)
    assert counts.min() > 20000 / 12 * 0.9 and counts.max() < 20000 / 12 * 1.1
    assert stable_draw(42, 0) != stable_draw(43, 0)


def test_stage3_latency_patch(tmp_path, monkeypatch, root_env):
    """注册后 StreamingS2STDataset 的 m 只取决于 (seed, index)，与进程的哈希种子无关。"""
    from simuls2st_omni.integration.latency import install, stable_draw
    from simuls2st_omni.train.datasets.streaming import StreamingS2STDataset
    original = StreamingS2STDataset._pick_latency
    try:
        assert install(StreamingS2STDataset)
        ds = StreamingS2STDataset.__new__(StreamingS2STDataset)
        ds.seed, ds.latency_multipliers = 42, list(range(1, 13))
        assert [ds._pick_latency(i) for i in range(50)] == [
            ds.latency_multipliers[min(int(stable_draw(42, i) * 12), 11)] for i in range(50)]
        monkeypatch.setenv('SIMULS2ST_LATENCY_HASH', 'python')
        StreamingS2STDataset._pick_latency = original
        assert not install(StreamingS2STDataset)
    finally:
        StreamingS2STDataset._pick_latency = original


def test_swift_registration():
    pytest.importorskip('swift')
    from swift.tuner_plugin import tuners_map

    from simuls2st_omni.integration import register as reg
    reg.register_simuls2st_model()
    reg.register_simuls2st_template()
    reg.register_simuls2st_tuner()
    reg.install_resume_patch()
    assert 'simuls2st_omni' in tuners_map
    from swift.trainers.mixin import SwiftMixin
    assert getattr(SwiftMixin, '_simuls2st_resume_patched', False)
    tpl = reg._template_cls()
    assert tpl.model_accepts_loss_kwargs is False   # 否则 swift 把 loss 乘以进程数、且 GA 不再除步数
