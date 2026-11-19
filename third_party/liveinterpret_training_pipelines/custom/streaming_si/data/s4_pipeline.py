"""S4 数据管线（不依赖 swift）：采样计划、指针行、按指针行编码样本。

swift 接线（数据集注册 / loader）见 `register_streaming_si_s4.py`。

设计：**样本编码完全复用源项目的 `S4StreamingDataset`**（已原样迁入，源单测全过），swift 侧的
HF 数据集只是「指针行」—— 每行一个 `(split, 记录下标, 视图描述, draw_seed)`；template 在
`_encode` 里用这些字段调用 `S4StreamingDataset.__getitem__`，得到与源逐 token 相同的样本。

为什么不让 swift 直接读 records jsonl：一条记录 ~4.5KB（`target_codec_codes` 占大头），两方向共
~5.8GB，转成 arrow 缓存要再占一份磁盘；源的 `JsonlIndex` 只存字节偏移（每行 8B）。

与源 trainer 的采样语义对应（全部在 loader 里一次性物化为指针行，训练中不再有 sampler 状态，
swift 的 `--resume_from_checkpoint` 跳过已消费 batch 即精确续训）：

  * 均匀（SI_S4_STRATIFIED=0）：每条记录一行，由 swift 打乱 == 源 shuffle / DistributedSampler。
  * 分层加权（SI_S4_STRATIFIED=1，默认）：源 `WeightedRandomSampler(w, num_samples=len,
    replacement=True)`；这里用同一权重（`si_sampler.weights_for_dataset`）有放回地抽 len 行。
    差异：源每个 epoch 重抽一次，这里多 epoch 共用同一份抽样（由 swift 每 epoch 重新打乱顺序）。
    默认 1 epoch 时二者同分布。
  * prefix-drop（SI_S4_P_DROP>0）：直接用源 `MixedContextViewSampler` 生成调度
    （world_size=1，即全局一份 Hamilton 精确配额），替补池与源 `configure_context_view_pools` 相同。
  * 延迟倍数 m：源用每个 dataset 实例一条顺序 RNG 流（多 worker 下各 worker 流相同）；
    这里每行带独立 draw_seed，`_sample_m` 按行确定 —— 分布相同、且与 worker 数/续训无关。
  * val 剔除：训练集按 val 源文本指纹跳过（源 `exclude_keys`，在 `__getitem__` 内有界扫描替补）。

验证集：与源一致按方向各取前 `SI_S4_VAL_LIMIT` 条（源 = val_batches × batch_size × world）。
swift 会把多个方向合并成一个 eval 集，汇报的是源 `evaluate_all` 的 ALL 口径（token 加权 +
累加混淆矩阵算 macro-F1）；分方向指标与 prefix-drop 的 paired(full/crop) 双臂验证未迁移。
"""
import hashlib
import json
import logging
import os
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..model.base_model import resolve_stream_token_ids, stream_special_tokens
from .s4_context_views import MixedContextViewSampler, S4ViewIndex
from .s4_streaming_dataset import JsonlIndex, S4StreamingDataset, s4_collate_fn
from .si_sampler import effective_sample_size, static_context_eligibility_for_dataset, weights_for_dataset

try:                                   # 训练时沿用 swift 的 logger（INFO 可见）；纯单测环境退回 logging
    from swift.utils import get_logger
    logger = get_logger()
except Exception:
    logger = logging.getLogger('streaming_si.s4')

_MESSAGES = [{'role': 'user', 'content': 's4'}, {'role': 'assistant', 'content': 's4'}]
_MAX_SEED = (1 << 63) - 1


# ---------------------------------------------------------------------------
# 1. 配置（全部来自 pipelines/streaming_si/train_simul.sh 导出的环境变量）
# ---------------------------------------------------------------------------
def s4_env() -> Dict[str, Any]:
    e = os.environ
    lm = e.get('SI_S4_LATENCY_MULTIPLIER', '1').strip()
    return dict(
        data=e.get('SI_S4_DATA', '').strip(),
        data_root=e.get('SI_S4_DATA_ROOT') or None,
        val_dir=e.get('SI_S4_VAL_DIR', '').strip(),
        index_dir=e.get('SI_S4_INDEX_DIR', '').strip(),
        stratified=e.get('SI_S4_STRATIFIED', '1') == '1',
        max_repeat=float(e.get('SI_S4_MAX_REPEAT', 2.0)),
        min_source_ms=int(e.get('SI_S4_MIN_SOURCE_MS', 0)),
        p_drop=float(e.get('SI_S4_P_DROP', 0.0)),
        paired_control=e.get('SI_S4_PAIRED_CONTROL', '0') == '1',
        prefix_context_turns=int(e.get('SI_S4_PREFIX_CONTEXT_TURNS', 16)),
        latency_multiplier=lm if lm == 'sample' else int(lm),
        m_max=int(e.get('SI_S4_M_MAX', 4)),
        m_schedule=e.get('SI_S4_M_SCHEDULE', 'uniform'),
        use_src_end=e.get('SI_S4_USE_SRC_END', '1') == '1',
        strict_text=e.get('SI_S4_STRICT_TEXT', '0') == '1',
        max_length=int(e.get('SI_MAX_LENGTH', 640)),
        max_audio_sec=float(e.get('SI_MAX_AUDIO_SEC', 20.0)),
        max_speech_len=int(e.get('SI_MAX_SPEECH_LEN', 750)),
        seed=int(e.get('SI_S4_SEED', 1337)),
        num_samples=int(e.get('SI_S4_NUM_SAMPLES', 0)),
        global_batch=int(e.get('SI_S4_GLOBAL_BATCH', 0)),    # per_device_bs × world × GA
        max_steps=int(e.get('SI_S4_MAX_STEPS', -1)),
        val_limit=int(e.get('SI_S4_VAL_LIMIT', 0)),
        limit_rows=int(e.get('SI_S4_LIMIT_ROWS', 0)),
    )


def _validate_positive_context(cfg):
    """源 validate_context_view_config 的约束（p_drop>0 的 H1 实验臂）。"""
    if not 0.0 <= cfg['p_drop'] <= 1.0:
        raise ValueError('SI_S4_P_DROP 必须在 [0, 1]')
    if cfg['paired_control'] and cfg['p_drop'] <= 0:
        raise ValueError('SI_S4_PAIRED_CONTROL=1 需要 SI_S4_P_DROP > 0')
    if cfg['p_drop'] <= 0:
        return False
    if cfg['latency_multiplier'] != 1:
        raise ValueError('prefix-drop 训练要求固定 m=1（SI_S4_LATENCY_MULTIPLIER=1）')
    if os.environ.get('SI_S4_TRAIN_SCOPE', 'joint') != 'thinker':
        raise ValueError('prefix-drop 训练要求 SI_S4_TRAIN_SCOPE=thinker')
    if float(os.environ.get('SI_SPEECH_LOSS_WEIGHT', '1')) != 0.0:
        raise ValueError('prefix-drop 训练要求 SI_SPEECH_LOSS_WEIGHT=0')
    if cfg['prefix_context_turns'] < 1:
        raise ValueError('SI_S4_PREFIX_CONTEXT_TURNS 必须 >= 1')
    return True


def read_val_meta(val_dir: str) -> List[Dict[str, Any]]:
    """`build_s4_val_split.py` 产物：`<dir>_val_keys.json` + `<dir>_val.jsonl`（源 build_val_sets 同规则）。"""
    import glob
    out = []
    if not val_dir or not os.path.isdir(val_dir):
        return out
    for keys_path in sorted(glob.glob(os.path.join(val_dir, '*_val_keys.json'))):
        with open(keys_path) as f:
            meta = json.load(f)
        jsonl = os.path.join(val_dir, f'{meta["direction"]}_val.jsonl')
        if not os.path.exists(jsonl):
            logger.warning(f'[S4] {keys_path} 有指纹但缺 {jsonl}，跳过该方向')
            continue
        out.append(dict(direction=meta['direction'], jsonl=jsonl, keys=list(meta['val_keys']),
                        audio_root=meta.get('audio_root')))
    return out


def add_s4_stream_tokens(tokenizer, capacity: int) -> None:
    """S4 的 tokenizer：等价于源 `build_qwen_tokenizer(add_stream_tokens=True, embedding_capacity=151936)`。

    S1-S3（及其导出目录）的 tokenizer 只注册了 `<|audio_pad|>`；源 S4 在原始 Qwen3 上一次性追加
    `[<|audio_pad|>] + stream_special_tokens()`。这里在已含 audio_pad 的 tokenizer 后按同一顺序
    追加其余 token，得到的 id 与源一致（`<|wait|>` = audio_pad + 1 ...），由调用方的断言兜底。
    """
    vocab = tokenizer.get_vocab()
    new = [t for t in stream_special_tokens() if t not in vocab]
    if new:
        free = capacity - len(tokenizer)
        if len(new) > free:
            raise ValueError(f'新增 {len(new)} 个流式 token 超出 embedding 空位 {free} 个')
        tokenizer.add_special_tokens({'additional_special_tokens': new},
                                     replace_additional_special_tokens=False)
    audio_id = tokenizer.convert_tokens_to_ids('<|audio_pad|>')
    for offset, tok in enumerate(stream_special_tokens(), start=1):
        got = tokenizer.encode(tok, add_special_tokens=False)
        if got != [audio_id + offset]:
            raise RuntimeError(f'{tok} -> {got}，预期单 token {audio_id + offset}（与源 build_qwen_tokenizer 顺序不一致）')


def s4_stream_token_ids(tokenizer) -> List[int]:
    """S4 真正使用且要学习的四个 Thinker 词表行，顺序是 adapter checkpoint 契约（源 active_stream_token_ids）。"""
    ids = resolve_stream_token_ids(tokenizer)
    names = ('wait_token_id', 'write_token_id', 'chunk_token_id', 'src_end_token_id')
    missing = [n for n in names if ids.get(n) is None]
    if missing:
        raise RuntimeError(f'tokenizer 缺少 S4 流式 token {missing}（需先 add_s4_stream_tokens）')
    return [int(ids[n]) for n in names]


# ---------------------------------------------------------------------------
# 2. 采样计划（loader 与 template 进程各自计算，结果由 digest 核对一致）
# ---------------------------------------------------------------------------
def _index_view(cfg):
    files = S4StreamingDataset._expand(cfg['data'])
    if not files:
        raise RuntimeError(f'SI_S4_DATA 没有找到 records jsonl: {cfg["data"]!r}')
    items = [(fi, li) for fi, f in enumerate(files) for li in range(len(JsonlIndex(f)))]
    if cfg['limit_rows'] > 0:
        items = items[:cfg['limit_rows']]
    return SimpleNamespace(items=items, _sharded_files=files)


def build_sampling_plan(cfg) -> Dict[str, Any]:
    """与源 train_s4_stream.main 的第 5 节同口径：分层权重 ->（prefix-drop 时）静态资格掩码。"""
    positive = _validate_positive_context(cfg)
    view = _index_view(cfg)
    n = len(view.items)
    weights = np.ones(n, dtype=np.float64)
    if cfg['stratified']:
        if not cfg['index_dir']:
            raise ValueError('分层采样需要 SI_S4_INDEX_DIR（build_record_index.py 的产出）')
        weights, _, _ = weights_for_dataset(view, cfg['index_dir'], min_source_ms=cfg['min_source_ms'],
                                            max_repeat=cfg['max_repeat'], logger=logger.info)
        pos = int((weights > 0).sum())
        logger.info(f'[S4] 分层采样: 有效样本量 {effective_sample_size(weights):.0f}/{pos}, '
                    f'单条最大被采 {weights[weights > 0].max() / weights[weights > 0].mean():.2f}x')
    thinker = np.zeros(0, dtype=np.int64)
    if positive:
        if not cfg['index_dir']:
            raise ValueError('prefix-drop 需要 SI_S4_INDEX_DIR')
        static = static_context_eligibility_for_dataset(
            view, cfg['index_dir'], thinker_ratio=cfg['p_drop'], talker_ratio=0.0,
            prefix_context_turns=cfg['prefix_context_turns'],
            min_source_ms=max(cfg['min_source_ms'], 500),     # S4StreamingDataset.min_audio_sec=0.5
            max_source_ms=int(cfg['max_audio_sec'] * 1000), max_speech_tokens=cfg['max_speech_len'])
        weights = np.asarray(weights, dtype=np.float64).copy()
        weights[~static.valid_mask] = 0.0
        thinker = np.flatnonzero(static.thinker_mask & (weights > 0)).astype(np.int64)
    h = hashlib.sha256()
    h.update(json.dumps({k: cfg[k] for k in sorted(cfg)
                         if k not in ('val_limit', 'num_samples', 'global_batch', 'max_steps')},
                        sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(weights, dtype='<f8').tobytes())
    h.update(thinker.astype('<i8').tobytes())
    return dict(n=n, weights=weights, thinker_eligible=thinker, positive=positive,
                digest=h.hexdigest()[:16])


def s4_schedule_length(cfg, n):
    """prefix-drop 调度长度，对应源 planned_batch_horizon：整 batch 数 × 全局 batch，受 max_steps 截断。

    源按 rank 计算 min_r(len_r // bs) × bs，全局 = 该值 × world；这里数据集全局一份，
    取 (n // global_batch) × global_batch，使 drop_last 不丢调度行、Hamilton 配额精确落地。
    """
    if cfg['num_samples'] > 0:
        return int(cfg['num_samples'])
    gb = cfg['global_batch']
    if gb <= 0:
        return n
    num = (n // gb) * gb
    if cfg['max_steps'] > 0:
        num = min(num, cfg['max_steps'] * gb)
    if num <= 0:
        raise ValueError(f'prefix-drop 调度为空：{n} 条不足一个全局 batch（{gb}）')
    return num


def build_train_rows(cfg) -> Dict[str, list]:
    plan = build_sampling_plan(cfg)
    n, w = plan['n'], plan['weights']
    g = torch.Generator().manual_seed(cfg['seed'])
    if plan['positive']:
        num = s4_schedule_length(cfg, n)
        sampler = MixedContextViewSampler(
            weights=w, thinker_eligible=plan['thinker_eligible'], talker_eligible=[],
            thinker_ratio=cfg['p_drop'], talker_ratio=0.0, num_samples=int(num), seed=cfg['seed'],
            rank=0, world_size=1, apply_crop=not cfg['paired_control'])
        sched = list(sampler)
        idx = [v.index for v in sched]
        designated = [v.designated_kind for v in sched]
        applied = [v.applied_kind for v in sched]
        seeds = [v.draw_seed for v in sched]
        n_crop = designated.count('thinker_crop')
        logger.info(f'[S4] prefix-drop 调度: {len(sched)} 行, designated thinker_crop={n_crop} '
                    f'({n_crop / max(len(sched), 1):.4f}), applied='
                    f'{"full control" if cfg["paired_control"] else "Thinker crop"}')
    else:
        if cfg['stratified']:
            idx = torch.multinomial(torch.as_tensor(w, dtype=torch.double), n, replacement=True,
                                    generator=g).tolist()
        else:
            idx = list(range(n))
        designated = applied = [''] * len(idx)
        seeds = torch.randint(0, _MAX_SEED, (len(idx),), generator=g, dtype=torch.int64).tolist()
    m = len(idx)
    return {'si_stage': ['s4'] * m, 's4_split': ['train'] * m, 's4_index': idx,
            's4_designated': list(designated), 's4_applied': list(applied), 's4_seed': seeds,
            's4_plan': [plan['digest']] * m, 'messages': [_MESSAGES] * m}


def build_val_rows(cfg) -> Dict[str, list]:
    rows = {k: [] for k in ('si_stage', 's4_split', 's4_index', 's4_designated', 's4_applied',
                            's4_seed', 's4_plan', 'messages')}
    for meta in read_val_meta(cfg['val_dir']):
        n = len(JsonlIndex(meta['jsonl']))
        take = min(n, cfg['val_limit']) if cfg['val_limit'] > 0 else n
        logger.info(f'[S4] Val[{meta["direction"]}]: 取前 {take}/{n} 条')
        for i in range(take):
            rows['si_stage'].append('s4')
            rows['s4_split'].append(f'val:{meta["direction"]}')
            rows['s4_index'].append(i)
            rows['s4_designated'].append('')
            rows['s4_applied'].append('')
            rows['s4_seed'].append(0)
            rows['s4_plan'].append('')
            rows['messages'].append(_MESSAGES)
    return rows


# ---------------------------------------------------------------------------
# 3. template 侧：按指针行构造样本
# ---------------------------------------------------------------------------
class S4Encoder:
    """每个进程（含 DataLoader worker）惰性构造一次源 S4StreamingDataset。"""

    def __init__(self, tokenizer, mel):
        self.tokenizer = tokenizer
        self.mel = mel
        self.cfg = s4_env()
        self._train = None
        self._train_digest = None
        self._val = {}

    def _common(self):
        c = self.cfg
        return dict(max_length=c['max_length'], max_speech_len=c['max_speech_len'],
                    max_audio_sec=c['max_audio_sec'], use_src_end=c['use_src_end'])

    def train_ds(self):
        if self._train is None:
            c = self.cfg
            exclude = set()
            for meta in read_val_meta(c['val_dir']):
                exclude |= set(meta['keys'])
            ds = S4StreamingDataset(
                c['data'], self.tokenizer, data_root=c['data_root'], audio_processor=self.mel,
                rank=0, world_size=1, strict_text=c['strict_text'], align_ranks=False,
                exclude_keys=exclude, latency_multiplier=c['latency_multiplier'], m_max=c['m_max'],
                m_schedule=c['m_schedule'], seed=c['seed'],
                prefix_context_turns=c['prefix_context_turns'], **self._common())
            if c['limit_rows'] > 0:
                ds.items = ds.items[:c['limit_rows']]
            plan = build_sampling_plan(c)
            if len(ds) != plan['n']:
                raise RuntimeError(f'S4 数据集行数 {len(ds)} 与采样计划 {plan["n"]} 不一致')
            if plan['positive']:
                ds.configure_context_view_pools(plan['weights'], plan['thinker_eligible'])
            self._train, self._train_digest = ds, plan['digest']
        return self._train

    def val_ds(self, direction):
        if direction not in self._val:
            meta = {m['direction']: m for m in read_val_meta(self.cfg['val_dir'])}.get(direction)
            if meta is None:
                raise RuntimeError(f'val 方向 {direction} 不在 SI_S4_VAL_DIR={self.cfg["val_dir"]}')
            # val jsonl 在数据树之外，音频根取划分时记下的 audio_root（源 build_val_sets 同规则）
            self._val[direction] = S4StreamingDataset(
                meta['jsonl'], self.tokenizer, data_root=meta['audio_root'] or self.cfg['data_root'],
                audio_processor=self.mel, **self._common())
        return self._val[direction]

    def encode(self, row) -> Dict[str, Any]:
        split = row['s4_split']
        index = int(row['s4_index'])
        if split == 'train':
            ds = self.train_ds()
            if row.get('s4_plan') != self._train_digest:
                raise RuntimeError('指针行的采样计划与本进程重建的不一致（环境变量在 loader 与 worker 间变了？）')
            ds._m_rng = random.Random(int(row['s4_seed']))    # 每行确定的 m（见模块 docstring）
            if row.get('s4_designated'):
                item = ds[S4ViewIndex(index, row['s4_designated'], row['s4_applied'], int(row['s4_seed']))]
            else:
                item = ds[index]
        elif split.startswith('val:'):
            item = self.val_ds(split[4:])[index]
        else:
            raise ValueError(f'未知 s4_split={split!r}')
        return unshift_s4_item(item, self.tokenizer.eos_token_id)


def unshift_s4_item(item, eos_token_id):
    """源样本是已 shift 的 X/Y/answer[:-1]；swift 约定 template 给未 shift 序列、模型内部 shift。

    源序列末位恒为 Thinker EOS、首位 label 恒为 -100（instruction），故逆变换精确：
        ids = X ++ [eos]；labels = [-100] ++ Y；answer = A ++ [0]
    `StreamingSIForConditionalGeneration._shifted_views` 再切回 X / Y / A，逐位还原。
    """
    out = dict(item)
    x, y, a = item['input_ids'], item['labels'], item['answer_mask']
    out['input_ids'] = torch.cat([x, x.new_tensor([int(eos_token_id)])])
    out['labels'] = torch.cat([y.new_tensor([-100]), y])
    out['answer_mask'] = torch.cat([a, a.new_tensor([0])])
    out['_s4'] = True
    return out


def collate_s4(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """源 `s4_collate_fn`（右 pad、分段长度不 pad、元信息透传）；不加 attention_mask，与源一致。"""
    for b in batch:
        b.pop('_s4', None)
    return s4_collate_fn(batch)
