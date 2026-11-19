"""NIR × 句长 联合分层采样权重（P1-2b + P1-3）。

论文依据（附录 §Monotonicity-Aware Trajectory Selection）
--------------------------------------------------------
论文按 **NIR × 句长** 两个维度分桶采样，难度权重
`{high:0.1, mid_high:0.3, mid_low:0.4, low:0.2}`，长度权重
`{short:0.1, medium:0.5, long:0.4}`；两维独立给出，故联合权重取乘积。

为什么这件事优先级最高：论文 §4（`ch04_experiments.tex:189`）实测**不做单调性过滤时
m=1 的 BLEU 崩到 4.59 / 3.56**，而 m 的采样 schedule 换来换去「negligible differences」。
m=1 正是我们的目标档（1 秒 chunk）。

关于分桶边界的一个诚实说明
--------------------------
论文没有给出 NIR 的绝对分桶边界，也没给 short/medium/long 的秒数阈值。我们用固定
语义边界（NIR: 0 / (0,10] / (10,25] / >25；长度: 4s / 8s），理由：
  * NIR 恰好为 0 的样本占 21-23%，用分位数会让边界退化（p25=0），`low` 桶恒空；
  * 固定边界在两个方向之间、以及将来数据增补后含义不变，可比。

在这组边界下我们的自然分布与论文权重**已经很接近**：
    en2zh 实测 {low .231, mid_low .445, mid_high .223, high .101}
    论文权重     {low .20,  mid_low .40,  mid_high .30,  high .10 }
所以 NIR 维只需轻度重加权（主要是把 mid_high 从 22% 提到 30%），代价很小。

另一件必须说清的事：论文 Table `tab:appendix_nir_random_stats` 报告
random 池 μ_NIR=9.66（En→Zh）、9.92（Zh→En），stratified 池 μ=13.79 / 15.86。
我们未加权的池（NIR≤50% 子集）实测 μ=9.19 / 7.90 —— **与它的 random 行几乎一致**，
说明 NIR 计算口径与论文相同。但把论文的权重向量套到我们池上只能把 μ 抬到约 10.1，
达不到 13.79。也就是说：论文的「权重向量」与它自己报告的「分层后均值」在
分布相同的池上无法同时成立，缺失的信息是它的分桶边界。
我们的取法是**照抄权重向量**（它是论文明确写出的），并把达成的 μ 一并报出来供审计，
而不是反推边界去硬凑 13.79 —— 后者会把 low 桶权重压到 ~0.01，等于丢掉全部单调样本。
"""
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

# NIR 分桶：low 单独把「恰好 0」隔出来（占 21-23%，是最稳定也最容易过采的一批）
NIR_EDGES = (1e-9, 10.0, 25.0)
NIR_NAMES = ('low', 'mid_low', 'mid_high', 'high')
NIR_WEIGHTS = {'low': 0.20, 'mid_low': 0.40, 'mid_high': 0.30, 'high': 0.10}

# 句长分桶（毫秒）
LEN_EDGES = (4000, 8000)
LEN_NAMES = ('short', 'medium', 'long')
LEN_WEIGHTS = {'short': 0.10, 'medium': 0.50, 'long': 0.40}

U16 = 65535


@dataclass(frozen=True)
class StaticContextEligibility:
    """Rank-local static index candidates; Task 4 applies final view admission."""

    thinker_mask: np.ndarray
    talker_mask: np.ndarray
    valid_mask: np.ndarray
    index_profile_digests: dict[str, str] = field(default_factory=dict)

    @property
    def thinker_count(self):
        return int(self.thinker_mask.sum())

    @property
    def talker_count(self):
        return int(self.talker_mask.sum())


def bucket_ids(nir_x100, dur_ms):
    """返回 (nir_bucket, len_bucket)，均为 0..k-1 的 int8。

    NIR 缺失（未对齐词 < 2 条，占 0.07-0.17%）归入 `mid_low` —— 它是占比最大的桶，
    归错的影响最小；不单独开桶是为了避免一个 0.1% 的桶把权重表搞复杂。
    """
    nir = np.where(nir_x100 == U16, 500, nir_x100).astype(np.float64) / 100.0
    nb = np.digitize(nir, NIR_EDGES).astype(np.int8)
    nb[nir_x100 == U16] = 1
    lb = np.digitize(dur_ms, LEN_EDGES).astype(np.int8)
    return nb, lb


def _ipf_cell_targets(counts, nir_target, len_target, iters=200, tol=1e-10):
    """用 IPF（迭代比例拟合 / raking）求各 cell 的目标份额。

    为什么不能直接用两个边际的**乘积**做 cell 目标：两个维度是相关的 —— 长句子词多、
    更容易出现跨语言逆序，所以 NIR 高；反过来 `low-NIR + long` 天然稀有。
    实测直接取乘积会要求 `low/long`（en2zh 仅 8406 条）占到 8%，等于一个 epoch 内把
    这 8 千条重复采 **15.4 倍**；`high/long`（zh2en 仅 4219 条）13.3 倍。那不是分层，
    是对少数样本过拟合。

    IPF 的解在「边际等于目标」的约束下与自然联合分布的 KL 散度最小，也就是
    **只动边际、尽量不动相关结构**。做法是交替按行、按列缩放，直到两个边际都收敛。

    counts: (n_nir, n_len) 自然计数；返回同形状的目标份额（和为 1）。
    """
    nat = counts.astype(np.float64)
    total = nat.sum()
    if total <= 0:
        return np.zeros_like(nat)
    p = nat / total
    rt = np.array([nir_target[k] for k in NIR_NAMES], dtype=np.float64)
    ct = np.array([len_target[k] for k in LEN_NAMES], dtype=np.float64)
    rt, ct = rt / rt.sum(), ct / ct.sum()

    # 空 cell 保持 0：IPF 不能把质量分给一条样本都没有的格子
    mask = nat > 0
    for _ in range(iters):
        prev = p.copy()
        rs = p.sum(axis=1, keepdims=True)
        p = np.where(mask & (rs > 0), p * (rt[:, None] / np.maximum(rs, 1e-300)), p)
        cs = p.sum(axis=0, keepdims=True)
        p = np.where(mask & (cs > 0), p * (ct[None, :] / np.maximum(cs, 1e-300)), p)
        if np.abs(p - prev).max() < tol:
            break
    s = p.sum()
    return p / s if s > 0 else p


def joint_weights(nir_x100, dur_ms, nir_weights=None, len_weights=None,
                  min_source_ms=0, drop_invalid=None, max_repeat=None,
                  use_ipf=True):
    """算每条样本的采样权重（未归一化，可直接喂 WeightedRandomSampler）。

    权重 = 目标份额 / 自然份额，即「这个桶要被放大多少倍」。采样后各桶份额收敛到
    目标份额，与桶的自然大小无关。空桶权重记 0。

    `use_ipf=True`（默认）用 IPF 求 cell 目标，只对齐两个**边际**；
    `False` 退回边际乘积（会产生 13-15x 的极端倍率，仅供对照，不要用于训练）。

    `max_repeat` 给每条样本的被采次数设上限（对 cell 倍率截断）。截断后边际会偏离
    目标，`summarize` 会把实际达成的份额打出来。

    `min_source_ms > 0` 时把更短的样本权重置 0（硬门槛）。默认 0，走软降权
    （short 桶权重 0.1）—— 硬丢会损失 36-42% 数据，而「该立刻说」也是合法监督，
    且短样本的译文/发声监督与句长无关。
    """
    nw = dict(NIR_WEIGHTS if nir_weights is None else nir_weights)
    lw = dict(LEN_WEIGHTS if len_weights is None else len_weights)
    nb, lb = bucket_ids(nir_x100, dur_ms)
    n = nb.size

    keep = np.ones(n, dtype=bool)
    if drop_invalid is not None:
        keep &= ~drop_invalid
    if min_source_ms > 0:
        keep &= dur_ms >= min_source_ms
    n_keep = max(int(keep.sum()), 1)

    counts = np.zeros((len(NIR_NAMES), len(LEN_NAMES)), dtype=np.int64)
    for i in range(len(NIR_NAMES)):
        for j in range(len(LEN_NAMES)):
            counts[i, j] = int((keep & (nb == i) & (lb == j)).sum())

    if use_ipf:
        cell_target = _ipf_cell_targets(counts, nw, lw)
    else:
        cell_target = np.zeros_like(counts, dtype=np.float64)
        for i, nn in enumerate(NIR_NAMES):
            for j, ln in enumerate(LEN_NAMES):
                cell_target[i, j] = nw[nn] * lw[ln] if counts[i, j] else 0.0
        s = cell_target.sum()
        cell_target = cell_target / s if s > 0 else cell_target

    w = np.zeros(n, dtype=np.float64)
    for i in range(len(NIR_NAMES)):
        for j in range(len(LEN_NAMES)):
            c = counts[i, j]
            if c == 0:
                continue
            r = cell_target[i, j] / (c / n_keep)      # 每条被采次数
            if max_repeat is not None:
                r = min(r, max_repeat)
            w[keep & (nb == i) & (lb == j)] = r
    return w, nb, lb


def effective_sample_size(w):
    """有效样本量 (Σw)² / Σw²。

    回答「重加权后相当于多少条独立样本」。注意它是个悲观指标：大量被降权的样本
    也会把它拉低。判断过拟合风险更该看 `summarize` 里的「单条最大被采次数」。
    """
    w = np.asarray(w, dtype=np.float64)
    s1, s2 = w.sum(), (w * w).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def _direction_of(path):
    """从 records 路径推方向。索引是按方向存的，一个 dataset 可能混了两个方向。"""
    p = path.replace('\\', '/')
    for d in ('en2zh', 'zh2en'):
        if f'/{d}/' in p or f'/{d}_' in p:
            return d
    return None


def weights_for_dataset(ds, index_dir, nir_weights=None, len_weights=None,
                        min_source_ms=0, max_repeat=2.0, use_ipf=True,
                        logger=print):
    """按 `ds.items` 的顺序产出采样权重，可直接喂 WeightedRandomSampler。

    对齐方式：`ds.items` 是 (file_idx, line_idx)，索引 npz 按 **part 文件名 + 行号**
    存，所以用 `basename(ds._sharded_files[fi])` 取到该 part 的数组再按 li 取值。
    这要求索引与 records 是同一批文件 —— 行数不一致会直接抛错而不是静默错位。

    索引缺失的方向（比如临时加了新数据还没建索引）会退化成权重 1，并打一条 warning：
    宁可不分层，也不要静默地把权重对错样本。
    """
    per_dir = {}
    for f in ds._sharded_files:
        d = _direction_of(f)
        if d is None or d in per_dir:
            continue
        p = os.path.join(index_dir, f'{d}_index.npz')
        per_dir[d] = np.load(p) if os.path.exists(p) else None
        if per_dir[d] is None:
            logger(f'[WARN] 缺索引 {p} —— {d} 方向不做分层采样（权重全 1）')

    n = len(ds.items)
    nir = np.full(n, U16, dtype=np.uint16)
    dur = np.zeros(n, dtype=np.uint32)
    flags = np.zeros(n, dtype=np.uint8)
    covered = np.zeros(n, dtype=bool)

    # 先按 part 归组，避免对每条样本反复 np.load 取数组
    by_file = {}
    for k, (fi, li) in enumerate(ds.items):
        by_file.setdefault(fi, []).append((k, li))
    for fi, pairs in by_file.items():
        path = ds._sharded_files[fi]
        d = _direction_of(path)
        z = per_dir.get(d)
        if z is None:
            continue
        base = os.path.basename(path)
        try:
            a_nir, a_dur, a_fl = (z[f'{base}::nir_x100'], z[f'{base}::dur_ms'],
                                  z[f'{base}::flags'])
        except KeyError:
            logger(f'[WARN] 索引里没有 {base} —— 该 part 不做分层采样')
            continue
        ks = np.array([k for k, _ in pairs], dtype=np.int64)
        ls = np.array([li for _, li in pairs], dtype=np.int64)
        if ls.max(initial=-1) >= a_nir.size:
            raise RuntimeError(
                f'索引与 records 行数不一致: {base} 索引 {a_nir.size} 行，'
                f'但 dataset 要读第 {int(ls.max())} 行。请重跑 build_record_index.py')
        nir[ks], dur[ks], flags[ks] = a_nir[ls], a_dur[ls], a_fl[ls]
        covered[ks] = True

    if not covered.any():
        logger('[WARN] 没有任何样本匹配到索引 —— 退回均匀采样')
        return np.ones(n, dtype=np.float64), None, None

    w, nb, lb = joint_weights(nir, dur, nir_weights, len_weights,
                              min_source_ms=min_source_ms,
                              drop_invalid=~covered | ((flags & 2) != 0),
                              max_repeat=max_repeat, use_ipf=use_ipf)
    # 未覆盖到的样本给平均权重：不分层，但也不丢
    if (~covered).any():
        w[~covered] = w[covered].mean() if covered.any() else 1.0
    return w, nb, lb


def _split_of(path):
    normalized = path.replace('\\', '/')
    for split in ('train', 'val', 'test'):
        if f'/{split}/' in normalized:
            return split
    return None


def _part_names(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values.tolist()]


def _validate_context_index_profile(meta, archive, direction, split, meta_path):
    def require(condition, message):
        if not condition:
            raise RuntimeError(f'invalid S4 index metadata {meta_path}: {message}')

    require(meta.get('index_schema_version') == 2, 'index schema must be 2')
    require(meta.get('direction') == direction,
            f'direction must be {direction!r}')
    require(split is not None and meta.get('split') == split,
            f'split must be {split!r}')
    fingerprint = meta.get('records_fingerprint')
    require(isinstance(fingerprint, dict), 'records_fingerprint is missing')
    require(fingerprint.get('chunk_ms') == 640, 'chunk_ms must be 640')
    require(meta.get('record_schema_version') == 'si-rw-m1-v0',
            'record_schema_version must be si-rw-m1-v0')
    require(meta.get('trajectory_base_chunk_ms') == 640,
            'trajectory_base_chunk_ms must be 640')
    require(meta.get('index_step_merge_m') == 1, 'index_step_merge_m must be 1')
    require(meta.get('codec_spec') == 'cosy3_fsq_25hz',
            'codec_spec must be cosy3_fsq_25hz')
    require(meta.get('codec_frame_ms') == 40.0, 'codec_frame_ms must be 40.0')

    declared = meta.get('fields')
    require(isinstance(declared, dict), 'fields declaration is missing')
    required_dtypes = {
        'n_step': 'uint8', 'speech_frames': 'uint16',
        'dur_ms': 'uint32', 'flags': 'uint8',
    }
    for field, dtype in required_dtypes.items():
        require(declared.get(field) == dtype,
                f'{field} declared dtype must be {dtype}')

    raw_parts = fingerprint.get('parts')
    require(isinstance(raw_parts, list), 'records_fingerprint parts are missing')
    try:
        profile_parts = [(str(name), int(count)) for name, count in raw_parts]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'invalid S4 index metadata {meta_path}: parts must be [name, row_count] pairs') from exc
    require(all(count >= 0 for _, count in profile_parts),
            'part row counts must be nonnegative')
    require(fingerprint.get('num_records') == sum(count for _, count in profile_parts),
            'num_records does not equal the part row count sum')
    require('__parts__' in archive, 'archive parts array is missing')
    archive_parts = _part_names(archive['__parts__'])
    require(archive_parts == [name for name, _ in profile_parts],
            'archive parts do not match metadata parts in order')

    for part, count in profile_parts:
        for field, dtype in required_dtypes.items():
            key = f'{part}::{field}'
            require(key in archive, f'{field} is missing for part {part}')
            values = archive[key]
            require(values.ndim == 1, f'{field} for {part} must be 1-D')
            require(values.dtype == np.dtype(dtype),
                    f'{field} dtype for {part} must be {dtype}, got {values.dtype}')
            require(values.size == count,
                    f'{field} row count for {part} is {values.size}, expected {count}')
    return dict(profile_parts)


def static_context_eligibility_for_dataset(
        ds, index_dir, thinker_ratio, talker_ratio,
        prefix_context_turns=16, talker_trigger_speech_tokens=500,
        min_source_ms=0, max_source_ms=None, max_speech_tokens=None):
    """Return static rank-local Thinker/Talker masks aligned to ``ds.items``.

    Any positive crop ratio requires the schema-2 640ms/m=1 profile and both compact
    fields. With both ratios zero this function deliberately does not inspect the index,
    leaving the production legacy weighting path unchanged.
    """
    n = len(ds.items)
    empty = StaticContextEligibility(
        thinker_mask=np.zeros(n, dtype=bool),
        talker_mask=np.zeros(n, dtype=bool),
        valid_mask=np.zeros(n, dtype=bool),
    )
    if float(thinker_ratio) <= 0 and float(talker_ratio) <= 0:
        return empty
    if prefix_context_turns < 0:
        raise ValueError('prefix_context_turns must be >= 0')
    if talker_trigger_speech_tokens < 0:
        raise ValueError('talker_trigger_speech_tokens must be >= 0')

    fields = {
        'n_step': np.empty(n, dtype=np.uint8),
        'speech_frames': np.empty(n, dtype=np.uint16),
        'dur_ms': np.empty(n, dtype=np.uint32),
        'flags': np.empty(n, dtype=np.uint8),
    }
    profile_digests = {}
    covered = np.zeros(n, dtype=bool)
    by_direction = {}
    for dataset_index, (file_index, line_index) in enumerate(ds.items):
        path = ds._sharded_files[file_index]
        direction = _direction_of(path)
        if direction is None:
            raise RuntimeError(f'cannot infer direction for crop index alignment: {path}')
        by_direction.setdefault(direction, []).append(
            (dataset_index, file_index, int(line_index)))

    for direction, entries in by_direction.items():
        index_path = os.path.join(index_dir, f'{direction}_index.npz')
        meta_path = os.path.join(index_dir, f'{direction}_index_meta.json')
        if not os.path.exists(meta_path):
            raise RuntimeError(f'S4 index metadata is missing: {meta_path}')
        if not os.path.exists(index_path):
            raise RuntimeError(f'S4 index archive is missing: {index_path}')
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'cannot read S4 index metadata {meta_path}: {exc}') from exc

        splits = {_split_of(ds._sharded_files[file_index])
                  for _, file_index, _ in entries}
        if len(splits) != 1:
            raise RuntimeError(
                f'dataset direction {direction} spans inconsistent splits: {sorted(splits)}')
        split = next(iter(splits))
        with np.load(index_path, allow_pickle=False) as archive:
            part_counts = _validate_context_index_profile(
                meta, archive, direction, split, meta_path)
            encoded_meta = json.dumps(
                meta, sort_keys=True, separators=(',', ':'),
                ensure_ascii=False).encode('utf-8')
            profile_digests[direction] = hashlib.sha256(encoded_meta).hexdigest()
            by_part = {}
            for dataset_index, file_index, line_index in entries:
                part = os.path.basename(ds._sharded_files[file_index])
                by_part.setdefault(part, []).append((dataset_index, line_index))
            for part, pairs in by_part.items():
                if part not in part_counts:
                    raise RuntimeError(
                        f'S4 index parts do not cover dataset part {part}')
                row_indices = np.asarray([line for _, line in pairs], dtype=np.int64)
                if (row_indices < 0).any() or row_indices.max(initial=-1) >= part_counts[part]:
                    raise RuntimeError(
                        f'S4 index row coverage failed for {part}: requested row '
                        f'{int(row_indices.max(initial=-1))}, row count {part_counts[part]}')
                output_indices = np.asarray([k for k, _ in pairs], dtype=np.int64)
                for field in fields:
                    fields[field][output_indices] = archive[f'{part}::{field}'][row_indices]
                covered[output_indices] = True

    if not covered.all():
        missing = np.flatnonzero(~covered)
        raise RuntimeError(
            f'S4 index coverage is incomplete for {missing.size} dataset rows; '
            f'first missing dataset index {int(missing[0])}')
    valid = (fields['flags'] & 2) == 0
    valid &= fields['dur_ms'] >= int(min_source_ms)
    if max_source_ms is not None:
        valid &= fields['dur_ms'] <= int(max_source_ms)
    if max_speech_tokens is not None:
        valid &= fields['speech_frames'] <= int(max_speech_tokens)
    return StaticContextEligibility(
        thinker_mask=valid & (fields['n_step'] >= int(prefix_context_turns) + 2),
        talker_mask=valid & (
            fields['speech_frames'] > int(talker_trigger_speech_tokens)),
        valid_mask=valid,
        index_profile_digests=profile_digests,
    )


def summarize(w, nb, lb, nir_x100, tag='', nir_weights=None, len_weights=None):
    """打印加权前后的桶份额、NIR 均值、有效样本量与最大重复倍率。

    最大重复倍率比 ESS 更能说明过拟合风险：ESS 是全局二阶量，会被大量被降权的
    样本拉低（看起来很吓人），而真正的风险是「某一小批样本被反复采」。
    """
    nw = NIR_WEIGHTS if nir_weights is None else nir_weights
    lw = LEN_WEIGHTS if len_weights is None else len_weights
    n = w.size
    pos = w > 0
    p = w / w.sum() if w.sum() > 0 else w
    nir = np.where(nir_x100 == U16, np.nan, nir_x100.astype(np.float64) / 100.0)
    le50 = np.isfinite(nir) & (nir <= 50.0)      # 论文口径：矩统计只用 NIR<=50%

    print(f'{tag}样本 {n}, 权重>0 {int(pos.sum())} ({100 * pos.mean():.1f}%)')
    print(f'  {"桶":<10}{"自然":>8}{"加权后":>9}{"目标":>8}{"桶内NIR":>9}')
    for i, nname in enumerate(NIR_NAMES):
        m = (nb == i) & pos
        nat = float(m.mean())
        got = float(p[m].sum())
        mu = float(np.nanmean(nir[m])) if m.any() else 0.0
        print(f'  {nname:<10}{100 * nat:7.1f}%{100 * got:8.1f}%'
              f'{100 * nw[nname]:7.0f}%{mu:8.2f}%')
    for j, lname in enumerate(LEN_NAMES):
        m = (lb == j) & pos
        print(f'  {lname:<10}{100 * float(m.mean()):7.1f}%{100 * float(p[m].sum()):8.1f}%'
              f'{100 * lw[lname]:7.0f}%')

    mu_raw = float(np.nanmean(nir[le50 & pos]))
    sel = le50 & pos
    mu_w = float((p[sel] * nir[sel]).sum() / p[sel].sum()) if p[sel].sum() else 0.0
    ess = effective_sample_size(w)
    mx = float(w[pos].max() / w[pos].mean()) if pos.any() else 0.0
    print(f'  NIR(<=50%) 均值: 加权前 {mu_raw:.2f}%  加权后 {mu_w:.2f}%')
    print(f'  有效样本量 {ess:.0f}/{int(pos.sum())} = {100 * ess / max(int(pos.sum()), 1):.1f}%'
          f'  |  单条最大被采次数 {mx:.2f}x  <- 这个才是过拟合风险的直接度量')
    return {'nir_mean_raw': mu_raw, 'nir_mean_weighted': mu_w,
            'ess': ess, 'ess_ratio': ess / max(int(pos.sum()), 1),
            'max_repeat': mx, 'n_positive': int(pos.sum())}
