#!/usr/bin/env python3
"""全量构建记录属性索引：NIR / 句长 / 延迟 / 对齐质量，供采样与 QC 共用。

为什么要一个索引，而不是训练时现算
----------------------------------
NIR 需要读 `10_word_ready.json`（每条一个小文件，~7KB）。训练时现读会让每个
dataloader worker 多一次随机小文件 IO，且 NIR 是**样本固有属性**、与 m 无关，
算一次就够。索引存成定长 numpy 数组，302 万条约 30 MB，各 rank 直接 mmap。

产出（每方向三个文件）
----------------------
`<out>/<dir>_index.npz`  —— 按 **part 文件名 + 行号** 对齐的定长数组：
    nir_x100   uint16   NIR × 100（0 表示完全单调；65535 = 缺失）
    dur_ms     uint32   源时长
    ready0_ms  uint16   首个目标词的 raw_ready_ms（>65534 截断）
    n_step     uint8    m=1 的步数
    n_wait     uint8    m=1 的 WAIT 步数
    speech_frames uint16 原始 25Hz Talker codec token 数
    lead_ms    uint16   前导等待（首个 WRITE 之前）
    inter_ms   uint16   内部等待（首个 WRITE 之后）
    laal_ms    uint16   策略口径 LAAL
    flags      uint8    bit0 = 首步即 WRITE，bit1 = trajectory 不合法（应跳过）
    unalign_x100 uint8  未对齐目标词占比 × 100（截断到 255）

`<out>/<dir>_index_stats.json` —— 分桶边界、分布、以及 P1-2 的归因相关性。

`<out>/<dir>_index_meta.json` —— schema 2 严格 profile：方向/split、源 manifest 的
有序 part/行数、640ms trajectory base、m=1 索引步长、CosyVoice3 25Hz codec 规格，
以及所有索引字段的声明 dtype。正 crop 比例的训练加载会强制校验这份 metadata。

用法
----
    PY=/opt/dlami/nvme/zeyangsong/envs/si_s4/bin/python
    cd projects/streaming-si-0.6b
    $PY scripts/build_record_index.py \
        --records $SI_S4/records/en2zh/chunk_640ms/train/records \
        --direction en2zh --data_root $SI_S4/audio --workers 32
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "custom"))

from streaming_si.data.si_traj_ops import compute_nir, latency_metrics, steps_until_thinker_eos  # noqa: E402

U16 = 65535
EXPECTED_RECORD_SCHEMA = 'si-rw-m1-v0'
EXPECTED_BASE_CHUNK_MS = 640
EXPECTED_CODEC_SPEC = 'cosy3_fsq_25hz'
EXPECTED_CODEC_FRAME_MS = 40.0


def clip16(v):
    return U16 if v is None or v > U16 - 1 else int(v)


def checked_unsigned(v, bits, field):
    value = int(v)
    maximum = (1 << bits) - 1
    if not 0 <= value <= maximum:
        raise ValueError(f'{field}={value} does not fit uint{bits}')
    return value


def record_profile(rec):
    trajectory = rec.get('trajectory') or {}
    codec = trajectory.get('codec') or {}
    return {
        'record_schema_version': rec.get('schema_version'),
        'trajectory_base_chunk_ms': trajectory.get('base_chunk_ms'),
        'codec_spec': codec.get('spec'),
        'codec_frame_ms': codec.get('frame_ms'),
    }


def scan_part(job):
    """扫一个 part 文件；返回定长数组。子进程入口，故参数打包成 tuple。"""
    path, data_root = job
    (nir, dur, ready0, nstep, nwait, speech_frames, lead, inter, laal, flags,
     unalign) = ([] for _ in range(11))
    observed_profile = None
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            rec = json.loads(line)
            current_profile = record_profile(rec)
            if observed_profile is None:
                observed_profile = current_profile
            elif current_profile != observed_profile:
                changed = next(k for k in observed_profile
                               if observed_profile[k] != current_profile[k])
                raise ValueError(
                    f'{os.path.basename(path)}:{line_number} uniform record profile '
                    f'violation for {changed}: {current_profile[changed]!r} != '
                    f'{observed_profile[changed]!r}')
            bad = 0
            steps = steps_until_thinker_eos(rec['trajectory']['steps'])
            d = int(rec.get('source_duration_ms') or 0)
            if steps is None:
                bad = 2
                mt = {}
            else:
                mt = latency_metrics(steps, d)

            # NIR 与首词 ready：来自词级中间产物
            v_nir, v_ready, v_unalign = None, None, 0
            src = rec.get('source_audio') or ''
            if src:
                wr = os.path.join(data_root, os.path.dirname(src), '10_word_ready.json')
                try:
                    with open(wr) as g:
                        words = json.load(g).get('words', [])
                    v_nir = compute_nir(words)
                    if words:
                        # 只看**已对齐**词的最早 ready。未对齐词的 ready 由 bridge_next
                        # 从右邻借来，不代表真实对齐关系，拿它判 FA 崩塌会误报。
                        al = [w['raw_ready_ms'] for w in words
                              if w.get('dep_src_word_idxs')
                              and w.get('raw_ready_ms') is not None]
                        v_ready = min(al) if al else None
                        n_un = sum(1 for w in words if not w.get('dep_src_word_idxs'))
                        v_unalign = min(255, round(100 * n_un / len(words)))
                except Exception:
                    pass

            nir.append(U16 if v_nir is None else min(U16 - 1, round(v_nir * 100)))
            dur.append(d)
            ready0.append(clip16(v_ready))
            nstep.append(checked_unsigned(mt.get('n_step', 0), 8, 'n_step'))
            nwait.append(checked_unsigned(mt.get('n_wait', 0), 8, 'n_wait'))
            raw_frames = checked_unsigned(
                rec['trajectory']['raw_target_num_frames'], 16, 'speech_frames')
            codes = rec.get('target_codec_codes')
            if steps is not None and codes is not None and raw_frames != len(codes):
                raise ValueError(
                    f'{os.path.basename(path)}:{line_number} raw_target_num_frames '
                    f'{raw_frames} != len(target_codec_codes) {len(codes)}')
            speech_frames.append(raw_frames)
            lead.append(clip16(mt.get('lead_wait_ms')))
            inter.append(clip16(mt.get('inter_wait_ms')))
            laal.append(clip16(None if mt.get('laal') is None else max(0, mt['laal'])))
            flags.append(bad | (1 if mt.get('first_step_is_write') else 0))
            unalign.append(v_unalign)

    return os.path.basename(path), {
        'nir_x100': np.array(nir, dtype=np.uint16),
        'dur_ms': np.array(dur, dtype=np.uint32),
        'ready0_ms': np.array(ready0, dtype=np.uint16),
        'n_step': np.array(nstep, dtype=np.uint8),
        'n_wait': np.array(nwait, dtype=np.uint8),
        'speech_frames': np.array(speech_frames, dtype=np.uint16),
        'lead_ms': np.array(lead, dtype=np.uint16),
        'inter_ms': np.array(inter, dtype=np.uint16),
        'laal_ms': np.array(laal, dtype=np.uint16),
        'flags': np.array(flags, dtype=np.uint8),
        'unalign_x100': np.array(unalign, dtype=np.uint8),
    }, observed_profile


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return 0.0
    da, db = a - a.mean(), b - b.mean()
    den = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / den) if den else 0.0


def load_source_manifest(records, parts, direction):
    if os.path.isdir(records):
        records_dir = records
    else:
        part_dirs = {os.path.dirname(os.path.abspath(path)) for path in parts}
        if len(part_dirs) != 1:
            raise ValueError('schema-2 index construction requires parts from one records directory')
        records_dir = next(iter(part_dirs))
    manifest_path = os.path.join(records_dir, '_manifest.json')
    if not os.path.exists(manifest_path):
        raise ValueError(f'source manifest is missing: {manifest_path}')
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get('direction') != direction:
        raise ValueError(
            f'source manifest direction {manifest.get("direction")!r} != {direction!r}')
    if not isinstance(manifest.get('split'), str):
        raise ValueError('source manifest split is missing')
    if manifest.get('chunk_ms') != EXPECTED_BASE_CHUNK_MS:
        raise ValueError(
            f'source manifest chunk_ms={manifest.get("chunk_ms")!r}; expected 640')
    manifest_parts = manifest.get('parts')
    if not isinstance(manifest_parts, list):
        raise ValueError('source manifest parts are missing')
    ordered = []
    for entry in manifest_parts:
        name = entry.get('file')
        count = entry.get('num_records')
        if not isinstance(name, str) or not isinstance(count, int) or count < 0:
            raise ValueError('source manifest parts require file and nonnegative num_records')
        ordered.append((name, count))
    if len({name for name, _ in ordered}) != len(ordered):
        raise ValueError('source manifest parts contain duplicate file names')
    if manifest.get('num_records') != sum(count for _, count in ordered):
        raise ValueError('source manifest num_records does not equal its part row count sum')
    selected_names = [os.path.basename(path) for path in parts]
    selected_names_set = set(selected_names)
    selected = [item for item in ordered if item[0] in selected_names_set]
    if [name for name, _ in selected] != selected_names:
        raise ValueError('selected part order does not match the source manifest')
    return manifest, selected


def build_index_metadata(manifest, manifest_parts, profile, results):
    expected_counts = dict(manifest_parts)
    actual_names = [name for name, _, _ in results]
    if actual_names != [name for name, _ in manifest_parts]:
        raise ValueError('emitted index parts do not match the source manifest order')
    for name, arrays, _ in results:
        actual = int(arrays['dur_ms'].size)
        if actual != expected_counts[name]:
            raise ValueError(
                f'source manifest row count mismatch for {name}: '
                f'manifest {expected_counts[name]}, scanned {actual}')
        sizes = {values.size for values in arrays.values()}
        if sizes != {actual}:
            raise ValueError(f'index fields have inconsistent row counts for {name}')
    expected_profile = {
        'record_schema_version': EXPECTED_RECORD_SCHEMA,
        'trajectory_base_chunk_ms': EXPECTED_BASE_CHUNK_MS,
        'codec_spec': EXPECTED_CODEC_SPEC,
        'codec_frame_ms': EXPECTED_CODEC_FRAME_MS,
    }
    if profile != expected_profile:
        raise ValueError(f'unsupported record profile: {profile!r}; expected {expected_profile!r}')
    fields = {key: str(values.dtype) for key, values in results[0][1].items()}
    return {
        'index_schema_version': 2,
        'direction': manifest['direction'],
        'split': manifest['split'],
        'records_fingerprint': {
            'chunk_ms': manifest['chunk_ms'],
            'parts': [[name, count] for name, count in manifest_parts],
            'num_records': sum(count for _, count in manifest_parts),
        },
        **expected_profile,
        'index_step_merge_m': 1,
        'fields': fields,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True, help='records 目录 / glob')
    ap.add_argument('--direction', required=True)
    ap.add_argument('--data_root', default=None,
                    help='默认由 records 路径上推两级（records -> <dir> -> root）')
    ap.add_argument('--out', default='../out/s4_index')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit_parts', type=int, default=0, help='>0 时只扫前 N 个 part（调试）')
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(args.records, 'part-*.jsonl'))
                   if os.path.isdir(args.records) else glob.glob(args.records))
    if args.limit_parts:
        parts = parts[:args.limit_parts]
    if not parts:
        print(f'[FAIL] 没找到 part 文件: {args.records}')
        return 1
    manifest, manifest_parts = load_source_manifest(args.records, parts, args.direction)
    root = args.data_root or os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(parts[0]))))
    print(f'{args.direction}: {len(parts)} 个 part, data_root={root}, workers={args.workers}')

    jobs = [(p, root) for p in parts]
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap(scan_part, jobs), 1):
            results.append(r)
            print(f'  [{i}/{len(parts)}] {r[0]}  n={len(r[1]["dur_ms"])}', flush=True)
    results.sort(key=lambda x: x[0])

    profiles = [profile for _, _, profile in results]
    if any(profile != profiles[0] for profile in profiles[1:]):
        raise ValueError('uniform record profile violation across source parts')
    meta = build_index_metadata(manifest, manifest_parts, profiles[0], results)

    os.makedirs(args.out, exist_ok=True)
    # npz 的 key 用 "<part>::<field>"，读回时按 part 名分组
    payload = {f'{name}::{k}': v for name, d, _ in results for k, v in d.items()}
    payload['__parts__'] = np.array([n for n, _, _ in results])
    npz = os.path.join(args.out, f'{args.direction}_index.npz')
    np.savez(npz, **payload)
    print(f'\n写出 {npz} ({os.path.getsize(npz) / 2**20:.1f} MiB)')
    mp = os.path.join(args.out, f'{args.direction}_index_meta.json')
    with open(mp, 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f'写出 {mp}')

    # ---------------- 统计与归因 ----------------
    cat = {k: np.concatenate([d[k] for _, d, _ in results])
           for k in results[0][1]}
    n = cat['dur_ms'].size
    ok = (cat['flags'] & 2) == 0
    has_nir = ok & (cat['nir_x100'] != U16)
    nir = cat['nir_x100'][has_nir].astype(np.float64) / 100.0
    print(f'\n样本 {n}, trajectory 合法 {int(ok.sum())} ({100 * ok.mean():.2f}%), '
          f'有 NIR {int(has_nir.sum())} ({100 * has_nir.mean():.2f}%)')
    print(f'静态候选: n_step>=18 {int((cat["n_step"] >= 18).sum())} | '
          f'speech_frames>500 {int((cat["speech_frames"] > 500).sum())}')

    # NIR 分桶用**固定语义边界**，不用分位数。
    # 原因：31.5% 的样本 NIR 恰好为 0（词对齐完全单调），p25 = 0，分位数边界退化 ——
    # np.digitize 会把 NIR=0 全部塞进第 1 桶，`low` 桶恒为空。
    # 固定边界还有个好处：两个方向、以及将来数据增补后，桶的含义不变，可比。
    NIR_EDGES = (1e-9, 10.0, 25.0)     # low = 恰好 0；(0,10]；(10,25]；>25
    names = ['low(=0)', 'mid_low(0-10]', 'mid_high(10-25]', 'high(>25)']
    bucket = np.digitize(nir, NIR_EDGES)
    print(f'NIR: mean={nir.mean():.2f}% p50={np.percentile(nir, 50):.2f}% '
          f'p90={np.percentile(nir, 90):.2f}% max={nir.max():.2f}%')
    nir_dist, nir_bucket_mean = {}, {}
    print(f'  {"桶":<18}{"占比":>8}{"桶内均值":>10}')
    for i, nm in enumerate(names):
        m = bucket == i
        cur = float(m.mean())
        mu = float(nir[m].mean()) if m.any() else 0.0
        nir_dist[nm], nir_bucket_mean[nm] = cur, mu
        print(f'  {nm:<18}{100 * cur:7.2f}%{mu:9.2f}%')

    # 与论文对齐的口径：论文的 NIR 矩统计只用 NIR <= 50% 的子集
    # （Table tab:appendix_nir_random_stats 的脚注）。这是唯一可比的数字。
    sub = nir[nir <= 50.0]
    paper_ref = {'en2zh': {'random': (9.66, 131.46), 'stratified': (13.79, 146.64)},
                 'zh2en': {'random': (9.92, 147.47), 'stratified': (15.86, 160.69)}}
    key = 'en2zh' if 'en2zh' in args.direction else 'zh2en'
    rnd, strat = paper_ref[key]['random'], paper_ref[key]['stratified']
    print(f'\n  [论文可比口径] NIR<=50% 子集 n={sub.size} ({100 * sub.size / nir.size:.1f}%): '
          f'mean={sub.mean():.2f}% var={sub.var():.1f}')
    print(f'    论文 random     mean={rnd[0]:.2f}% var={rnd[1]:.1f}  '
          f'<- 我们未加权时应当接近这一行')
    print(f'    论文 stratified mean={strat[0]:.2f}% var={strat[1]:.1f}  '
          f'<- 加权采样后要达到的目标')
    print(f'    结论：NIR 分层是**提高**平均 NIR（{rnd[0]:.2f} -> {strat[0]:.2f}），'
          f'机制是压低 NIR=0 的那 {100 * nir_dist[names[0]]:.0f}% 质量')

    # 等待位置分解（全量）：前导 = 首个 WRITE 之前，内部 = 之后
    ld, it = cat['lead_ms'][ok].astype(np.float64), cat['inter_ms'][ok].astype(np.float64)
    ld[ld == U16] = np.nan
    it[it == U16] = np.nan
    lm, im = float(np.nanmean(ld)), float(np.nanmean(it))
    print(f'\n等待位置分解: 前导 {lm:.0f}ms ({100 * lm / max(lm + im, 1e-9):.0f}%) | '
          f'内部 {im:.0f}ms ({100 * im / max(lm + im, 1e-9):.0f}%) | '
          f'总 {lm + im:.0f}ms')

    dur = cat['dur_ms'][ok]
    len_edges = (4000, 8000)
    len_names = ['short(<4s)', 'medium(4-8s)', 'long(>8s)']
    len_want = [0.1, 0.5, 0.4]
    lb = np.digitize(dur, len_edges)
    print('\n句长分桶（论文权重 short/medium/long = 0.1/0.5/0.4）:')
    len_dist = {}
    for i, nm in enumerate(len_names):
        cur = float((lb == i).mean())
        len_dist[nm] = cur
        mult = len_want[i] / max(cur, 1e-9)
        flag = '  <- 不可行，见下' if mult > 4 else ''
        print(f'  {nm:<14} {100 * cur:5.1f}%  目标 {len_want[i]:.2f}  '
              f'需要 {mult:5.2f}x{flag}')
    if len_want[2] / max(len_dist[len_names[2]], 1e-9) > 4:
        print(f'  注：我们 >8s 的样本只有 {100 * len_dist[len_names[2]]:.1f}%，'
              f'要凑到 40% 需重复采样 {len_want[2] / len_dist[len_names[2]]:.1f} 倍，'
              f'必然过拟合那批样本。论文的长度权重**不能照搬**，需按我们的分布重定。')

    # P1-2 归因：源侧 FA 崩塌（首词被压得极短 -> ready 过早）是否是首步即 WRITE 的主因
    fw = ((cat['flags'] & 1) == 1)
    r0 = cat['ready0_ms']
    valid_r0 = ok & (r0 != U16)
    base_fw = float(fw[ok].mean())
    print(f'\nP1-2 归因：首个**已对齐**目标词的 raw_ready_ms（n={int(valid_r0.sum())}）')
    print(f'  首步即 WRITE 总占比 = {100 * base_fw:.1f}%')
    print(f'  {"阈值":<14}{"占比":>8}{"其中首步WRITE":>14}{"其余样本":>10}{"可解释份额":>11}')
    attribution = {}
    for th in (50, 100, 200, 300, 500):
        sel = valid_r0 & (r0 < th)
        if not sel.any():
            continue
        p_in = float(fw[sel].mean())
        p_out = float(fw[valid_r0 & ~sel].mean())
        # 这批样本贡献的「首步即 WRITE」占全部首步即 WRITE 的比例
        share_of_fw = float((sel & fw).sum()) / max(float(fw[ok].sum()), 1.0)
        attribution[f'ready0<{th}ms'] = {
            'share': float(sel.mean()), 'p_first_write': p_in,
            'p_first_write_others': p_out, 'share_of_all_first_write': share_of_fw}
        print(f'  ready0<{th:>4}ms {100 * sel.mean():6.2f}%'
              f'{100 * p_in:13.1f}%{100 * p_out:9.1f}%{100 * share_of_fw:10.2f}%')
    print(f'  corr(ready0, 首步即WRITE) = {pearson(r0[valid_r0], fw[valid_r0]):+.3f}')
    print(f'  未对齐目标词占比: mean={cat["unalign_x100"][ok].mean():.1f}% '
          f'p90={np.percentile(cat["unalign_x100"][ok], 90):.0f}%')

    stats = {'direction': args.direction, 'n_records': int(n),
             'n_valid_traj': int(ok.sum()), 'n_with_nir': int(has_nir.sum()),
             'n_static_thinker_ge18': int((cat['n_step'] >= 18).sum()),
             'n_static_talker_gt500': int((cat['speech_frames'] > 500).sum()),
             'nir_edges': list(NIR_EDGES), 'nir_dist': nir_dist,
             'nir_bucket_mean': nir_bucket_mean,
             'nir_mean': float(nir.mean()), 'nir_p50': float(np.percentile(nir, 50)),
             'nir_le50_mean': float(sub.mean()), 'nir_le50_var': float(sub.var()),
             'paper_random': list(rnd), 'paper_stratified': list(strat),
             'len_edges': list(len_edges), 'len_dist': len_dist,
             'len_target': dict(zip(len_names, len_want)),
             'first_write_share': base_fw,
             'lead_wait_ms_mean': lm, 'inter_wait_ms_mean': im,
             'attribution_ready0': attribution,
             'corr_ready0_firstwrite': pearson(r0[valid_r0], fw[valid_r0])}
    sp = os.path.join(args.out, f'{args.direction}_index_stats.json')
    with open(sp, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f'写出 {sp}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
