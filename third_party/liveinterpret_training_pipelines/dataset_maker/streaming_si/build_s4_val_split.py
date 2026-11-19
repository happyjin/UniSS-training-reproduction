"""切出 S4 的 held-out 验证集（**按源文本分组**，不是按 record_id / part 文件）。

为什么必须按源文本分组：同一句源文本在库里会重复出现（en→zh 实测重复率 21.52%），
且重复项散落在不同 part 里。按 id 或按文件切 val，val 里的句子有很大概率也在 train
里出现过 —— BLEU/CER 会**虚高**，而且不会有任何报错提示。

产物（每个方向一份）：
  * `<out>/<dir>_val.jsonl`    —— 抽出来的完整记录，直接喂 `S4StreamingDataset`
  * `<out>/<dir>_val_keys.json` —— val 占用的源文本指纹集合；训练集用
    `S4StreamingDataset(..., exclude_keys=...)` 把这些组整体剔掉

用法:
    PY=/opt/dlami/nvme/zeyangsong/envs/si_s4/bin/python
    cd projects/streaming-si-0.6b
    $PY scripts/build_s4_val_split.py \
        --records $SI_S4/records/en2zh/chunk_640ms/val/records \
        --direction en2zh --n_val 1500 --data_root $SI_S4/audio
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "custom"))

from streaming_si.data.s4_streaming_dataset import S4StreamingDataset, source_key  # noqa: E402


def scan(files):
    """扫一遍 records，按源文本指纹分组。Returns: (groups, n_total, audio_root)。

    只保留定位信息（文件下标 + 字节偏移），不缓存记录本身 —— 单方向 108 万条 × 4.5KB
    有 5GB，整表进内存会 OOM。

    顺带取出音频根：val jsonl 会被写到数据树**之外**（`out/s4_val_640_si_s4/`），
    `S4StreamingDataset` 从 `si_s4/records/.../records/part-*.jsonl` 上推猜不出
    `si_s4/audio`。优先用 `--data_root`；没有时才退回记录里的 `audio_root_hint`
    （旧 jsonl 里可能仍是本机 `rw_data` 绝对路径，换机器会错）。
    """
    groups, n_total, audio_root = {}, 0, None
    for fi, path in enumerate(files):
        with open(path, 'rb') as f:
            pos = 0
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    groups.setdefault(source_key(rec), []).append((fi, pos, len(line)))
                    audio_root = audio_root or rec.get('audio_root_hint')
                    n_total += 1
                pos += len(line)
        print(f'  扫完 {os.path.basename(path)}: 累计 {n_total} 条 / {len(groups)} 组', flush=True)
    return groups, n_total, audio_root


def pick(groups, n_val):
    """按指纹字典序**确定性**地选组，直到凑够 n_val 条。

    确定性很重要：重跑要能得到同一份 val，否则续训/复评会悄悄换掉 held-out。
    优先选小组（多数是唯一句），避免一个高频重复组一口气吃掉整个配额。
    """
    ordered = sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
    chosen, n = {}, 0
    for key, locs in ordered:
        if n >= n_val:
            break
        chosen[key] = locs
        n += len(locs)
    return chosen, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True, help='records 目录 / glob / 逗号分隔')
    ap.add_argument('--direction', required=True, help='方向名，用作输出文件前缀，如 en2zh')
    ap.add_argument('--out', default='../out/s4_val_640_si_s4', help='输出目录')
    ap.add_argument('--n_val', type=int, default=1500, help='目标 val 条数（按组取，会略微超出）')
    ap.add_argument('--data_root', default=None,
                    help='音频相对路径根（si_s4/audio）。覆盖 jsonl 里的 audio_root_hint')
    args = ap.parse_args()

    files = S4StreamingDataset._expand(args.records)
    if not files:
        raise SystemExit(f'没找到 records: {args.records}')
    print(f'[{args.direction}] {len(files)} 个 part，开始扫描…')
    groups, n_total, hinted = scan(files)
    if args.data_root:
        audio_root = os.path.abspath(args.data_root)
    else:
        audio_root = hinted or S4StreamingDataset._infer_root(files[0])
    dup = sum(len(v) for v in groups.values() if len(v) > 1)
    print(f'[{args.direction}] 共 {n_total} 条 / {len(groups)} 个源文本组；'
          f'处于重复组中的记录 {dup} 条（{dup / max(n_total, 1) * 100:.2f}%）')

    chosen, n_val = pick(groups, args.n_val)
    os.makedirs(args.out, exist_ok=True)
    jsonl_path = os.path.join(args.out, f'{args.direction}_val.jsonl')
    keys_path = os.path.join(args.out, f'{args.direction}_val_keys.json')

    # 按 (文件, 偏移) 排序后顺序读，避免在 5GB 文件上乱序 seek
    locs = sorted((loc, key) for key, ls in chosen.items() for loc in ls)
    with open(jsonl_path + '.tmp', 'wb') as out:
        cur_fi, fp = None, None
        for (fi, pos, ln), _ in locs:
            if fi != cur_fi:
                fp and fp.close()
                fp, cur_fi = open(files[fi], 'rb'), fi
            fp.seek(pos)
            out.write(fp.read(ln))
        fp and fp.close()
    os.replace(jsonl_path + '.tmp', jsonl_path)     # 原子替换，训练侧不会读到半截文件

    meta = {'direction': args.direction, 'n_val': n_val, 'n_val_groups': len(chosen),
            'n_total': n_total, 'n_groups': len(groups), 'files': files,
            'audio_root': audio_root, 'val_keys': sorted(chosen)}
    with open(keys_path + '.tmp', 'w') as f:
        json.dump(meta, f, indent=1)
    os.replace(keys_path + '.tmp', keys_path)
    print(f'[{args.direction}] val {n_val} 条 / {len(chosen)} 组 -> {jsonl_path}')
    print(f'[{args.direction}] 指纹清单 -> {keys_path}'
          f'（训练侧传 --val_keys 即可整组剔除）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
