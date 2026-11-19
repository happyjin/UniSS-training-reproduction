"""把三个数据集的清单合并成一份，再按「总时长均衡」切成 N 个分片。

三个系统都读同一份合并清单（每行自带 `tgt_lang`），这样 7 张卡一次把 1193 句分完，
不会出现「某个方向先跑完、卡闲着」的情况。切分用最长优先 + 贪心填最短分片，
保证各分片音频总时长接近（耗时与音频时长近似成正比）。

用法：python make_shards.py <work_dir> <num_shards>
"""
import json
import os
import sys


def main():
    work, n = sys.argv[1], int(sys.argv[2])
    man_dir = os.path.join(work, 'manifests')
    rows = []
    for name in ('realsi_en2zh', 'realsi_zh2en', 'acl6060_en2zh'):
        p = os.path.join(man_dir, f'{name}.jsonl')
        for l in open(p, encoding='utf-8'):
            if l.strip():
                r = json.loads(l)
                r['dataset'] = name
                rows.append(r)
    assert len({r['id'] for r in rows}) == len(rows), 'id 在三个数据集间必须唯一'
    with open(os.path.join(man_dir, 'all.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    shards = [[] for _ in range(n)]
    load = [0.0] * n
    for r in sorted(rows, key=lambda x: -x['source_sec']):
        k = load.index(min(load))
        shards[k].append(r)
        load[k] += r['source_sec']
    out = os.path.join(work, 'shards')
    os.makedirs(out, exist_ok=True)
    for k, sh in enumerate(shards):
        with open(os.path.join(out, f'all.{k}.jsonl'), 'w', encoding='utf-8') as f:
            for r in sh:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'分片 {k}: {len(sh)} 句，{load[k] / 60:.1f} 分钟')
    print(f'合计 {len(rows)} 句，{sum(load) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
