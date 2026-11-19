#!/usr/bin/env python3
"""还原各补跑 worker 的待处理队列，估算现状与重新分配两种方案的完成时间。

分配规则与 tts_omni_driver.py 完全一致：list_items 顺序 → 筛选 → Random(42).shuffle → i % NWT。
单项记录数按「已完成数据的 记录/GB（ZH、KO）或 记录/tar（JA）」估算，属于估计值。
"""
import json, random, datetime as dt, collections
import boto3

B = 'stevehguo-ap-southeast-3'
BASE = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS'
SOURCES = [('ZH', f'{BASE}/zh/emilia/', '.tar'), ('JA', f'{BASE}/jp/audio/', '.tar'),
           ('KO', f'{BASE}/ko/patterned_utterance_with_number/data/', '.parquet')]
K8S = '/Users/stevehguo/Documents/coding-project/liveinterpret_training_pipelines/dataset_maker/simuls2st_omni/k8s'

s3 = boto3.client('s3', region_name='ap-southeast-3')
items = []
for lang, pre, ext in SOURCES:
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=B, Prefix=pre):
        for o in page.get('Contents', []):
            if o['Key'].endswith(ext):
                items.append({'lang': lang, 'key': o['Key'], 'gb': o['Size'] / 1e9})
assert len(items) == 1152, len(items)

done_now = {l.strip() for l in open('/tmp/done_now.txt') if l.strip()}
prog = {}; fin = {}
for l in open('/tmp/wprog.jsonl'): prog.update(json.loads(l))
for l in open('/tmp/wfin.jsonl'): fin.update(json.loads(l))

# ---- 记录数估算：用审计时的 union 记录数 / 已完成项体量 ----
union = {'ZH': 18_269_396, 'JA': 7_518_186, 'KO': 3_596_698}
done_items = [it for it in items if it['key'] in done_now]
gb = collections.Counter(); cnt = collections.Counter()
for it in done_items:
    gb[it['lang']] += it['gb']; cnt[it['lang']] += 1
per_gb = {'ZH': union['ZH'] / gb['ZH'], 'KO': union['KO'] / gb['KO']}
per_tar_ja = union['JA'] / cnt['JA']
est = lambda it: it['gb'] * per_gb[it['lang']] if it['lang'] != 'JA' else per_tar_ja
print(f'估算口径: ZH {per_gb["ZH"]:,.0f} 条/GB | KO {per_gb["KO"]:,.0f} 条/GB | JA {per_tar_ja:,.0f} 条/tar')

# ---- 还原队列 ----
def shuffled(lst):
    lst = list(lst); random.Random(42).shuffle(lst); return lst

sweep_keys = {l.strip() for l in open(f'{K8S}/sweep_keys.txt') if l.strip()}
sw = shuffled([it for it in items if it['key'] in sweep_keys])
queues = {}
for w in range(32):
    pod = 'tts-omni-sweep1' if w < 16 else 'tts-omni-sweep2'
    queues[f'{pod}:{w}'] = [it for i, it in enumerate(sw) if i % 32 == w]

start = dt.datetime(2026, 11, 2, 2, 11, 8, tzinfo=dt.timezone.utc).timestamp()
main_done_at_start = set()
for l in open('/tmp/main_done_mtime.txt'):
    t, k = l.split(' ', 1)
    if float(t) < start: main_done_at_start.add(k.strip())
cu = shuffled([it for i, it in enumerate(items) if 40 <= i % 48 <= 47])
for w in (2, 4):
    mine = [it for i, it in enumerate(cu) if i % 32 == w]
    queues[f'tts-omni-catchup:{w}'] = [it for it in mine if it['key'] not in main_done_at_start]

# ---- 速率：按当前项类型取各 worker 实测速率的中位数 ----
rates = collections.defaultdict(list)
cur_of = {}
for wk, q in queues.items():
    f = fin.get(wk)
    if f is None or f >= len(q):
        continue
    cur_of[wk] = q[f]
    r = prog.get(wk, {}).get('rate', 0)
    if r > 0: rates[q[f]['lang']].append(r)
med = {k: sorted(v)[len(v) // 2] for k, v in rates.items()}
print('实测速率中位数(条/秒/worker):', {k: round(v, 1) for k, v in med.items()})

# ---- 方案 A：现状 ----
finish = {}; cur_left = {}; queued = []
for wk, q in queues.items():
    f = fin.get(wk, 0)
    if wk not in cur_of:
        finish[wk] = 0; continue
    cur = cur_of[wk]
    left = max(est(cur) - prog.get(wk, {}).get('cur_done', 0), est(cur) * 0.05)
    r = prog[wk]['rate'] or med[cur['lang']]
    cur_left[wk] = left / r / 3600
    rest = q[f + 1:]
    queued += rest
    finish[wk] = cur_left[wk] + sum(est(it) / med[it['lang']] for it in rest) / 3600
crit = max(finish, key=finish.get)
print(f'\n[A 现状] 剩余记录 ≈ {sum(est(it) for it in queued) + sum(max(est(cur_of[w]) - prog[w]["cur_done"], 0) for w in cur_of):,.0f} 条；'
      f'排队未开始的项 {len(queued)} 个')
print(f'[A 现状] 完成时间 ≈ {finish[crit]:.1f} 小时（瓶颈 {crit}，当前项 {cur_of[crit]["key"].split("/")[-1]}，'
      f'其后还排 {len(queues[crit]) - fin[crit] - 1} 项）')
top = sorted(finish.items(), key=lambda x: -x[1])[:5]
print('   最晚的 5 个 worker:', [(k.split('-')[-1], round(v, 1)) for k, v in top])
print(f'   当前项最晚结束 ≈ {max(cur_left.values()):.1f} 小时')

# ---- 方案 B：排队项交给 i-04ed 的 N 个新 worker（LPT 贪心），在途项原地跑完 ----
for n in (16, 32):
    boot = 0.25
    heap = [boot] * n
    for it in sorted(queued, key=lambda x: -est(x) / med[x['lang']]):
        i = heap.index(min(heap)); heap[i] += est(it) / med[it['lang']] / 3600
    mk = max(max(heap), max(cur_left.values()))
    print(f'[B 重新分配] i-04ed 起 {n} 个 worker 接 {len(queued)} 个排队项 → 完成 ≈ {mk:.1f} 小时'
          f'（新 worker 最晚 {max(heap):.1f}h，在途项最晚 {max(cur_left.values()):.1f}h）')

# ---- 方案 B'：按驱动真实规则（only-keys 筛选 → shuffle(42) → i % N）分配排队项 ----
qkeys = {it['key'] for it in queued}
qs = shuffled([it for it in items if it['key'] in qkeys])
for n in (24, 32):
    t = [0.25 + sum(est(it) / med[it['lang']] for i, it in enumerate(qs) if i % n == w) / 3600 for w in range(n)]
    print(f"[B' 驱动分配] {n} 个 worker → 新 worker 最晚 {max(t):.1f}h，整体 ≈ {max(max(t), max(cur_left.values())):.1f} 小时")
lang_q = collections.Counter(it['lang'] for it in queued)
print('排队项构成:', dict(lang_q))
open('/tmp/queued_keys.txt', 'w').write('\n'.join(sorted(qkeys)) + '\n')
