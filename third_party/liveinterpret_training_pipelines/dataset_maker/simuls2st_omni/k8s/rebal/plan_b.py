#!/usr/bin/env python3
"""方案 B 规划：还原补跑队列 → 校验 → 把未开始的项按估算耗时均衡分给 i-04ed 上的新 worker。

产出（写到 rebal/ 目录）：
  bkeys_w{N}.txt         新 Pod 每个 worker 的显式清单
  watch_tts-omni-sweep{1,2}.py  在 sweep Pod 内运行的停止脚本（占位标记 + 当前项完成即停）
  plan.json              规划明细，供事后核对
"""
import collections
import hashlib
import json
import os
import random
import sys

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
K8S = os.path.dirname(HERE)
B = 'stevehguo-ap-southeast-3'
BASE = 'speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS'
SOURCES = [('ZH', f'{BASE}/zh/emilia/', '.tar'), ('JA', f'{BASE}/jp/audio/', '.tar'),
           ('KO', f'{BASE}/ko/patterned_utterance_with_number/data/', '.parquet')]
N_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 28
UNION = {'ZH': 18_269_396, 'JA': 7_518_186, 'KO': 3_596_698}   # 审计时的唯一记录数

s3 = boto3.client('s3', region_name='ap-southeast-3')
items = []
for lang, pre, ext in SOURCES:
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=B, Prefix=pre):
        for o in page.get('Contents', []):
            if o['Key'].endswith(ext):
                items.append({'lang': lang, 'key': o['Key'], 'gb': o['Size'] / 1e9})
assert len(items) == 1152, len(items)
by_key = {it['key']: it for it in items}

snap = {p: json.load(open(f'{HERE}/state/{p}.json'))
        for p in ('tts-omni-catchup', 'tts-omni-sweep1', 'tts-omni-sweep2')}
node_done = {p: {k for k, _t in s['done'] if not k.startswith('MOVED_TO_B:')} for p, s in snap.items()}
done_all = set().union(*node_done.values())
missing = [it['key'] for it in items if it['key'] not in done_all]
import datetime as _dt
CATCHUP_START = _dt.datetime(2026, 11, 2, 2, 11, 8, tzinfo=_dt.timezone.utc).timestamp()   # Pod startTime


def shuffled(lst):
    lst = list(lst)
    random.Random(42).shuffle(lst)
    return lst


# ---- 还原 sweep 队列（only-keys 筛选 → shuffle(42) → i % 32）----
sweep_keys = {l.strip() for l in open(f'{K8S}/sweep_keys.txt') if l.strip()}
sw = shuffled([it for it in items if it['key'] in sweep_keys])
errs, inflight, queued, cur_map = [], {}, collections.defaultdict(list), collections.defaultdict(dict)
for pod in ('tts-omni-sweep1', 'tts-omni-sweep2'):
    for ws, st in snap[pod]['workers'].items():
        w = int(ws)
        q = [it['key'] for i, it in enumerate(sw) if i % 32 == w]
        if st['todo'] != len(q):
            errs.append(f'{pod} w{w}: 日志待处理 {st["todo"]} != 还原 {len(q)}')
            continue
        f = st['fin']
        if f >= len(q):
            errs.append(f'{pod} w{w}: fin={f} 已超出队列')
            continue
        if f > 0 and q[f - 1] not in node_done[pod]:
            errs.append(f'{pod} w{w}: 前一项 {q[f-1].rsplit("/",1)[-1]} 无完成标记')
        if q[f] in node_done[pod]:
            errs.append(f'{pod} w{w}: 当前项 {q[f].rsplit("/",1)[-1]} 已有完成标记')
        inflight[q[f]] = (pod, w, st)
        cur_map[pod][w] = q[f]
        queued[pod] += q[f + 1:]
# catchup：残差 [40,47] 筛选 → shuffle(42) → i % 32，再去掉启动时主节点已完成的项
pre_done = {k for k, t in snap['tts-omni-catchup']['done'] if t < CATCHUP_START}
cu = shuffled([it for i, it in enumerate(items) if 40 <= i % 48 <= 47])
for ws, st in snap['tts-omni-catchup']['workers'].items():
    w = int(ws)
    q = [it['key'] for i, it in enumerate(cu) if i % 32 == w and it['key'] not in pre_done]
    f = st['fin']
    if st['todo'] != len(q) or f >= len(q):
        errs.append(f'catchup w{w}: 日志待处理 {st["todo"]} 还原 {len(q)} fin={f}')
        continue
    if f > 0 and q[f - 1] not in node_done['tts-omni-catchup']:
        errs.append(f'catchup w{w}: 前一项无完成标记')
    if q[f + 1:]:
        errs.append(f'catchup w{w}: 当前项之后还有 {len(q) - f - 1} 项排队（预期为 0）')
    inflight[q[f]] = ('tts-omni-catchup', w, st)
if errs:
    print('校验失败：'); [print('  ', e) for e in errs]; sys.exit(1)
print(f'校验通过：sweep {sum(len(v) for v in cur_map.values())} 个 + catchup '
      f'{len(snap["tts-omni-catchup"]["workers"])} 个 worker 的队列与日志一致')

# 在途项若已在任一节点有真实完成标记，说明是重复处理 → 立即停止该 worker
kill_now = collections.defaultdict(list)
for k, (p, w, st) in list(inflight.items()):
    where = [q for q in node_done if k in node_done[q]]
    if where:
        kill_now[p].append(w)
        print(f'  重复在途：{p} w{w} 的 {k.rsplit("/", 1)[-1]} 已在 {where} 完成（白做 {st["cur_done"]:,} 条）'
              + (' → 立即停止' if p != 'tts-omni-catchup' else ' → 不停：停 catchup worker 会触发整 Pod 重启，让它自然结束'))
        inflight.pop(k)
        if p == 'tts-omni-catchup':
            kill_now[p].pop()
        if p in cur_map:
            cur_map[p].pop(w, None)

queued_all = [k for p in queued for k in queued[p]]
new_keys = sorted(set(missing) - set(inflight))
orphans = sorted(set(new_keys) - set(queued_all))
print(f'未完成 {len(missing)} = 在途 {len(inflight)} + 待分配 {len(new_keys)}'
      f'（其中排队项 {len(set(queued_all) & set(new_keys))}，不在任何队列的孤项 {len(orphans)}）')
assert not (set(inflight) - set(missing)), '有在途项已带完成标记'

# ---- 估算 ----
gb, cnt = collections.Counter(), collections.Counter()
for k in done_all:
    it = by_key.get(k)
    if it:
        gb[it['lang']] += it['gb']; cnt[it['lang']] += 1
per_gb = {'ZH': UNION['ZH'] / gb['ZH'], 'KO': UNION['KO'] / gb['KO']}
per_ja = UNION['JA'] / cnt['JA']
est = lambda it: it['gb'] * per_gb[it['lang']] if it['lang'] != 'JA' else per_ja
rates = collections.defaultdict(list)
for k, (_p, _w, st) in inflight.items():
    if st['rate'] > 0:
        rates[by_key[k]['lang']].append(st['rate'])
med = {l: sorted(v)[len(v) // 2] for l, v in rates.items()}
hours = lambda it: est(it) / med[it['lang']] / 3600
inflight_left = {k: max(est(by_key[k]) - st['cur_done'], est(by_key[k]) * 0.05) / (st['rate'] or med[by_key[k]['lang']]) / 3600
                 for k, (_p, _w, st) in inflight.items()}

# ---- LPT 均衡分配 ----
bins = [[0.25, []] for _ in range(N_NEW)]
for k in sorted(new_keys, key=lambda k: -hours(by_key[k])):
    b = min(bins, key=lambda x: x[0])
    b[0] += hours(by_key[k]); b[1].append(k)
mk = max(max(b[0] for b in bins), max(inflight_left.values()))
print(f'速率中位数 {dict((l, round(v, 1)) for l, v in med.items())}；新 Pod {N_NEW} worker 最晚 {max(b[0] for b in bins):.1f}h，'
      f'在途项最晚 {max(inflight_left.values()):.1f}h → 预计整体 {mk:.1f}h')
print('每 worker 项数:', [len(b[1]) for b in bins])

# ---- 写文件 ----
for f in os.listdir(HERE):
    if f.startswith('bkeys_w'):
        os.remove(os.path.join(HERE, f))
for w, (_t, ks) in enumerate(bins):
    open(f'{HERE}/bkeys_w{w}.txt', 'w').write(''.join(k + '\n' for k in ks))
tmpl = open(f'{HERE}/watch_tmpl.py').read()
for pod in ('tts-omni-sweep1', 'tts-omni-sweep2'):
    # 占位标记：本 Pod 所有 worker 队列中、本节点无真实标记、且不在监视名单里的项。
    # 容器在所有 worker 停止后会被 OnFailure 重启，驱动只看本节点标记——
    # 不仅排队项，连「已在别的节点完成」的项（如重复在途项）也必须占位，否则重启后会重做。
    lo, hi = (0, 16) if pod.endswith('1') else (16, 32)
    full = {it['key'] for i, it in enumerate(sw) if lo <= i % 32 < hi}
    moved = sorted(full - node_done[pod] - set(cur_map[pod].values()))
    assert not (set(moved) & set(cur_map[pod].values()))
    body = (tmpl.replace('__CUR__', json.dumps({str(w): k for w, k in cur_map[pod].items()}))
                .replace('__MOVED__', json.dumps(moved))
                .replace('__KILL__', json.dumps(kill_now.get(pod, []))))
    open(f'{HERE}/watch_{pod}.py', 'w').write(body)
    print(f'{pod}: 停止脚本监视 {len(cur_map[pod])} 个 worker，立即停止 {kill_now.get(pod, [])}，占位标记 {len(moved)} 项')
json.dump({'n_new': N_NEW, 'new_keys': new_keys, 'orphans': orphans, 'kill_now': kill_now,
           'inflight': {k: [p, w] for k, (p, w, _s) in inflight.items()},
           'bins': [[round(t, 2), ks] for t, ks in bins], 'est_hours': round(mk, 2)},
          open(f'{HERE}/plan.json', 'w'), ensure_ascii=False, indent=1)
