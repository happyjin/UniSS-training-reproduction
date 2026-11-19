#!/usr/bin/env python3
"""在补跑 Pod 内采集快照：存活 worker 的待处理数/已完成项数/当前项进度/速率，以及本节点完成标记。"""
import glob
import json
import os
import re
import subprocess

R = '/opt/dlami/nvme/stevehguo/tts_omni'
ps = subprocess.run(['pgrep', '-af', 'tts_omni_driver.py'], capture_output=True, text=True).stdout.splitlines()
out = {}
for line in ps:
    m = re.search(r'--worker-id (\d+)', line)
    if not m or 'skip-ids' in line:
        continue
    w = int(m.group(1))
    t = open(f'{R}/logs/worker-{w}.log', errors='ignore').read()
    L = t.splitlines()
    todo = re.search(r'待处理 (\d+)', t)
    done_lines = [i for i, s in enumerate(L) if re.search(r'\[(?:tar|parquet) done\]', s)]
    oks = [(i, int(x.group(1)), int(x.group(2))) for i, s in enumerate(L)
           for x in [re.search(r'ok=(\d+) in=\d+ skip=\d+ items=\d+ elapsed=(\d+)s', s)] if x]
    last_done = max(done_lines or [-1])
    base = max([o for i, o, _ in oks if i < last_done] or [0])
    cur = oks[-1][1] if oks else 0
    rate = 0.0
    if len(oks) > 8:
        a, b = oks[-1], oks[-8]
        rate = (a[1] - b[1]) / max(a[2] - b[2], 1)
    dls = re.findall(r'下载 (\S+) \(', t)
    out[w] = {'todo': int(todo.group(1)) if todo else None, 'fin': len(done_lines),
              'cur_done': max(cur - base, 0), 'rate': round(rate, 2),
              'dl_last': dls[-1] if dls else None}
done = [[open(f).read().strip(), os.path.getmtime(f)] for f in glob.glob(R + '/done/*.done')]
print(json.dumps({'workers': out, 'done': done}))
