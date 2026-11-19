"""判定复现 stage1 / stage2 的起点：与官方 offline 的 talker / thinker 做逐张量比较。"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

E = '/fsx/stevehguo/eval/models'
OFF = f'{E}/SimulS2ST-Omni/offline'
REP = f'{E}/repro'


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def talker(p):
    d = torch.load(p, map_location='cpu', weights_only=True)
    return d['talker'] if 'talker' in d else d


print('== sha256')
for p in (f'{OFF}/talker_components.pt', f'{REP}/stage1_talker_warmup/talker_components.pt',
          f'{REP}/stage2_multitask_full/talker_components.pt'):
    if os.path.isfile(p):
        print(' ', sha(p)[:16], p.replace(E, ''))

off_t = talker(f'{OFF}/talker_components.pt')
print('官方 offline talker 张量数', len(off_t))
for st in ('stage1_talker_warmup', 'stage2_multitask_full'):
    p = f'{REP}/{st}/talker_components.pt'
    if not os.path.isfile(p):
        continue
    t = talker(p)
    rel, same, n = [], 0, 0
    for k, v in t.items():
        if k not in off_t or off_t[k].shape != v.shape:
            continue
        a, b = off_t[k].float(), v.float()
        d = (a - b).norm() / (a.norm() + 1e-8)
        rel.append((d.item(), k)); n += 1
        same += int(torch.equal(off_t[k], v))
    rel.sort()
    print(f'== {st} talker vs 官方 offline talker：共同张量 {n}，完全相同 {same}，'
          f'相对差中位 {rel[len(rel)//2][0]:.4f}，最大 {rel[-1][0]:.4f} ({rel[-1][1]})')
    for key in ('model.embed_tokens.weight', 'codec_head.weight'):
        kk = [k for k in t if k.endswith(key)]
        if kk and kk[0] in off_t:
            a, b = off_t[kk[0]].float(), t[kk[0]].float()
            print(f'   {key}: shape {tuple(b.shape)} 相对差 {((a-b).norm()/a.norm()).item():.4f}')

# thinker：官方 offline 的 thinker 是否已含 Stage 2（与原版 Qwen 比，若有原版的话）
raw = sys.argv[1] if len(sys.argv) > 1 else None
if raw and os.path.isdir(raw):
    idx_o = json.load(open(f'{OFF}/model.safetensors.index.json'))['weight_map']
    idx_r = json.load(open(f'{raw}/model.safetensors.index.json'))['weight_map']
    keys = [k for k in idx_o if k.startswith('thinker.model.layers.10.self_attn.q_proj.weight')
            or k.startswith('thinker.model.layers.20.mlp.down_proj.weight')
            or k.startswith('talker.codec_head.weight')]
    for k in keys:
        with safe_open(f'{OFF}/{idx_o[k]}', 'pt') as fo, safe_open(f'{raw}/{idx_r[k]}', 'pt') as fr:
            a, b = fo.get_tensor(k).float(), fr.get_tensor(k).float()
            print(f'== 官方 offline vs 原版 Qwen  {k}: shape {tuple(a.shape)} vs {tuple(b.shape)}',
                  f'相对差 {((a-b).norm()/b.norm()).item():.4f}' if a.shape == b.shape else '形状不同')
