"""三方对照：复现 stage1/stage2 talker、官方 offline talker、原版 Qwen2.5-Omni-3B talker。
看复现 talker 的 decoder 层离谁更近（同形状张量的相对 L2 差），以及完全相同的张量是哪些。
同时比较官方 offline 与原版 Qwen 的 thinker，确认官方 offline 的 thinker 已被训练过。"""
import json
import sys

import torch
from safetensors import safe_open

E = '/fsx/stevehguo/eval/models'
OFF = f'{E}/SimulS2ST-Omni/offline'
RAW = sys.argv[1]
REP = f'{E}/repro'


def load_raw(prefix):
    idx = json.load(open(f'{RAW}/model.safetensors.index.json'))['weight_map']
    out = {}
    for shard in sorted({v for k, v in idx.items() if k.startswith(prefix)}):
        with safe_open(f'{RAW}/{shard}', 'pt') as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = f.get_tensor(k)
    return out


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


raw_t = load_raw('talker.')
off_t = torch.load(f'{OFF}/talker_components.pt', map_location='cpu', weights_only=True)['talker']
print('原版 talker 张量', len(raw_t), '| 官方 offline talker 张量', len(off_t))
print('原版 codec 词表', tuple(raw_t['model.embed_tokens.weight'].shape), '官方', tuple(off_t['model.embed_tokens.weight'].shape))
for st in ('stage1_talker_warmup', 'stage2_multitask_full'):
    t = torch.load(f'{REP}/{st}/talker_components.pt', map_location='cpu', weights_only=True)['talker']
    keys = [k for k in t if k in raw_t and k in off_t and raw_t[k].shape == t[k].shape == off_t[k].shape]
    d_raw = sorted(rel(t[k], raw_t[k]) for k in keys)
    d_off = sorted(rel(t[k], off_t[k]) for k in keys)
    eq_raw = [k for k in keys if torch.equal(t[k], raw_t[k].to(t[k].dtype))]
    eq_off = [k for k in keys if torch.equal(t[k], off_t[k])]
    m = lambda x: x[len(x) // 2]
    print(f'== 复现 {st} talker（同形状张量 {len(keys)} 个，不含 codec embed/head）')
    print(f'   vs 原版 Qwen   ：相对差中位 {m(d_raw):.4f}，完全相同 {len(eq_raw)}')
    print(f'   vs 官方 offline：相对差中位 {m(d_off):.4f}，完全相同 {len(eq_off)}')
    print('   与官方完全相同的张量示例:', eq_off[:6])
d = sorted(rel(off_t[k], raw_t[k]) for k in off_t if k in raw_t and raw_t[k].shape == off_t[k].shape)
print(f'== 官方 offline talker vs 原版 Qwen talker：相对差中位 {d[len(d)//2]:.4f}')

idx_o = json.load(open(f'{OFF}/model.safetensors.index.json'))['weight_map']
idx_r = json.load(open(f'{RAW}/model.safetensors.index.json'))['weight_map']
for k in ('thinker.model.layers.10.self_attn.q_proj.weight', 'thinker.model.layers.20.mlp.down_proj.weight',
          'thinker.audio_tower.layers.10.self_attn.q_proj.weight'):
    if k in idx_o and k in idx_r:
        with safe_open(f'{OFF}/{idx_o[k]}', 'pt') as fo, safe_open(f'{RAW}/{idx_r[k]}', 'pt') as fr:
            print(f'== 官方 offline vs 原版 Qwen  {k}: 相对差 {rel(fo.get_tensor(k), fr.get_tensor(k)):.5f}')
