"""把一个 S4 checkpoint 的 talker 桥接模块重新初始化，用于「换条件通路」的实验。

为什么需要
----------
从已收敛的 S4 继续训练、同时更换 talker 的条件通路（换取层 / 换取位 / 加旁路）时，
`talker.fusion` 的门控是针对**旧通路**（末层译文位置 hidden）调好的。沿用旧权重时门控
可能继续按旧解工作，把新通路的信息压掉 —— 结果是**测不出新通路的好处，但原因是门控没动**。

这不是推测，有实测依据（`pipelines/streaming_si/train_tts.sh` L15）：

    fusion 必须和 thinker_proj 一起预热：只放 thinker_proj 时，fusion 的门控还停在
    「cond 恒零」的解上。

所以换通路实验的正确做法是：**重初始化 `thinker_proj` + `fusion`**（talker 主干与 thinker
保留 S4 的能力），再跑 S4-w 式预热（只训这两个模块），过门 A 后才进 joint。

对照臂必须**同样重初始化**，否则对照保留了调好的门控、实验臂从零开始，不可比。

初始化口径与 `export_streaming_si.py` 建模时一致（`nn.Linear` 默认 Kaiming uniform、
`RMSNorm.weight` 置 1），即等价于「刚建好、还没训过」的状态。

用法
----
    python reinit_talker_bridge.py --src <s4 ckpt 目录> --dst <输出目录> \
        [--modules thinker_proj fusion] [--layer 6] [--seed 0]

`--layer` 顺带把 `config.talker_bridge_layer` 写进输出目录的 config（省掉训练时再传环境变量，
也让推理/门 A 自动沿用）。不传则保持原值。
"""
import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 这些前缀下的张量会被重置；与 common_args.sh 的 scope=talker_bridge 可训范围一致
DEFAULT_MODULES = ('thinker_proj', 'fusion')
SHARD_INDEX = 'model.safetensors.index.json'


def reinit_tensor(name, t, gen):
    """按建模时的默认初始化重置一个张量。

    * RMSNorm.weight（1 维，名字以 norm*.weight 结尾）-> 全 1
    * nn.Linear.weight（2 维）-> Kaiming uniform（PyTorch Linear 的默认）
    """
    if t.ndim == 1:
        return torch.ones_like(t)
    if t.ndim == 2:
        fan_in = t.shape[1]
        bound = (1.0 / fan_in) ** 0.5 * (3.0 ** 0.5)      # kaiming_uniform_(a=sqrt(5)) 的等价边界
        out = torch.empty_like(t, dtype=torch.float32).uniform_(-bound, bound, generator=gen)
        return out.to(t.dtype)
    raise ValueError(f'{name}: 不支持 {t.ndim} 维张量')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='S4 checkpoint 目录')
    ap.add_argument('--dst', required=True, help='输出目录（会整目录复制后改权重）')
    ap.add_argument('--modules', nargs='+', default=list(DEFAULT_MODULES),
                    help=f'要重置的 talker 子模块名（默认 {" ".join(DEFAULT_MODULES)}）')
    ap.add_argument('--layer', type=int, default=None,
                    help='顺带写入 config.talker_bridge_layer（-1 = 末层）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    from safetensors.torch import load_file, save_file

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit('--src 与 --dst 不能相同')
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)) or '.', exist_ok=True)
    if os.path.exists(args.dst):
        raise SystemExit(f'{args.dst} 已存在，先删除或换路径（避免覆盖已有实验）')
    print(f'[reinit] 复制 {args.src} -> {args.dst}', flush=True)
    shutil.copytree(args.src, args.dst)

    pats = tuple(f'talker.{m}.' for m in args.modules)
    gen = torch.Generator().manual_seed(args.seed)

    idx_p = os.path.join(args.dst, SHARD_INDEX)
    if os.path.isfile(idx_p):
        with open(idx_p, encoding='utf-8') as f:
            wmap = json.load(f)['weight_map']
        shards = sorted({v for k, v in wmap.items() if k.startswith(pats)})
    else:
        shards = ['model.safetensors']
    if not shards:
        raise SystemExit(f'没有包含 {pats} 的分片，检查 --modules 是否写对')

    total = 0
    for sh in shards:
        p = os.path.join(args.dst, sh)
        sd = load_file(p)
        hit = [k for k in sd if k.startswith(pats)]
        for k in hit:
            sd[k] = reinit_tensor(k, sd[k], gen)
            print(f'  重置 {k}  {tuple(sd[k].shape)}', flush=True)
        if hit:
            save_file(sd, p, metadata={'format': 'pt'})
        total += len(hit)
    if not total:
        raise SystemExit(f'一个张量都没匹配到 {pats}，请核对模块名')

    if args.layer is not None:
        cp = os.path.join(args.dst, 'config.json')
        with open(cp, encoding='utf-8') as f:
            cfg = json.load(f)
        old = cfg.get('talker_bridge_layer')
        cfg['talker_bridge_layer'] = args.layer
        with open(cp, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f'[reinit] config.talker_bridge_layer {old} -> {args.layer}', flush=True)

    print(f'[reinit] 共重置 {total} 个张量 -> {args.dst}')
    print('[reinit] 下一步：以此目录为 MODEL_PATH 跑 S4-w 式预热（只训桥接），过门 A 后再 joint')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
