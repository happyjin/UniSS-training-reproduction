"""在真实 checkpoint 上验证 Fusion 幅值通道的“起步等价”：
开关关 / 开分别加载同一 checkpoint，scale_gain 必须为全零，且同一输入下 cond 逐位相同；
再把 scale_gain 设成非零，确认通道确实生效（输出改变）。

用法（GPU Pod 内）：
  PYTHONPATH=/workspace/custom python check_fusion_scale_equiv.py --ck <S4 checkpoint>
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ck', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    from streaming_si.model.modeling_streaming_si import StreamingSIForConditionalGeneration

    def load(flag):
        m = StreamingSIForConditionalGeneration.from_pretrained(
            args.ck, dtype=torch.bfloat16, talker_fusion_scale=flag).to(args.device).eval()
        return m

    m0, m1 = load(False), load(True)
    f0, f1 = m0.talker.fusion, m1.talker.fusion
    assert not f0.scale_channel and f1.scale_channel, '开关未生效'
    assert torch.count_nonzero(f1.scale_gain) == 0, 'scale_gain 未零初始化'
    torch.manual_seed(0)
    H = m0.config.talker_hidden_size
    cond = torch.randn(2, 7, H, device=args.device, dtype=torch.bfloat16)
    text = torch.randn(2, 7, H, device=args.device, dtype=torch.bfloat16)
    with torch.no_grad():
        y0, y1 = f0(cond, text), f1(cond, text)
        assert torch.equal(y0, y1), f'起步不等价：max|Δ|={(y0 - y1).abs().max().item()}'
        f1.scale_gain.fill_(0.1)
        y2 = f1(cond, text)
    diff = (y2 - y0).abs().max().item()
    assert diff > 0, '非零 scale_gain 下输出未变化'
    print(f'OK：scale_gain 零初始化，起步逐位等价；置 0.1 后 max|Δ|={diff:.4f}（通道生效）')
    n_new = sum(p.numel() for n, p in m1.named_parameters() if 'scale_gain' in n)
    print(f'新增参数：{n_new}（talker.fusion.scale_gain）')


if __name__ == '__main__':
    main()
