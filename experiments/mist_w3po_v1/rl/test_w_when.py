"""--w-when (2026-11-07): the default 1.0 must give the pilot's loss bit for bit; other values scale only the when
term. Also checks adv_magnitudes. Run (CPU):
    python -m experiments.mist_w3po_v1.rl.test_w_when
"""
from __future__ import annotations

import inspect
import re

import torch

from experiments.mist_w3po_v1.rl import train_w3po as T


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def pilot_total(l_when, l_who, l_what, cnt, w_who, w_what):
    """the pilot's expression (train_w3po.py before --w-when), verbatim"""
    return l_when / cnt[0] + w_who * l_who / cnt[1] + w_what * l_what / cnt[2]


def leaves(seed):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(1000, generator=g, dtype=torch.float32).requires_grad_() for _ in range(3)]
    cnt = torch.tensor([4013.0, 1959.0, 19005.0])
    return x, cnt


def run(fn, seed):
    x, cnt = leaves(seed)
    tot = fn((x[0] * 1.7).sum(), (x[1] ** 2).sum(), torch.tanh(x[2]).sum(), cnt)
    tot.backward()
    return tot.detach(), [t.grad.clone() for t in x]


def main():
    for seed in range(5):
        for w_who, w_what in ((1.0, 1.0), (0.7, 1.3)):
            a, ga = run(lambda lw, lh, lt, c: pilot_total(lw, lh, lt, c, w_who, w_what), seed)
            b, gb = run(lambda lw, lh, lt, c: T.w3_total(lw, lh, lt, c, 1.0, w_who, w_what), seed)
            check(f"seed {seed} w_who {w_who}: loss bit-identical at w_when=1.0", torch.equal(a, b))
            check(f"seed {seed} w_who {w_who}: grads bit-identical at w_when=1.0",
                  all(torch.equal(p, q) for p, q in zip(ga, gb)))
    a, ga = run(lambda lw, lh, lt, c: T.w3_total(lw, lh, lt, c, 1.0), 0)
    b, gb = run(lambda lw, lh, lt, c: T.w3_total(lw, lh, lt, c, 0.3), 0)
    check("w_when=0.3 scales the when gradient by 0.3", torch.allclose(gb[0], 0.3 * ga[0], rtol=1e-6, atol=0))
    check("w_when=0.3 leaves who/what gradients unchanged", torch.equal(ga[1], gb[1]) and torch.equal(ga[2], gb[2]))
    src = inspect.getsource(T.main)
    check("--w-when default is 1.0", re.search(r'"--w-when", type=float, default=1\.0', src) is not None)
    check("micro_loss uses w3_total with args.w_when",
          "w3_total(l_when, l_who, l_what, cnt, args.w_when, args.w_who, args.w_what)" in inspect.getsource(T.micro_loss))
    # adv_magnitudes: (when?, who?, a_when, a_who, a_what) per token; anchors skipped
    seqs = [(None, 0, [(True, False, -2.0, 0.0, 0.5), (False, True, 0.0, 1.0, 0.0), (False, False, 0.0, 0.0, -1.5)], False, None),
            (None, 0, [(True, True, 9.0, 9.0, 9.0)], True, None)]
    check("adv_magnitudes", T.adv_magnitudes(seqs) == [2.0, 1, 1.0, 1, 2.0, 2])
    print("ALL PASS")


if __name__ == "__main__":
    main()
