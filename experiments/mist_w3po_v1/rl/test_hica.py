"""--hica-ms (2026-11-09, HiCA-GRPO: hindsight credit allocation with a lookahead self-teacher, with --mode dia).
  1. advantage.py = the version before HiCA (argv[1]) + an appended block; train_w3po.py differs from its version
     before HiCA (argv[2]) by insertions plus one equivalent rewrite (logs[k] += -> logs.get), and every inserted line
     that runs in main() / micro_loss() is behind `hica_ms > 0` (or is the empty hica_logs dict and logs.update of it):
     --hica-ms 0 (default) is dia (and every other mode) exactly;
  2. hica_weights on hand-made values: exp(clip(sign(A) delta, +-c)), 1 where not applied or A = 0;
  3. hica_items: lookahead time min(t + D, dur), akey = the rollout key of that read time (a read time itself);
  4. hica_reweight with a stub engine (embedding + linear head, audio tokens growing with t): the batched teacher
     log-probabilities equal a per-item recomputation; rows whose lookahead is capped at t (final read) keep weight 1;
     pure when tokens and anchors keep weight 1; who tokens use the 8-way speaker projection.

    CUDA_VISIBLE_DEVICES=0 python -m experiments.mist_w3po_v1.rl.test_hica <advantage.py before> <train_w3po.py before>
"""
from __future__ import annotations

import difflib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.mist_w3po_v1.core.engine import ReadItem, read_times
from experiments.mist_w3po_v1.rl import advantage as A
from experiments.mist_w3po_v1.rl import train_w3po as T
from experiments.uniss_qwen3_omni_v1.training.thinker_gstar_model import AUDIO_PAD, NEW_END, SPK0

HERE = Path(__file__).parent


def check_diffs(adv_old: str, tr_old: str):
    new_adv = (HERE / "advantage.py").read_text()
    old_adv = Path(adv_old).read_text()
    assert new_adv.startswith(old_adv) and "def hica_weights" in new_adv[len(old_adv):], "advantage.py: not a pure append"
    old = Path(tr_old).read_text().splitlines()
    new = (HERE / "train_w3po.py").read_text().splitlines()
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    inserted = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            assert old[i1:i2] == ["                    logs[k] += lg[k]"], old[i1:i2]
            assert new[j1:j2] == ["                    logs[k] = logs.get(k, 0.0) + lg[k]"], new[j1:j2]
            continue
        assert tag == "insert", (tag, old[i1:i2])
        inserted.append(new[j1:j2])
    allowed_first = ("    ap.add_argument(\"--hica-ms\"", "    if args.hica_ms > 0 and", "        HK = (",
                     "    hica_logs = {}", "    logs.update(hica_logs)", "            if args.hica_ms > 0:",
                     "from experiments.mist_w3po_v1.rl.advantage import hica_weights", "", "def hica_items",
                     "@torch.no_grad()")
    for blk in inserted:
        first = next((x for x in blk if x.strip()), "")
        assert first.startswith(allowed_first), f"unguarded insertion: {first!r}"
        if first.startswith("    hica_logs = {}"):
            assert blk[1].startswith('    if getattr(args, "hica_ms", 0) > 0:'), blk[1]
        if first.startswith("        HK = ("):
            assert blk[1].startswith("        if args.hica_ms > 0:"), blk[1]
    print(f"[1] advantage.py append-only; train_w3po.py: {len(inserted)} guarded insertions + 1 equivalent rewrite")


def check_weights():
    adv = torch.tensor([1.0, -1.0, -1.0, 1.0, 0.0, 2.0])
    d = torch.tensor([0.5, -0.5, 2.0, -3.0, 0.7, 0.3])
    ap = torch.tensor([True, True, True, True, True, False])
    w, cl = A.hica_weights(adv, d, ap, 1.0)
    exp = [math.exp(0.5), math.exp(0.5), math.exp(-1.0), math.exp(-1.0), 1.0, 1.0]
    assert torch.allclose(w, torch.tensor(exp)), w
    assert cl.tolist() == [False, False, True, True, False, False], cl
    print("[2] hica_weights matches hand values")


def check_items():
    w = {"id": "W1", "dur_ms": 4300}
    its = [ReadItem(wav=None, post=None, t_ms=t, final=t == 4300, committed=[], akey=f"W1|{t}") for t in read_times(4300)]
    rows = [(None, r, None, False, w) for r in range(len(its))]
    out = T.hica_items(its, rows, 2000)
    assert [o.t_ms for o in out] == [3000, 4000, 4300, 4300, 4300], [o.t_ms for o in out]
    assert all(o.akey == f"W1|{o.t_ms}" and o.t_ms in read_times(4300) for o in out)
    print("[3] hica_items: lookahead read times / rollout akeys")


class StubEng:
    def __init__(self, V=NEW_END + 4, D=6):
        g = torch.Generator().manual_seed(0)
        self.emb_layer = torch.nn.Embedding(V, D)
        torch.nn.init.normal_(self.emb_layer.weight, generator=g)
        head = torch.nn.Linear(D, V, bias=False)
        torch.nn.init.normal_(head.weight, generator=g)
        self.thinker = SimpleNamespace(
            model=lambda inputs_embeds, attention_mask, use_cache, return_dict:
                SimpleNamespace(last_hidden_state=torch.tanh(inputs_embeds.cumsum(1))),
            lm_head=head)
        self.tok = SimpleNamespace(pad_token_id=0)
        self.dev = torch.device("cpu")
        self.pr = SimpleNamespace(ids=lambda n, ctx, committed: [AUDIO_PAD] * n + [11, 12] + list(committed))
        self.calls = []

    def audio_embeds(self, items, dtype, cache=None):
        out = []
        for it in items:
            if cache is not None and it.akey in cache:
                out.append(cache[it.akey])
                continue
            n = it.t_ms // 500
            e = torch.sin(torch.arange(n * 6, dtype=torch.float32).reshape(n, 6) * 0.01 * it.t_ms / 1000).to(dtype)
            if cache is not None:
                cache[it.akey] = (n, e)
            out.append((n, e))
        return out

    def logp(self, item, blk):
        """per-item, unpadded teacher-forced log-probs and 8-way speaker log-probs of blk"""
        (n, e), = self.audio_embeds([item], torch.float32)
        p = self.pr.ids(n, item.ctx, item.committed)
        ids = torch.tensor([p + blk[:-1]])
        emb = self.emb_layer(ids)
        emb = emb.masked_scatter((ids == AUDIO_PAD).unsqueeze(-1), e)
        h = self.thinker.model(inputs_embeds=emb, attention_mask=None, use_cache=False, return_dict=True).last_hidden_state
        lg = self.thinker.lm_head(h[0, len(p) - 1: len(p) - 1 + len(blk)]).float()
        ls = torch.log_softmax(lg, -1)
        lp = ls.gather(1, torch.tensor(blk)[:, None])[:, 0]
        spk = torch.log_softmax(lg[:, SPK0:NEW_END], -1)
        return lp, spk


@torch.no_grad()
def check_reweight():
    eng = StubEng()
    w = {"id": "W2", "dur_ms": 3500}
    # read 1 s (lookahead 3 s), read 2 s (lookahead capped 3.5 s), final read 3.5 s (lookahead = itself)
    items = [ReadItem(wav=None, post=None, t_ms=t, final=t == 3500, committed=c, akey=f"W2|{t}")
             for t, c in ((1000, []), (2000, [21, 22]), (3500, [21, 22, 23]))]
    blocks = [[SPK0 + 1, 31, 32, 99], [SPK0 + 2, 41, 99], [51, SPK0, 99]]
    # (when, who) per token; anchors: none except row 2 token 0 marked anchor below
    cats = [[(True, True), (False, False), (False, False), (True, False)],
            [(True, True), (False, False), (True, False)],
            [(False, False), (False, True), (True, False)]]
    advs = [0.8, -1.2, 0.5]
    seqrows, flat = [], []
    for r, (blk, cs, a) in enumerate(zip(blocks, cats, advs)):
        toks = [(c[0], c[1], 0.0, 0.0, a) for c in cs]
        seqrows.append((None, r, toks, False, w))
    cache = {}   # as in training: the rollout encoded every read time of the window
    eng.audio_embeds([ReadItem(wav=None, post=None, t_ms=t, final=False, committed=[], akey=f"W2|{t}")
                      for t in read_times(3500)], torch.float32, cache=cache)
    tg, ref_lp, ref_spk, spk_k, is_w, is_h, anc, at = [], [], [], [], [], [], [], []
    for r, (it, blk, cs, a) in enumerate(zip(items, blocks, cats, advs)):
        lp, spk = eng.logp(it, blk)
        for k, t in enumerate(blk):
            tg.append(t)
            ref_lp.append(lp[k])
            ref_spk.append(spk[k])
            spk_k.append(max(0, min(7, t - SPK0)))
            is_w.append(cs[k][0])
            is_h.append(cs[k][1])
            anc.append(r == 2 and k == 0)
            at.append(a)
    tg_t, ref_lp, ref_spk = torch.tensor(tg), torch.stack(ref_lp), torch.stack(ref_spk)
    spk_k, is_w, is_h, anc, at = (torch.tensor(spk_k), torch.tensor(is_w), torch.tensor(is_h), torch.tensor(anc),
                                  torch.tensor(at))
    args = SimpleNamespace(hica_ms=2000, hica_clip=1.0)
    new_at, logs = T.hica_reweight(eng, [], items, blocks, seqrows, cache, args, at, tg_t, ref_lp, ref_spk, spk_k,
                                   is_w, is_h, anc)
    # manual: per item, teacher at min(t + 2000, 3500)
    exp_w, pos = [], 0
    for r, (it, blk) in enumerate(zip(items, blocks)):
        t2 = min(it.t_ms + 2000, 3500)
        it2 = ReadItem(wav=None, post=None, t_ms=t2, final=it.final, committed=it.committed, akey=f"W2|{t2}")
        lp2, spk2 = eng.logp(it2, blk)
        for k, t in enumerate(blk):
            wh, who = cats[r][k]
            if anc[pos] or (wh and not who):
                exp_w.append(1.0)
            else:
                kk = spk_k[pos]
                d = float(spk2[k, kk] - ref_spk[pos, kk]) if who else float(lp2[k] - ref_lp[pos])
                exp_w.append(math.exp(max(-1.0, min(1.0, math.copysign(1.0, advs[r]) * d))))
            pos += 1
    got_w = (new_at / at).tolist()
    assert all(abs(g - e) < 1e-4 for g, e in zip(got_w, exp_w)), (got_w, exp_w)
    assert all(abs(x - 1.0) < 1e-6 for x in got_w[7:]), "final read (lookahead = itself) must keep weight 1"
    assert abs(got_w[3] - 1.0) < 1e-9 and abs(got_w[6] - 1.0) < 1e-9, "pure when tokens keep weight 1"
    assert any(abs(x - 1.0) > 1e-3 for x in got_w[:3]), "lookahead rows should move some weight"
    assert logs["hica_n"] == 6.0, logs
    print(f"[4] hica_reweight = per-item recomputation; weights {[round(x, 3) for x in got_w]}")


if __name__ == "__main__":
    check_diffs(sys.argv[1], sys.argv[2])
    check_weights()
    check_items()
    check_reweight()
    print("test_hica OK")
