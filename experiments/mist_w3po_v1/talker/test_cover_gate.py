"""Unit test for cover_gate.CoverGate on a tiny random Qwen2 (CPU): positions match the training rows'
(pretrain_talker_crf Groups: text after the last START_CONTENT, frames from the prompt end), P in [0, 1]."""
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.mist_w3po_v1.talker import align_crf
from experiments.mist_w3po_v1.talker.cover_gate import START_CONTENT, SEMANTIC_0, CoverGate


def test_gate():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=180407, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg).eval()
    wq, wk = torch.randn(8, 32), torch.randn(8, 32)
    g = CoverGate(m, wq, wk, 2.0, torch.device("cpu"), batch=2)
    p1 = [5, 6, START_CONTENT, 10, 11, 12, 180390, 7, 8, 180388]
    o1 = [SEMANTIC_0 + i for i in range(9)] + [180395]
    p2 = [5, START_CONTENT, 13, 180390, 180388]
    o2 = [SEMANTIC_0 + 3, SEMANTIC_0 + 4, 180392]
    lens = [[2, 3, 1], [4]]
    P = g.p_cover([p1, p2], [o1, o2], lens)
    assert len(P) == 2 and all(0.0 <= x <= 1.0 for x in P) and abs(P[1] - 1.0) < 1e-6   # one unit: always covered
    # direct computation for row 0
    with torch.no_grad():
        h = m.model(input_ids=torch.tensor([p1 + o1]), return_dict=True).last_hidden_state[0]
        e = (F.linear(h[10:19], wq) @ F.linear(h[3:6], wk).T / 16.0)[None]
        zc, za = align_crf.forward(e, torch.tensor([9]), torch.tensor([3]), torch.tensor([lens[0]]), kappa=2.0)
    assert abs(float((zc - za).exp()) - P[0]) < 1e-4


if __name__ == "__main__":
    test_gate()
    print("ok")
