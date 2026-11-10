"""Unit tests for pretrain_talker_crf's rows / groups (label positions, KL alignment, alignment row tensors); CPU."""
import sys

sys.path.insert(0, "third_party/Megatron-LM")
from experiments.mist_w3po_v1.talker import pretrain_talker_crf as P  # noqa: E402
from experiments.mist_w3po_v1.talker import talker_rows as T  # noqa: E402
from experiments.mist_w3po_v1.talker.test_talker_rows import dump_item  # noqa: E402


def test_plem_groups():
    it = dump_item()
    W = {"M_on_0000": {"open_detect_ms": 900, "close_detect_ms": 1600, "dur_ms": 1500}}
    blocks, _ = T.plem_rows([it], W)
    rows = P.flat_rows({"mode": "plem", "blocks": blocks}, 3000)
    g = P.Groups(rows, "plem", 100000, 2000, 3, 0)
    x = g[0]
    ids, rids = x["ids"], x["rids"]
    # the label at p+1 of the own sequence is the label; the same label at pr+1 of the reference sequence
    assert (ids[x["b"], x["p"] + 1] == x["labels"]).all() and (rids[x["b"], x["pr"] + 1] == x["labels"]).all()
    assert x["sb"].numel() == len(blocks) and abs(float(x["sq"].sum()) - len(blocks)) < 1e-5
    # the speed position's next own token is a speed token
    nxt = ids[x["sb"], x["sp"] + 1]
    assert ((nxt >= T.SPEED_0) & (nxt < T.SPEED_0 + 35)).all()


def test_align_groups():
    it = dump_item()
    blocks, _ = T.align_rows([it])
    rows = P.flat_rows({"mode": "align", "blocks": blocks}, 3000)
    g = P.Groups(rows, "align", 100000, 2000, 2, 0)
    x = g[0]
    assert (x["ids"][x["b"], x["p"] + 1] == x["labels"]).all()
    for j in range(x["a_b"].numel()):
        b, t0, M, c0, Tn = (int(x[k][j]) for k in ("a_b", "a_t0", "a_M", "a_c0", "a_T"))
        assert int(x["ids"][b, t0 - 1]) == T.START_CONTENT and int(x["ids"][b, t0 + M]) == T.END_CONTENT
        assert int(x["ids"][b, c0 - 1]) == T.START_SEMANTIC
        codes = x["ids"][b, c0: c0 + Tn]
        assert ((codes >= T.SEMANTIC_0) & (codes <= 163952)).all()
        assert int(x["ids"][b, c0 + Tn]) in (T.WAIT_READ, T.END_SEMANTIC)
    # CE only on the chosen rows
    assert sorted(set(int(v) for v in x["b"])) == [j for j, r in enumerate([g.rows[k] for k in g.groups[0]]) if r["ce"]]


if __name__ == "__main__":
    test_plem_groups()
    test_align_groups()
    print("ok")
