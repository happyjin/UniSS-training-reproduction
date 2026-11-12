"""Test of merge_read_segments --sep (CPU): the default joins with one space exactly as before (same output as an
explicit sep " " on the toy and on the real A' dev decode); sep "" joins Chinese members without a space and leaves
every other field identical to the default merge.

    PYTHONPATH=. python -m experiments.mist_w3po_v1.crf.test_merge_read_segments_sep
"""
from __future__ import annotations

import json

from experiments.mist_w3po_v1.crf import merge_read_segments as MR


def seg(spk, w, a, b, t):
    return {"spk": spk, "write_ms": w, "start_s": a, "end_s": b, "text": t}


def main() -> None:
    S = [seg(1, 1000, 0.5, 0.9, "我们"), seg(2, 1000, 0.0, 0.4, "好"), seg(1, 1000, 0.0, 0.4, "所以"),
         seg(1, 1000, 0.9, 1.0, "开会。"), seg(1, 2000, 1.2, 1.6, "对")]
    d = MR.merge_entries(S, "end_s")
    assert d == MR.merge_entries(S, "end_s", " ")
    z = MR.merge_entries(S, "end_s", "")
    assert [x["text"] for x in d] == ["所以 我们 开会。", "好", "对"], d
    assert [x["text"] for x in z] == ["所以我们开会。", "好", "对"], z
    assert [{k: v for k, v in x.items() if k != "text"} for x in z] == [{k: v for k, v in x.items() if k != "text"} for x in d]
    n = 0
    for l in open(f"{MR.DEC}/dev_crf_v2p_s3000_finalpost_s8_c2g06/hyp.jsonl", encoding="utf-8"):
        h = json.loads(l)
        assert MR.merge_hyp(h) == MR.merge_hyp(h, " ")
        n += 1
        if n >= 500:
            break
    print("ok")


if __name__ == "__main__":
    main()
