"""E9 cascade (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §1 S3): gstar_talker_v6 (B's Talker input
prep), unchanged, made to accept the relabelled speaker "unk" (relabel_causal_run.py: a segment for which the causal
rule finds no active slot). Without this, prep fails on "unk" (slot arithmetic k - 1, sorting with integer slots).

  * on load, a hypothesis line's segment / block speaker "unk" becomes UNK: an int subclass (value 99, sorts after the
    8 slots) that formats as "unk", so the stream id is "<window id>__Sunk" -- the id asr_hyp.py and
    speech_latency_lit look up for the "unk" segments (f"{id}__S{spk}"), as in E7 (build_e7.py)
  * solo_runs for a slot the posterior does not have (UNK - 1 = 98) returns no runs, so the voice prompt is prep's own
    fallback for a slot heard too little alone (--causal: the window's audio up to the stream's first commit)
Slots 1..8 are untouched: with no "unk" in the input the output is byte-identical to gstar_talker_v6's.

    GSTAR_DATA=... python -m experiments.mist_w3po_v1.evaluation.gstar_talker_v6_unk prep <gstar_talker_v6 prep args>
"""
from __future__ import annotations

from experiments.uniss_qwen3_omni_v1.evaluation import gstar_talker_v6 as G


class Unk(int):
    def __new__(cls):
        return int.__new__(cls, 99)

    def __format__(self, spec: str) -> str:
        return "unk"

    def __str__(self) -> str:
        return "unk"

    __repr__ = __str__


UNK = Unk()


def _fix(r):
    if isinstance(r, dict):
        for key in ("segments", "blocks"):
            for s in r.get(key) or []:
                if isinstance(s, dict) and s.get("spk") == "unk":
                    s["spk"] = UNK
    return r


def install():
    load0, solo0 = G.load, G.solo_runs

    def load(p):
        return [_fix(r) for r in load0(p)]

    def solo_runs(post, k, until_frame=None):
        if k >= post.shape[1]:
            return []
        return solo0(post, k, until_frame)

    G.load, G.solo_runs = load, solo_runs


def main() -> None:
    install()
    G.main()


if __name__ == "__main__":
    main()
