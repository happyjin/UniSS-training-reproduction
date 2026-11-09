"""Scheme 2, end-to-end TurnCRF (reports/mist_w3po_v1/crf_novel_ideas.zh-CN.md §3.2, §9, §11): continue v8 with the
shared trainer crf/train_crf_sft.py in its control mode (--objective heuristic = C0, plain v8 SFT on the sched_v1 AMI
train windows) plus the TurnCRF layer through the trainer's HOOK API:

  post_transform  TurnHook: P_prefix [T, 8] (80 ms, after v8's perturbation; T = max(0, t - 1040) // 80) and the
                  audio-encoder states enc [T, 2048] -> TurnLayer: P + g (pi - P), g = 0 at start (bit-identical to v8,
                  test_turn_layer.py (a); the trainer's torch post_to_tokens equals v8's numpy one bit for bit)
  extra_loss      the CRF auxiliary loss, weight 1 (pre-registered): per row (log Z~_T - log Z~_T,y) / T at the row's
                  read prefix, y = reference frame labels (turn_refs_e2e.RecRefs: the window's complete v8 target from
                  meta["rec"]["S"], window-level slot re-map); summed over the micro-batch and divided by the step's
                  global row count (so the trainer's sum over micro-batches and ranks is the mean over the step's rows;
                  rows with T = 0 add 0)
  extra_params    [the layer] (float32 parameters, CRF math in float64 inside): same AdamW / lr schedule / clip as the
                  LoRA, saved in <out>/ckpt/step_N/extra.pt["extra_params"][0] (with its config buffer), restored by
                  --resume; the decode wrapper loads it from the checkpoint dir (turn_decode.py --turn-crf <ckpt>)

The LoRA, the speaker projector and the 259 new token rows train exactly as in C0 (the shared trainer). The control is
the shared C0 run (train_crf_sft.py --objective heuristic, same steps / lr / data / seed).

Layer options (one pre-registered setting = the defaults): --turn-init neutral (the CPU module's independent-P init:
a = 1, everything else 0; the CPU-stage trained parameters FAILED that stage's pre-registered stop rule, <= 1 s turn
recall -0.21 vs raw P, heavy smoothing: reports/mist_w3po_v1/turn_crf_cpu_stage.zh-CN.md, so they are opt-in only:
--turn-init cpu_stage = eval_outputs/mist_w3po_v1/crf/turn_v1/semi/params.pt, or a params.pt path),
--turn-enc 1 (use the encoder states: zero-initialised MLP 2048 -> 256 -> 9),
--turn-labels window. Everything else goes to train_crf_sft.py unchanged (--objective must be heuristic or absent).
Side log (rank 0): <out>/turn_layer.jsonl, the layer's init, then per step g, the CRF parameters, the auxiliary loss of
the step's micro-batches on rank 0 and the layer's local gradient norms.

    torchrun --nproc_per_node 4 -m experiments.mist_w3po_v1.crf.train_turncrf_sft --out <new dir>
    (launcher: experiments/mist_w3po_v1/scripts/run_turncrf_sft.sh)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from experiments.mist_w3po_v1.crf.turn_layer import CPU_STAGE_SEMI, TurnLayer
from experiments.mist_w3po_v1.crf.turn_refs_e2e import RecRefs


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0


class TurnHook:
    """post_transform (callable) + extra_loss for train_crf_sft.main."""

    def __init__(self, layer: TurnLayer, refs: RecRefs, use_enc: bool, log_path: str | None = None,
                 init_note: dict | None = None):
        self.layer, self.refs, self.use_enc = layer, refs, use_enc
        self.init_note, self.noted = init_note, False
        self.wants_enc = use_enc
        self.cache: dict[int, tuple] = {}
        self.log_path = log_path
        self.cur_step, self.acc = None, None
        self.gnorm: dict[str, float] = {}
        for n, p in layer.named_parameters():
            p.register_post_accumulate_grad_hook(lambda q, n=n: self.gnorm.__setitem__(n, float(q.grad.norm())))

    # post_transform(P_prefix, enc, meta) -> [T, 8]
    def __call__(self, P: torch.Tensor, enc, meta: dict) -> torch.Tensor:
        if P.shape[0] == 0:
            return P
        e = enc if self.use_enc else None
        if torch.is_grad_enabled() and meta.get("phase", "train") == "train":
            out, cached = self.layer(P, e, return_aux=True)
            self.cache[id(out)] = (out, cached)
            return out
        return self.layer(P, e)

    def extra_loss(self, info: dict) -> torch.Tensor:
        dev = info["device"]
        tot = torch.zeros((), dtype=torch.float64, device=dev)
        n_rows = n_frames = 0
        for meta, Pin, Pout, enc in zip(info["metas"], info["P_in"], info["P_out"], info["enc"]):
            T = Pin.shape[0]
            if T == 0:
                continue
            y = torch.as_tensor(self.refs.labels(meta["rec"], T), device=dev)
            c = self.cache.get(id(Pout))
            cached = c[1] if c is not None and c[0] is Pout else None
            tot = tot + self.layer.aux_loss(Pin, y, enc if self.use_enc else None, cached=cached).sum()
            n_rows += 1
            n_frames += T
        self.cache.clear()
        self._log(info, float(tot), n_rows, n_frames)
        return (tot / max(1, int(info["rows_global"]))).to(torch.float32)

    def _log(self, info, aux_sum, n_rows, n_frames):
        if self.log_path is None or _rank() != 0:
            return
        if not self.noted:                    # first record: the layer's init (the out dir exists by now)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"init": self.init_note}) + "\n")
            self.noted = True
        step = int(info["step"])
        if self.cur_step is not None and step != self.cur_step:
            self._flush()
        if self.cur_step != step:
            self.cur_step, self.acc = step, {"aux_sum": 0.0, "rows": 0, "frames": 0, "mbs": 0}
        a = self.acc
        a["aux_sum"] += aux_sum
        a["rows"] += n_rows
        a["frames"] += n_frames
        a["mbs"] += 1

    def _flush(self):
        """Called at the first micro-batch of the next step (the previous step's gradients are then complete)."""
        if self.cur_step is None or self.log_path is None:
            return
        a = self.acc
        rec = {"step": self.cur_step + 1, "rank0_aux_mean_per_row": a["aux_sum"] / max(1, a["rows"]),
               "rank0_rows_with_frames": a["rows"], "rank0_frames": a["frames"], "rank0_mbs": a["mbs"],
               **self.layer.summary(), "grad_norm_local": {k: round(v, 6) for k, v in self.gnorm.items()},
               "refs": dict(self.refs.stats), "time": time.time()}
        rec.pop("init_src", None)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def build_layer(init: str, use_enc: bool, e_hidden: int = 256, e_dim: int = 2048) -> TurnLayer:
    layer = TurnLayer(K=8, L=257, Q=50, e_dim=e_dim if use_enc else 0, e_hidden=e_hidden)
    if init == "cpu_stage":                  # opt-in (the CPU stage failed its stop rule)
        layer.init_from(f"{CPU_STAGE_SEMI}/params.pt")
    elif init != "neutral":
        layer.init_from(init)
    return layer.float()          # float32 parameters (one dtype in the trainer's grad all-reduce); float64 math inside


def split_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--turn-init", default="neutral", help="neutral (default) | cpu_stage | <params.pt>")
    ap.add_argument("--turn-enc", type=int, default=1)
    ap.add_argument("--turn-e-hidden", type=int, default=256)
    ap.add_argument("--turn-labels", default="window", choices=["window", "session"])
    ap.add_argument("--objective", default="heuristic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    mine, rest = ap.parse_known_args(argv)
    if mine.objective != "heuristic":
        raise SystemExit("train_turncrf_sft runs the shared trainer with --objective heuristic only (scheme 2 alone)")
    rest = ["--objective", "heuristic", "--out", mine.out] + (["--resume"] if mine.resume else []) + rest
    return mine, rest


def main(argv=None) -> None:
    from experiments.mist_w3po_v1.crf import train_crf_sft as T

    import sys
    mine, rest = split_args(list(sys.argv[1:] if argv is None else argv))
    torch.manual_seed(20261109)      # the e-MLP's first layer (identical on every rank; the trainer broadcasts too)
    layer = build_layer(mine.turn_init, bool(mine.turn_enc), mine.turn_e_hidden)
    refs = RecRefs(mine.turn_labels)
    out = Path(mine.out)
    note = {"init_src": layer.init_src, "cfg": layer.cfg, "labels": mine.turn_labels, "use_enc": bool(mine.turn_enc),
            "trainer_args": rest}
    hook = TurnHook(layer, refs, bool(mine.turn_enc), str(out / "turn_layer.jsonl"), note)
    if int(os.environ.get("RANK", "0")) == 0:
        print("TURNCRF " + json.dumps(note), flush=True)
    try:
        T.main(rest, post_transform=hook, extra_loss=hook.extra_loss, extra_params=[layer])
    finally:
        hook._flush()


if __name__ == "__main__":
    main()
