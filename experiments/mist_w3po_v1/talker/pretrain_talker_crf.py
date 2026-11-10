"""Megatron entry point of the two talker schemes (reports/mist_w3po_v1/talker_crf_prereg_v1.zh-CN.md §0, §1.3, §2.3):
the T1c / Plan 2 shape (TP=PP=1, a HuggingFaceModule holding the talker, only trainable tensors saved, the batch
built by the dataset; pretrain_selfmon.py's row groups).

--crf-mode plem   rows of talker_rows --mode plem: per candidate row, CE on its codes + closing (closing x close_weight)
                  weighted by q(c | b_n); 1.0 x speed loss (soft labels q(v | b_n) on the 35 speed tokens at the TTS
                  speed position, once per block); w_ref x KL(T1c || talker) on code positions, T1c reading the native
                  sequence (positions aligned from the end), weighted by q.
--crf-mode align  rows of talker_rows --mode align: CE + w_ref x KL on the chosen candidate; w_align x the alignment
                  CRF loss (align_crf.losses) on every labelled row; w_dur x the duration posterior regularisation on
                  label-1 rows. The alignment head (align_q, align_k: hidden -> 256, e = q.k / 16) is trained with the
                  talker and saved as added tensors (export_e2 --added-prefixes align_).
"""
from __future__ import annotations

import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from training.pretrain_uniss_megatron import load_megatron_runtime

from experiments.uniss_omni_s2st_v1.training.pretrain_omni_talker_megatron import disable_dataset_helpers

END_SEMANTIC, WAIT_READ, SEMANTIC_0, SEMANTIC_LAST = 180392, 180395, 155761, 163952
SPEED_0, N_SPEED = 180337, 35
METRICS = ("loss", "ce", "kl_ref", "spd", "align", "dur_pr", "p_cover_pos", "p_cover_neg", "close_acc", "labels")


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group(title="talker CRF schemes")
    g.add_argument("--crf-mode", choices=("plem", "align"), required=True)
    g.add_argument("--uniss-path", required=True, help="the talker to start from (T1c's export)")
    g.add_argument("--ref-path", default="", help="the frozen KL reference; default --uniss-path")
    g.add_argument("--crf-rows", required=True, help="talker_rows output (torch.save)")
    g.add_argument("--crf-w-ref", type=float, default=0.5)
    g.add_argument("--crf-w-speed", type=float, default=1.0)
    g.add_argument("--crf-w-align", type=float, default=1.0)
    g.add_argument("--crf-w-dur", type=float, default=0.1)
    g.add_argument("--crf-close-weight", type=float, default=3.0)
    g.add_argument("--crf-token-budget", type=int, default=24000)
    g.add_argument("--crf-max-len", type=int, default=3000)
    g.add_argument("--crf-window", type=int, default=2000)
    g.add_argument("--crf-seed", type=int, default=20261110)
    g.add_argument("--omni-no-recompute", action="store_true")
    return parser


def flat_rows(data: dict, max_len: int) -> list[dict]:
    """one entry per training sequence: plem -> per candidate {own, ref, y, w, spd (pos, qv) on the block's first};
    align -> per row {seq, y, ce, lab, text_start, M, lens}."""
    rows = []
    if data["mode"] == "plem":
        for bk in data["blocks"]:
            for i, c in enumerate(bk["cands"]):
                if len(c["own"]) + len(c["y"]) > max_len:
                    continue
                rows.append({"own": np.asarray(c["own"], np.int64), "ref": np.asarray(c["ref"], np.int64),
                             "y": np.asarray(c["y"], np.int64), "w": float(c["w"]),
                             "spd": (bk["spd_pos"], np.asarray(bk["qv"], np.float32)) if i == 0 else None})
    else:
        for bk in data["blocks"]:
            for x in bk["rows"]:
                if len(x["seq"]) + len(x["y"]) > max_len:
                    continue
                rows.append({"seq": np.asarray(x["seq"], np.int64), "y": np.asarray(x["y"], np.int64),
                             "ce": int(x["ce"]), "lab": int(x["lab"]), "text_start": int(x["text_start"]),
                             "M": int(x["M"]), "lens": np.asarray(x["lens"], np.int64)})
    return rows


class Groups(torch.utils.data.Dataset):
    """pretrain_selfmon.RowGroups over (own) sequence lengths: groups of <= budget padded tokens, a fresh permutation
    per pass, sorted by length only inside windows."""

    def __init__(self, rows, mode: str, budget: int, window: int, n_samples: int, seed: int):
        self.rows, self.mode = rows, mode
        key = "own" if mode == "plem" else "seq"
        lens = np.asarray([len(r[key]) + len(r["y"]) for r in rows])
        rng = np.random.default_rng(seed)
        self.groups, self.passes = [], 0
        while len(self.groups) < n_samples:
            perm = rng.permutation(len(rows))
            out = []
            for s in range(0, len(perm), window):
                w = perm[s: s + window]
                w = w[np.argsort(lens[w], kind="stable")]
                cur, longest = [], 0
                for i in w:
                    nl = max(longest, int(lens[i]))
                    if cur and nl * (len(cur) + 1) > budget:
                        out.append(cur)
                        cur, nl = [], int(lens[i])
                    cur.append(int(i))
                    longest = nl
                if cur:
                    out.append(cur)
            self.groups += [out[k] for k in rng.permutation(len(out))]
            self.passes += 1
        self.n = n_samples

    def __len__(self):
        return self.n

    @staticmethod
    def _pad(seqs):
        T = max(len(s) for s in seqs)
        ids = np.zeros((len(seqs), T), np.int64)
        mask = np.zeros((len(seqs), T), np.int64)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = s
            mask[j, : len(s)] = 1
        return torch.from_numpy(ids), torch.from_numpy(mask)

    def __getitem__(self, i):
        g = [self.rows[k] for k in self.groups[int(i)]]
        if self.mode == "plem":
            own = [np.concatenate([r["own"], r["y"]]) for r in g]
            ref = [np.concatenate([r["ref"], r["y"]]) for r in g]
            ids, mask = self._pad(own)
            rids, rmask = self._pad(ref)
            b, p, pr, lab, w, close, sb, sp, sq = [], [], [], [], [], [], [], [], []
            for j, r in enumerate(g):
                n0, r0 = len(r["own"]), len(r["ref"])
                for q in range(len(r["y"])):
                    b.append(j)
                    p.append(n0 + q - 1)
                    pr.append(r0 + q - 1)
                    lab.append(int(r["y"][q]))
                    w.append(r["w"])
                    close.append(int(r["y"][q]) in (END_SEMANTIC, WAIT_READ))
                if r["spd"] is not None:
                    sb.append(j)
                    sp.append(r["spd"][0])
                    sq.append(r["spd"][1])
            return {"ids": ids, "mask": mask, "rids": rids, "rmask": rmask, "b": torch.tensor(b), "p": torch.tensor(p),
                    "pr": torch.tensor(pr), "labels": torch.tensor(lab), "w": torch.tensor(w, dtype=torch.float32),
                    "is_close": torch.tensor(close), "sb": torch.tensor(sb, dtype=torch.long),
                    "sp": torch.tensor(sp, dtype=torch.long),
                    "sq": torch.from_numpy(np.stack(sq)) if sq else torch.zeros(0, N_SPEED)}
        seqs = [np.concatenate([r["seq"], r["y"]]) for r in g]
        ids, mask = self._pad(seqs)
        b, p, lab, close = [], [], [], []
        Tm = max(len(r["y"]) - 1 for r in g) or 1
        Mm = max(r["M"] for r in g) or 1
        ab, at0, aM, ac0, aT, ay = [], [], [], [], [], []
        alens = np.ones((len(g), Mm), np.int64)
        for j, r in enumerate(g):
            n0 = len(r["seq"])
            if r["ce"]:
                for q in range(len(r["y"])):
                    b.append(j)
                    p.append(n0 + q - 1)
                    lab.append(int(r["y"][q]))
                    close.append(int(r["y"][q]) in (END_SEMANTIC, WAIT_READ))
            ab.append(j)
            at0.append(r["text_start"])
            aM.append(r["M"])
            ac0.append(n0)
            aT.append(len(r["y"]) - 1)    # code frames (the closing token excluded)
            ay.append(r["lab"])
            alens[j, : r["M"]] = r["lens"][: r["M"]]
        return {"ids": ids, "mask": mask, "b": torch.tensor(b, dtype=torch.long), "p": torch.tensor(p, dtype=torch.long),
                "labels": torch.tensor(lab, dtype=torch.long), "is_close": torch.tensor(close, dtype=torch.bool),
                "a_b": torch.tensor(ab), "a_t0": torch.tensor(at0), "a_M": torch.tensor(aM), "a_c0": torch.tensor(ac0),
                "a_T": torch.tensor(aT), "a_y": torch.tensor(ay), "a_lens": torch.from_numpy(alens),
                "a_Tm": torch.tensor(Tm), "a_Mm": torch.tensor(Mm)}


def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("restricted to TP=PP=1")
    from experiments.mist_w3po_v1.talker.talker_crf_model import TalkerCRFModel

    if config is None:
        config = runtime.gpt_config_from_args(args)
    return TalkerCRFModel(config, mode=args.crf_mode, uniss_path=args.uniss_path, ref_path=args.ref_path or args.uniss_path,
                          w_ref=args.crf_w_ref, w_speed=args.crf_w_speed, w_align=args.crf_w_align, w_dur=args.crf_w_dur,
                          close_weight=args.crf_close_weight, recompute=not args.omni_no_recompute)


def loss_func(output_tensor):
    return output_tensor[0], OrderedDict((n, output_tensor[i].detach().clone()) for i, n in enumerate(METRICS))


def forward_step(data_iterator, model):
    batch = next(data_iterator)
    dev = torch.cuda.current_device()
    t = {k: v[0].to(dev, non_blocking=True) for k, v in batch.items()}
    return model(t), loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    runtime = load_megatron_runtime()
    args = runtime.megatron_gpt.get_args()
    data = torch.load(args.crf_rows, weights_only=False)
    if data["mode"] != args.crf_mode:
        raise SystemExit(f"{args.crf_rows} holds {data['mode']} rows, not {args.crf_mode}")
    rows = flat_rows(data, args.crf_max_len)
    n = int(args.train_iters) * int(args.global_batch_size)
    train = Groups(rows, args.crf_mode, args.crf_token_budget, args.crf_window, n, args.crf_seed)
    runtime.print_rank_0(f"talker crf {args.crf_mode}: {len(rows):,} rows, {len(train.groups):,} groups over "
                         f"{train.passes} passes for {n:,} samples")
    return train, None, None


train_valid_test_datasets_provider.is_distributed = True


def main() -> None:
    runtime = load_megatron_runtime()
    args = runtime.parse_and_validate_args(extra_args_provider=add_args, args_defaults={"tokenizer_type": "NullTokenizer"})
    from experiments.uniss_qwen3_omni_v1.training.pretrain_e2 import keep_lr_groups_apart

    keep_lr_groups_apart()
    disable_dataset_helpers()
    model_config = runtime.gpt_config_from_args(args)
    full_config = runtime.pretrain_cfg_container_from_args(args, model_config)
    full_config.model = None
    runtime.pretrain(full_config, train_valid_test_datasets_provider, runtime.ModelType.encoder_or_decoder,
                     forward_step, model_provider=model_provider)
    print("TALKER_CRF_MEGATRON_DONE", flush=True)


if __name__ == "__main__":
    main()
