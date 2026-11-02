"""QABC v11 (prereg_v6): QABC's rules on one clock per talk, the queue carried over from one sentence to the next.

si_qabc_rules.py decodes every sentence on its own with an empty queue at its start. Here the sentences of a talk are
decoded in order and the tracked queue of reports/qabc_v11/cont_queue.py (one Talker per talk; a committed unit adds
RATE * exp(-gamma * shift) s; q = what is committed and not yet played at talk time offset + t) is carried over, so
every decision sees the backlog the earlier sentences left. Each read is si_qabc_rules's: the sentence's audio up to
t, the sentence's own committed text, greedy with the wait marker barred and its raw log-probability asked for. The
throughput control (--ctrl, cont_queue.CONTROLS) gives each block its shift from q at its commit; the Talker reads it
(si_infer_cont --speed-policy tracked), and the same function annotates LA2's blocks, so every policy gets the same
control. Rules (all with tau0 = sigma(b0), Delta the read window, TAU_MAX = 0.9975):

  p5        the deployed QABC rule (si_qabc_rules p5 = si_thinker_stream selfwait): greedy with the bias logit tau(q),
            tau(q) = 1 - (1 - tau0) max(0, Delta - q) / Delta
  gate      the binary queue gate of v10 (si_thinker_stream --sw-queue-binary): tau = TAU_MAX while q >= Delta, else tau0
  linlogit  logit tau linear in the queue: b0 + (logit TAU_MAX - b0) min(1, q / Delta)
  seqb      the within-block rule of v10 (si_qabc_rules seqb): unit u+1 commits while rho(u) > tau(q + r u)

Streams are (window, rule, talk); all streams of a process are decoded together, a read each per round. --dump writes
every read (q, tau, what each of the four rules would commit from the same draft and queue). Outputs refuse to
overwrite: <out>/blocks_<d>_w<w>_c<rule><tag>b<b0>q<rate>_<ctrl>.jsonl, sentences in talk order, every block with
q_ms (tracked queue at its commit) and shift.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_qabc_cont --direction en2zh --windows 1,2 --rules p5,gate \
        --thinker <merged> --thinker-tag s0 --ctrl fr --gamma 0.017 --out <dir> [--talks pseudo] [--limit-talks 2] [--dump <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from experiments.uniss_qwen3_omni_v1.evaluation.si_qabc_rules import TAU_MAX, choose, make_head, read_numbers, tau_of
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import THINKER_EOS, WAIT_MARK, join, units

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports/qabc_v11"))
import cont_queue as cq  # noqa: E402

RULES = ("p5", "gate", "linlogit", "seqb")


def tau_rule(rule: str, tau0: float, window_s: float, q_s: float) -> float:
    if rule == "gate":
        return TAU_MAX if q_s >= window_s else tau0
    if rule == "linlogit":
        b0, bm = math.log(tau0 / (1 - tau0)), math.log(TAU_MAX / (1 - TAU_MAX))
        return 1 / (1 + math.exp(-(b0 + (bm - b0) * min(1.0, q_s / window_s))))
    return tau_of(tau0, window_s, q_s)


def keep_text(rule: str, nums: dict, tau0: float, window_s: float, q_s: float, rate: float) -> str:
    """The prefix of the continuation this rule commits (p5 mechanics for p5/gate/linlogit, as selfwait decodes)."""
    if rule == "seqb":
        return choose("seqb", nums, tau0, window_s, q_s, rate)
    tau = tau_rule(rule, tau0, window_s, q_s)
    b = math.log(tau / (1 - tau))
    for j, (lt, lw) in enumerate(zip(nums["lp_tok"], nums["lp_wait"])):
        if lw + b > lt:
            return "".join(nums["toks"][:j])
    return nums["cont"]


def load_audio(talks, direction, pseudo):
    import soundfile as sf
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items
    if pseudo:
        rows = {r["id"]: r for r in map(json.loads, open(cq.SPONT[direction], encoding="utf-8"))}
        src = {k: (r["audio"], 0, int(r["dur_ms"])) for k, r in rows.items()}
    else:
        src = {it["id"]: (it["wav"], it["start"], it["end"]) for it in realsi_items(direction)}
    audio, cache = {}, {}
    for _, sents in talks:
        for sid, _, _ in sents:
            path, a, b = src[sid]
            if path not in cache:
                x, sr = sf.read(path, dtype="float32", always_2d=True)
                assert sr == 16000, path
                cache = {path: x.mean(axis=1)}
            audio[sid] = cache[path][a * 16: b * 16] if not pseudo else cache[path]
    return audio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--rules", required=True, help=f"comma-separated, of {RULES}")
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--thinker-tag", required=True)
    ap.add_argument("--b0", type=float, default=1.0)
    ap.add_argument("--ctrl", required=True, choices=sorted(cq.CONTROLS))
    ap.add_argument("--gamma", type=float, required=True, help="speed gained per shift step (prereg_v6: fitted on dev)")
    ap.add_argument("--talks", choices=("realsi", "pseudo"), default="realsi")
    ap.add_argument("--limit-talks", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    d, zh = a.direction, a.direction.endswith("zh")
    rules = a.rules.split(",")
    assert all(r in RULES for r in rules), rules
    windows = [float(w) for w in a.windows.split(",")]
    talks = cq.pseudo_talks(d) if a.talks == "pseudo" else cq.realsi_talks(d)
    if a.limit_talks:
        talks = talks[: a.limit_talks]
    rate = cq.RATE[d]
    outs = {}
    for w in windows:
        for rule in rules:
            outs[(w, rule)] = Path(a.out) / f"blocks_{d}_w{w:g}_c{rule}{a.thinker_tag}b{a.b0:g}q{rate:g}_{a.ctrl}.jsonl"
    for o in outs.values():
        if o.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.dump:
        Path(a.dump).mkdir(parents=True, exist_ok=True)
    audio = load_audio(talks, d, a.talks == "pseudo")
    head = make_head(d)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.thinker, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=a.seed, tensor_parallel_size=a.tp)
    sp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logit_bias={WAIT_MARK: -100.0},
                        logprob_token_ids=[WAIT_MARK])
    tau0 = 1 / (1 + math.exp(-a.b0))
    streams = []
    for (w, rule) in outs:
        for talk, sents in talks:
            streams.append({"w": w, "rule": rule, "talk": talk, "sents": sents, "si": 0, "r": 0, "busy": -1e18,
                            "committed": [], "blocks": [], "drafts": [], "recs": [], "reads": [], "done": not sents})
    t0, rnd = time.time(), 0
    while any(not s["done"] for s in streams):
        live = [s for s in streams if not s["done"]]
        prompts = []
        for s in live:
            sid, off, dur = s["sents"][s["si"]]
            s["t"] = min((s["r"] + 1) * int(round(s["w"] * 1000)), dur)
            prompts.append({"prompt": head + join(s["committed"], zh),
                            "multi_modal_data": {"audio": (audio[sid][: s["t"] * 16], 16000)}})
        results = llm.generate(prompts, sp, use_tqdm=False)
        for s, res in zip(live, results):
            sid, off, dur = s["sents"][s["si"]]
            w_s, t = s["w"], s["t"]
            nums = read_numbers(res.outputs[0], zh)
            n = len(s["committed"])
            final = t >= dur
            q = max(0.0, s["busy"] - (off + t)) / 1000
            sh = cq.shift_of(q, a.ctrl)
            r_eff = cq.unit_s(d, sh, a.gamma)
            said = nums["cont"] if final else keep_text(s["rule"], nums, tau0, w_s, q, r_eff)
            sep = "" if zh or not s["committed"] or said[:1].isspace() or said[:1] in ",.;:!?)" else " "
            draft_all = units(join(s["committed"], zh) + sep + nums["cont"], zh)
            s["drafts"].append({"t": t, "draft": join(draft_all, zh)})
            new = units(join(s["committed"], zh) + sep + said, zh)[n:] if said.strip() else []
            if a.dump and not final:
                alt = {}
                for rr in RULES:
                    x = keep_text(rr, nums, tau0, w_s, q, r_eff)
                    alt[rr] = len(units(join(s["committed"], zh) + sep + x, zh)[n:]) if x.strip() else 0
                s["reads"].append({"id": sid, "t": t, "q_s": round(q, 3), "shift": sh, "tau": round(tau_rule(s["rule"], tau0, w_s, q), 6),
                                   "keep": len(new), "alt": alt, "n_draft": len(units(nums["cont"], zh))})
            if new or final:
                text = ("" if zh or not n else " ") + join(new, zh)
                s["blocks"].append({"write_ms": t, "text": text, "final": final, "q_ms": int(round(q * 1000)), "shift": sh})
                s["committed"] += new
                if new:
                    s["busy"] = max(s["busy"], off + t) + len(new) * r_eff * 1000
            s["r"] += 1
            if final:
                blocks = [b for b in s["blocks"] if b["text"].strip() or b["final"]]
                if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
                    blocks = blocks[:-1]
                    blocks[-1]["final"] = True
                s["recs"].append({"id": sid, "talk": s["talk"], "offset_ms": off, "translation": join(s["committed"], zh),
                                  "blocks": blocks, "drafts": s["drafts"]})
                s.update(si=s["si"] + 1, r=0, committed=[], blocks=[], drafts=[])
                if s["si"] >= len(s["sents"]):
                    s["done"] = True
        rnd += 1
        if rnd % 20 == 0:
            print(f"{d} round {rnd}: {len(live)} streams, {time.time() - t0:.0f}s", flush=True)
    for (w, rule), o in outs.items():
        mine = [s for s in streams if s["w"] == w and s["rule"] == rule]
        with open(o, "x", encoding="utf-8") as fh:
            for s in mine:
                fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in s["recs"])
        if a.dump:
            with open(Path(a.dump) / f"{o.stem}.jsonl", "x", encoding="utf-8") as fh:
                for s in mine:
                    fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in s["reads"])
        print(f"{o.name}: {sum(len(s['recs']) for s in mine)} sentences", flush=True)
    print(f"{d}: {rnd} rounds, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
