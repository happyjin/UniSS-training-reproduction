"""The whole system's text side on RealSI: the frozen Thinker translating as the source arrives.

At the end of every read window (--window seconds, SimulS2ST-Omni's m) the
Thinker (Qwen3-Omni, the E1 prompt) is given the source heard so far and the
translation committed so far as the start of its reply, and continues it
greedily. With --policy la2, what the last two continuations agree on
(LocalAgreement-2: the longest common prefix, in English words or Chinese
characters) is committed; with --policy hold, all of the continuation but
its last --hold units (Hold-n, Liu et al. 2020) -- sooner, and more exposed
to the Thinker completing a sentence it has not heard. With --policy vote,
--n samples are drawn at once and a unit is committed while at least --gamma of
them agree on it (RALCP, Wang et al. 2024; TAF's majority vote, Ouyang et al.
2025): agreement across possible futures at one moment, instead of agreement
across two moments, which costs LocalAgreement a window. --snap ends each read
at the quietest frame of its last 200 ms when that frame is a real pause (30 dB
under the peak), so the Thinker is not handed half a word (Simul-Whisper's
truncation problem); --prompt stream adds to the instruction that the speech
may stop mid-sentence and only what was said is to be translated. What is committed
becomes a block for the talker; when the source has ended, the rest of
the translation is committed as the last block. Nothing committed is ever
revised. The Thinker is not trained for this: it is the training-free text
side the talker is evaluated with, until the Thinker gets its own streaming
training.

With --policy head (CRISP's C2/C3) one greedy draft is drawn with its top-5
log-probabilities and a small head (si_commit_head.py) scores each token: the
prefix whose tokens all score at least --tau is committed -- one pass, no
agreement to wait for. --dump writes, beside a vote run, that same greedy draft
with its features and two labels per unit: inside the vote's consensus (what the
head distils) and consistent with the finished translation (what the risk
control bounds). --items-file runs on training rows instead of RealSI.
With --decision bayes the head commits the k units maximising P(first k agree)
+ lam k instead. --relabel-n with --dump on a head run labels the head's own
draft by the consensus of n samples drawn at the head's own state: labels from
the states the head itself reaches, for training the next head (DAgger).
With --policy selfwait (Plan 4A) a Thinker trained to end each read's reply with
the wait marker <|box_end|> (--thinker, a merged copy) is asked once per read:
what it says before the marker is committed; --wait-biases shifts the marker's
logit, so one model gives the whole latency/quality curve.

Commits do not depend on what the talker says, so the text side runs first
for every sentence and window, and si_infer --text-policy file speaks it.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream --direction zh2en --windows 2,3,4,5,6 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
THINKER_EOS = 151645
WAIT_MARK = 151649   # <|box_end|>: Plan 4A's "say no more yet"


def units(text: str, zh: bool) -> list[str]:
    return [c for c in text if not c.isspace()] if zh else text.split()


def join(us: list[str], zh: bool) -> str:
    return "".join(us) if zh else " ".join(us)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--windows", required=True, help="read windows in seconds, comma-separated; one file each")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--policy", choices=("la2", "hold", "vote", "head", "selfwait", "belief", "oracle"),
                    default="la2", help="oracle: commit what the Thinker would still say with the whole sentence heard -- "
                                        "the stability label itself as the policy, an upper bound, not a system")
    ap.add_argument("--belief", default="", help="belief policy: an si_belief checkpoint (prior g + likelihood ratio h)")
    ap.add_argument("--belief-min", type=int, default=1,
                    help="belief policy: a commit shorter than this many units waits unless it ends a clause")
    ap.add_argument("--belief-sigma", type=float, default=0.0,
                    help="belief policy: prior sd of the talk-level offset, updated after each sentence (0: off)")
    ap.add_argument("--thinker", default=QWEN3, help="the Thinker vLLM serves (Plan 4A: a merged self-distilled copy)")
    ap.add_argument("--thinker-tag", default="", help="selfwait policy: names the Thinker in the output files")
    ap.add_argument("--wait-biases", default="0",
                    help="selfwait policy: logit biases on the wait marker <|box_end|>, comma-separated; one file each")
    ap.add_argument("--items-file", default="", help="jsonl of {id, audio, dur_ms, translation} instead of RealSI")
    ap.add_argument("--dump", default="", help="vote policy: write the greedy draft's features and labels here")
    ap.add_argument("--head", default="", help="head policy: a si_commit_head.py checkpoint")
    ap.add_argument("--taus", default="0.5", help="head policy: commit thresholds, comma-separated; one file each")
    ap.add_argument("--decision", choices=("threshold", "bayes"), default="threshold",
                    help="head policy: commit the leading units scoring >= tau, or (bayes) the k maximising "
                         "P(first k agree) + lam k, with --taus read as lam")
    ap.add_argument("--agree-beta", type=float, default=None,
                    help="head policy: fuse the head (prior) with cross-read agreement (evidence, log-likelihood "
                         "ratio +-beta); named in the file as a<beta>")
    ap.add_argument("--first-tau", type=float, default=None,
                    help="head policy: the threshold while nothing is committed and no earlier draft can be compared "
                         "(no evidence, the prior alone); named in the file as f<tau>")
    ap.add_argument("--min-units", type=int, default=1,
                    help="head policy: any commit shorter than this many units waits unless it ends a clause (the "
                         "talker's fragment cost); named as u<n>")
    ap.add_argument("--voi-kappa", type=float, default=0.0,
                    help="head policy with --min-units: instead of always holding a short commit, hold it only when "
                         "waiting is worth it -- q * kappa > k, q the posterior chance the next read completes it to "
                         "--min-units (each missing unit surviving and clearing tau after one more agreeing read), "
                         "kappa the fragment's cost in units of one read's delay of one unit; named as v<kappa>")
    ap.add_argument("--first-min", type=int, default=0,
                    help="head policy: the first commit waits until it has at least this many units; named as m<n>")
    ap.add_argument("--virtual-ms", type=int, default=0,
                    help="head policy: the cross-read evidence comes from a draft of the same read on the audio minus its "
                         "last this many ms (a virtual earlier read, computed now), not from the previous read -- the "
                         "agreement LocalAgreement waits a whole read for, without the wait; named as vr<ms>")
    ap.add_argument("--virtual-ms-list", default="",
                    help="head policy: also draft this read on the audio minus each of these ms (comma-separated); with "
                         "--dump --oracle they are dumped as the evidence the likelihood is fitted on; named as vl<ms-ms>")
    ap.add_argument("--lr-table", default="",
                    help="head policy: Route 1's posterior (si_bayes_lr fit): --head is the prior without the agreement "
                         "feature, the previous read's draft and the table's virtual drafts are evidence through the "
                         "table's log-likelihood ratios; replaces --agree-beta / --virtual-ms; named as lr<stem>")
    ap.add_argument("--adapt", choices=("none", "bayes", "aci", "both"), default="none",
                    help="with --lr-table: Route 2 (si_bayes_adapt) -- each talk's sentences run one after another and, "
                         "once a sentence ends, its reads are labelled by the oracle and adapt the talk's prior offset "
                         "(bayes), its threshold (aci, --taus the start), or both; named as ad<mode>...")
    ap.add_argument("--adapt-sigma", type=float, default=1.0, help="bayes: prior sd of the talk's log-odds offset")
    ap.add_argument("--aci-gamma", type=float, default=0.02, help="aci: threshold step per labelled read")
    ap.add_argument("--aci-eps", type=float, default=0.1, help="aci: target rate of reads committing past the oracle")
    ap.add_argument("--queue-rate", type=float, default=0.0,
                    help="head policy (Route 3): seconds of speech a committed unit takes; the threshold then counts only "
                         "the delay the talker would sit idle -- tau_t = 1 - (1 - tau) max(0, window - queue) / window, the "
                         "queue being the committed speech not yet played; 0: off; named as q<rate>")
    ap.add_argument("--queue-tau-max", type=float, default=1.0,
                    help="with --queue-rate: the queue raises the threshold at most to this (1: no cap); named as c<cap>")
    ap.add_argument("--mbr", type=int, default=0,
                    help="minimum Bayes risk over what to say: each read also samples this many continuations and keeps, "
                         "of them and the greedy draft, the one with the highest mean chrF to the others (the candidate "
                         "with the least expected loss under the Thinker's own posterior); 0: greedy; named as _mbr<k>")
    ap.add_argument("--mbr-final-only", action="store_true",
                    help="with --mbr: only the read that has heard the whole sentence picks by MBR; the commits before it "
                         "are untouched (their latency is the greedy run's); named as _mbrf<k>")
    ap.add_argument("--sw-queue-rate", type=float, default=0.0,
                    help="selfwait policy (Route 3 on the amortised posterior): the wait-marker bias b is the commit "
                         "threshold's log-odds, tau = sigmoid(b); with this many seconds of speech a committed unit, "
                         "tau_t = 1 - (1 - tau) max(0, window - queue) / window (capped by --queue-tau-max); named q<rate>")
    ap.add_argument("--sw-adapt", choices=("none", "aci", "beta"), default="none",
                    help="selfwait policy: each talk's sentences in turn; when one ends its reads are labelled by the "
                         "Thinker continuing each committed prefix with the whole sentence heard (no reference), a read "
                         "being wrong when it committed past that; the talk's bias then follows -- aci: b += gamma "
                         "(err - eps) a read (Gibbs & Candes 2021); beta: the talk's revision rate pi ~ Beta(n0 eps, "
                         "n0 (1 - eps)), b = kappa (logit E[pi | reads] - logit eps); named ad<mode>")
    ap.add_argument("--sw-gamma", type=float, default=0.1)
    ap.add_argument("--sw-eps", type=float, default=0.1)
    ap.add_argument("--sw-kappa", type=float, default=1.0)
    ap.add_argument("--sw-n0", type=float, default=20.0, help="beta: prior strength in reads")
    ap.add_argument("--hygiene", action="store_true",
                    help="selfwait policy (fluency step 1, si_fluency_rules.hygiene): a non-final block does not end on a "
                         "function word or inside a Chinese word, is not punctuation alone, and while the talker is idle is "
                         "not shorter than --hyg-min units unless it ends a clause; named _hyg<min>")
    ap.add_argument("--hyg-min", type=int, default=0, help="with --hygiene: the shortest block (0: 3 Chinese characters / 2 words)")
    ap.add_argument("--hyg-hold-ms", type=int, default=0,
                    help="with --hygiene: text the draft has but a rule holds back goes out once it has been held this long "
                         "while the talker is idle (0: no limit); named hold<ms>")
    ap.add_argument("--sw-urgency", type=float, default=0.0,
                    help="selfwait policy with --sw-queue-rate: the wait-marker bias falls by this much per second the talker "
                         "has been idle (fluency plan v2), down to --sw-urgency-min; named _urg<k>m<min>")
    ap.add_argument("--sw-urgency-min", type=float, default=-3.0)
    ap.add_argument("--clean", action="store_true",
                    help="selfwait policy (fluency step 2, si_fluency_rules.clean): the talker's text loses fillers and "
                         "immediate repeats; the Thinker's committed text is unchanged; named _cln")
    ap.add_argument("--look-units", type=int, default=0,
                    help="head / belief policy: beside each committed block, keep the next this many draft units it holds "
                         "back as \"look\" (the talker's lookahead; si_infer --lookahead); 0: none")
    ap.add_argument("--n", type=int, default=8, help="vote policy: samples per read")
    ap.add_argument("--relabel-n", type=int, default=0,
                    help="head policy with --dump: also draw this many samples a read and label the head's own "
                         "draft with their consensus (on-policy labels for the next head)")
    ap.add_argument("--relabel-gamma", type=float, default=0.8, help="the consensus threshold for --relabel-n")
    ap.add_argument("--oracle", action="store_true",
                    help="head policy with --dump: also continue each read's committed prefix greedily with the whole "
                         "sentence heard, and dump it beside the draft (the label: does the draft survive the future)")
    ap.add_argument("--gammas", default="0.8", help="vote policy: agreement thresholds, comma-separated; one file each")
    ap.add_argument("--temperature", type=float, default=0.7, help="vote policy: sampling temperature (named in the file when not 0.7)")
    ap.add_argument("--snap", action="store_true", help="end each read at a pause within its last 200 ms")
    ap.add_argument("--prompt", choices=("base", "stream"), default="base")
    ap.add_argument("--hold", type=int, default=0, help="hold policy: units held back (default 2 English words, 3 Chinese characters)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    windows = [float(w) for w in args.windows.split(",")]
    gammas = ([float(g) for g in args.gammas.split(",")] if args.policy == "vote" else
              [float(x) for x in args.taus.split(",")] if args.policy in ("head", "belief") else
              [float(x) for x in args.wait_biases.split(",")] if args.policy == "selfwait" else [None])
    extra = ("_snap" if args.snap else "") + ("_sp" if args.prompt == "stream" else "") + (f"_mbr{'f' if args.mbr_final_only else ''}{args.mbr}" if args.mbr else "")
    if (args.hygiene or args.clean) and args.policy != "selfwait":
        raise SystemExit("--hygiene / --clean are implemented for the selfwait policy only")
    hyg_min = (args.hyg_min or (3 if args.direction.endswith("zh") else 2)) if args.hygiene else 0
    extra += (f"_hyg{hyg_min}" if hyg_min else "") + (f"hold{args.hyg_hold_ms}" if hyg_min and args.hyg_hold_ms else "") \
        + ("_cln" if args.clean else "") + (f"_urg{args.sw_urgency:g}m{args.sw_urgency_min:g}" if args.sw_urgency else "")
    if args.sw_urgency and not (args.policy == "selfwait" and args.sw_queue_rate):
        raise SystemExit("--sw-urgency needs the selfwait policy with --sw-queue-rate (the talker's queue)")
    sw_extra = ("" if args.policy != "selfwait" else
                (f"q{args.sw_queue_rate:g}" + (f"c{args.queue_tau_max:g}" if args.queue_tau_max < 1 else "") if args.sw_queue_rate else "")
                + (f"ad{args.sw_adapt}e{args.sw_eps:g}" + (f"g{args.sw_gamma:g}" if args.sw_adapt == "aci" else
                   f"k{args.sw_kappa:g}n{args.sw_n0:g}") if args.sw_adapt != "none" else ""))

    def tag_of(g):
        base = ("" if args.policy == "la2" else "_oraclecommit" if args.policy == "oracle" else f"_hold{args.hold or ''}" if args.policy == "hold" else
                f"_vote{args.n}g{g:g}" + (f"T{args.temperature:g}" if args.temperature != 0.7 else "")
                if args.policy == "vote" else f"_sw{args.thinker_tag}b{g:g}" if args.policy == "selfwait"
                else f"_bel{Path(args.belief).stem}t{g:g}m{args.belief_min}s{args.belief_sigma:g}" if args.policy == "belief"
                else f"_head{Path(args.head).stem}{'l' if args.decision == 'bayes' else 't'}{g:g}"
                + (f"a{args.agree_beta:g}" if args.agree_beta is not None else "")
                + (f"f{args.first_tau:g}" if args.first_tau is not None else "")
                + (f"m{args.first_min}" if args.first_min else "")
                + (f"u{args.min_units}" if args.min_units > 1 else "")
                + (f"v{args.voi_kappa:g}" if args.voi_kappa > 0 else "")
                + (f"vr{args.virtual_ms}" if args.virtual_ms else "")
                + (f"vl{args.virtual_ms_list.replace(',', '-')}" if args.virtual_ms_list and not args.lr_table else "")
                + (f"lr{Path(args.lr_table).stem}" if args.lr_table else "")
                + (f"ad{args.adapt}" + (f"s{args.adapt_sigma:g}" if args.adapt in ("bayes", "both") else "")
                   + (f"g{args.aci_gamma:g}e{args.aci_eps:g}" if args.adapt in ("aci", "both") else "")
                   if args.lr_table and args.adapt != "none" else ""))
        return base + sw_extra + (f"q{args.queue_rate:g}" + (f"c{args.queue_tau_max:g}" if args.queue_tau_max < 1 else "")
                       if args.queue_rate and args.policy == "head" else "") + extra + (
            f"_look{args.look_units}" if args.look_units else "")
    runs = [(w, g, Path(args.out) / f"blocks_{args.direction}_w{w:g}{tag_of(g)}.jsonl") for w in windows for g in gammas]
    outs = [o for _, _, o in runs]
    for out in outs:
        if out.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {out}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.evaluation.si_infer import realsi_items
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    zh = args.direction.endswith("zh")
    tgt = "cmn" if zh else "eng"
    if args.items_file:
        items = [{"id": r["id"], "wav": r["audio"], "start": 0, "end": int(r["dur_ms"]), "trg_text": r.get("translation", "")}
                 for r in map(json.loads, open(args.items_file, encoding="utf-8"))]
    else:
        items = realsi_items(args.direction)
    if args.limit and args.limit < len(items):
        items = [items[round(i * (len(items) - 1) / (args.limit - 1))] for i in range(args.limit)]
    cache = {}
    for it in items:
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache = {it["wav"]: x.mean(axis=1)}
        it["audio"] = cache[it["wav"]][it["start"] * 16: it["end"] * 16]
        it["dur_ms"] = len(it["audio"]) // 16
        f = it["audio"][: len(it["audio"]) // 320 * 320].reshape(-1, 320)
        it["frame_db"] = 10 * np.log10((f ** 2).mean(1) + 1e-10) if len(f) else np.zeros(1)
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    instruction = inference_instruction()[tgt]
    if args.prompt == "stream":
        instruction += (" The speech may stop in the middle of a sentence: translate only what has been said so far, "
                        "and do not guess or complete what has not been said.")
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                         {"type": "text", "text": instruction}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=args.thinker, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=args.seed)
    params = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS])
    for w, g, out in runs:
        if args.policy == "belief":
            from experiments.uniss_qwen3_omni_v1.evaluation.si_belief import load_belief, run_belief
            run_belief(llm, SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logprobs=5),
                       items, head, zh, int(round(w * 1000)), out, f"{args.direction} {out.stem}", load_belief(args.belief),
                       g, args.belief_min, args.belief_sigma or None, args.look_units)
            continue
        hold = (args.hold or (3 if zh else 2)) if args.policy == "hold" else None
        vote = (args.n, g) if args.policy == "vote" else None
        scorer = None
        if args.policy == "head":
            from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import load_head
            scorer = (load_head(args.head), g, args.decision, args.agree_beta, args.first_tau, args.first_min,
                      args.min_units, args.voi_kappa)
        dump = Path(args.dump) / f"{out.stem}.jsonl" if args.dump else None
        lr = None
        if args.lr_table and scorer is not None:
            from experiments.uniss_qwen3_omni_v1.evaluation.si_bayes_lr import load_table
            lr = load_table(args.lr_table)
        adapt = None
        if lr is not None and args.adapt != "none":
            from experiments.uniss_qwen3_omni_v1.evaluation.si_bayes_adapt import TalkAdapt
            adapt = TalkAdapt(args.adapt, args.adapt_sigma, args.aci_gamma, args.aci_eps, g)
        vlist = (lr["virtual"] if lr is not None else
                 [int(v) for v in args.virtual_ms_list.split(",")] if args.virtual_ms_list and scorer is not None else [])
        run_window(llm, params, items, head, zh, int(round(w * 1000)), out, f"{args.direction} {out.stem}", hold,
                   vote=vote, snap=args.snap, seed=args.seed, dump=dump, scorer=scorer, temperature=args.temperature,
                   relabel=(args.relabel_n, args.relabel_gamma) if args.relabel_n and scorer else None,
                   selfwait=g if args.policy == "selfwait" else None, oracle=args.oracle and scorer is not None,
                   oracle_commit=args.policy == "oracle", queue_rate=args.queue_rate, queue_tau_max=args.queue_tau_max, mbr=args.mbr, mbr_final_only=args.mbr_final_only,
                   sw_cfg={"rate": args.sw_queue_rate, "adapt": args.sw_adapt, "gamma": args.sw_gamma, "eps": args.sw_eps,
                           "kappa": args.sw_kappa, "n0": args.sw_n0, "urgency": args.sw_urgency, "urgency_min": args.sw_urgency_min}
                   if args.policy == "selfwait" else None, look_units=args.look_units, virtual_ms=0 if lr is not None else args.virtual_ms,
                   fluency={"hyg_min": hyg_min, "clean": args.clean, "hold_ms": args.hyg_hold_ms} if (hyg_min or args.clean) else None,
                   virtual_list=vlist, lr=lr, adapt=adapt)


class _Pick:
    def __init__(self, out):
        self.outputs = [out]


def chrf(h: str, r: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (Popovic 2015), spaces ignored."""
    from collections import Counter

    h, r = h.replace(" ", ""), r.replace(" ", "")
    ps, rs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i: i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i: i + k] for i in range(len(r) - k + 1))
        m = sum((hc & rc).values())
        if hc:
            ps.append(m / sum(hc.values()))
        if rc:
            rs.append(m / sum(rc.values()))
    if not ps or not rs:
        return float(h == r)
    p, q = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + q == 0 else (1 + beta ** 2) * p * q / (beta ** 2 * p + q)


def mbr_pick(outs: list):
    """The candidate with the highest mean utility to the others; a tie keeps the first (the greedy draft)."""
    texts = [o.text for o in outs]
    score = [sum(chrf(a, b) for j, b in enumerate(texts) if j != i) for i, a in enumerate(texts)]
    return outs[int(np.argmax(score))]


def snapped(it: dict, t: int) -> int:
    """The quietest 20 ms frame of the read's last 200 ms, when it is a pause; else t."""
    db = it["frame_db"]
    hi, lo = min(t // 20, len(db)), max(0, (t - 200) // 20)
    if t >= it["dur_ms"] or hi <= lo:
        return t
    k = lo + int(np.argmin(db[lo:hi]))
    return (k + 1) * 20 if db[k] < db.max() - 30 else t


def ralcp(cands: list[list[str]], n: int, gamma: float) -> list[str]:
    """The prefix at least gamma of the n candidates agree on, unit by unit."""
    alive, out, j = cands, [], 0
    while True:
        votes = {}
        for c in alive:
            if len(c) > j:
                votes[c[j]] = votes.get(c[j], 0) + 1
        if not votes:
            return out
        tok, k = max(votes.items(), key=lambda kv: kv[1])
        if k < gamma * n:
            return out
        out.append(tok)
        alive = [c for c in alive if len(c) > j and c[j] == tok]
        j += 1


def draft_tokens(out, cont: str, zh: bool) -> dict:
    """A greedy draft's tokens: text, log-probability, top-5 log-probabilities, and the unit each starts."""
    toks, lp, top, unit_of = [], [], [], []
    spans = ([(m.start(), m.end()) for m in re.finditer(r"\S", cont)] if zh else
             [(m.start(), m.end()) for m in re.finditer(r"\S+", cont)])
    pos = 0
    for tid, alts in zip(out.token_ids, out.logprobs or []):
        if tid == THINKER_EOS:
            break
        chosen = alts.get(tid)
        text = chosen.decoded_token if chosen is not None and chosen.decoded_token is not None else ""
        lead = len(text) - len(text.lstrip())
        u = next((k for k, (a, b) in enumerate(spans) if a <= pos + lead < b), len(spans) - 1 if spans else 0)
        toks.append(text)
        lp.append(float(chosen.logprob) if chosen is not None else -20.0)
        top.append(sorted((float(a.logprob) for a in alts.values()), reverse=True)[:5])
        unit_of.append(u)
        pos += len(text)
    return {"tokens": toks, "lp": lp, "top": top, "unit_of": unit_of}


def run_window(llm, params, items, head, zh, window, out, tag, hold=None, vote=None, snap=False, seed=0,
               dump=None, scorer=None, temperature=0.7, relabel=None, selfwait=None, oracle=False, look_units=0,
               virtual_ms=0, virtual_list=(), lr=None, adapt=None, oracle_commit=False, queue_rate=0.0, queue_tau_max=1.0, mbr=0, mbr_final_only=False, sw_cfg=None,
               fluency=None):
    from vllm import SamplingParams
    # Plan 4A: one greedy pass that stops at the wait marker; at the last read the marker is barred
    sw_read = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS, WAIT_MARK],
                             logit_bias={WAIT_MARK: selfwait}) if selfwait is not None else None
    sw_last = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS],
                             logit_bias={WAIT_MARK: -100.0}) if selfwait is not None else None
    sample = SamplingParams(n=vote[0], temperature=temperature, top_p=0.95, max_tokens=160, stop_token_ids=[THINKER_EOS],
                            seed=seed) if vote else None
    greedy_lp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logprobs=5)
    records = []
    for it in items:
        it.update(committed=[], prev=None, prev_greedy=None, blocks=[], drafts=[], done=False, busy=0, spoken="", held_t=None)   # nothing from an earlier run
    nxt = {}
    sw_seq = bool(sw_cfg and sw_cfg["adapt"] != "none")
    sw_talk = {}   # selfwait adaptation: talk -> {"b": offset, "reads", "errs", "trace"}
    if adapt is not None or sw_seq:   # Route 2: a talk's sentences one after another, so its labels reach only later sentences
        last = {}
        for it in items:
            talk = it["id"].rsplit("_", 1)[0]
            it.update(r=0, reads=[], waiting=talk in last)
            if talk in last:
                nxt[last[talk]["id"]] = it
            last[talk] = it
    t0, r = time.time(), 0
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"] and not it.get("waiting")]
        prompts = []
        for it in live:
            t = min(((it["r"] if adapt is not None or sw_seq else r) + 1) * window, it["dur_ms"])
            it["t"] = t
            heard = snapped(it, t) if snap else t
            prompts.append({"prompt": head + join(it["committed"], zh),
                            "multi_modal_data": {"audio": (it["audio"][: heard * 16], 16000)}})
        plist = [(sw_read if it["t"] < it["dur_ms"] else sw_last) if selfwait is not None else
                 sample if vote and it["t"] < it["dur_ms"] else greedy_lp if scorer else params for it in live]
        if sw_cfg and (sw_cfg["rate"] or sw_seq):   # the bias as a Bayes threshold that moves with the talk and the queue
            for k, it in enumerate(live):
                if it["t"] >= it["dur_ms"]:
                    continue
                b = selfwait + sw_talk.get(it["id"].rsplit("_", 1)[0], {}).get("b", 0.0)
                if sw_cfg["rate"]:
                    tau = 1.0 / (1.0 + np.exp(-b))
                    tau = max(tau, min(queue_tau_max, 1 - (1 - tau) * max(0.0, window - max(0, it["busy"] - it["t"])) / window))
                    tau = min(tau, 0.9975)
                    b = float(np.log(tau / (1 - tau)))
                if sw_cfg.get("urgency"):   # the longer the talker has sat idle, the cheaper committing (fluency plan v2)
                    b = max(sw_cfg["urgency_min"], b - sw_cfg["urgency"] * max(0, it["t"] - it["busy"]) / 1000)
                plist[k] = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS, WAIT_MARK],
                                          logit_bias={WAIT_MARK: float(b)})
        results = llm.generate(prompts, plist, use_tqdm=False)
        if mbr and selfwait is None and vote is None:   # Bayes risk over what to say, before deciding what to commit
            mp = SamplingParams(n=mbr, temperature=0.7, top_p=0.95, max_tokens=160, stop_token_ids=[THINKER_EOS],
                                seed=seed + r, logprobs=5 if scorer else None)
            todo = [k for k, it in enumerate(live) if not mbr_final_only or it["t"] >= it["dur_ms"]]
            for k, sm in zip(todo, llm.generate([prompts[k] for k in todo], mp, use_tqdm=False) if todo else []):
                results[k] = _Pick(mbr_pick([results[k].outputs[0]] + list(sm.outputs)))
        greedy = [None] * len(live)
        if dump is not None and vote is not None:   # the single greedy draft the head will see, beside the vote's samples
            todo = [k for k, it in enumerate(live) if it["t"] < it["dur_ms"]]
            for k, g in zip(todo, llm.generate([prompts[k] for k in todo], greedy_lp, use_tqdm=False)):
                greedy[k] = g
        rel = [None] * len(live)
        if relabel is not None and dump is not None:   # the head's own draft, labelled by the consensus at its state
            todo = [k for k, it in enumerate(live) if it["t"] < it["dur_ms"]]
            rsp = SamplingParams(n=relabel[0], temperature=temperature, top_p=0.95, max_tokens=160,
                                 stop_token_ids=[THINKER_EOS], seed=seed)
            for k, g in zip(todo, llm.generate([prompts[k] for k in todo], rsp, use_tqdm=False)):
                rel[k] = g
        virt = [None] * len(live)
        vmulti = [{} for _ in live]
        if virtual_list and scorer is not None:   # several shorter prefixes at once (Route 1's evidence), one batch
            todo = [(k, d) for d in sorted(set(virtual_list) | ({virtual_ms} if virtual_ms else set()))
                    for k, it in enumerate(live) if it["dur_ms"] > it["t"] > d]
            vp = [{"prompt": prompts[k]["prompt"],
                   "multi_modal_data": {"audio": (live[k]["audio"][: (live[k]["t"] - d) * 16], 16000)}} for k, d in todo]
            for (k, d), g in zip(todo, llm.generate(vp, params, use_tqdm=False) if vp else []):
                vmulti[k][d] = units(g.outputs[0].text, zh)
            if virtual_ms:
                virt = [vm.get(virtual_ms) for vm in vmulti]
        elif virtual_ms and scorer is not None:   # the same read on a shorter prefix: evidence now, not one read later
            todo = [k for k, it in enumerate(live) if it["dur_ms"] > it["t"] > virtual_ms]
            vp = [{"prompt": prompts[k]["prompt"],
                   "multi_modal_data": {"audio": (live[k]["audio"][: (live[k]["t"] - virtual_ms) * 16], 16000)}} for k in todo]
            for k, g in zip(todo, llm.generate(vp, params, use_tqdm=False)):
                virt[k] = units(g.outputs[0].text, zh)
        orc = [None] * len(live)
        if (oracle and dump is not None) or oracle_commit:   # the same prompt with the whole sentence heard: where the future takes the draft
            todo = [k for k, it in enumerate(live) if it["t"] < it["dur_ms"]]
            full = [{"prompt": prompts[k]["prompt"], "multi_modal_data": {"audio": (live[k]["audio"], 16000)}} for k in todo]
            for k, g in zip(todo, llm.generate(full, params, use_tqdm=False)):
                orc[k] = units(g.outputs[0].text, zh)
        for k, (it, res) in enumerate(zip(live, results)):
            cont = res.outputs[0].text
            sep = "" if zh or not it["committed"] or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
            draft = units(join(it["committed"], zh) + sep + cont, zh)
            it["drafts"].append({"t": it["t"], "draft": join(draft, zh)})
            n = len(it["committed"])
            if it["t"] >= it["dur_ms"]:
                new, it["done"] = draft[n:], True
            elif hold is not None:
                new = draft[n: max(n, len(draft) - hold)]
            elif vote is not None:
                cands = []
                for o in res.outputs:
                    sep_o = "" if zh or not it["committed"] or o.text[:1].isspace() or o.text[:1] in ",.;:!?)" else " "
                    cands.append(units(join(it["committed"], zh) + sep_o + o.text, zh)[n:])
                new = ralcp(cands, vote[0], vote[1])
                if greedy[k] is not None:
                    g = greedy[k].outputs[0]
                    gu = units(g.text, zh)
                    cons = 0
                    while cons < min(len(gu), len(new)) and gu[cons] == new[cons]:
                        cons += 1
                    prev = it.get("prev_greedy") or []
                    agree = 0
                    while agree < min(len(gu), len(prev)) and gu[agree] == prev[agree]:
                        agree += 1
                    records.append({"id": it["id"], "t": it["t"], "heard_s": it["t"] / 1000, "n_committed": n,
                                    "draft_units": gu, "cons": cons, "prev_agree": agree, **draft_tokens(g, g.text, zh)})
                    it["prev_greedy"] = gu[len(new):]   # what the next draft continues from
            elif selfwait is not None:   # everything said before the wait marker is committed
                new = draft[n:]
                if fluency and fluency["hyg_min"]:   # no dangling end, no fragment while the talker is idle (si_fluency_rules)
                    from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_rules import hygiene
                    raw_new, idle = new, it["busy"] <= it["t"]
                    new = hygiene(it["committed"], new, zh, idle, fluency["hyg_min"])
                    if fluency.get("hold_ms") and idle and len(new) < len(raw_new) and any(
                            re.search(r"[A-Za-z0-9一-鿿]", u) for u in raw_new[len(new):]):
                        if it["held_t"] is None:   # a rule holds text the draft has: at most hold_ms while the talker is idle
                            it["held_t"] = it["t"]
                        elif it["t"] - it["held_t"] >= fluency["hold_ms"]:
                            new = raw_new
                    if new:
                        it["held_t"] = None
                if sw_seq and new and it["t"] < it["dur_ms"]:
                    it["reads"].append((join(it["committed"], zh), list(new)))
            elif scorer is not None:
                tok = draft_tokens(res.outputs[0], cont, zh)
                gu = units(cont, zh)
                prev = (virt[k] or []) if virtual_ms else (it.get("prev_greedy") or [])
                agree = 0
                while agree < min(len(gu), len(prev)) and gu[agree] == prev[agree]:
                    agree += 1
                if lr is not None:   # Route 1: prior x the joint likelihood of the real and virtual drafts
                    from experiments.uniss_qwen3_omni_v1.evaluation.si_bayes_lr import lr_commit
                    others = [it.get("prev_greedy")] + [vmulti[k].get(d) for d in lr["virtual"]]
                    opening = n == 0 and all(o is None for o in others)
                    tau = scorer[4] if opening and scorer[4] is not None else scorer[1]
                    if queue_rate:   # Route 3: waiting costs only the time the talker would be idle
                        tau = max(tau, min(queue_tau_max, 1 - (1 - tau) * max(0.0, window - max(0, it["busy"] - it["t"])) / window))
                    if adapt is not None:
                        talk = it["id"].rsplit("_", 1)[0]
                        tau = adapt.tau(talk)
                        keep, unit_p, z = lr_commit(scorer[0], lr, tok, gu, others, window, it["t"] / 1000, n, zh, tau,
                                                    to_probs=lambda z, s=talk: adapt.probs(s, z))
                        it["reads"].append((join(it["committed"], zh), gu, z, list(tok["unit_of"]), keep))
                    else:
                        keep, unit_p, _ = lr_commit(scorer[0], lr, tok, gu, others, window, it["t"] / 1000, n, zh, tau)
                else:
                    from experiments.uniss_qwen3_omni_v1.evaluation.si_commit_head import commit_units
                    opening = n == 0 and not prev   # nothing said yet and nothing to compare: the prior alone
                    tau = scorer[4] if opening and scorer[4] is not None else scorer[1]
                    if queue_rate:   # Route 3: waiting costs only the time the talker would be idle
                        tau = max(tau, min(queue_tau_max, 1 - (1 - tau) * max(0.0, window - max(0, it["busy"] - it["t"])) / window))
                    keep, unit_p = commit_units(scorer[0], tok, len(gu), it["t"] / 1000, n, agree, zh, tau, scorer[2],
                                                agree_beta=scorer[3], has_prev=bool(prev), return_probs=True)
                if n == 0 and keep < scorer[5]:
                    keep = 0
                if 0 < keep < scorer[6] and not gu[keep - 1].endswith(tuple(",.;:!?，。；：！？、…")):
                    if scorer[7] > 0:   # value of waiting: hold only if the next read is likely to complete the phrase
                        need = unit_p[keep: scorer[6]]
                        beta = scorer[3] or 0.0
                        lift = 1.0 / (1.0 + np.exp(-(np.log(need / (1 - need + 1e-9) + 1e-9) + beta)))
                        q = float(np.prod(need * (lift >= tau))) if len(need) == scorer[6] - keep else 0.0
                        if q * scorer[7] > keep:
                            keep = 0
                    else:
                        keep = 0
                new = draft[n: n + keep]
                if rel[k] is not None:
                    cands = []
                    for o in rel[k].outputs:
                        sep_o = "" if zh or not it["committed"] or o.text[:1].isspace() or o.text[:1] in ",.;:!?)" else " "
                        cands.append(units(join(it["committed"], zh) + sep_o + o.text, zh)[n:])
                    agreed = ralcp(cands, relabel[0], relabel[1])
                    cons = 0
                    while cons < min(len(gu), len(agreed)) and gu[cons] == agreed[cons]:
                        cons += 1
                    records.append({"id": it["id"], "t": it["t"], "heard_s": it["t"] / 1000, "n_committed": n,
                                    "draft_units": gu, "cons": cons, "prev_agree": agree, "policy_keep": keep, **tok})
                elif orc[k] is not None:
                    records.append({"id": it["id"], "t": it["t"], "dur_ms": it["dur_ms"], "heard_s": it["t"] / 1000,
                                    "n_committed": n, "draft_units": gu, "oracle_units": orc[k], "prev_units": prev,
                                    "prev_agree": agree, "policy_keep": keep, **tok,
                                    **({"virt": {str(d): v for d, v in vmulti[k].items()}, "prev_real": it.get("prev_greedy"),
                                        "window_ms": window} if virtual_list else {})})
                it["prev_greedy"] = gu[keep:]
                it["look"] = join(gu[keep: keep + look_units], zh) if look_units else ""
            elif oracle_commit:   # the upper bound: commit the draft's prefix the whole-sentence continuation keeps
                gu, ou, L = units(cont, zh), orc[k] or [], 0
                while L < min(len(gu), len(ou)) and gu[L] == ou[L]:
                    L += 1
                new = draft[n: n + L]
            else:
                agreed = 0
                if it["prev"] is not None:
                    while agreed < min(len(draft), len(it["prev"])) and draft[agreed] == it["prev"][agreed]:
                        agreed += 1
                new = draft[n: agreed] if agreed > n else []
                it["prev"] = draft
                it["look"] = join(draft[max(n, agreed): max(n, agreed) + look_units], zh) if look_units else ""
            if new or it["done"]:
                text = ("" if zh or not n else " ") + join(new, zh)
                said = len(new)
                if fluency and fluency["clean"]:   # the talker says the block without fillers and repeats; the Thinker keeps them
                    from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_rules import clean, content_units
                    raw, text = text, clean(it["spoken"], text, zh)
                    said = len(units(text, zh))
                    it["spoken"] += text
                    if content_units(units(text, zh), zh) or it["done"]:
                        it["blocks"].append({"write_ms": it["t"], "text": text, "raw": raw, "final": it["done"]})
                else:
                    it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["done"],
                                         **({"look": it.get("look", "")} if look_units and not it["done"] else {})})
                it["committed"] += new
                if sw_cfg and sw_cfg["rate"] and said:
                    it["busy"] = max(it["busy"], it["t"]) + said * sw_cfg["rate"] * 1000
                if queue_rate:
                    it["busy"] = max(it["busy"], it["t"]) + len(new) * queue_rate * 1000
        if sw_seq:   # sentences that ended: label their reads by the whole-sentence continuation, move the talk's bias
            for it in live:
                it["r"] += 1
            fin = [it for it in live if it["done"]]
            todo = [(it, rd) for it in fin for rd in it["reads"]]
            full = [{"prompt": head + rd[0], "multi_modal_data": {"audio": (it["audio"], 16000)}} for it, rd in todo]
            outs = llm.generate(full, sw_last, use_tqdm=False) if full else []
            for (it, (_, said)), o in zip(todo, outs):
                ou, L = units(o.outputs[0].text, zh), 0
                while L < min(len(said), len(ou)) and said[L] == ou[L]:
                    L += 1
                st = sw_talk.setdefault(it["id"].rsplit("_", 1)[0], {"b": 0.0, "reads": 0, "errs": 0, "trace": []})
                err = float(len(said) > L)
                st["reads"] += 1
                st["errs"] += int(err)
                if sw_cfg["adapt"] == "aci":
                    st["b"] = float(np.clip(st["b"] + sw_cfg["gamma"] * (err - sw_cfg["eps"]), -6, 6))
                else:   # Beta posterior mean of the talk's revision rate, against the target
                    n0, e = sw_cfg["n0"], sw_cfg["eps"]
                    m = (n0 * e + st["errs"]) / (n0 + st["reads"])
                    st["b"] = float(sw_cfg["kappa"] * (np.log(m / (1 - m)) - np.log(e / (1 - e))))
            for it in fin:
                st = sw_talk.get(it["id"].rsplit("_", 1)[0])
                if st is not None:
                    st["trace"].append({"b": round(st["b"], 3), "reads": st["reads"], "errs": st["errs"]})
                it["reads"] = []
                if it["id"] in nxt:
                    nxt[it["id"]]["waiting"] = False
        if adapt is not None:   # sentences that ended: label their reads by the oracle, adapt the talk, start the next
            for it in live:
                it["r"] += 1
            fin = [it for it in live if it["done"]]
            todo = [(it, rd) for it in fin for rd in it["reads"]]
            full = [{"prompt": head + rd[0], "multi_modal_data": {"audio": (it["audio"], 16000)}} for it, rd in todo]
            per = {}
            for (it, (_, gu, z, unit_of, keep)), o in zip(todo, llm.generate(full, params, use_tqdm=False) if full else []):
                ou, L = units(o.outputs[0].text, zh), 0
                while L < min(len(gu), len(ou)) and gu[L] == ou[L]:
                    L += 1
                per.setdefault(it["id"], []).append((np.asarray(z, dtype=np.float64),
                                                     (np.asarray(unit_of) < L).astype(np.float64), keep > L))
            for it in fin:
                adapt.update(it["id"].rsplit("_", 1)[0], per.get(it["id"], []))
                it["reads"] = []
                if it["id"] in nxt:
                    nxt[it["id"]]["waiting"] = False
        r += 1
        print(f"{tag} round {r}: {len(live)} sentences, {time.time() - t0:.0f}s", flush=True)
    if dump is not None:   # second label, once each translation is finished: the draft agrees with it
        final = {it["id"]: it["committed"] for it in items}
        dump.parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w", encoding="utf-8") as fh:
            for rec in records:
                f, n0, ok = final[rec["id"]], rec["n_committed"], 0
                while ok < len(rec["draft_units"]) and n0 + ok < len(f) and f[n0 + ok] == rec["draft_units"][ok]:
                    ok += 1
                rec["final_ok"] = ok
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:   # an empty last commit: the one before is the last
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fh.write(json.dumps({"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks,
                                 **({"spoken": it["spoken"]} if fluency and fluency["clean"] else {}),
                                 "drafts": it["drafts"]}, ensure_ascii=False) + "\n")
    if adapt is not None:
        out.with_suffix(".adapt.json").write_text(json.dumps(adapt.summary()), encoding="utf-8")
    if sw_seq:
        out.with_suffix(".adapt.json").write_text(json.dumps({t: {**v, "err_rate": round(v["errs"] / max(v["reads"], 1), 4)}
                                                              for t, v in sw_talk.items()}), encoding="utf-8")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
