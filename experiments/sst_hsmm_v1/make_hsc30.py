"""change 67 / 68: build hsc30_si_thinker.py from hsc22_si_thinker.py (not modified) by exact, asserted replacements.
    python3 experiments/sst_hsmm_v1/make_hsc30.py        (refuses to overwrite hsc30)"""
from pathlib import Path

D = Path(__file__).resolve().parent
src = (D / "hsc22_si_thinker.py").read_text(encoding="utf-8")
dst = D / "hsc30_si_thinker.py"
if dst.exists():
    raise SystemExit(f"refuse to overwrite {dst}")


def rep(old, new, s):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


s = src
s = rep('"""portfolio 2026-11-27 (change 55): one Thinker script',
        '"""sst_hsmm_v1 change 67 / 68: a copy of hsc22_si_thinker.py (not modified) with\n'
        '--taf K (change 67, TAF-S): at every NON-FINAL read the same Thinker transcribes the audio prefix ("Transcribe this speech."),\n'
        'continues the transcript as raw text K times (T --taf-temp, top-p 0.95, <= 24 tokens, stop at . ? ! 。 ？ ！), translates every\n'
        '"transcript + continuation" with a text-translation prompt under the same healed committed prefix (pp), and the commit cut is\n'
        'capped at max(n, vote), vote = the ceil(K/2)-th largest unit LCP between a future translation and the draft d0 (--taf-shadow:\n'
        'record rec["taf"] = [cut, cap, n_valid, n] only). --ctx N (change 68): the instruction of every prompt of an item carries the\n'
        'previous N final outputs of its session (id without the last _<n>, ordered by n, chunks of <= 60 items); an item starts after\n'
        'its predecessor finished, with its own clock (first read at its own 1000 ms). Per-round wall times -> combo json round_s.\n'
        'Original header follows.\n'
        'portfolio 2026-11-27 (change 55): one Thinker script', s)
s = rep('    ap.add_argument("--draft-lora", default="", help="change 48: PEFT dir served for non-final drafter prompts")\n',
        '    ap.add_argument("--draft-lora", default="", help="change 48: PEFT dir served for non-final drafter prompts")\n'
        '    ap.add_argument("--taf", type=int, default=0, help="change 67: K sampled source futures per non-final read, majority veto")\n'
        '    ap.add_argument("--taf-shadow", action="store_true", help="change 67 P1: compute the TAF veto, do not apply it")\n'
        '    ap.add_argument("--taf-temp", type=float, default=0.9)\n'
        '    ap.add_argument("--ctx", type=int, default=0, help="change 68: previous N final outputs of the same session in the prompt")\n', s)
s = rep('    assert not (args.bias_redraft > 0 and args.mid_mbr > 0), "A1 and mid-MBR are separate methods"\n',
        '    assert not (args.draft_tag and args.ctx), "change 68: --ctx with the plain head only"\n'
        '    if args.taf > 0:                                                 # change 67 TAF-S\n'
        '        conv_t = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": "Transcribe this speech."}]}]\n'
        '        HASR_T = processor.apply_chat_template([conv_t], add_generation_prompt=True, tokenize=False)[0]\n'
        '        TMT = ("Translate the following English text into Chinese. Reply with the Chinese translation only.\\n\\n" if zh else\n'
        '               "Translate the following Chinese text into English. Reply with the English translation only.\\n\\n")\n'
        '\n'
        '        def tmt_head(src_):\n'
        '            c_ = [{"role": "user", "content": [{"type": "text", "text": TMT + src_}]}]\n'
        '            return processor.apply_chat_template([c_], add_generation_prompt=True, tokenize=False)[0]\n'
        '        TAF_STOP = [".", "?", "!", "\u3002", "\uff1f", "\uff01"]\n'
        '    CTXH = {}\n'
        '\n'
        '    def head_of(it):                                                 # change 68: session-context head (cached per context)\n'
        '        c_ = it.get("ctx") or ""\n'
        '        if not args.ctx or not c_:\n'
        '            return head\n'
        '        if c_ not in CTXH:\n'
        '            cv_ = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": instruction\n'
        '                    + "\\nFor reference, the translation of the preceding speech in this conversation: " + c_}]}]\n'
        '            CTXH[c_] = processor.apply_chat_template([cv_], add_generation_prompt=True, tokenize=False)[0]\n'
        '        return CTXH[c_]\n'
        '    assert not (args.bias_redraft > 0 and args.mid_mbr > 0), "A1 and mid-MBR are separate methods"\n', s)
s = rep('        if not args.draft_tag:\n            return head\n        return head_p',
        '        if not args.draft_tag:\n            return head_of(it)\n        return head_p', s)
s = rep('        PS = [head + join(hb(it), zh) for it in live]',
        '        PS = [head_of(it) + join(hb(it), zh) for it in live]', s)
s = rep('        it.update(committed=[], prev=None, blocks=[], drafts=[], done=False, hist=[], H={}, qend=0.0, fork=None, sample_now=False, heal_k=None)\n',
        '        it.update(committed=[], prev=None, blocks=[], drafts=[], done=False, hist=[], H={}, qend=0.0, fork=None, sample_now=False, heal_k=None,\n'
        '                  start_r=0, ctx="", sess_next=None, ctx_hist=[])\n', s)
s = rep('    t0, r = time.time(), 0\n',
        '    if args.ctx:                                                     # change 68: sessions, chunks of <= 60, sequential start\n'
        '        import re as _re3\n'
        '        grp_ = {}\n'
        '        for it in items:\n'
        '            m_ = _re3.match(r"^(.*)_(\\d+)$", it["id"])\n'
        '            key_, num_ = (m_.group(1), int(m_.group(2))) if m_ else (it["id"], 0)\n'
        '            grp_.setdefault((it["v"], key_), []).append((num_, it["id"], it))\n'
        '        for L_ in grp_.values():\n'
        '            L_.sort(key=lambda x_: (x_[0], x_[1]))\n'
        '            for c0_ in range(0, len(L_), 60):\n'
        '                ch_ = [x_[2] for x_ in L_[c0_:c0_ + 60]]\n'
        '                for j_, it in enumerate(ch_):\n'
        '                    it["start_r"] = 0 if j_ == 0 else None\n'
        '                    it["sess_next"] = ch_[j_ + 1] if j_ + 1 < len(ch_) else None\n'
        '        print("ctx sessions:", len(grp_), "chunks start:", sum(1 for it in items if it["start_r"] == 0), flush=True)\n'
        '    t0, r = time.time(), 0\n'
        '    RT, RTN, tr0_ = [], [], time.time()\n', s)
s = rep('        live = [it for it in items + forks if not it["done"]]\n        for it in live:\n            it["t"] = min((r + 1) * window, it["dur_ms"])\n',
        '        live = [it for it in items + forks if not it["done"] and it.get("start_r", 0) is not None and it.get("start_r", 0) <= r]\n'
        '        for it in live:\n            it["t"] = min((r - it.get("start_r", 0) + 1) * window, it["dur_ms"])\n', s)
s = rep('                        gk.append(k)\n                        owners.append(it)\n',
        '                        gk.append(k)\n                        owners.append(it)\n'
        '            tk = []\n'
        '            if args.taf > 0:                                          # change 67: source transcript for the TAF futures\n'
        '                for k, it in enumerate(live):\n'
        '                    if it["t"] < it["dur_ms"]:\n'
        '                        prompts.append({"prompt": HASR_T, "multi_modal_data": {"audio": AUD[k]}})\n'
        '                        tk.append(k)\n'
        '                        owners.append(it)\n', s)
s = rep('            sps += [params] * len(gk)\n', '            sps += [params] * len(gk)\n            sps += [params] * len(tk)\n', s)
s = rep(' + [None] * len(gk))\n', ' + [None] * len(gk) + [None] * len(tk))\n', s)
s = rep('            resg = {k: results[o_ + j] for j, k in enumerate(gk)}; o_ += len(gk)\n',
        '            resg = {k: results[o_ + j] for j, k in enumerate(gk)}; o_ += len(gk)\n'
        '            rest = {k: results[o_ + j] for j, k in enumerate(tk)}; o_ += len(tk)\n', s)
s = rep('                LPP[k_] = SD.unit_lp(tok, units, join, hb(it_), SEL[k_].text, fake_, zh)[1]\n        for k, (it, res) in enumerate(zip(live, results[: len(live)])):\n',
        '                LPP[k_] = SD.unit_lp(tok, units, join, hb(it_), SEL[k_].text, fake_, zh)[1]\n'
        '        TAFF, TAFTR, TAFSRC = {}, {}, {}\n'
        '        if args.taf > 0 and rest:                                     # change 67: K source futures -> K text translations\n'
        '            cps_, cks_, csp_ = [], [], []\n'
        '            for k_, rr_ in rest.items():\n'
        '                tr_ = rr_.outputs[0].text.strip().rstrip(".?!\u3002\uff1f\uff01\u2026,\uff0c\u3001;\uff1b:\uff1a\\"\'\u201d\u2019 ")\n'
        '                if not tr_:\n'
        '                    continue\n'
        '                TAFTR[k_] = tr_\n'
        '                cps_.append({"prompt": tr_})\n'
        '                cks_.append(k_)\n'
        '                csp_.append(SamplingParams(n=args.taf, temperature=args.taf_temp, top_p=0.95, max_tokens=24, stop=TAF_STOP,\n'
        '                                           include_stop_str_in_output=True,\n'
        '                                           seed=int(hashlib.sha1(f"taf|{live[k_][\'id\']}|{live[k_][\'t\']}".encode()).hexdigest(), 16) % (2 ** 31)))\n'
        '            if cps_:\n'
        '                cres_ = llm.generate(cps_, csp_, use_tqdm=False)\n'
        '                tps_, tks_, tsp_ = [], [], []\n'
        '                for k_, cr_ in zip(cks_, cres_):\n'
        '                    it_ = live[k_]\n'
        '                    for oc_ in cr_.outputs:\n'
        '                        src_ = TAFTR[k_] + oc_.text\n'
        '                        TAFSRC.setdefault(k_, []).append(src_)\n'
        '                        tps_.append({"prompt": tmt_head(src_) + join(hb(it_), zh)})\n'
        '                        tks_.append(k_)\n'
        '                        tsp_.append(pp(it_))\n'
        '                tres_ = llm.generate(tps_, tsp_, use_tqdm=False)\n'
        '                for k_, tr2_ in zip(tks_, tres_):\n'
        '                    TAFF.setdefault(k_, []).append(tr2_.outputs[0])\n'
        '        for k, (it, res) in enumerate(zip(live, results[: len(live)])):\n', s)
s = rep('                    rec.update(sv0=sv0)\n',
        '                    if args.taf > 0 and args.mode != "rollout" and k in rest:      # change 67: future-vote veto\n'
        '                        Ls_ = []\n'
        '                        for of_ in TAFF.get(k, []):\n'
        '                            cf_, okf_ = strip_(it, of_.text)\n'
        '                            if not okf_:\n'
        '                                continue\n'
        '                            sepf_ = "" if zh or not it["committed"] or cf_[:1].isspace() or cf_[:1] in ",.;:!?)" else " "\n'
        '                            Ls_.append(lcp(draft, units(join(it["committed"], zh) + sepf_ + cf_, zh)))\n'
        '                        need_ = (args.taf + 1) // 2\n'
        '                        vote_ = sorted(Ls_, reverse=True)[need_ - 1] if len(Ls_) >= need_ else n\n'
        '                        capt_ = max(n, vote_)\n'
        '                        rec["taf"] = [int(cut), int(capt_), len(Ls_), int(n)]\n'
        '                        rec["taf_tr"] = TAFTR.get(k, "")[-80:]\n'
        '                        rec["taf_src"] = [x_[-80:] for x_ in TAFSRC.get(k, [])[:2]]\n'
        '                        st["taf_reads"] += 1\n'
        '                        st["taf_capped"] += int(cut > capt_)\n'
        '                        st["taf_units_removed"] += max(0, cut - capt_)\n'
        '                        st["taf_units_hsmm"] += max(0, cut - n)\n'
        '                        if not args.taf_shadow:\n'
        '                            cut = min(cut, capt_)\n'
        '                    rec.update(sv0=sv0)\n', s)
s = rep('                if new:\n                    it["qend"] = max(it["qend"], it["t"] / 1000.0) + len(new) * UNIT_S\n        r += 1\n',
        '                if new:\n                    it["qend"] = max(it["qend"], it["t"] / 1000.0) + len(new) * UNIT_S\n'
        '            if args.ctx and it["done"] and it.get("sess_next") is not None:     # change 68: start the next item of the session\n'
        '                nx_ = it["sess_next"]\n'
        '                nx_["ctx_hist"] = it["ctx_hist"] + [join(it["committed"], zh).strip()]\n'
        '                nx_["ctx"] = ("" if zh else " ").join(h_ for h_ in nx_["ctx_hist"][-args.ctx:] if h_)\n'
        '                nx_["start_r"] = r + 1\n'
        '                st["ctx_started"] += 1\n'
        '        RT.append(round(time.time() - tr0_, 3)); RTN.append(len(live)); tr0_ = time.time()\n'
        '        r += 1\n', s)
s = rep('            if args.assign:\n                row["alpha"] = alpha_of[it["id"]]\n',
        '            if args.assign:\n                row["alpha"] = alpha_of[it["id"]]\n'
        '            if args.ctx:\n                row["ctx"] = it.get("ctx", "")\n', s)
s = rep('    st = dict(st)\n',
        '    st = dict(st)\n    st.update(round_s=RT, round_n=RTN, hsc30=dict(taf=args.taf, taf_shadow=args.taf_shadow, taf_temp=args.taf_temp, ctx=args.ctx))\n', s)
with open(dst, "x", encoding="utf-8") as fh:
    fh.write(s)
print("wrote", dst, len(s.splitlines()), "lines")
