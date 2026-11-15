# CHANGELOG v10 (paper/gstar_si_v10, from gstar_si_v9)

Written 2026-11-14 (Prague night), by the paper agent, following `docs/mist_v10_plan.zh-CN.md` §4. v9 is untouched.

Build:

```bash
cd paper/gstar_si_v10
/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python scripts/make_tables_v10.py   # needs numpy (system python3 has none)
/opt/dlami/nvme/jasonleeeli/tools/tectonic/tectonic --keep-logs main.tex
```

State of this draft: builds without errors, no undefined references or citations, no overfull boxes, no undefined
`\num` keys (main.tex now prints `??` and a log warning for an undefined key instead of nothing). Main content (abstract
to the end of the Conclusion) is exactly 8 pages; Limitations start at the top of page 9; 33 pages in total.

## 1. What changed, per file

| File | Change |
|---|---|
| `main.tex` | New title (user's); `\input` of `macros_v9` (v9 defect: `v9:` macros printed nothing) and `macros_v10`; `\bias` = `\delta_{\mathrm{w}}`; `\num` prints `\textbf{??}` + warning for unknown keys; appendix numbering beyond Z (AA, AB, ...); `appendix_v10` first. |
| `sections/abstract.tex` | Rewritten (plan §4.2): task, v9 attribution results (v7/v8 macros), the two latent-structure modules + write bias, test result sentence for the two AMI directions (v10 macros), AliMeeting sentence behind `\ifAliwin`, consecutive-interpretation caveat. |
| `sections/introduction.tex` | Compressed v9 paragraphs; new motivation paragraph (deadline decoder commits only `v10:dev:z:b:pre`% of Zh→En words before the utterance ends; speech LAAL of the baseline); four contributions (plan §2); result sentences with v10 macros; Ali sentence behind `\ifAliwin`. Removed the v9 claim "text lag does not grow with overlap" from the intro. |
| `sections/related.tex` | New paragraph "Latent timing" (MILk, HMT, posterior regularisation, test-time thresholds incl. SeamlessStreaming). The capability table (`tab:related8`) moved to Appendix "Relation to Prior Work" (space); it is now defined (v9 check item). |
| `sections/method.tex` | v9 parts compressed (Task + two streams + fusion in one subsection; projector, slot map, deadline fraction and read-target equations moved to new Appendix H "Model Details"); new subsections *Latent write alignment* (lattice Eq. 8, CRF posterior Eq. 9, M-step Eq. 10, write bias Eq. 11) and *Speaking on the playback timeline* (backlog Eq. 14, potential Eq. 15, PR budget, PL-EM training, inference); online windows: 0.6 s close, chunk protocol, read-level merging; latencies stated as computation-unaware. Notation: chunk index j with due read r(j) (v9 Eq. due); stream blocks i with release time ν_i, backlog h_i, words W_i, durations D(c) (avoids v9's a_n/b_n/β). |
| `figures/arch.tex` | Talker drawn as trained (orange); equation references for the encoder/diarizer/projector point to §3.1 / App. H; loop label mentions Eq. bias and §3.4; caption updated. |
| `sections/setup.tex` | Rewritten compactly: joint two-stage training, the 3500-vs-4000 deviation in one sentence, strict baseline / baseline + PL-EM / optional SeamlessStreaming (`\ifEseven`), v9 baselines kept "as run with the v9 system and protocol", new latency definitions (speech LAAL primary, EndOffset, text LAAL), statistics for v10. |
| `sections/results.tex` | Offline (Table 1, now a compact single-column version of `tab_v7_main`) and overlap sections kept (Figure 2 → appendix); v9 online/latency subsections reduced to one sentence, full text moved to Appendix F; Real Mandarin compressed; new §5.4 *Learning when to write and when to speak* (`tab_v10_main`, `tab_v10_ablation`, E2 split, plan B, PMC, AliMeeting paragraph behind `\ifAliwin`, disclosure that the bias came after the first test run); new §5.5 *Why latent alignment writes late* (posterior drift from the training logs, pre-end share, start offset, utterance-level decomposition). |
| `sections/conclusion.tex` | New conclusion (RL latency reward as one-sentence future work); Limitations rewritten with plan §4.9 items (bias introduced after first test run; 3500 override with +0.137 s; one ρ; LAAL 3–5 s / consecutive; compute not measured; point-estimate latencies; one seed; no human eval / COMET / MetricX; E7 not finished (auto-hidden once E7 exists); Ali text loss behind `\ifAliwin`). |
| `sections/appendix_v10.tex` (new) | A v10 training and checkpoint selection (rule, deviation 3500 vs 4000, Talker selection, other deviations) + `tab_v10_s2sel`; B latent write alignment details (scoring, exactness, plan B objective) + `tab_v10_drift`, B.1 PL-EM (candidates, f, forward/backward, weights, objective, data, λ*) + `tab_v10_talker`; C write-bias calibration and the δ=0 run (`tab_v10_calib`, `tab_v10_delta0`); D latency protocol (chunk protocol, speech LAAL formula, `tab_v10_m2laal`), D.1 delay decomposition (`tab_v10_decomp`); E further results: E.1 AliMeeting BP/duplicates (`tab_v10_dup`), E.2 significance (`tab_v10_sig`), E.3 SeamlessStreaming slot (`\ifEseven`); F v9 system under the v9 protocol (moved v9 §5.3/§5.4 text, `tab_v8_latency`, full `tab_v7_main`, speaker-blind Simul control); G AliMeeting human references (v9) (fixes the undefined `app:v9`) + `tab_v9_reg`. |
| `sections/appendix.tex` | New first section H "Model Details" (equations moved out of the method, labels `eq:enc`, `eq:diar`, `eq:interp`, `eq:conv`, `eq:mlp`, `eq:ar`, `eq:slotmap`, `eq:frac`, `eq:readtarget`, single-channel layout); capability table `tab:related8` moved into "Relation to Prior Work"; `tab:related7` gains rows for latent read/write models and speech timing. |
| `sections/appendix_v8.tex` | Now also holds Figure `fig:overlap8`. |
| `references.bib` | + `ganchev2010posterior` (JMLR 11:2001–2049), `zhang2023hmt` (ICLR 2023), `arivazhagan2019milk` (ACL 2019; page range 1313–1323 marked TODO-verify in a bib comment). |
| `scripts/make_tables_v10.py` (new) | Reads every source listed in the task; never fails on a missing file; writes `generated/macros_v10.tex` (1,389 macros), `generated/tab_v10_*.tex`, `generated/tab_v7_main_{compact,full}.tex`, `generated/v10_tests.json`, `generated/v10_missing.json`; copies the block-start LAAL JSONs into `generated/v10_inputs/` (never overwrites). Rounds half-up as the reports do. |
| `generated/v10_inputs/` (new) | 24 `*_online.json` copied from the scratchpad m2laal run. |
| `.gitignore` | + `.git_copied_from_v9/` (see decision 2). |

`sections/analysis.tex` is still not `\input` (as in v9); its one new point (utterance-level near-consecutive
decomposition) is in §5.5 and Appendix D.1.

## 2. Decisions taken (best option, user asleep)

1. **Title**: the user's title, verbatim.
2. **Nested git repo**: `cp -r gstar_si_v9` also copied v9's own `.git` (each paper version is a local clone of the previous one), so the first outer commit (00fccdea4) recorded only a gitlink. I renamed the copy to `paper/gstar_si_v10/.git_copied_from_v9` (nothing deleted; v9's `.git` untouched), ignored it, removed the gitlink from the index only (`git update-index --force-remove`) and committed the files (f824e8178). To restore the per-version repo convention later, rename it back.
3. **8-page budget**: the draft was 10.7 pages before compression. Moved to the appendix instead of dropping: capability table, overlap-bin figure, simultaneous rows of the offline table (full table in App. F), v9 online/latency/commit-policy text and table, v9 projector/slot-map/deadline equations. Shortened intro, related work, setup, results prose. No result was removed from the paper.
4. **Statistics (E3)**: implemented in `make_tables_v10.py` rather than read from `gate_*.json`: exact two-sided sign-flip over all 2^M meeting flips + 10k paired meeting bootstrap, speech pooling the two TTS seeds per meeting (as gate_2seed), Holm over 12 test comparisons (3 sets × {text M−B, speech M−B, speech E2−B, speech M−E2}). Checked against gate_2seed on the δ=0 run: AMI speech +2.41, interval [1.53, 3.31] here vs [1.56, 3.30] there. Text E2−B is excluded (identical text). Latency has no per-meeting data in the lat files, so latency differences are point estimates (stated in the paper).
5. **E1 / E2 inputs**: read from the per-run score / latency files (the same files `wb_test.sh` and `e2_baseline_plem.sh` summarise), so each number appears as soon as its file exists; `wb_test_summary_v1.json` is not needed. AMI Zh→En and En→Zh E1 *text* numbers are already in (25.34 / 41.74 tcpBLEU).
6. **AliMeeting claim switch**: `\ifAliwin` is set by the script: true only if E1's Ali text tcpBLEU ≥ the strict baseline's. While E1 is pending it is false → narrow (AMI-only) wording. The two wordings sit in the abstract, intro, results (AliMeeting paragraph), conclusion and limitations.
7. **E7 slot**: `\ifEseven` is set when `eval_outputs/mist_w3po_v1/decode/jtest_<set>_seamless_default/score_text.json` exists; it adds rows to Table 3, a baseline sentence in §4 and App. E.3 text, and removes the "not finished" Limitations sentence. Latency for E7 is looked up in `crf/t_eval/jtest_<set>_seamless_default*/*.json` or `eval_outputs/mist_w3po_v1/e7_seamless/**` by field name (`unit.speech_laal_online` etc.); check the row once E7 lands, because the other agent's latency file names are not known yet.
8. **Drift statistics**: computed from the training logs. The plan's "≈8 deferred chunks per window" is per training step (≈9.6 windows); per window it is 0.86 (v2′) vs 0.43 (ρ = 0.5). The paper reports per-window values and the first-100 / last-500-step trend.
9. **"Reads with output" (45.9 → 36.2 %)**: kept as cited constants in the script but not used in the text; the paper uses the data-derived share of words committed before the source unit ends (`pre_end_share` of the score files) and the speech start offset instead.
10. **Single-speaker-style LAAL**: only δ = 0 test runs were rescored with that convention; the appendix table says so.
11. **PL-EM inference** described as in the runs: two samples per block, queue-dependent rate, better CTC fidelity kept (c1+b1+s2 config).
12. **Latency is computation-unaware** in v10 (audio and playback time only); stated in §3.7 and Limitations.
13. **E4 (compute cost)** and **E5 (DiariST SAgBLEU)** were not run: E4 is a Limitation; no `app:cost` section was written; E5 is not mentioned.
14. **Bias wording**: "the decision to introduce a bias was influenced by the first test run; its value was not" (App. C), plus the δ = 0 test table, per plan §3.6.

## 3. Pending numbers (placeholders `??`) and their sources

Re-run `scripts/make_tables_v10.py` and tectonic when these exist; `generated/v10_missing.json` lists every pending key with its source file.

| Pending | Macros (examples) | File(s) that fill them |
|---|---|---|
| **E1** text, AliMeeting | `v10:test:ali:m:{all,ov,1spk,tlaal}`, `v10:sig:test:ali:text:m-b:*`, `v10:test:ali:m-b:all`, `\ifAliwin` | `eval_outputs/mist_w3po_v1/decode/jtest_mali_joint_v2p_20261113T120415Z_s03500_wbzm1em1/score_text.json`, `crf/t_eval/<same>/textlat_chunk_<same>.json` |
| **E1** speech, all 3 test sets | `v10:test:{ami,ali,en}:m:{asr,asr_ov,slaal,endoff,sstart,p_*,load}`, `v10:test:*:m-b:{asr,slaal,endoff}`, `v10:sig:test:*:speech:m-b:*`, abstract/intro ranges | `decode/jtest_m{ami,ali,en}_joint_v2p_20261113T120415Z_s03500_wbzm1em1__rlplem{,_tts2}/score_asr.json`, `crf/t_eval/jtest_m<set>_..._wbzm1em1__rlplem/lat_chunk_<name>{,_tts2}.json` |
| **E2** dev | `v10:dev:{z,e}:e2:*`, `v10:dev:*:m0-e2:asr`, `v10:dev:z:e2:p_queue_wait` | `decode/jdev_b{z,e}_joint_v8h6k_20261113T073658Z_s06000__rlplem{,_tts2}/score_asr.json`, `crf/t_eval/jdev_b{z,e}_..._s06000__rlplem/lat_chunk_*.json` |
| **E2** test | `v10:test:*:e2:*`, `v10:test:*:e2-b:*`, `v10:test:*:m-e2:*`, `v10:sig:test:*:speech:{e2-b,m-e2}:*` | `decode/jtest_b{ami,ali,en}_joint_v8h6k_20261113T073658Z_s06000__rlplem{,_tts2}/score_asr.json`, `crf/t_eval/jtest_b<set>_..._s06000__rlplem/lat_chunk_*.json` |
| **E7** (optional) | `v10:test:*:e7:*`, `\ifEseven` | `decode/jtest_{ami,ali,en}_seamless_default/{score_text,score_asr}.json`; latency JSON (see decision 7); report `reports/mist_w3po_v1/e7_seamless_multispk_results.zh-CN.md` |

33 of the 403 placeholder macros are used in running text; the rest are table cells.

## 4. To check after the numbers land

- Sentences whose wording assumes the dev direction (verify against the filled numbers): §5.4 "The speech layer mainly removes queueing" (needs `v10:dev:z:e2:p_queue_wait` ≪ baseline); Limitations "Speech LAAL is still three to five seconds"; the AliMeeting paragraph (the `\ifAliwin` false branch explains the δ = 0 run; check Table 12 for δ = −1 before keeping "duplicates" as the explanation in the abstract).
- If E7 rows and the E7 setup sentence push the Conclusion past page 8, move the "Where the delay remains" paragraph of §5.5 to Appendix D.1 (it is duplicated there in substance).
- `arivazhagan2019milk` page range (bib comment).
- Optional: an Overleaf zip under a new file name (not done).

## 5. 2026-11-15 (Prague 01:45 CET): prose made consistent with the final E1 / E2 / E7 numbers

All experiments finished; `generated/v10_missing.json` is `{}`; switches `\Eseventrue`, `\Eonedonetrue`,
`\Etwodonetrue`, `\Aliwinfalse`. No number was typed by hand: every number in the text is a macro from
`scripts/make_tables_v10.py` (no new macro was needed; existing `v10:sig:test:*`, `v10:test:*:{m,b,e2,e7,m-b,m-e2,e2-b}:*`,
`v10:test:ali:*:{nobp,bp,lenratio,dup}` cover every sentence).

Substantive prose changes:

1. **Credit split (E2) made explicit everywhere.** Abstract, introduction, §5.4 and the conclusion now say that the
   speech-layer (read-level merging + PL-EM, "playback alignment") brings the speech ASR-BLEU gain and the whole
   speech-latency reduction, and that the latent-write-aligned Thinker ("write alignment") contributes the text tcpBLEU
   gain only: on top of E2 it changes ASR-BLEU by +0.10 / +0.11 / −0.24 (intervals include zero) and is heard 0.08 /
   0.10 / 0.26 s later. The two are presented as complementary (write alignment: what/when text is written; playback
   alignment: when it is heard). The old intro sentence "latent alignment changes speech ASR-BLEU ..." (which credited
   both to "latent alignment") was replaced. §5.4 paragraph renamed "Which part brings what?". Limitations: new sentence
   that the write-aligned Thinker costs speech latency and that its text gain is small and not significant after Holm.
2. **Significance stated honestly.** §5.4 now says the AMI text gains (+0.32 / +0.58) have bootstrap intervals that
   exclude zero but are not significant after Holm (p_Holm 0.403 / 0.080); speech gains are significant in Zh→En
   (<0.001) and borderline in En→Zh (0.053). The abstract states the text and speech gains as numbers, not "wins".
3. **AliMeeting (`\Aliwinfalse` branch) rewritten for δ = −1.** The old explanation (baseline duplicates inflating its
   length) held for δ = 0 only; with δ = −1 the system duplicates as many units as the baseline (19.2 vs 18.1 %), its
   precision part is equal (BLEU/BP 21.42 vs 21.43) and the −0.36 gap is a brevity penalty (BP 0.984). Abstract,
   §5.4, Limitations and App. E.1 now say this; AliMeeting is framed as cross-corpus generalisation (never trained or
   tuned on), text claims restricted to AMI, speech ASR-BLEU (+1.11) and speech LAAL (−0.37 s) still improve but less
   than E2.
4. **δ disclosure** kept and tightened in §5.4 (bias introduced after the first δ = 0 test decoding, value set on dev
   only, each test set decoded once with it; Table 9 gives δ = 0), Limitations and App. C unchanged in substance.
5. **E7 SeamlessStreaming** described as a zero-shot, single-speaker external reference with no speaker attribution
   of its own (setup, §5.4, Table 3 footnote, App. E.3). Its speech LAAL is lower only on AMI Zh→En (3.29 vs 3.69 s);
   it is about equal on AliMeeting (3.96 vs 3.91) and higher on En→Zh (3.19 vs 3.05), so the text no longer implies it
   is faster overall, and says LAAL at less than half our ASR-BLEU does not place it on our trade-off. Note: the E7
   report's "语音 LAAL 比我们低 0.0–0.5 s" is not true for En→Zh (3.19 > 3.05).
6. **Table 3 (`make_tables_v10.py`, formatting only):** bold is now "best of our three systems per set and column"
   (was: every MIST v10 cell bold, which bolded worse latency values); E2 is thus bold on speech LAAL / EndOffset.
   Caption shortened; † footnote rewritten (zero-shot, no own speaker attribution, deterministic single run instead of
   "one TTS draw"). Table 4 caption points to App. C only.
7. **Page budget.** The draft with these changes was one column over. Moved to the appendix (nothing dropped):
   §5.5 "Why latent alignment writes late" → merged into the §5.4 paragraph "Ablation and the write bias" (label
   `sec:res-drift` kept there) + full drift paragraph in App. B ("Drift towards later writes"); "Where the delay
   remains" → App. D.1 (adds the last-read share); E-step budget and early-write lattice numbers → App. C ("Other ways
   to control the write time"); §5.3 compressed (relabelling-overall sentence → App. G, v6 synthesised-speech
   sentence → App. F); the v9 RTF numbers in §5.2 → pointer to App. F (already there). Shorter conclusion, intro
   contribution list, related-work threshold sentence; method's "latencies count computation" sentence removed (it is
   in Limitations and setup).
8. Title line break moved after the colon (the forced break produced a three-line title). Title text unchanged.

State: builds without errors, no undefined references/citations/number keys, no overfull boxes; main body (abstract
to the end of the Conclusion) is exactly 8 pages (Conclusion ends at the bottom of page 8, Limitations start at the top
of page 9); 33 pages in total. Still open: `arivazhagan2019milk` page range (bib TODO comment); the inactive
`\ifAliwin`-true / `\ifEseven`-false branches are kept in the source for robustness.

## 6. 2026-11-15: revision after the strict review (`review_v10.md`)

Author decisions applied: all [TEXT] fixes; no [EXPERIMENT] item run (test hygiene; baseline + δw control declined);
M2/M5 analyses (block-start speech LAAL for B/E2/M/M0 on test, latency CIs, speech rate, UTMOS) are computed by another
agent and only get a guarded slot here; no COMET/MetricX anywhere. No number was typed into the tex; every new number is
a macro (new: `v10:tk:{ami,ali}:c-a:{asr,slaal}`; `crf:pass1/2` now printed as whole rows). Build state: no errors, no
undefined references/citations/number keys, no overfull boxes; **main body (abstract to end of Conclusion) = exactly 8
pages, Limitations start at the top of page 9**; 36 pages in total.

**Title** (M1/M11): "Simultaneous" → "Streaming": *Who Said What, and When to Say It: Latent Write and Playback
Alignment for Speaker-Attributed Streaming Meeting Interpretation*. The method names stay (they name the components, not
a claimed gain).

**System names** (M10): `MIST v10` → MIST; strict baseline → deadline baseline (B); baseline + PL-EM → B+PA (playback
alignment); `PL-EM` → PA Talker / "one round of posterior-weighted fine-tuning"; `T1c` → base Talker; `SerialCRF v2′` →
write alignment (WA); "v9 MIST-Simul" → the per-direction streaming model MIST-Simul; "v9 protocol" → look-ahead
protocol; `ρ` (E-step budget ratio) → `ξ`; "plan B" → training-time / E-step latency budget.

| Item | What was done / why not |
|---|---|
| M1 | Abstract, intro, results, conclusion rewritten: each number carries its status (offline 6.52 = exploratory, 2.17 = confirmatory Holm; overlap 3.58/2.82 exploratory); text +0.32/+0.58 stated as **not significant after Holm**; "complementary" dropped everywhere; plain statement that write alignment yields no significant gain, no speech gain, slightly later speech, and that the speech gains of MIST over B are those of the playback layer; decision-rule difference stated (offline rule incl. decoder-noise widening vs. streaming tests without it, "which can only turn more differences into ties"); text gains compared with the per-direction seed spread (macros `v8:seed:*`); a possible quality–latency shift named. Table 2 bold replaced by significance marks. |
| M2 | [TEXT] PL-EM described as one round of posterior-weighted fine-tuning on sampled candidates conditioned on the rule path's backlog, not iterated EM, no CRF at inference (method §3.6 + App. C.1); abstract no longer says "exact EM" for playback. The four-way confound of B→B+PA, the Table 6 evidence that a rule layer gets most of the gain (new macros), and the CTC-selection/ASR-scoring circularity are stated in §5.3. [ANALYSIS] (UTMOS, rate, similarity) and [EXPERIMENT] (component ladder, listening test) not done → Limitations; PostReview slot (below). |
| M3 | [EXPERIMENT] not run (author decision). §5.3 says MIST writes later on every set, so the text difference may be a quality–latency shift; Limitations lists the missing δw-sweep-on-baseline, compute-matched and lag-mixed controls. Drift framed as the known effect of likelihood-trained policies (related work + §5.3). |
| M4 | New §5.4 "Test-set hygiene" (main text) + Limitations: δw added after a first test run (and what it repaired); 3,500 vs 4,000 with the J margin within decoder noise, the +0.137 s excess over the rule's bound, the bound being relative to a later-writing system, 4,000 never decoded on test; reuse of the same 16+20 test meetings while closing gap / latency protocol / merging / fallback / Ali references changed; analysis plans internal and mostly unexecuted. "held out from all training and tuning" / "never used for training or tuning" removed. δw = 0 run now in the main table (italic rows). Closing gap: "chosen on development meetings" (from `reports/mist_w3po_v1/c2_and_a_points_dev.zh-CN.md`). "Pre-registered" no longer used for the analysis plans (App. Y explains they are internal, not publicly verifiable). [EXPERIMENT] (step 4,000 on test, fresh test set) not run → Limitations. |
| M5 | Per-unit speech LAAL renamed LAAL$_\text{spread}$ everywhere and defined against block-start LAAL (setup §4, App. E); Table 2 adds a "speech start" column (existing macros) and the text states speech starts hardly earlier and that the reduction comes from shorter queues/blocks; block-start dev result cited; Limitations: not the conventional metric, gains largely disappear under block-start, LAAL may degenerate, point estimates, no computation. **PostReview slot**: `\newif\ifPostReview` (main.tex, default false); guarded sentence in §5.3 uses `\num{v10:pr:test:ami:{b,e2,m}:bslaal}` and refers to App. F.3 (`app:postreview`), which inputs `generated/tab_v10_postreview.tex` (label `tab:postreview`) if it exists. To fill: extend `make_tables_v10.py` to read `eval_outputs/mist_w3po_v1/joint/post_review_v10/analysis_v1.json`, write the `v10:pr:*` macros and that table, then set `\PostReviewtrue`. |
| M6 | [EXPERIMENT] not run; Limitations states that fixed-lag / LA2 / cascade / relabelling baselines exist only for the per-direction model under the look-ahead protocol. SeamlessStreaming moved out of Table 2 into App. F.4 (`tab:e7`). |
| M7 | AliMeeting contradiction resolved from the eval scripts: the streaming (Table 2) and per-direction whole-meeting results are scored with the **DiariST human English translations** (`experiments/mist_w3po_v1/scripts/joint_lib.sh`: `RT_ALI=data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl`; `final_test_orch.py`); the offline/overlap studies used Hy-MT2 machine translations (Table 1, Tables real7/overlap8/online7 captions now say so; App. AC retitled "Machine-Translated AliMeeting References"). Audio condition stated (headset channels mixed, not far-field). tcpBLEU rewarding duplication stated (setup, App. F.1, Limitations); "no learned metric (COMET, MetricX) is used" stated; sacreBLEU with 13a / zh tokenisation stated (from `reports/gstar_si_v8/overlap_score.py`). Duplicate-aware scores for all systems, bilingual check: not done. |
| M8 | Related work expanded (~12 lines): Hibiki contextual alignment, MILk/MMA/EMMA latency losses (drift as known effect), EDAtt/StreamAtt, SimulEval, t-SOT for joint streaming ASR+ST (Papi et al. 2023), DiarizationLM, dubbing length control (Isometric MT); the delta of the set lattice and why a decoding-time bias. "No multi-speaker system reports latency" → "We are not aware of". New bib entries, each checked against the publisher/arXiv page: `ma2020mma`, `ma2023emma` (arXiv 2312.04515), `papi2023edatt` (ACL 2023, 13340–13356), `papi2023tsot` (ASRU 2023, arXiv 2307.03354), `lakew2022isometric` (ICASSP 2022, 6242–6246), `baevski2020wav2vec`, `guerreiro2024xcomet` (TACL 12). AlignAtt not added (venue details not verified). |
| M9 | Notation de-collided: diarizer look-ahead λ → γ (λ stays the PR dual), E-step multiplier → μ, budget ratio → ξ, frame length δ removed (frame time t_f), token centres c_i → t̄_i, instruction c → ι, loss normaliser D → Ω, backlog transition T_i → g_i, release time T(f) → t^rel_f, LAAL cut-offs τ/τ′ → i*/k*, PA forward/backward messages → m→/m←. "Anneal" replaced by a fixed temperature and an explicit statement that τ≠1 is stochastic EM-style self-training on a tempered posterior, E-step per minibatch with current θ. κ(A′) (ordered by due read, due time, source start; from `crf/precheck_serial_v2.py::kap`) and the binary o_r (one choice per read for any number of speakers; from `precheck_serial_v2p.py`) defined; q_0 defined; weights 0.5/5 stated as fixed, not ablated. New App. A "Notation" table and Algorithm 2/3 box (one write-alignment step; PA data generation). Figure 1 redrawn as two rows (smallest label font 3.3–5 pt → 4.9/7.4 pt) and its caption names both alignments. Not done: release statement (needs an author decision), chunk time tokens α_j/β_j, the inference rate rule, stage-2/PA compute. |
| M10 | All internal traces removed from the PDF (checked by a text scan): re-review IDs (M09, M17), "the user", colleagues, git hashes/UTC (the `v7:prereg:list` macro is no longer used), `prereg_v6–v9`, amendment numbers, run IDs (`sp2_on_*` → readable row label), "our previous version", "v5 matched…", B5/B6/B8/C2 labels, all v5–v10 version names, "SerialCRF v2′", "T1c", "PL-EM". Old generated tables are not edited: `make_tables_v10.py::sanitize_legacy()` writes cleaned copies to `generated/clean/` (wording + p 0.000 → <0.001 + 4-decimal intervals rounded half-up) and the tex inputs those. Appendix sections retitled by topic; offline Table (main7) moved out of the main text (two-sentence summary kept, full table in App. G); the Ali offline paragraph folded into §5.2. Tex comments with internal paths removed. Not done: full topical reordering of the appendix (m9). |
| M11 | "Simultaneous" → "streaming" for our system (title, method §3.3, figure, setup); abstract/intro/conclusion say per utterance it is closer to consecutive interpretation and that latencies assume one channel per speaker; conclusion (main text) discusses several overlapping translated voices vs. a single channel falling behind. |
| m1 | "text BLEU" → text tcpBLEU; "speaker-attributed BLEU" → cpBLEU. |
| m2 | Bold removed from Table 2; ∗ (p_Holm<0.05) / ∘ (n.s.) marks against B, from the Holm macros. |
| m3 | SeamlessStreaming rows moved to App. F.4 (`tab_v10_e7.tex`). |
| m4 | SerialCRF v2′, T1c removed; rl defined in method and tables; systems defined in setup. |
| m5 | Table 10 caption states δw = −1 on dev, δw = 0 on test; rescoring δw = −1 left to the PostReview analysis. |
| m6 | Figure 1 fonts enlarged (see M9). Figure 2 (trade-off panels): not changed. |
| m7 | App. "Relation to Prior Work" now has introductory text before its floats; capability table: "compute" latency footnoted as per-direction model only. |
| m8 | Chinese-audit table: ragged-right p-columns, cell reworded. |
| m9 | Appendix titles made topical and version-free; full reordering not done (page/label risk). |
| m10 | "61.50 / 100.40 rows" → whole rows; Table 44 (reg8) p 0.000 → <0.001 and intervals to 3 decimals (via `sanitize_legacy`). |
| m11 | X-Translator duplicated note removed; "Proc." venues normalised to "Proceedings of …"; MILk pages verified against the ACL Anthology (1313–1323) and the TODO comment removed. Not done: "et al." truncations of the Seamless / Qwen3-Omni / UniSS / Seed LiveInterpret / Spark-TTS / SimulS2ST-Omni entries (long author lists not verified) and Liu et al. 2020 pages (not verified). |
| m12 | Now cited: Whisper, Paraformer, sacreBLEU, SimulEval, StreamAtt, DiarizationLM, WavLM, TitaNet, UTMOS, wav2vec 2.0, xCOMET. VoxCPM2, Hy-MT2, LMT, MiLMMT, CometKiwi: left uncited (no verified reference). |
| m13 | Method states Eq. 5 uses the bound γ, actual release times (App. E) time commits for scoring. |
| m14 | Limitations + App. C.1: the Zh→En PA Talker was trained on outputs of the separately trained write-aligned Thinker, not on the baseline's text. |
| m15 | Calibration: the full-dev check contains dev5 and is not independent (App. D, Table 8 caption). |
| m16 | Method: time tokens up to 20 s, padding beyond is clipped. |
| m17 | "Our first simultaneous decoder" → "A decoder trained with a wait token on deadlines". |
| m18 | Ethics: synthetic Chinese AMI audio clones participants' voices; AliMeeting licence CC BY-SA 4.0 (checked on openslr.org/119). Release terms: open (author decision). |
| m19 | DiariST no longer said to "introduce speaker-attributed BLEU"; Limitations: DiariST's own metric not reported, so AliMeeting numbers are not comparable. |
| m20 | Limitations condensed to what affects the claims (test hygiene / missing controls, latency, evaluation, system). |
| m21 | "Hesitations produce no segment" → "By design, hesitations have no target segment." |

Still open: PostReview numbers (other agent); release statement; full appendix reorder; Figure 2; unverified long
author lists; all [EXPERIMENT] items (by decision). `docs/` (old plans/reviews) and `review_v10.md` must not go into a
submission archive.

## 7. 2026-11-15: post-review speech analyses integrated (`\PostReviewtrue`)

Source: `eval_outputs/mist_w3po_v1/joint/post_review_v10/analysis_v1.json` (method in
`reports/mist_w3po_v1/post_review_v10_analysis.zh-CN.md`: per-unit latency under both definitions, two TTS seeds pooled
per meeting, 10k meeting bootstrap, exact sign-flip p, **no Holm**; UTMOS/SIM on seed 1, SIM against the Talker's voice
prompt; pauses from `talker_block_pauses`). New `make_tables_v10.py::postreview()` writes 426 `v10:pr:*` macros
(latency means and paired diff/ci/p for m-b, e2-b, m-e2, m0-b, m-m0; rate, speech minutes, rate %, speech-time %, UTMOS,
SIM, UTMOS paired diff/ci/p, pauses and mid-sentence pauses per minute); `write_postreview_tables()` writes
`generated/tab_v10_postreview.tex` (`tab:postreview`: per-system latency/rate/UTMOS/SIM/pauses; `tab:postreview2`:
paired latency differences with intervals). No number typed by hand. `\PostReviewtrue` in main.tex
(`\PostReviewfalse` restores the shorter text).

Prose (all numbers are macros; checked against the JSON):
- Table 2 gains a LAAL$_\text{block}$ column; caption points to the interval table.
- §5.3 speech layer: B+PA ends speech earlier mainly by speaking faster (2.40 → 3.49 words/s AMI Zh→En; 3.90 → 5.04
  char/s En→Zh; 31 / 23 / 32 % less speech), queues less; under LAAL$_\text{block}$ B+PA − B is −0.09 / −0.11 / −0.33 s
  and MIST − B only −0.02 [−0.04, +0.01] / −0.02 [−0.04, +0.00] / −0.08 [−0.15, −0.02] s, first words later
  (+0.06 / +0.06 / +0.14 s); LAAL$_\text{spread}$ / EndOffset gains are a playback effect. UTMOS +0.12 / +0.12 / +0.19
  (MIST − B), but pauses ≥ 0.5 s per minute 3.3 → 7.7, 4.6 → 8.9, 4.4 → 11.0. Note: the coordinator's "~45 % faster" holds
  for Zh→En (45 / 47 %); En→Zh is 29 %, and En→Zh speech time falls 23 %, not 31–33 % — the text uses the macros.
- Abstract: latency claim rewritten (ends speech earlier mainly by speaking faster, about twice as many pauses,
  block-start LAAL of the full system unchanged vs B, not significant). Intro and conclusion likewise; no claim of a
  latency reduction under the standard definition remains.
- App. F.3 (`app:postreview`, retitled "Speech latency under both definitions, speech rate and voice quality"): method,
  all block-start / rate / UTMOS / pause results incl. δw=0, SIM vs voice prompt, low absolute AMI UTMOS, no listening
  test, no forced alignment / τ′ distribution. Table m2laal10 caption points to the new tables.
- Setup statistics: speech-latency differences now have bootstrap + sign-flip tests, uncorrected.
- Limitations "Latency and speech": block-start result, speech-rate / pause trade-off (incl. mid-sentence pauses), SIM
  against the prompt, uncorrected latency tests, no forced alignment.
- To stay at 8 pages: AliMeeting human-reference sentence in §5.2 shortened, ablation paragraph and B+PA confound
  sentence shortened, AliMeeting brevity-penalty sentence moved out of §5.3 (still App. F.1), hygiene (iii) and
  conclusion tightened.

Title kept ("When to Say It"): the system does decide when each block is heard, and the playback layer measurably
changes it (block-start LAAL of B+PA −0.09 / −0.11 / −0.33 s, EndOffset −0.96 / −0.58 / −1.34 s); the text no longer
claims that this makes the full system start speaking earlier.

State: no errors, no undefined refs / citations / number keys, no overfull boxes; main body = 8 pages (Limitations at the
top of page 9); 37 pages in total.
