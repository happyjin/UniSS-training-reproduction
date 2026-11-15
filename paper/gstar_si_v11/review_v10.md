# Review: "Who Said What, and When to Say It: Latent Write and Playback Alignment for Speaker-Attributed Simultaneous Meeting Interpretation" (v10)

Reviewer role: senior reviewer / area-chair level, ACL ARR long paper.
Basis: `main.pdf` (33 pages: 8-page main body, references, Limitations/Ethics, appendices A–AD with 45 tables) and the LaTeX source (`main.tex`, `sections/*.tex`, `generated/*.tex`). Line numbers refer to the review-mode line numbers in the PDF. Nothing outside the paper was consulted.

---

## 1. Summary of the paper's claims

The paper presents MIST, a system for simultaneous speech-to-speech interpretation of multi-party meetings. A Qwen3-Omni "Thinker" reads acoustic tokens interleaved with speaker tokens from a frozen streaming diarizer (G-STAR-style). It writes speaker-first, time-stamped translation segments for every speaker in a window. A UniSS "Talker" then speaks each segment with that speaker's voice prompt. The paper makes three groups of claims:

1. **Offline system and overlap evaluation.** On AMI reference windows, offline joint decoding beats the strongest non-oracle diarize-then-translate cascade by 6.52 (En→Zh) and 2.17 (Zh→En) cpBLEU. On overlapped units it beats a speaker-blind model relabelled by the same diarizer by 3.58 and 2.82 tcpBLEU (Tables 1–2).
2. **The titled contribution: two latent timing alignments.**
   - *Write alignment*: an exact-EM, linear-chain CRF over sets of written chunks and speaker order, with one constant write bias δw = −1 at decoding to undo EM's drift towards later writes.
   - *Playback alignment (PL-EM)*: a backlog CRF with a posterior-regularised latency budget, used to fine-tune the Talker.
   On AMI test (Zh→En / En→Zh), the full system improves on the jointly trained deadline baseline by +0.32 / +0.58 text tcpBLEU and +2.38 / +0.69 ASR-BLEU, and changes speech LAAL by −0.33 / −0.16 s (Table 3).
3. **Decomposition.** The paper claims the two parts are "complementary": playback alignment gives the speech and latency gains, write alignment the text gain. On AliMeeting the text score falls (−0.36, n.s.) while speech improves.

## 2. Scores (ARR style)

| Criterion | Score | One-line justification |
|---|---|---|
| **Soundness** | **2.5 / 5** | The older offline and overlap claims (Tables 1, 2, 25, 43) are well supported. The titled v10 claims are not. The write-alignment effects are not significant after the paper's own correction, and the latency results are not compared at matched quality. The speech gains are confounded with heuristic components, and a test-informed design change was made. |
| **Excitement** | **2.5 / 5** | The problem (multi-speaker, overlap-aware SimulS2ST with attribution and per-speaker voices) is new and worthwhile, and the evaluation framework is useful. The latent-alignment ideas are technically neat, but on the evidence shown they deliver little beyond a speech-layer heuristic. |
| **Overall assessment** | **2.5 / 5** (borderline Findings) | A substantial, unusually candid systems paper whose headline method contribution is not yet demonstrated. Recalibrating the claims alone ([TEXT]) would make it a solid Findings paper. A main-conference case needs the controls in M2–M3 and a clean test (M4). |
| **Reviewer confidence** | **4 / 5** | I checked every main-text number against the appendix tables and the source. I did not run the code. |

## 3. Strengths

- **S1. New and well-motivated problem.** Translating overlapping multi-party speech with speaker attribution and per-speaker voices, and measuring latency on overlapping meetings, is under-explored. The paper's capability comparison (Tables 40–41) makes the gap visible.
- **S2. Clean joint output format.** Speaker-first, time-stamped serialisation conditioned on a persistent streaming-diarizer stream is a simple, sensible design. The offline gains over the cascades are large and survive the pre-registered, Holm-corrected tests (Table 25: E-cas1/2/3 all p_Holm ≤ 0.001).
- **S3. Overlap-aware evaluation and honest decomposition.** The paper splits tcpBLEU into overlapped and single-speaker units, and adds concurrent-speaker recall and collapse (§4). The "Labels or translation?" analysis (App. AD) is exemplary: it shows the offline overlap gain is mainly attribution, not translation (34.56 → 31.94 when MIST's own text is relabelled, vs 30.98 for the speaker-blind text).
- **S4. Unusual statistical candour.** The paper reports exact sign-flip tests, a meeting bootstrap and Holm correction (Table 13). It reports non-significant and negative results (AliMeeting text, Table 16's failed or unrun tests), the δw = 0 test run (Table 9) and the checkpoint deviation (Table 5). It also checks diarizer causality (App. S), decoder run-to-run noise (App. T) and deadline accuracy (App. U). Few submissions document this much.
- **S5. A technically neat write-alignment construction.** Keying the history on the *set* of written chunks (canonical serialisation) makes the multi-speaker plan posterior a chain over set states with an exact partition function (Eq. 9, App. B). The paper also states the train–test mismatch this causes.
- **S6. A principled playback formulation and an informative delay decomposition.** The backlog CRF with a posterior-regularised budget (Eqs. 14–15) is a principled way to pose playback scheduling. The delay decomposition (Table 11: model lag / detector wait / queue / last block) is genuinely useful to the community.
- **S7. Realistic online protocol.** The online runs cover whole meetings and all audio, with detector release times (chunk protocol, App. D) and per-speaker streams. This is more realistic than the segmented evaluation common in SimulST (cf. Papi et al., 2025).

## 4. Major weaknesses

### M1. The headline v10 claims are stronger than the evidence, by the paper's own statistics [TEXT]

**Problem.**
- **Text gains are stated without qualification.** The abstract (l.030–039), introduction (l.120–129) and conclusion (l.620–626) say MIST "raises text BLEU by +0.32 / +0.58" and that write alignment "improves the text". Table 13 gives p_Holm = 0.403 (AMI Zh→En) and 0.080 (AMI En→Zh), so neither text gain is significant. The En→Zh speech gain (+0.69) has p_Holm = 0.053, also not significant, yet the abstract reports it as a gain.
- **Write alignment itself does not improve anything significantly.** Its isolated contribution (MIST vs B+PL, §5.4 l.578–582, Table 13) is:
  - ASR-BLEU +0.10 / +0.11 / −0.24, all intervals including 0, p_Holm = 1.0 / 1.0 / 0.512;
  - text tcpBLEU: the n.s. differences above;
  - speech LAAL: worse by +0.08 / +0.10 / +0.26 s.
- **"Complementary" is not supported.** Complementarity would require the combination to beat each part alone. Here B+PL dominates MIST on latency, ties it on speech quality, and the text difference is n.s. The data are equally consistent with write alignment having no effect beyond a small shift along the quality–latency curve (see M3).
- **The decision rule was relaxed for v10.** For v7–v9 the paper calls a difference real only if p_Holm < 0.05 *and* the decoder-noise-widened interval excludes 0 (l.491–494; App. N, T). The v10 tests drop the widening (§4 "Statistics", l.485–490). Appendix T estimates the decoder noise at 0.20–0.32 cpBLEU, the same size as the v10 text gains. Under the v7–v9 rule, every write-alignment effect would be reported as a tie.
- **Seed variance is not accounted for.** Every v10 model was trained once (Limitations l.664). The v9 deadline model's whole-meeting tcpBLEU varies across training seeds by s.d. 0.52 (En→Zh, 3 seeds), 0.22 (Zh→En, 2 seeds) and 0.63 (AliMeeting) (App. AD). The +0.32 / +0.58 text gains are of that order. The meeting-level sign-flip test treats each model as fixed, so it says nothing about seed variance.
- **The abstract mixes test families.** The 6.52 En→Zh cascade gain comes from the exploratory, uncorrected Table 26, while the 2.17 Zh→En gain is the pre-registered E-cas2. The 3.58 / 2.82 overlap gains are exploratory "x" comparisons that appear in the main text with intervals only.

**Why it matters.** The title and abstract present latent write alignment as a contribution that improves translation. Taken at face value, the paper's own tables show it has no significant benefit on any metric and costs latency relative to the speech layer alone.

**Fix [TEXT].**
- Rewrite the abstract, introduction and conclusion so each number carries its significance status.
- Lead with what holds: the speech-layer gains on AMI Zh→En and AliMeeting, the offline and overlap results, and the overlap-aware evaluation.
- State plainly that write alignment yields no significant improvement in this study, and drop "complementary".
- Apply one decision rule across all versions, or justify dropping the decoder-noise widening.
- Mark which abstract numbers are pre-registered or corrected and which are exploratory.
- If the M3 controls do not change the picture, retitle (e.g. foreground the speaker-attributed format and the playback layer).

### M2. The speech gains credited to playback alignment are confounded with heuristic components and with ASR-in-the-loop selection [EXPERIMENT (Talker-only re-runs) + ANALYSIS]

**Problem.**
- **The baseline and B+PL differ in four ways at once.** The strict baseline Talker uses one sample per block, a fixed rate and no merging (l.429–432). "B+PL" adds:
  - (i) read-level merging;
  - (ii) the PL-EM-fine-tuned Talker;
  - (iii) a backlog-dependent rate;
  - (iv) two samples per block with the better one chosen by wav2vec2-CTC fidelity (l.369–372).
  No experiment isolates these four on the strict-baseline text.
- **Table 7 suggests the learned part contributes little.** It compares speech layers on a single fixed text (the separately trained Zh→En v2′ Thinker):
  - AMI: the rule-based "queue rate, 2-best CTC" already reaches 20.52 ASR-BLEU / 4.00 s; PL-EM reaches 20.64 / 3.92; rl + PL-EM reaches 20.68 / 3.90.
  - AliMeeting: 15.83 → 16.26 → 16.22.
  So the learned PL-EM Talker adds about +0.1–0.4 ASR-BLEU and −0.1 s over a heuristic. Most of the +2.28 ASR-BLEU attributed to "playback alignment" in Table 3 is probably due to best-of-2 reranking and the rate rule. Neither is the posterior-regularised CRF.
- **ASR-based selection is then scored with ASR.** Choosing among candidates with a CTC recogniser and then scoring with ASR-BLEU (Whisper or Paraformer) rewards ASR-friendly audio. That is not the same as intelligibility or naturalness for listeners.
- **B+PL produces a third less audio for the same text.** Table 11's "load" falls from 1.41 to 0.94 s of synthesised speech per second of source on dev Zh→En, and similarly elsewhere. That implies much faster speech or removed silences. No v10 naturalness (UTMOS is reported only for older systems, Table 35), speaking rate or speaker similarity is reported, although the Talker is fully fine-tuned ("All Talker parameters are trained", App. B.1).
- **"Exact EM" does not describe PL-EM.** Appendix B.1 (l.1053–1057) shows PL-EM is:
  - a single round of posterior-weighted fine-tuning;
  - over 8 *sampled* candidates per block;
  - conditioned on the backlog the *rule-based* path reached (h_rule);
  - neither iterated nor marginalised over the backlog.
  At inference no CRF is used: the Talker samples two candidates and keeps the better CTC score. The abstract's "We learn both timing decisions as latent structure with exact EM" therefore overstates the playback half.

**Why it matters.** The paper's main positive finding (speech quality and latency) may come from simple reranking and speed-up heuristics rather than from the proposed latent-structure method.

**Fix.**
- **[EXPERIMENT, Talker-only, no training]** On the strict-baseline text, run a ladder on 3 test sets × 2 TTS seeds:
  1. T1c, 1 sample;
  2. + rl;
  3. + 2-best CTC;
  4. + rule-based queue rate;
  5. PL-EM Talker, 1 sample;
  6. PL-EM, 2-best (= B+PL).

  This is roughly 5 extra Talker passes per set, a few GPU-hours each.
- Also give the baseline Talker best-of-2 CTC selection, so that "B" is not handicapped.
- **[ANALYSIS, existing audio]** Report UTMOS, characters or words per second, WavLM/TitaNet speaker similarity to the attributed speaker, and the abrupt-pause rate for B, B+PL and MIST.
- **[EXPERIMENT, human]** Run a small listening test (MOS or MUSHRA for naturalness and comprehensibility, 20–30 raters × 50 clips) before claiming speech-quality gains.
- **[TEXT]** Call PL-EM what it is ("posterior-weighted Talker fine-tuning"), unless iterated EM is actually run.

### M3. Write alignment lacks matched-latency and compute-matched controls [EXPERIMENT]

**Problem.**
- **MIST writes later than the baseline on every test set.** Text LAAL is 3.07 vs 2.99 (AMI Zh→En), 3.06 vs 2.82 (AliMeeting) and 2.47 vs 2.37 (AMI En→Zh) (Table 3). In this regime later writing normally buys quality: Figure 2 shows it, and so does App. C ("earlier writes cost quality"; δw = −2 / −3 drop dev5 Zh→En tcpBLEU to 25.25 / 14.66, Table 8).
- **δw only roughly matches the baseline's latency.** It was calibrated to match dev5 text LAAL on the coarse grid {−1, −2, −3, −5} and lands +0.09 / +0.10 s later than the baseline on dev5 (Table 8).
- **The lattice can only delay.** Eq. 8 allows each chunk to be written only "on time or one read late". Maximum-likelihood alignment can therefore only make writes later. The observed drift (Table 6) is the expected consequence, and δw then pushes the decoder back.
- **Without controls, the text gain cannot be attributed to latent alignment.** It could equally come from any of:
  - a point further along the same quality–latency curve;
  - 3,500 extra training steps with fresh LoRA adapters (stage 2 vs stage 1, App. A);
  - a likelihood-weighted mix of ε and ε + Δ deadlines;
  - the speaker-order latent alone (6,801 of 27,953 windows have a real order choice, App. B).

**Why it matters.** Without these controls, the central method has not been shown to do anything that a cheaper baseline cannot.

**Fix [EXPERIMENT]**, in increasing order of cost:
- (a) **Decode only.** Apply the same wait-logit bias family to the deadline baseline, e.g. δw ∈ {+0.5, +1, −0.5, −1}, on dev5 and full dev. Plot text and speech quality–latency curves for B and MIST, and compare at equal text and speech LAAL. Choose a finer δw grid (including −0.5) for MIST. Cost: a few GPU-hours per direction.
- (b) **Compute-matched control.** Continue deadline SFT from stage 1 for 3,500 steps with fresh LoRA and the same schedule, then decode. Cost: one stage-2-length run (~3,500 steps on 8 H200) plus decoding.
- (c) **Lag-mixture control.** Run deadline SFT with ε drawn from {1, 2} s (or ε = 1.5 s). Same cost as (b).
- (d) **Lattice ablation.** Train with the order latent only and with deferral only. Two runs; run them only if (a)–(c) leave a positive residual.

If write alignment does not beat (a)–(b) at matched latency, the paper should say so and reframe.

### M4. Test-set hygiene: a test-informed design change, a post-hoc checkpoint choice, and repeated reuse of the same test meetings [TEXT + EXPERIMENT]

**Problem.**
- **The write bias was added after a test result.** δw was introduced after the first test decoding showed that the system "started speaking later than the strict baseline, most on AliMeeting" (§5.4 l.562–568; App. C l.1073–1087; Limitations l.631–635).
  - The paper argues that only the *decision* to add δw, not its value, was test-influenced (App. C l.1086–1087). But adding a component is a tuning decision.
  - The test result that motivated δw is exactly what δw repaired: AliMeeting text 18.99 → 21.07, speech LAAL 4.85 → 3.91 s (Table 9).
  - The abstract nevertheless calls AliMeeting "held out from all training and tuning" (l.039–041), and §5.4 says "never used for training or tuning" (l.603). For the final system this is inaccurate.
- **The stage-2 checkpoint deviates from the pre-registered rule.** The rule selected step 4,000; the authors used 3,500 (§4 l.421–423; App. A; Table 5).
  - The difference in the selection criterion is J = 35.59 vs 35.31 (0.28), within the decoder noise the paper itself estimates (0.20–0.32).
  - Step 3,500 violates the pre-registered feasibility bound (+0.137 s > 0.10 s).
  - The "principle that BLEU takes precedence over a small excess of lag" is stated after the fact.
  - The rule's latency reference L_ref is the separately trained latent-write system (3.648 / 2.841 s on dev5), not the deadline baseline (3.07 / 2.35 s). The latency gate therefore admitted checkpoints about 0.5 s later than the baseline.
- **The same test meetings have been reused across many versions.** The same 16 AMI and 20 AliMeeting test meetings have served versions v6–v10, between which the paper reports changes to:
  - the online window closing gap (1 s → 0.6 s, §3.7 l.376, with no stated selection data);
  - the latency protocol (chunk protocol, App. D);
  - read-level merging;
  - fallback adoption (App. AD);
  - the AliMeeting references (App. G).

  Adaptive overfitting to these test meetings cannot be ruled out, and the paper offers no untouched test set for the final system.
- **The pre-registrations cannot be verified and were mostly not executed.** They are identified by private git hashes and UTC timestamps (App. X), which reviewers cannot check. For v9, 10 of the 11 confirmatory tests were "not run" and the one that was run failed (Table 16). This weakens the "pre-registered" label used in §4 (l.485).

**Why it matters.** The final-system numbers on AliMeeting (and partly on AMI) are post hoc. Reviewers will discount them.

**Fix.**
- **[TEXT]** Remove "held out from all ... tuning" for the final system and state upfront that the AliMeeting result for MIST v10 is post hoc. Report the δw = 0 run in the main table, not only in the appendix.
- **[EXPERIMENT, decode only]** Report test results for the pre-registered checkpoint (4,000) next to 3,500.
- **[EXPERIMENT]** Freeze the final system and evaluate it once on a test set never used in v6–v10, for example:
  - another English meeting corpus (e.g. ICSI or NOTSOFAR-1, with translated references) for En→Zh;
  - another real Mandarin meeting corpus (e.g. AISHELL-4) for Zh→En.

  Decoding costs about a GPU-day. Creating the references is the main cost.
- **[TEXT]** State how the 0.6 s closing gap was chosen. Put pre-registrations in an anonymised public registry (e.g. OSF), or stop calling them pre-registered in the main text.

### M5. The primary latency metric is non-standard, and its conclusions change under the standard convention [ANALYSIS]

**Problem.**
- **The primary metric spreads words evenly over playback.** "Speech LAAL" charges each word at the moment it is heard, with a block's words spread evenly over its playback (§4 l.462–469; App. D). Shorter or faster speech therefore lowers it directly, even when the first word is heard no earlier.
- **Under the block-start convention, the gain mostly disappears.** Table 10 reports the convention used in single-speaker work, where every word is charged at its block's start:
  - Dev Zh→En: B 3.10 vs MIST 3.06 s (−0.04), against 3.97 vs 3.61 under the paper's convention (−0.36).
  - Dev En→Zh: 2.49 vs 2.48 (−0.01), against 3.12 vs 2.96.
  - Test, with MIST at δw = 0: MIST is *later* than B by 0.27 / 0.73 / 0.26 s.
  - The δw = −1 test runs "were not rescored with this convention" (Table 10 caption).

  So most of the reported speech-latency reduction comes from how long blocks take to say (cf. load 1.41 → 0.94, M2), not from speech starting earlier.
- **LAAL may degenerate in this regime.** 56% of words are written at a window's last read, the mean unit lasts 3.48 s (median 2.11 s), and text LAAL after the detector equals the mean unit duration (App. D.1). With delays this large, LAAL's cut-off τ′ (first word heard after the source ends) is likely to be small for many units, so LAAL approaches first-word delay plus within-block spreading. The paper does not report the distribution of τ′.
- **No latency figure has an interval.** All latency differences are point estimates (§4 l.488–490; Limitations l.654–655), yet the abstract leads with them.
- **v10 latency ignores computation.** No v10 latency accounts for computation (Limitations l.649–654). The system re-encodes the full prefix with a 30B MoE at every read without a KV cache (Fig. 3), and PL-EM doubles Talker sampling. The v9 real-time factor was already 0.72–0.98 on two H200s at batch 1 (App. F).

**Why it matters.** The "whole latency reduction" claim rests on a convention the authors introduced, and the conventional measure largely reverses it.

**Fix.**
- **[ANALYSIS, existing outputs]**
  - Rescore every Table 3 run (including δw = −1 on test and B+PL) under both conventions.
  - Add StartOffset, EndOffset and ATD (or a computation-unaware Average Token Delay), each with meeting-bootstrap intervals and sign-flip tests.
  - Use forced-aligned word times from the synthesised audio instead of even spreading.
  - Report the τ′ distribution.
- **[EXPERIMENT, small]** Measure computation-aware latency or the real-time factor for the v10 decoder plus Talker on an uncontended node. A few GPU-hours for one meeting per set.

### M6. The final system is compared only with its own stage-1 baseline and a zero-shot single-speaker system [EXPERIMENT (mostly decode only)]

**Problem.**
- **The cascade and relabelling claims in the abstract are not about the final system.** They are for the offline model (v7/v8) on reference windows. These windows keep only 45.6% of En→Zh test speech and 35.6% of its overlapped speech (Table 37; App. AD reports a whole-meeting rescoring for v9 only).
- **The policy comparisons use an older system and protocol.** All simultaneous comparisons with LocalAgreement, fixed lag and speaker-blind + causal relabelling are on the v9 system under the v9 protocol, which the paper itself calls "not comparable with the v10 numbers" (App. F l.1216–1217).
- **A trivial policy already looked competitive.** On the meeting subsets, a fixed 2 s lag is at least as good as MIST-Simul on overall tcpBLEU (Table 44: 40.28 vs 38.84 on 8 AMI En→Zh meetings; 12.83 vs 11.55 on AliMeeting), at a longer overlapped-unit lag. It was not re-run for v10.
- **SeamlessStreaming cannot calibrate the result.** It is a weak, zero-shot, relabelled reference that "says little" (l.568–572).
- **No v10-protocol baseline shows that a strong but simple system would not do as well.** Nothing compares a streaming cascade, a separation-based cascade or a fixed-lag policy with a per-speaker Talker.

**Fix [EXPERIMENT].** Under the v10 protocol and with the v10 speech layer, re-run the existing checkpoints (decoding plus Talker; roughly 1–2 GPU-days in total):
- (a) speaker-blind Simul + causal relabelling;
- (b) the streaming cascade (diarizer turns + LA2);
- (c) fixed lag 1 / 2 s on the deadline model;
- (d) LA2 on the joint model;
- (e) if feasible, a separation- or target-speaker-extraction cascade into an off-the-shelf SimulS2ST system.

Present quality–latency curves rather than single operating points.

### M7. The evaluation rests on machine-translated or synthetic material and a single string metric [ANALYSIS + TEXT]

**Problem.**
- **En→Zh references are machine translations.** They are chosen by QE from three MT systems (App. I), with no bilingual check (Limitations l.659–662). Training targets come from the same pipeline, so BLEU partly rewards matching that MT style.
- **The Zh→En source is synthetic.** It is Chinese MT re-voiced with the UniSS Talker, the same model family as the output Talker (§4, App. Q). The references are the original English, so the task is translating translationese back towards its source, with synthetic voices that are easier to diarize. The only real-speech Zh→En set, AliMeeting, is where write alignment fails.
- **The AliMeeting references are described inconsistently.**
  - §4 (l.402–405) and §5.3 (l.536–538) say "human English translations released with DiariST".
  - The Table 20 caption and App. AB (around l.1820–1836) say "machine translations (Hy-MT2-30B) of the human transcripts; no human check was made".
  - It is not clear which references score Table 3.
  - The AliMeeting input is mixed close-talk audio (Limitations l.689–691), not the far-field condition of M2MeT or DiariST, so the results are not comparable with published AliMeeting numbers.
- **tcpBLEU keeps duplicate translations.** A unit translated again in a neighbouring window keeps both translations (App. E.1). The baseline had 27,697 duplicated words (33% of attached words) on AliMeeting. Duplicates lengthen the hypothesis and avoid the brevity penalty, so the metric partly rewards duplication. That interacts directly with the paper's explanation of the AliMeeting gap as a "brevity penalty" (Table 12).
- **No learned metric is reported.** The authors have XCOMET (used for data selection, App. I), but report no COMET, MetricX or xCOMET scores. For deltas of 0.3–0.6 BLEU a second metric is essential.
- **The BLEU implementation is unspecified.** No sacreBLEU signature or tokenisation (Chinese character vs `zh` tokenizer) is given. The ASR-BLEU pipeline is also underspecified: is ASR run per block or per stream, and are source or playback timestamps used for the collar? See Q5.

**Fix.**
- **[ANALYSIS, existing outputs]**
  - Score all Table 3 systems with COMET-22 or MetricX (reference-based) and CometKiwi (reference-free), using the same sign-flip and Holm tests.
  - Report a duplicate-aware tcpBLEU (keeping one translation per unit) as a secondary metric for *all* systems, including δw = −1.
  - Give the sacreBLEU signatures.
- **[TEXT]** Resolve the AliMeeting reference contradiction and state the audio condition clearly.
- **[EXPERIMENT, human]** Run a bilingual check on a few hundred En→Zh references.

### M8. Novelty and positioning are under-argued, and related work is thin [TEXT]

**Problem.** Section 2 is 0.6 column long, and the contribution delta is argued in one sentence ("differs in its state space", l.149–153). Prior work that should be discussed includes:
- **Learned write timing.** Hibiki's *contextual alignment* (Labiausse et al., 2025, already cited) uses a text model's likelihood to place each target word after the source that makes it predictable. That is very close in spirit to timing chunks with the model's own likelihood.
- **Latent policies with latency regularisation.** MILk and monotonic multihead attention (MMA, Ma et al., 2020; EMMA, Ma et al., 2023, the policy inside SeamlessStreaming) add explicit latency losses precisely because likelihood alone prefers late writes. The "drift" in §5.4 is this known phenomenon and should be framed as such, not presented as a finding.
- **Joint streaming ASR + ST with token-level serialisation** (e.g. Papi et al., 2023/2024, on t-SOT for joint recognition and translation with timestamps). This is closer to the decoding format than the cited ASR-only SOT papers.
- **LLM post-processing of diarization** (DiarizationLM, Wang et al., 2024). It is in `references.bib` but never cited.
- **Attention-based policies for speech LLMs** (AlignAtt / EDAtt / StreamAtt; StreamAtt is in the bib but not cited) and the SimulEval and computation-aware latency literature (SimulEval is in the bib but not cited).
- **Speech-rate and duration control** in dubbing and S2ST (isochrony, duration-controlled TTS). This is directly relevant to playback alignment.

The claim that "no multi-speaker system reports latency on overlapping meetings" (l.066–068) is narrow and hard to verify. Soften it to "we are not aware of".

**Fix [TEXT].** Expand related work by about 10–12 lines (space can come from M10), and state the delta precisely:
- what the multi-speaker set lattice adds over HMT, MMA and contextual alignment;
- that the drift was expected, and why a decoding-time bias was chosen over a latency term;
- what the PR budget adds over rate-control heuristics.

### M9. The method cannot be reimplemented from the main text [TEXT]

**Problem.**
- **Notation collides throughout.**
  - λ is the diarizer look-ahead (1.04 s, Eq. 5 and §3.7), the posterior-regularisation dual variable (Eq. 15) and the deferral-cost multiplier (App. B).
  - δ is the 80 ms frame and δw the write bias.
  - τ is the posterior temperature (Eq. 9), and τ / τ′ are LAAL indices (Eq. 26, App. D).
  - D is the loss normaliser (Eq. 10) and also the durations Dₙ and D(c).
  - c is the instruction, the token centres cᵢ and the TTS candidates.
  - ρ is a reference speaker (Eq. 22) and also the budget ratio (App. B).
  - T is the recording duration, the backlog transition Tᵢ and the frame release time T(f).
- **"Anneal" is a misnomer.** "We anneal with τ = 0.5" (l.311) describes a fixed temperature, not a schedule. With τ ≠ 1 the procedure is not EM on the marginal likelihood. The paper should state which objective is optimised, and that the E-step uses a minibatch with the current θ (stochastic or online EM).
- **Several components are undefined or unjustified.**
  - the canonical serialisation κ(A′);
  - how oᵣ works when more than two speakers have deferred chunks;
  - how chunk-level time tokens αⱼ, βⱼ are set;
  - the "posterior perturbation";
  - q₀ in the posterior-regularisation problem (presumably the CRF at λ = 0);
  - how the weights 0.5 and 5 in Eq. 15 and the 0.3 s budget offset were chosen;
  - how the "backlog-dependent rate" is picked at inference (the rate token among 35 rates?).
- **Figure 1 is unreadable and out of date.** It shows neither v10 contribution (write lattice, playback CRF), and its fonts measure 3.3–5 pt, illegible when printed.
- **Reproducibility details are missing.** There is no code, model or data release statement. Training compute for stage 2 (E-step rows scored per step) and for PL-EM data generation is not reported. Several components are proprietary pipelines (three MT systems, QE, VoxCPM2, the T1c Talker).

**Fix [TEXT].**
- Add a notation table in the appendix and de-collide the symbols.
- Add an algorithm box for one write-alignment EM step and one for PL-EM data generation and training.
- Redraw Figure 1 (or add a small Figure 2) showing the two lattices.
- State what will be released.

### M10. The paper reads as a cumulative lab notebook across v5–v10, with internal artefacts [TEXT]

**Problem.**
- **The main text mixes four system generations and protocols.**
  - v7 offline (Table 1), v8 overlap (Table 2), v9 Simul (end of §5.2, §5.3) and v10 (Tables 3–4).
  - The online windows differ (1 s vs 0.6 s closing gap), and so do the latency protocols.
- **Readers must track internal names.** "v9 MIST-Simul", "MIST v10", "SerialCRF v2′" (Tables 4, 8), "T1c", "B5 / B6 / B8", "C2", "E-ov1", "prereg_v7 amendment 6".
- **The appendix carries core material.** It runs A–AD with 45 tables, and core definitions (tcpBLEU, chunk protocol, the LAAL variant) appear only there.
- **Process artefacts and possible anonymity problems remain in the text.**
  - "re-review M17" (App. W, ~l.1682) and "re-review M09" (App. AD, ~l.1866);
  - "The user then stopped the remaining B6 runs" (App. AD, ~l.1904);
  - "colleagues' jobs on the shared cards" (App. X, ~l.1712);
  - private git commit hashes with UTC timestamps (App. X);
  - internal run IDs "sp2_on_S2dl vs. sp2_on_O2_LA2" (Table 26);
  - "the 1,000-step Zh→En model of our previous version" (Table 22 caption);
  - "v5 matched Zh→En output against the Chinese source" (Table 34 caption).

  "Our previous version" may signal an earlier submission or preprint, which works against double-blind review.
- **The core story is lost.** The 8-page body spends about half its results space (§5.1–5.3, Tables 1–2) on older systems, while the titled contribution lacks the controls in M2–M3.

**Fix [TEXT].**
- Decide the story. Either:
  - (A) the joint speaker-attributed format plus overlap-aware evaluation plus a speech layer, with timing as one section; or
  - (B) latent timing, which needs M2–M3.
- Rename systems descriptively (Deadline-SFT, +WA, +PA) and drop version numbers.
- Remove all internal artefacts.
- Collapse the v6–v9 history into one "development history" appendix, or remove it.
- Move Table 1 to the appendix and keep a two-sentence summary.

### M11. "Simultaneous" framing and listener realism [TEXT]

**Problem.**
- **Per utterance, the system is consecutive.** The abstract's last sentence and App. D.1 concede this: 56% of words are written at the last read, and speech LAAL is 3.05–3.91 s against a mean unit of 3.48 s. The title nevertheless says "Simultaneous".
- **The latencies assume one channel per speaker.** All v10 latencies assume per-speaker channels (Limitations l.656–657), so a listener hears several synthetic voices at once during overlap. A single channel "cannot keep up": median completion lag is 142 s (En→Zh) and 605 s (AliMeeting) (App. F, Table 45).
- **The listener experience is not discussed.** Overlapping translated voices are a usability problem, and the paper does not address it.

**Fix [TEXT].** Use "streaming" or "online" in the title, or defend "simultaneous" with a clear definition. Discuss the per-speaker-channel assumption and the listener experience in the main text, not only in Limitations.

## 5. Minor issues

- **m1.** The abstract says "text BLEU" (l.032); the metric is tcpBLEU. Elsewhere the abstract says "speaker-attributed BLEU" for cpBLEU_sess. Use the metric names consistently.
- **m2.** Table 3: "Bold: best of our three systems" bolds non-significant differences (e.g. 25.34 vs 25.02) and both tied rows. Mark significance with symbols instead.
- **m3.** Table 3: SeamlessStreaming's LAAL is compared at less than half the quality. Move the row to a separate block or the appendix (the paper itself says LAAL is only comparable at similar quality, App. E.3).
- **m4.** Table 4 and Table 8: "SerialCRF v2′" is undefined in the main text. "T1c" (l.425, l.430) is also undefined. "rl" is introduced only in the last line of §3.7.
- **m5.** Table 10 uses one "MIST" label for δw = −1 (dev) and δw = 0 (test). Split the column or rescore (see M5).
- **m6.** Figure 1's fonts are 3.3–5 pt. Figure 2 (p. 21) stacks two small panels in one column and leaves large white space.
- **m7.** The heading of Appendix AC, "Relation to Prior Work" (p. 28), is empty. Its Tables 40–41 float two to three pages later (pp. 30–31).
- **m8.** Table 31 (App. Q, p. 26) has broken cell wrapping ("≤ 1.25×: −12 to −17%" splits across lines). Use p{}-columns or tabularx.
- **m9.** Appendix order and labels: the v10 appendices come first (A–E), v9 next (F–G), then older material (H–AD). The AA–AD labels come from a custom counter. Reorder by topic.
- **m10.** Spurious precision: "about 61.50 rows and ... 100.40" (App. B, l.1005–1006); interval endpoints to four decimals in Table 43 ([0.1451, 0.1939]); "0.000" p-values in Table 43. Use "<0.001".
- **m11.** Bibliography:
  - The X-Translator entry prints "Preprint, arXiv:2607.17544. ArXiv preprint, July 2026." (duplicated note).
  - Several entries truncate authors to "et al." (Cheng 2025a/b, Xu 2025, Wang 2025, He 2026, Seamless 2023).
  - Venue styles are inconsistent ("Proc. Interspeech" vs full names); some entries lack pages (Liu et al., 2020).
- **m12.** Tools used without citation:
  - In `references.bib` but never cited: Whisper-large-v3, Paraformer, WavLM, TitaNet, UTMOS, sacreBLEU, SimulEval, StreamAtt, DiarizationLM.
  - Not in the bib at all: wav2vec2, XCOMET, CometKiwi, VoxCPM2, Hy-MT2, LMT, MiLMMT.
- **m13.** Eq. 5 uses a constant λ = 1.04 s, but App. D's chunk protocol releases frames at T(f) = (9⌊f/9⌋ + 13)·80 ms ≤ f·80 ms + 1.04 s. Say in the main text that Eq. 5 is the bound and which rule is used for decoding and for scoring.
- **m14.** §5.4 (l.550–551) says "Our systems in Table 3 share training data, online windows and diarizer". The Zh→En PL-EM Talker, however, was trained on outputs of the *separately trained* Zh→En latent-write Thinker (App. A "Talkers"; App. B.1 "Data"), so the speech layer was adapted to a different text distribution. Say so.
- **m15.** δw was chosen on dev5 and "confirmed" on the full development set, which contains dev5. This confirmation is not independent.
- **m16.** Time tokens reach 20 s, while windows reach 20 s plus 0.2 / 0.3 s padding (App. I). Table 39 shows zero-length segments piling up "at 20 s". Mention the vocabulary limit in the main text.
- **m17.** The intro (l.083–090) describes "our first simultaneous decoder" with dev numbers. If this is the authors' earlier published system, anonymity needs care. If not, call it the deadline baseline.
- **m18.** Ethics: the Zh→En test set clones the voices of AMI participants into Chinese. State whether this synthetic audio will be released and under which terms, and give AliMeeting's licence explicitly.
- **m19.** The DiariST metric: the paper says DiariST "introduces speaker-attributed BLEU" (l.174) but uses its own cpBLEU and tcpBLEU. Explain why, or also report DiariST's metric (see Q10).
- **m20.** The Limitations section reports compute-forced cuts, colleague contention and "by decision" choices (App. AB). This is fine in a lab report, but in a submission it should be condensed into what affects the claims.
- **m21.** Minor wording: "Hesitations produce no segment" (App. H) is ambiguous: is that by design or an observation?

## 6. Questions to the authors

1. On the *strict-baseline text*, how much of the B → B+PL ASR-BLEU gain comes from best-of-2 CTC selection, the rule-based rate, read-level merging and PL-EM training respectively? (M2)
2. What does the deadline baseline achieve with its own wait-logit bias (δw > 0 or < 0)? At equal text LAAL, is MIST still ahead? (M3)
3. What is the test result of the pre-registered stage-2 checkpoint (step 4,000)? (M4)
4. How was the 0.6 s window closing gap (v9: 1 s) chosen, and on which data? Were any other v10 settings chosen after looking at v6–v9 test results on the same meetings?
5. ASR-BLEU: is ASR run per block or per stream, and which timestamps (source or playback) decide the tcpBLEU collar for speech segments? If playback time is used, does lower latency mechanically raise ASR-BLEU?
6. Which AliMeeting references score Table 3: the DiariST human translations or Hy-MT2 machine translations? Is the audio mixed headset or far-field?
7. What are speaking rate, UTMOS and speaker similarity for B, B+PL and MIST? Does the fully fine-tuned PL-EM Talker reduce voice similarity?
8. Is the write-alignment E-step re-run with the current θ at every step (online EM)? With τ = 0.5, what objective does the procedure optimise, and does training improve the tempered marginal likelihood?
9. What does stage 2 cost relative to stage 1 (GPU-hours, rows scored per step), and what does PL-EM data generation cost?
10. Since AliMeeting is scored with DiariST's references, can you report DiariST's own metric under matched conditions so that readers can relate the numbers to published DiariST results? If not, why not?
11. In the 6,801 windows with a real order choice, which order does the posterior prefer? Does the order latent change decoding behaviour?
12. How often does the decoder's own serialisation at test time differ from the canonical κ(A′) used in training?
13. What is the distribution of the LAAL cut-off τ′ for speech LAAL? What fraction of units have τ′ = 1?
14. Will code, model weights, the re-voiced Chinese AMI meetings and the translated references be released?
15. Table 7 (v2′ text) gives the rule-based layer with queue rate and 2-best CTC 4.00 s speech LAAL, and rl + PL-EM only 0.10 s less. On the strict-baseline text, B → B+PL gains 0.40 s (Table 3). The texts differ, but which part of the speech layer actually removes the queue, and what does the rule-based layer achieve on the strict-baseline text?

## 7. Prioritised revision list (by impact on the review outcome)

The main body is limited to 8 pages; Limitations, Ethics, references and appendices do not count. A page budget is suggested for each item.

1. **[TEXT] Recalibrate all claims (M1, M4, M11).** Rewrite the abstract, introduction, conclusion and possibly the title:
   - qualify each number with its corrected significance;
   - say write alignment yields no significant gain;
   - drop "complementary";
   - remove "held out from all tuning" for AliMeeting;
   - report the δw = 0 run in the main table.

   No page cost. *This alone moves the paper from overclaiming to sound.*
2. **[ANALYSIS] Latency rescoring (M5).** Rescore all Table 3 runs, including δw = −1 test and B+PL:
   - under both speech-LAAL conventions, plus StartOffset and EndOffset;
   - with meeting-bootstrap intervals and sign-flip tests;
   - with forced-aligned word times and the τ′ distribution.

   Page cost about 0.1 (extra Table 3 columns; details in the appendix).
3. **[EXPERIMENT, Talker-only] Speech-layer ladder plus speech-quality diagnostics (M2).**
   - Run T1c → +rl → +2-best CTC → +rate rule → PL-EM on the strict-baseline text, and give the baseline best-of-2 CTC.
   - Report UTMOS, speaking rate and speaker similarity.
   - Ideally add a small listening test.

   A few GPU-hours, plus a few days for a listening test. Replace Table 4 in the main body with this ladder (≈0.3 page) and move the dev ablation to the appendix.
4. **[EXPERIMENT] Controls for write alignment (M3).**
   - (a) Run a δw sweep on the deadline baseline and compare at matched latency (decode only, cheapest; do first).
   - (b) Continue deadline SFT for 3,500 steps as a compute-matched control (one stage-2-length run).
   - (c) Run a lag-mixture deadline SFT.

   One quality–latency figure, about 0.3 page. If write alignment does not beat (a)–(b), reframe the paper around story (A) of M10.
5. **[EXPERIMENT, decode only] Hygiene runs (M4).** Decode the pre-registered checkpoint 4,000 on test. Evaluate the frozen final system on an untouched meeting test set. Reference creation is the main cost. About 0.1 page.
6. **[EXPERIMENT, decode only] v10-protocol baselines (M6).** Run speaker-blind + causal relabelling, the streaming cascade, fixed lag 1 / 2 s and LA2 with the v10 speech layer, as quality–latency points in the item-4 figure. About 1–2 GPU-days. No extra page if merged into that figure.
7. **[ANALYSIS] Second metric and metric hygiene (M7).**
   - Score COMET or MetricX and CometKiwi on all Table 3 systems, with the same tests.
   - Add a duplicate-aware tcpBLEU for all systems.
   - Give sacreBLEU signatures.
   - Resolve the AliMeeting reference contradiction.

   About 0.1 page.
8. **[TEXT] Restructure and clean up (M9, M10).**
   - Drop version numbers and internal artefacts (re-review IDs, "the user", "colleagues", run IDs, git hashes, "our previous version").
   - Move Table 1 to the appendix and shrink Table 2 to its overlap columns.
   - Cut §5.3 to about three lines.
   - Redraw Figure 1 with the two lattices at legible font size.
   - Add a notation table and algorithm boxes in the appendix.
   - Consolidate the appendix by topic.

   This frees about 0.6–0.8 page for items 3–4 and 9.
9. **[TEXT] Related work and novelty (M8).** Discuss contextual alignment (Hibiki), MMA / EMMA latency regularisation, joint streaming ASR + ST with t-SOT, DiarizationLM, attention-based SimulST policies and dubbing / rate control. Frame the drift as expected and state the delta precisely. About 10–12 lines.
10. **[TEXT] Minor issues m1–m21.** Fix the bibliography, uncited tools, table formatting, the empty appendix heading and the notation. No page cost.

**Bottom line.** The multi-speaker format and the overlap-aware evaluation are a real contribution, and the paper is commendably transparent. As written, however, the titled method contribution is not demonstrated:
- write alignment has no significant effect and costs latency;
- the playback gains are confounded with heuristics and with a favourable latency convention;
- the final configuration was shaped by a test result.

With claims recalibrated (item 1), this is a reasonable Findings paper. With items 2–5 completed and positive, it could become a main-conference paper.
