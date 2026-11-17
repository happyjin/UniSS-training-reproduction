# v16 (2026-11-17), from v15

Plan: ../gstar_si_v16_revision_plan.zh-CN.md (round 2) plus the user's last decisions: no factorial (R3') table anywhere;
Table 1 shows that the other systems have no speaker attribution of their own. No new experiments.
Untouched: abstract, introduction, Figure 1, related work (byte-identical to v15).

Method
- Symbols: rate token tau (q is now only the PR distribution), transcripts o_k / references g_rho (h is only the
  backlog), projector Proj_psi (g_i is the backlog transition), W_S = total target units, CE for cross-entropy, chunk
  index l. Dimensions added for x, y, v_k; messages, marginals, candidate weights and rate posteriors are vectors /
  matrices (alpha_i, beta_i in R^|H|, mu_i in [0,1]^{|H|x|C_i|}, w_i and rho_i in the simplex); forward-backward in
  matrix-vector form with 0/1 transition matrices T_i^(c); stream observation U = (nu, W, C_1:I).
- Problem formulation and every subsection opening shortened.
- 3.4 restores the latency-budget derivation in the body: Lagrangian -> Gibbs form (the price of delay is the
  multiplier), dual and derivatives (Eq. 10), Proposition 1 (latency budget) stated in the body with its purpose;
  modelling approximations stated; "one M-step of PR-EM" now "analogous to".
- 3.6 contains only the proposed metrics (cpASR-BLEU, tcpASR-BLEU), inspired by cpWER / tcpWER, each with a formula;
  the scorer will be released. LAAL (Papi et al.) and EndOffset (SimulEval) moved to the setup.

Experiments (one chapter, SEE-style): 4.1 Setup, 4.2 Main Results, 4.3 Where the Listener-Side Delay Is Removed,
4.4 Out-of-Domain Generalisation and Robustness, 4.5 Ablation Analysis (speaker tokens; playback-layer components).
- Table 1: column "Own speaker attribution" (x / voice only / check); caption: systems a-c have none and are given
  our diarizer's slots, which favours them.
- 4.4: our metrics rank the four systems on AliMeeting as DiariST's official speaker-attributed BLEU does.
- 4.5: the budget set lambda* without tuning.

Appendix
- B: hyperparameter table uses the new symbols; Algorithm 1 gives the bisection range.
- C: computation (wall-clock replay; not real time as implemented) moved here from the limitations.
- D: formal CRF factorisation, modelling assumptions (sampled candidates, ceil-rounded capped backlog grid, drift
  note), Proposition 2 (exact inference) and Proposition 3 (posterior-weighted training) with proofs, proof of
  Proposition 1, implementation (log space, bisection on [0, 50] with 40 halvings, brute-force unit tests).
- E: complete definitions of cpASR-BLEU / tcpASR-BLEU (inputs, unit chaining < 1 s, chrF Hungarian matching,
  2 s collar assignment, drop / insertion / split rules, pair sets, per-meeting statistics averaged over TTS runs),
  Algorithm 2 and a worked example; checked against experiments/mist_w3po_v1/evaluation/cp_asr_bleu.py and
  v11_metrics.score_ext.
- F: causal relabelling rule stated precisely (reports/gstar_si_v8/overlap_score.relabel_causal).

Closing sections: about 300 words (Conclusion 60, Limitations 187 in three paragraphs, Ethics 54).

Body: 8 pages (Conclusion ends on page 8). Build: TinyTeX pdflatex + bibtex, no errors, no undefined references,
no overfull boxes.
