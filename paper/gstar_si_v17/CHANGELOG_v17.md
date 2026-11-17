# v17 (2026-11-17), from v16

Response to the v16 mock review, with the user's decisions (plan: ../gstar_si_v17_revision_plan.zh-CN.md).
No new experiments. Untouched: abstract, introduction, related work, method, Figures 1 and 2.

- CRF evidence (4.5 Playback-layer components; sections/results.tex): only the one-sample comparison is used. Under
  the same rate rule, the PA Talker trained on the CRF posterior raises tcpASR-BLEU on every set and ends speech
  0.15-0.23 s earlier (Table 5, R3 against R2), most on AliMeeting. No significance markers, no extra table.
- Table 5 (ablation): bold now marks the best value of each column (v16 bolded the whole MIST row, although R3 has the
  lower En->Zh EndOffset, 2.90 vs 2.91); caption names R3 as the PA Talker under R2's rate rule.
- Computation: 4.1 states that the simultaneous evaluation is computation-excluded, as SimulEval's default, with a
  computation-aware replay in Appendix C; Table 1 caption: LAAL computation-excluded. Appendix C paragraph renamed
  "Computation-aware analysis": keeps the per-second compute (Thinker 0.65 / 0.80 / 1.20 s, PA Talker
  0.18 / 0.26 / 0.31 s) and the cause (prefix re-encoding, not real time); the median block-end delay clause
  (3.4 / 2.9 / 52.8 s) is removed. Limitations: one sentence. Precedent: SimulS2ST-Omni reports non-CA latency in the
  body and CA latency / RTF (RTF > 1 at its lowest tier) in its appendix F.
- Pauses removed everywhere (user decision): Table 2 column, 4.3 sentence, Limitations sentence, appendix Table 12
  columns (>= 0.5 s and mid-sentence) and G.2 sentence, locked-test Table 15 column.
- Overlap study (appendix G.6): the offline rows are removed; what remained duplicated Table 4, so the whole former
  Table 18 and the offline-relabelling paragraph are removed; the twelve confirmatory tests remain (now Table 18).
- Appendix layout: \clearpage before the appendix (no appendix table among the references); placeins \FloatBarrier
  before every appendix section and subsection; appendix floats [!htbp] / [!t]; Table 6 placed after its first
  mention; float-only pages top-aligned.

Body: 8 pages (Conclusion ends on page 8). Build: TinyTeX pdflatex + bibtex (placeins installed via tlmgr; Overleaf has
it), no errors, no undefined references, no overfull boxes.
