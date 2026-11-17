# v15 (2026-11-17), from the user-edited gstar_si_v14_overleaf_v1

Plan: ../gstar_si_v15_revision_plan.zh-CN.md (round 3, decisions adopted). No new experiments; every number comes
from existing JSON (eval_outputs/mist_w3po_v1/joint/table1_v14, v11f_analysis) or v14 generated tables, through
scripts/make_tables_v15.py, which checks each copied number against its source table.

Untouched: abstract, introduction and Figure 1, except the "distil" wording (one phrase in the abstract, two in the
introduction, one in the Figure 1 caption). The Figure 1 image still reads "CRF-distilled Talker"; the user edits it.

Method (sections/method.tex, rewritten)
- 3.1 Problem Formulation; 3.2 Speaker-Attributed Simultaneous Translation; 3.3 Speaker Streams and Playback Queues;
  3.4 Playback Alignment: A Latency-Constrained CRF; 3.5 Training the Talker on the CRF Posterior;
  3.6 Speaker-Attributed Speech Metrics (moved from Setup).
- Each subsection opens with the problem it solves and ends with what it leaves to the next.
- Symbols defined where introduced, dimensions as letters; notation table removed; all values in Appendix B (Table 7).
- Symbol clashes fixed: time-token indices \hat a_n, \hat b_n; Talker parameters \xi; stream \mathcal{U}; deadline
  chunks u_j; layout start o_n; reference speaker \varrho.
- Backlog-aware speaking rate is now a full paragraph with Eq. (5) (rule from si_infer.py --speed-policy queue).
- No "distil": the Talker is trained on the CRF posterior (one M-step of PR-EM; cf. Hu et al. 2016).
- Moved to the appendix: projector architecture, slot mapping, chunk timing, training rows, voice-prompt thresholds,
  online windows / chunk protocol, message definitions, dual derivation, all numeric values.

Diarizer: Nemotron 3 Diarization (Medennikov et al., 2026, arXiv:2611.12246) cited in the method, setup, Figure 2,
appendix and related work; its training data include AMI and AliMeeting training recordings (disclosed in Setup and
Limitations). AliMeeting is framed as out-of-domain for every model we train.

Setup: about half a page; details to Appendices A, B, C, E, F.

Results: five result floats in the body (was two): Table 1 systems (cpASR-BLEU / tcpASR-BLEU / LAAL only, no stars,
AliMeeting marked out-of-domain), Table 2 speaker tokens under overlap, Table 3 delay decomposition + speech properties,
Table 4 ablation, Table 5 robustness (second recogniser, COMET, locked test). Text: one question per subsection, few
numbers, mechanism tied to the method's equations. EndOffset appears only in comparisons with B.
The quality-LAAL figure (Figure 5) went to Appendix G to keep the body within eight pages.

Fixes: Table 1 and ablation now agree (31.65: half-up rounding of 31.645); delta_w rows dropped from the speech table;
Set column filled in the significance tables; ASR-BLEU renamed tcpASR-BLEU in appendix tables; DiariST mode column
"simultaneous"; new Table 10 with the MIST-vs-system tests behind Table 1; LAAL statements checked against it
(SeamlessStreaming is earlier only on AMI Zh->En; X-Translator on AMI Zh->En and AliMeeting).

Streaming -> simultaneous for our task, model, mode and title; proper names and other work's streaming systems kept;
the two occurrences in the introduction are the user's and were left as they are.

Body: 8 pages (Conclusion ends on page 8). Build: TinyTeX pdflatex + bibtex (Overleaf default), no errors, no
undefined references, no overfull boxes.
