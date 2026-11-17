# v18 (2026-11-17), from v17

Response to the v17 mock review with the user's decisions (plan: ../gstar_si_v18_revision_plan.zh-CN.md, round 2).
No new experiments. Introduction, related work, Figures 1-2 unchanged; the abstract only gains a footnote.

- Abstract: footnote after "(MIST)": "Code and models will be released upon acceptance." (only release statement left).
- Method 3.6: "we will release the scorer" removed; appendix E's matching clause removed too.
- Table 1 caption: "without / with source-time constraints".
- Results 4.4: opening question now "Do the gains over B hold with another recogniser and on meetings never decoded
  before?" (no mention of test meetings used during development).
- Results 4.5, AliMeeting overlap: "recall rises by 17 points and overlapped units slightly (9.18 vs. 8.55), while all
  units are 1.2 points lower".
- Limitations (Evaluation): sentence on test meetings scored during development removed.
- Appendix: private-repository / pre-registration sentence removed; Table 18 (overlap tests, now Table 17) keeps its
  p-values but drops the verdict column (holds / tie / fails) and the caption sentence defining it; the MIST-vs-B
  test table (former Table 11) removed because its rows are the B rows of Table 10 (two interval ends differed by
  0.01 between the two bootstrap runs); locked test (Table 15) transposed to a one-column table, pauses left out.
- Appendix layout: no per-(sub)section \FloatBarrier; appendix-only float parameters (more floats per page, text fills
  around them, float pages >= 85% full, top-aligned); Supplementary Results moved before Evaluation Protocols and
  Baseline Systems so their text fills the space around its tables. Appendix 23 -> 21 pages; every column except the
  last page <= 20% empty; tables 0-3 pages after their first appendix mention.

Body: 8 pages. Build: TinyTeX pdflatex + bibtex, no errors, no undefined references, no overfull boxes.
