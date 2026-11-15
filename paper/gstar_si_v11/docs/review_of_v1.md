# Review of v1: "Who Said What, in Their Own Voice: End-to-End Speaker-Attributed Simultaneous Speech-to-Speech Translation of Meetings"

Written as a strict ACL reviewer, against `paper/gstar_si_v1/main.pdf` (commit 1e2cfbd).

## Summary

The paper proposes MIST. MIST interleaves the posteriors of a frozen streaming diarizer, projected to tokens, with the acoustic tokens of Qwen3-Omni. The Thinker is fine-tuned with LoRA to write speaker-attributed, time-stamped Chinese translations of AMI meeting windows.

A second stage trains a read/wait policy from alignment-derived chunk deadlines. A UniSS Talker then speaks each speaker slot with a voice prompt cut from the diarizer's solo regions.

Main results on AMI En→Zh:

| Setting | System | Meeting-level cpBLEU |
|---|---|---|
| Offline | MIST | 42.3 |
| Offline | prompted Qwen3-Omni | 12.8 |
| Simultaneous (LAAL 2.4 s) | MIST-Simul | 39.8 |
| Simultaneous (LAAL 2.4 s) | MIST + LocalAgreement | 29.5 |
| Ablation | MIST without the speaker stream | 18.3 |

## Strengths

- **S1.** The task is new and relevant: speaker-attributed simultaneous S2ST of multi-party meetings, evaluated on both text and speech.
- **S2.** The method section is clear. Each component is motivated and every equation states symbols and shapes.
- **S3.** The ablation without the speaker stream is clean. It shows that the speaker stream contributes meeting-level identity, not translation quality.
- **S4.** The analysis is honest, including the repetition artefact of LocalAgreement and the Talker misread rates.

## Weaknesses (major)

### W1. No pipeline baseline
The paper motivates an end-to-end model against "diarize → recognise → translate → synthesise" cascades, but never compares with one.

The only offline baseline is a prompted model that cannot represent speakers at all, which is a strawman. Two baselines are needed:
- **(a)** diarizer turns translated by the base model;
- **(b)** the same turns translated by a model fine-tuned on the same data.

Baseline (b) separates "end-to-end joint modelling" from "fine-tuning on in-domain data with this reference style". Without it, the central claim (§1, contribution 2) is not supported.

### W2. Translation quality is confounded with reference style
The references are machine translations chosen from three MT systems by QE. MIST is fine-tuned on references from the same pipeline, the prompted baseline is not. The jump in speaker-blind BLEU from 29.0 to 43.7 is therefore mostly adaptation to the reference style.

This must be stated plainly, and every claim about translation *quality* should rest on comparisons between systems trained on the same data. A human-verified subset or a reference-free metric would help. If neither is possible, the limitation must be explicit in the main text, not only under Limitations.

### W3. No significance tests and few test meetings
The test set has only 16 meetings, and windows within a meeting are correlated. A meeting-level cluster bootstrap (or a test over windows and over meetings) is needed for every claim:

| Comparison | Difference to test |
|---|---|
| oracle diarizer vs. diarizer | −0.21 / +1.00 |
| offline vs. simultaneous | 2.5 |
| MIST-Simul vs. LA2 | 10.3 |

### W4. Oracle segmentation at test time
Test windows are AMI reference utterance groups. They start and end at utterance boundaries, exclude windows whose utterances straddle the boundary, and drop windows that fail data review. A deployed system would cut windows with its own diarizer.

The model is trained partly on such diarizer windows, but never evaluated on them. Report results on diarizer-defined windows, or at least discuss the size of the oracle advantage.

### W5. Latency evaluation is thin
The latency evaluation has five gaps:
- There is one operating point (Δ = 1 s) per system and no latency–quality trade-off.
- LAAL is computation-unaware.
- When the listener actually hears the speech (speech-level lag after the queue on one channel) is not reported, although the paper is about S2ST.
- Whether the 1.04 s diarizer look-ahead is included in the reported LAAL needs a precise statement.
- Decoding cost is mentioned once ("2.5× faster"), but no real-time factor is given.

### W6. Fairness of the streaming comparison
MIST-Simul receives 2,000 additional training steps (about 129k rows) beyond MIST (offline), whereas LocalAgreement uses the offline model as is. Part of the 10.3-point gain may come from the extra training rather than from the learned commit policy.

A control is needed: MIST-Simul decoded offline (a single read). It shows how much of the gain is policy and how much is model.

### W7. The title promises "in their own voice", but voice is never measured
Speech is scored only with ASR-BLEU. Three things are missing:
- a speaker-similarity measure between each output speaker's synthesised voice and the attributed source speaker;
- a check of whether voices are confused between speakers;
- any measure of speech quality.

The voice prompts are cut from the whole session, which is not causal in the simultaneous setting. This must be stated in the main text.

### W8. Novelty needs sharper positioning
Interleaving diarizer posteriors with audio tokens is G-STAR's design. Serialised, time-stamped multi-speaker output is SOT / TagSpeech. Learning wait tokens from alignment-derived deadlines is close to existing read/write supervision.

The contributions should be framed as follows: the task and benchmark; transferring speaker-stream fusion from ASR to *simultaneous translation and speech*, with the specific streaming problems this raises (a causal speaker stream with look-ahead, chunk deadlines, speaker-first serialisation for the Talker); and the analysis of which component provides what. The paper should state explicitly what differs from G-STAR.

### W9. Presentation and format
- The main text runs to about 9.5 pages; the ACL limit is 8.
- In Figure 1 the Chinese example text is not rendered, and labels overlap ("s_{1..M}" over the projector box).
- The figure is too small to read in print.

## Weaknesses (minor)

- **m1.** "Speaker accuracy" is really speaker-*count* accuracy; rename it.
- **m2.** The diarizer's DER of 27.7% is surprising until one reads that AMI utterances include pauses. Report standard DER details (collar, overlap scoring) or explain the gap in one sentence in §5.
- **m3.** The analysis claims that MIST-Simul attributes better "because of finer supervision". This is a hypothesis; present it as one or test it.
- **m4.** The Talker "misread" rate mixes Talker and ASR errors; say so.
- **m5.** There are no qualitative examples. One window with overlap, with reference, MIST output and baseline output, would help readers.
- **m6.** Only one training seed is used and run-to-run variance is not reported; state this.
- **m7.** In Eq. (14), state the frame-index convention (frame f covers ((f−1)δ, fδ]).
- **m8.** The abstract is dense with numbers. Keep the key ones and state the speech result in one clause.

## Questions to the authors

- **Q1.** How does MIST compare with a cascade that has the same translation training (W1)?
- **Q2.** How do results change on diarizer-defined windows (W4)?
- **Q3.** What is the latency the listener experiences (W5)?
- **Q4.** How similar are the synthesised voices to the attributed speakers, and how often is a voice confused (W7)?

## Scores (v1)

| Criterion | Score |
|---|---|
| Soundness | 2.5 / 5 (claims rest on a strawman baseline, no significance, oracle segmentation) |
| Excitement | 3.5 / 5 (new and useful task, a convincing ablation) |
| Overall assessment | 2.5 (borderline, below acceptance in current form) |
| Confidence | 4 |

## What would raise the score

Addressing W1, W3, W5–W7 with experiments and W2, W4, W8, W9 with clear statements (or experiments where possible) would make this a solid accept.
