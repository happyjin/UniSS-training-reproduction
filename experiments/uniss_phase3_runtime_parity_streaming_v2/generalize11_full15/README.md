# Runtime-parity streaming generalize11 full15

V10 learned low-latency natural WRITE/EOS behavior, but its semantic head was
trained on only five 18k packs (128 sessions) and collapsed to repeated unit
IDs on a held-out utterance.  This isolated experiment restarts from the
completed dense-aligned fixed-15 checkpoint and trains only the natural-length
parallel semantic content/length head on all 59,576 bilingual dense packs.

The successful Phase3 model, streaming frontend, action head, text path, and
speaker path stay frozen.  Training covers the full fixed-15 trajectory set
once with a deterministic global permutation over complete packs.  Internal
READ/WRITE order remains unchanged.  One percent replay is retained only for
schedule compatibility; the frozen base receives no replay gradient.

Success is not defined by training loss.  Candidate checkpoints must pass the
real PCM held-out runtime gate with natural WRITE, first source-time WRITE and
first wall-clock PCM below one second, natural EOS, no revision, RTF below one,
and correct translation/audio.
