# Runtime-parity streaming v12: causal semantic microblocks

V11 met the natural action, first-PCM, EOS and RTF gates but its independent
24-slot semantic head collapsed to one frequent unit on held-out speech.  V12
does not continue that checkpoint.  It starts from the completed dense-aligned
fixed-15 checkpoint and freezes Phase3, LoRA, frontend, action, safe-commit,
text and speaker parameters.

The only trainable module predicts target BiCodec units in four-unit causal
microblocks.  Each later microblock is conditioned on the preceding units at
their real teacher-forced Qwen positions during training and on units actually
committed to the persistent Qwen KV cache at runtime.  Within a microblock a
small causal transition consumes the previous unit.  The content classifier
is tied to the frozen Phase3 semantic embedding rows and starts as the Phase3
next-token classifier instead of an unrelated random 8192-way projection.
Mild clipped inverse-square-root weighting prevents a single frequent codec
unit from dominating the content loss.

CONTINUE/END is a learned binary posterior.  When CONTINUE is selected, all
four units are committed and the resulting Qwen hidden state starts the next
microblock.  When END is selected, a separate learned 1..4 posterior chooses
the final natural block length.  A runtime safety ceiling raises a failure; it
never truncates an unterminated prediction into a successful result.

The canary must first demonstrate learnability and natural termination on real
PCM.  The full15 run then covers all 59,576 trajectory packs once with strict
global shuffle.  Success still requires held-out translation correctness,
playable non-collapsed audio, natural WRITE/EOS, no revision, first source-time
WRITE and wall-clock PCM below one second, and RTF below one.
