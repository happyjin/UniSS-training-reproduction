# UniSS Phase3 v4 quality-first true-streaming pilot15 v8

V8 is an isolated repair for the two V7 formal failures. It always starts
from the immutable Phase3 checkpoint and never resumes a V7 checkpoint.

The V7 formal trace proved that its mean-posterior blank budget could remain
zero while framewise CTC argmax collapsed to blank. It also showed slow code
geometry drift during the 254-update LR-floor hold. V8 therefore changes only
the failed constraints:

- keep a 0.10 monotonic-seed floor after curriculum saturation;
- tighten the mean blank-posterior target to 0.20;
- add a differentiable decision-margin penalty that requires at least 80% of
  valid frames to prefer some non-blank class;
- increase codebook commitment from 0.10 to 0.30;
- increase codebook identity CE from 0.30 to 0.50;
- increase adapter residual control from 0.01 to 0.05.

All Phase3 replay, same-prefix teacher KL, data, exact global shuffle,
Megatron topology, sequence length, batch geometry, curriculum, frozen
Whisper frontend, and optimizer horizons remain unchanged.

The first V8 job is a 255-update diagnostic prefix. It reproduces the complete
127-update curriculum/optimizer clock and then holds at the LR floor for 128
updates, crossing V7's first blank-gate failure at update 231. Its canary gate
never authorizes Stage B; it only authorizes a new three-epoch formal V8 run.
