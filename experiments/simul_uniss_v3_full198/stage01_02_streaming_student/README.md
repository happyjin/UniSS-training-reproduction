# Stage 1/2 — full-data streaming students

Both launchers reuse the v2 DDP implementations and all eight GPUs. The token
student reads the full merged schedule with a distributed random sampler. The
audio student uses the Stage 0 stratified manifest. Their step budgets remain
explicit research budgets; unlike Stage 3/4/6 they are not claimed as a full
epoch over 19.8 million schedule records.
