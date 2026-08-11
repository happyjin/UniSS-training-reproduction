# Runtime-parity streaming v9

This is a runtime-only, lossless optimization of the completed v8 checkpoint.
It does not retrain or mutate v8.  The semantic codes selected by the natural
length posterior and their `END_SEMANTIC` marker are appended in one causal
forward rather than two.  The strict evaluator keeps all quality and latency
gates unchanged and writes to a new report directory.
