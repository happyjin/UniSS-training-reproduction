# UniSS Offline--Streaming S2ST English Technical Paper

This directory contains an independent English LaTeX paper covering Offline Phase 1--3, Streaming Stage A/B, and the proposed quality-aware RL/GRPO stage. It does not replace or modify the Chinese paper in `docs/uniss_streaming_s2st_technical_paper_v1/`.

## Files

- `main.tex`: complete English paper.
- `references.bib`: bibliography.
- `figures/architecture.tikz.tex`: reproducible English TikZ system diagram.
- `build/main.pdf`: compiled PDF.
- `build/main.log`: compilation log.

## Build

The local compiler is installed under the required user root:

```text
/opt/dlami/nvme/jasonleeeli/tools/tectonic-0.15.0/tectonic
```

Run:

```bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS/docs/uniss_streaming_s2st_technical_paper_en_v1
make
```

The paper explicitly separates completed experiments from proposed RL work and records all failed gates, proxy-metric limitations, and unavailable Stage A loss denominators.
