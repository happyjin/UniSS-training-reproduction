"""Take one family from a second pool, leaving the rest of the first alone.

Why this exists
---------------
The first rejection-sampling fine-tune concatenated gold and winner
trajectories into one jsonl, so every family saw both.  Measured on the same
3000 source utterances, the two disagree completely about when to commit text:

    gold     34.6 events, 1.4 characters each, 70.8% of events commit nothing
    winners   5.6 events, 8.0 characters each,  0.0% commit nothing

Mixed in one family that is two contradictory policies for the same audio, and
the model learned neither -- the translation collapsed from 60 characters to 7
and ASR-BLEU from 32.93 to 1.10 over 180 steps.

The winners' value was never their commit policy; it is the speech they
produce once they have committed.  So the text families stay pure gold and only
``p2st_streaming_tts`` is taken from the winners.  Family names are unchanged,
so the trainer needs no new registration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def graft(base: dict, donor: dict, families: list[str]) -> dict:
    out = json.loads(json.dumps(base))
    missing = [f for f in families if f not in donor.get("families", {})]
    if missing:
        raise SystemExit(f"donor pool has no family {missing}")
    unknown = [f for f in families if f not in out.get("families", {})]
    if unknown:
        raise SystemExit(f"base pool has no family {unknown}")
    for family in families:
        out["families"][family] = json.loads(json.dumps(donor["families"][family]))
    out["grafted_families"] = list(families)
    out["grafted_from"] = donor.get("gold")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="pool that supplies every other family")
    parser.add_argument("--donor", required=True, help="pool that supplies --family")
    parser.add_argument("--family", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    donor = json.loads(Path(args.donor).read_text(encoding="utf-8"))
    merged = graft(base, donor, args.family)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    for family, entry in merged["families"].items():
        mark = "  <- donor" if family in args.family else ""
        print(f"  {family:<28} {entry.get('samples', '?'):>8} samples{mark}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
