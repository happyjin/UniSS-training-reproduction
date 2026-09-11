"""Turn the reranked winners into trajectories a fine-tune can consume.

The structural simplification this rests on
-------------------------------------------
A trajectory's source side -- ``source_glm_delta``, ``gold_source_delta``, the
PCM and GLM ranges -- comes from the source recording and is the same whichever
candidate won.  Only the target side differs.  So a winner's trajectory is the
gold trajectory with its events re-cut at the winner's own fragment boundaries
and its target fields replaced.

Re-cutting is exact rather than approximate: every fragment is emitted at a
read step, which is a multiple of the 640 ms grid the pool builder re-bins onto,
so ``chunk_windows`` sees the boundaries it would have chosen anyway.

What the target side becomes
----------------------------
``target_text_delta`` is the growth of the committed text between consecutive
spoken fragments, and ``target_semantic_delta`` the codes of that fragment.
Text that was committed while nothing was spoken lands on the next spoken
fragment, which is what the cascade itself does.

Why not simply replace gold
---------------------------
The winners are better than the model's own greedy output -- text BLEU 40.87
against 38.04 on the 400-utterance slice, and silence 0.207 against 0.281 --
but worse than gold, which is the reference itself.  So these records are meant
to be trained *alongside* the gold pool as a separate family, teaching the model
which branch of its own output space to take, not replacing the answer key.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    hash_int_sequence,
)


def merge_source(events: list[dict], start_ms: float, end_ms: float) -> dict:
    """Source-side fields for the window ``(start_ms, end_ms]``."""
    members = [
        e for e in events
        if float(e["source_end_ms"]) > start_ms and float(e["source_end_ms"]) <= end_ms
    ]
    if not members:
        return {}
    first, last = members[0], members[-1]
    return {
        "source_start_ms": int(first["source_start_ms"]),
        "source_end_ms": int(last["source_end_ms"]),
        "source_pcm_start": int(first["source_pcm_start"]),
        "source_pcm_end": int(last["source_pcm_end"]),
        "source_glm_start": int(first["source_glm_start"]),
        "source_glm_end": int(last["source_glm_end"]),
        "source_glm_delta": [int(v) for e in members for v in e["source_glm_delta"]],
        "gold_source_word_start": int(first["gold_source_word_start"]),
        "gold_source_word_end": int(last["gold_source_word_end"]),
        "gold_source_delta": "".join(str(e["gold_source_delta"]) for e in members),
        "gold_source_prefix": str(first["gold_source_prefix"]),
        "v1_source_delta": last.get("v1_source_delta"),
        "v1_source_prefix": first.get("v1_source_prefix"),
        "target_support_end_ms": last.get("target_support_end_ms"),
        "alignment_confidence": float(last.get("alignment_confidence") or 0.0),
        "noise_severity": str(last.get("noise_severity") or "none"),
    }


def rebuild(gold: dict, starts_ms: list[float], texts: list[str],
            codes: list[list[int]]) -> dict | None:
    """The gold trajectory with the winner's target side."""
    events = list(gold["events"])
    if not events or not starts_ms:
        return None
    out_events: list[dict] = []
    previous_text = ""
    code_cursor = 0
    previous_ms = -1.0
    for index, (end_ms, text, delta) in enumerate(zip(starts_ms, texts, codes)):
        last = index == len(starts_ms) - 1
        # the final window absorbs whatever source events remain, because the
        # last fragment is emitted once the source is exhausted
        window_end = float(end_ms) if not last else float(
            max(float(e["source_end_ms"]) for e in events)
        )
        source = merge_source(events, previous_ms, window_end)
        if not source:
            previous_ms = window_end
            continue
        text_delta = text[len(previous_text):] if text.startswith(previous_text) else text
        out_events.append({
            "event_index": len(out_events),
            **source,
            "target_text_delta": text_delta,
            "target_text_prefix": previous_text,
            "target_semantic_start": code_cursor,
            "target_semantic_end": code_cursor + len(delta),
            "target_semantic_delta": [int(v) for v in delta],
            "source_final": bool(last),
            "target_final": bool(last),
        })
        previous_text = text
        code_cursor += len(delta)
        previous_ms = window_end
    if not out_events:
        return None
    record = {k: v for k, v in gold.items() if k != "events"}
    record["events"] = out_events
    record["full_translation"] = previous_text
    record["normalized_translation"] = previous_text
    record["target_semantic_length"] = code_cursor
    # target_semantic_sha256 is *checked* against the codes by the schema, so it
    # has to be recomputed rather than dropped -- the first attempt removed both
    # hashes as "honest" and the pool builder rejected every record.
    # phase3_teacher_sha256 only has its hex format validated; it identifies the
    # teacher cache the source side came from, which these records still share
    # with their gold originals, so it is carried through unchanged.
    record["target_semantic_sha256"] = hash_int_sequence(
        [int(v) for e in out_events for v in e["target_semantic_delta"]]
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="gold trajectory jsonl")
    parser.add_argument("--picks", required=True, help="rerank_candidates output")
    parser.add_argument("--codes-root", required=True, help="directory of code dumps")
    parser.add_argument("--arm-to-tag", required=True,
                        help="python format string mapping an arm name to a code file glob")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    picks = {p["sample_id"]: p for p in
             json.loads(Path(args.picks).read_text(encoding="utf-8"))["picks"]}
    # Codes, keyed by (arm, sample_id).  Two layouts exist and both are in use:
    # separate code_dump files named ``<tag>_s<shard>.json`` directly under the
    # root, and rollout manifests at ``<arm>/MANIFEST.json`` carrying the codes
    # inline because ``--dump-codes`` was passed.  The 20k run used the second
    # and the first version of this loader found nothing, skipping all 20000
    # records silently, so it now reads whichever is there.
    codes: dict[tuple[str, str], dict] = {}
    root = Path(args.codes_root)
    for path in sorted(root.glob("*.json")):
        tag = path.stem.split("_s")[0]
        for row in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            codes[(tag, row["sample_id"])] = row
    for path in sorted(root.glob("*/MANIFEST.json")):
        tag = path.parent.name
        for row in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            if row.get("codes") is None:
                continue
            codes[(tag, row["sample_id"])] = {
                "sample_id": row["sample_id"],
                "fragments": row["codes"],
                "starts_ms": row["starts_ms"],
                "texts": row["texts"],
            }
    if not codes:
        raise SystemExit(
            f"no codes under {root}: expected <tag>_s<shard>.json files or "
            "<arm>/MANIFEST.json with a 'codes' field from --dump-codes"
        )
    written = 0
    skipped = 0
    lengths: list[int] = []
    with open(args.output, "w", encoding="utf-8") as out:
        with open(args.gold, encoding="utf-8") as fh:
            for line in fh:
                gold = json.loads(line)
                pick = picks.get(gold["sample_id"])
                if pick is None:
                    continue
                tag = args.arm_to_tag.format(arm=pick["chosen_arm"])
                row = codes.get((tag, gold["sample_id"]))
                if row is None:
                    skipped += 1
                    continue
                record = rebuild(gold, [float(v) for v in row["starts_ms"]],
                                 [str(v) for v in row["texts"]],
                                 [[int(c) for c in f] for f in row["fragments"]])
                if record is None:
                    skipped += 1
                    continue
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                lengths.append(len(record["events"]))
    print(f"wrote {written} trajectories, skipped {skipped}")
    if lengths:
        print(f"  events per trajectory: median {statistics.median(lengths):.0f}, "
              f"min {min(lengths)}, max {max(lengths)}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
