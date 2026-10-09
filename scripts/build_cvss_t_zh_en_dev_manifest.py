"""Build the CVSS-T zh/en *dev* pending manifest, for training-time curves.

The only CVSS-T material staged on this box is the test split, which is also
the number the project reports. Watching it during training would contaminate
it, so training needs a disjoint split to curve against.

This assembles one from two sources that are already the project's own:

  - the Chinese source audio and transcripts, from the same pinned
    ``fixie-ai/covost2`` mirror revision the test split came from, but its
    ``validation`` rows (``download_cvss_t_zh_source_dev_fixie.sh``);
  - the English target speech and reference text, from ``dev`` / ``dev.tsv``
    inside the official ``cvss_t_zh_en_v1.0`` archive.

The reference text deliberately comes from CVSS-T's ``dev.tsv`` rather than
the parquet's ``translation`` column: the tsv carries the lowercased,
unpunctuated form the test references use, and a dev curve scored against a
differently-normalised reference is not comparable to the test number.

Output is a *pending* manifest in the schema ``evaluation.cvss_t.canonicalize``
consumes, so dev audio goes through exactly the same 16 kHz mono PCM16 path as
test audio rather than a parallel implementation of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def read_translations(tsv: Path) -> dict[str, str]:
    """``<common_voice_filename>\\t<english reference>``, no header."""
    rows: dict[str, str] = {}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, _, text = line.partition("\t")
        text = text.strip()
        if name and text:
            rows[name] = text
    return rows


def export_clips(parquet_dir: Path, clips_dir: Path) -> dict[str, dict]:
    """Write the mp3 bytes out of the parquets and index them by id."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    for shard in sorted(parquet_dir.glob("validation-*.parquet")):
        table = pq.read_table(shard, columns=["audio", "sentence", "id"])
        for row in table.to_pylist():
            sample_id = row["id"]
            sentence = (row["sentence"] or "").strip()
            audio = row["audio"] or {}
            payload = audio.get("bytes")
            if not sample_id or not sentence or not payload:
                continue
            path = clips_dir / f"{sample_id}.mp3"
            if not path.is_file() or path.stat().st_size != len(payload):
                path.write_bytes(payload)
            found[sample_id] = {"path": path, "sentence": sentence}
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet-dir",
        default="/opt/dlami/nvme/jasonleeeli/CVSS/source/"
        "common_voice_v4_zh-CN_dev_fixie_parquet",
    )
    ap.add_argument(
        "--clips-dir",
        default="/opt/dlami/nvme/jasonleeeli/CVSS/source/"
        "common_voice_v4_zh-CN_dev/clips",
    )
    ap.add_argument(
        "--cvss-dev-root",
        default="/opt/dlami/nvme/jasonleeeli/CVSS/extracted/cvss_t_zh_en_v1.0",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cvss_root = Path(args.cvss_dev_root)
    translations = read_translations(cvss_root / "dev.tsv")
    clips = export_clips(Path(args.parquet_dir), Path(args.clips_dir))

    rows: list[dict] = []
    missing_target = missing_source = 0
    for sample_id, clip in sorted(clips.items()):
        filename = f"{sample_id}.mp3"
        reference = translations.get(filename)
        target_wav = cvss_root / "dev" / f"{filename}.wav"
        if reference is None:
            missing_target += 1
            continue
        if not target_wav.is_file():
            missing_source += 1
            continue
        rows.append(
            {
                "id": filename,
                "common_voice_filename": filename,
                "source_zh_audio_path": str(clip["path"]),
                "source_zh_text": clip["sentence"],
                "target_en_audio_path": str(target_wav),
                "target_en_text": reference,
                "source_zh_duration_seconds": float(sf.info(clip["path"]).duration),
                "target_en_duration_seconds": float(sf.info(target_wav).duration),
                "source_available": True,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{len(rows)} dev pairs"
        f"  (clips {len(clips)}, tsv {len(translations)},"
        f" no reference {missing_target}, no target wav {missing_source})"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
