"""The URI parsing and byte-range reading the whole data path rests on."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from experiments.uniss_omni_s2st_v1.data.wds_resolver import (
    WdsRef,
    WdsResolveError,
    available_shards,
    parse_wds_uri,
    read_audio,
    read_range,
    resolve_path,
)

URI = (
    "wds://unist/audio/processed/shards/dataset=unist/"
    "process_version=unist_bicodec_qwen3_v2/rank03/train-00001-00000.tar"
    "?offset=512&length=12720&format=flac"
)


def test_parses_the_path_after_process_version():
    ref = parse_wds_uri(URI)
    assert ref.relative_path == "unist_bicodec_qwen3_v2/rank03/train-00001-00000.tar"
    assert (ref.offset, ref.length, ref.audio_format) == (512, 12720, "flac")
    assert ref.shard == "train-00001-00000.tar"


def test_resolves_under_the_audio_root_not_a_rank_directory():
    ref = parse_wds_uri(URI)
    assert str(resolve_path(ref, "/data/speech")) == (
        "/data/speech/unist_bicodec_qwen3_v2/rank03/train-00001-00000.tar"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/key?offset=0&length=10",
        "wds://unist/no/marker/here.tar?offset=0&length=10",
        URI.replace("offset=512", "offset=-1"),
        URI.replace("length=12720", "length=0"),
        URI.replace("?offset=512&length=12720", "?length=12720"),
    ],
)
def test_rejects_uris_it_cannot_resolve(uri):
    with pytest.raises(WdsResolveError):
        parse_wds_uri(uri)


def _tar_with_flac(tmp_path, waveform, sample_rate=16_000, pad=512):
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="FLAC")
    payload = buffer.getvalue()
    path = tmp_path / "unist_bicodec_qwen3_v2" / "rank03" / "train-00001-00000.tar"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * pad + payload + b"\0" * 1024)
    return path, len(payload)


def test_reads_the_range_and_decodes_it(tmp_path):
    expected = np.sin(np.linspace(0, 8, 4000)).astype("float32")
    _, length = _tar_with_flac(tmp_path, expected)
    ref = WdsRef("unist_bicodec_qwen3_v2/rank03/train-00001-00000.tar", 512, length, "flac")

    assert len(read_range(ref, tmp_path)) == length
    waveform, sample_rate = read_audio(ref, tmp_path)
    assert sample_rate == 16_000
    assert waveform.shape == expected.shape
    assert np.allclose(waveform, expected, atol=1e-3)


def test_a_short_read_raises_rather_than_decoding_clipped_audio(tmp_path):
    _, length = _tar_with_flac(tmp_path, np.zeros(4000, dtype="float32"), pad=512)
    # Ask for more than the file holds past the offset.
    ref = WdsRef(
        "unist_bicodec_qwen3_v2/rank03/train-00001-00000.tar",
        512,
        length + 1_000_000,
        "flac",
    )
    with pytest.raises(WdsResolveError, match="wanted"):
        read_range(ref, tmp_path)


def test_available_shards_finds_tars_at_any_depth(tmp_path):
    (tmp_path / "a" / "rank00").mkdir(parents=True)
    (tmp_path / "a" / "rank00" / "train-1.tar").write_bytes(b"")
    (tmp_path / "b.tar").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    assert available_shards(tmp_path) == {"train-1.tar", "b.tar"}
