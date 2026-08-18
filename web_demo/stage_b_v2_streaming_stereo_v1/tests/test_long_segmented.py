import numpy as np

from web_demo.stage_b_v2_streaming_stereo_v1.long_segmented import (
    place_without_overlap,
    render_placements,
)


def test_place_without_overlap_respects_availability_and_cursor():
    placements = []
    first_start, first_end = place_without_overlap(
        placements,
        np.ones(4, dtype=np.float32),
        available_sample=3,
        cursor=0,
    )
    second_start, second_end = place_without_overlap(
        placements,
        np.ones(2, dtype=np.float32),
        available_sample=4,
        cursor=first_end,
    )
    assert (first_start, first_end) == (3, 7)
    assert (second_start, second_end) == (7, 9)
    assert np.array_equal(
        render_placements(placements, minimum_samples=5),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    )
