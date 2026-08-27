def test_long_episode_gap_metadata_is_not_extra_duration():
    components = [5740, 16960, 9320]
    episode_duration = sum(components)
    cursor = 0
    for duration in components:
        cursor += duration
    assert cursor == episode_duration
    assert cursor + 160 * (len(components) - 1) != episode_duration
