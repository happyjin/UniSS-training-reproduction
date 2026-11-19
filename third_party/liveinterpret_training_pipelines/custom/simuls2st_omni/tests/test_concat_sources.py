from simuls2st_omni.train.datasets.concat import ConcatSourcesDataset


class _Rows:
    def __init__(self, tag: str, n: int):
        self.rows = [f"{tag}{i}" for i in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def code_length(self, i):
        return i + 1


def test_repeat_layout_and_lookup():
    ds = ConcatSourcesDataset([_Rows("a", 4), _Rows("b", 3)], repeats=[1, 2])
    assert len(ds) == 4 + 6
    assert [ds[i] for i in range(len(ds))] == ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b0", "b1", "b2"]
    assert ds.code_length(9) == 3
    assert ds.token_length(9) == 3


def test_eval_split_is_proportional_and_disjoint_from_all_copies():
    ds = ConcatSourcesDataset([_Rows("a", 90), _Rows("b", 10)], repeats=[1, 3])
    train, evals = ds.train_eval_split(10)
    eval_rows = {ds[i] for i in evals}
    assert eval_rows == {f"a{i}" for i in range(81, 90)} | {"b9"}
    assert not eval_rows & {ds[i] for i in train}
    assert len(train) == 81 + 3 * 9
