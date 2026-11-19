"""Tests for the DataLoader-batch Lance resolver patch."""

import os
import unittest
from unittest import mock

from custom import lance_lazy_batch_patch


class _ColumnarDataset:
    def __init__(self, rows):
        self.rows = rows
        self.batch_fetches = []
        self.single_fetches = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, list):
            self.batch_fetches.append(tuple(index))
            keys = self.rows[0].keys()
            return {key: [self.rows[i][key] for i in index] for key in keys}
        self.single_fetches.append(index)
        return dict(self.rows[index])


def _make_lazy_class():
    class FakeLazy:
        def __init__(self, dataset, encode_func):
            self.dataset = dataset
            self.encode_func = encode_func
            self.original_calls = []

        def __getitem__(self, index):
            self.original_calls.append(index)
            return self.encode_func(self.dataset[index], return_length=True)

    return FakeLazy


class LanceLazyBatchPatchTest(unittest.TestCase):

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {
            'LANCE_TRAIN_SOURCE_URI': '/tmp/fake.lance',
            'LANCE_DATALOADER_BATCH_RESOLVE': '1',
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_getitems_batch_fetches_resolves_then_encodes_in_order(self):
        lazy_cls = _make_lazy_class()
        self.assertTrue(lance_lazy_batch_patch._install_on_class(lazy_cls))
        self.assertFalse(lance_lazy_batch_patch._install_on_class(lazy_cls))
        dataset = _ColumnarDataset([
            {'id': 0, 'source_input_features': 'lance://10/features'},
            {'id': 1, 'source_input_features': 'lance://11/features'},
            {'id': 2, 'source_input_features': 'lance://12/features'},
        ])

        def materialize(rows):
            for row in rows:
                row['source_input_features'] = {'frames': row['id'] + 1}
            return rows

        lazy = lazy_cls(
            dataset,
            lambda row, return_length=False: {
                'id': row['id'],
                'frames': row['source_input_features']['frames'],
                'return_length': return_length,
            })
        with mock.patch.object(
                lance_lazy_batch_patch, 'resolve_lance_training_rows_batch',
                side_effect=materialize) as resolver:
            output = lazy.__getitems__([2, 0])

        resolver.assert_called_once()
        self.assertEqual(dataset.batch_fetches, [(2, 0)])
        self.assertEqual(dataset.single_fetches, [])
        self.assertEqual(lazy.original_calls, [])
        self.assertEqual(output, [
            {'id': 2, 'frames': 3, 'return_length': True},
            {'id': 0, 'frames': 1, 'return_length': True},
        ])

    def test_encode_error_delegates_to_original_getitem(self):
        lazy_cls = _make_lazy_class()
        lance_lazy_batch_patch._install_on_class(lazy_cls)
        dataset = _ColumnarDataset([
            {'id': 0, 'source_input_features': 'lance://10/features'},
            {'id': 1, 'source_input_features': 'lance://11/features'},
        ])
        attempts = {0: 0, 1: 0}

        def encode(row, return_length=False):
            attempts[row['id']] += 1
            if row['id'] == 1 and attempts[1] == 1:
                raise ValueError('first encode fails')
            return {'id': row['id']}

        lazy = lazy_cls(dataset, encode)
        with mock.patch.object(
                lance_lazy_batch_patch, 'resolve_lance_training_rows_batch',
                side_effect=lambda rows: rows):
            output = lazy.__getitems__([0, 1])

        self.assertEqual(output, [{'id': 0}, {'id': 1}])
        self.assertEqual(lazy.original_calls, [1])
        self.assertEqual(attempts[1], 2)

    def test_getitems_attaches_native_mel_after_resolve(self):
        lazy_cls = _make_lazy_class()
        lance_lazy_batch_patch._install_on_class(lazy_cls)
        dataset = _ColumnarDataset([
            {'id': 0, 'source_input_features': 'lance://10/features'},
            {'id': 1, 'source_input_features': 'lance://11/features'},
        ])

        def materialize(rows):
            for row in rows:
                row['source_input_features'] = {
                    'flat': [0.0] * 128,
                    'frames': row['id'] + 1,
                    'mask': [1],
                }
            return rows

        seen = []

        def fake_attach(rows, dtype):
            seen.append((len(rows), [int(r['source_input_features']['frames'])
                                     for r in rows]))

        lazy = lazy_cls(dataset, lambda row, return_length=False: {'id': row['id']})
        with mock.patch.object(
                lance_lazy_batch_patch, 'resolve_lance_training_rows_batch',
                side_effect=materialize), \
             mock.patch.object(
                 lance_lazy_batch_patch, '_attach_native_mel_tensors',
                 side_effect=fake_attach):
            output = lazy.__getitems__([0, 1])

        self.assertEqual(output, [{'id': 0}, {'id': 1}])
        self.assertEqual(seen, [(2, [1, 2])])

    def test_disabled_path_uses_original_getitem(self):
        lazy_cls = _make_lazy_class()
        lance_lazy_batch_patch._install_on_class(lazy_cls)
        dataset = _ColumnarDataset([
            {'id': 0, 'source_input_features': 'lance://10/features'},
            {'id': 1, 'source_input_features': 'lance://11/features'},
        ])
        lazy = lazy_cls(dataset, lambda row, return_length=False: {'id': row['id']})
        with mock.patch.dict(
                os.environ, {'LANCE_DATALOADER_BATCH_RESOLVE': '0'}), \
             mock.patch.object(
                 lance_lazy_batch_patch, 'resolve_lance_training_rows_batch',
                 side_effect=AssertionError('resolver must not run')):
            output = lazy.__getitems__([1, 0])

        self.assertEqual(output, [{'id': 1}, {'id': 0}])
        self.assertEqual(lazy.original_calls, [1, 0])
        self.assertEqual(dataset.batch_fetches, [])


if __name__ == '__main__':
    unittest.main()
