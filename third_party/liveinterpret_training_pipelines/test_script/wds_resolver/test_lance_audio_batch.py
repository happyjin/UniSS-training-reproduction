"""Unit tests for lance:// batch audio / codec resolution helpers."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

from custom import wds_audio


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows
        self.num_rows = len(rows)

    def to_pylist(self):
        return list(self._rows)


class _FakeReader:
    def __init__(self, rows_by_id):
        self.rows_by_id = rows_by_id
        self.calls = []

    def take_by_stable_rowid(self, rowids, columns=None):
        self.calls.append((tuple(rowids), tuple(columns or ())))
        out = []
        for rid in rowids:
            row = dict(self.rows_by_id[int(rid)])
            row['_rowid'] = int(rid)
            if columns:
                filtered = {'_rowid': row['_rowid']}
                for c in columns:
                    filtered[c] = row.get(c)
                row = filtered
            out.append(row)
        return _FakeTable(out)


class LanceAudioBatchTest(unittest.TestCase):

    def setUp(self):
        wds_audio._LANCE_READER = None
        wds_audio._LANCE_READER_PID = None
        wds_audio._LANCE_BYTE_READER = None
        wds_audio._FORK_ORPHANS.clear()
        self.env = mock.patch.dict(os.environ, {
            'LANCE_TRAIN_SOURCE_URI': '/tmp/fake.lance',
            'WDS_AUDIO_MODE': 'auto',
            # These cases drive the in-process resolvers directly (they patch
            # _get_lance_reader); the helper-child hop is covered separately.
            'LANCE_MAIN_PROCESS_ISOLATION': '0',
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        wds_audio._LANCE_READER = None
        wds_audio._LANCE_READER_PID = None
        wds_audio._LANCE_BYTE_READER = None

    def test_audio_batch_coalesces_unique_rowids_per_side(self):
        reader = _FakeReader({
            1: {'source_audio_bytes': b'a1', 'source_audio_shard_uri': None,
                'source_audio_byte_offset': None, 'source_audio_byte_length': None},
            2: {'source_audio_bytes': b'a2', 'source_audio_shard_uri': None,
                'source_audio_byte_offset': None, 'source_audio_byte_length': None},
            3: {'target_audio_bytes': b't3', 'target_audio_shard_uri': None,
                'target_audio_byte_offset': None, 'target_audio_byte_length': None},
        })

        def fake_import():
            def resolve_audio(row, side, byte_reader=None):
                return row.get(f'{side}_audio_bytes')
            return type('R', (), {}), resolve_audio

        with mock.patch.object(wds_audio, '_get_lance_reader', return_value=reader), \
             mock.patch.object(wds_audio, '_import_lance', side_effect=fake_import):
            out = wds_audio.resolve_lance_audio_batch([
                'lance://1/source',
                'lance://2/source',
                'lance://1/source',  # dup
                'lance://3/target',
                '/legacy.wav',
            ])

        self.assertEqual(out[0], b'a1')
        self.assertEqual(out[1], b'a2')
        self.assertEqual(out[2], b'a1')
        self.assertEqual(out[3], b't3')
        self.assertEqual(out[4], '/legacy.wav')
        # one take per side; source take de-duped to (1, 2)
        sides = {cols[0].split('_')[0] for _, cols in reader.calls}
        self.assertEqual(sides, {'source', 'target'})
        source_call = next(c for c in reader.calls if c[1][0].startswith('source_'))
        self.assertEqual(source_call[0], (1, 2))

    def test_resolve_audio_references_mixes_lance_and_passthrough(self):
        reader = _FakeReader({
            9: {'source_audio_bytes': b'wav9', 'source_audio_shard_uri': None,
                'source_audio_byte_offset': None, 'source_audio_byte_length': None},
        })

        def fake_import():
            def resolve_audio(row, side, byte_reader=None):
                return row.get(f'{side}_audio_bytes')
            return type('R', (), {}), resolve_audio

        with mock.patch.object(wds_audio, '_get_lance_reader', return_value=reader), \
             mock.patch.object(wds_audio, '_import_lance', side_effect=fake_import):
            out = wds_audio.resolve_audio_references([
                'lance://9/source',
                b'already',
                '/fsx/x.wav',
            ])
        self.assertEqual(out[0], b'wav9')
        self.assertEqual(out[1], b'already')
        self.assertEqual(out[2], '/fsx/x.wav')

    def test_training_batch_resolves_all_hot_fields_with_one_take(self):
        reader = _FakeReader({
            11: {
                'source_input_features': [0.25] * 256,
                'source_feature_attention_mask': [1, 1],
                'source_input_features_frames': 2,
                'source_speaker_embedding': [0.5] * 2048,
                'codes': [[1] * 16, [2] * 16],
            },
            12: {
                'source_input_features': [0.75] * 128,
                'source_feature_attention_mask': [1],
                'source_input_features_frames': 1,
                'source_speaker_embedding': [1.5] * 2048,
                'codes': [[3] * 16],
            },
        })
        rows = [
            {
                'source_input_features': 'lance://11/features',
                'source_speaker_embedding': 'lance://11/speaker',
                'target_codec_codes': 'lance://11/codes',
            },
            {
                'source_input_features': 'lance://12/features',
                'source_speaker_embedding': 'lance://12/speaker',
                'target_codec_codes': 'lance://12/codes',
            },
        ]

        with mock.patch.object(wds_audio, '_get_lance_reader', return_value=reader):
            returned = wds_audio.resolve_lance_training_rows_batch(rows)

        self.assertIs(returned, rows)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(reader.calls[0][0], (11, 12))
        self.assertEqual(reader.calls[0][1], (
            'source_input_features',
            'source_feature_attention_mask',
            'source_input_features_frames',
            'source_speaker_embedding',
            'codes',
        ))
        self.assertEqual(rows[0]['source_input_features']['frames'], 2)
        self.assertEqual(len(rows[0]['source_input_features']['flat']), 256)
        self.assertIsInstance(rows[0]['source_input_features']['flat'], np.ndarray)
        np.testing.assert_array_equal(
            rows[1]['source_speaker_embedding'], np.float32(1.5) * np.ones(2048))
        np.testing.assert_array_equal(
            rows[0]['target_codec_codes'], np.asarray([[1] * 16, [2] * 16]))

    def test_training_batch_sorts_unique_rowids_before_take(self):
        reader = _FakeReader({
            11: {
                'source_input_features': [0.25] * 256,
                'source_feature_attention_mask': [1, 1],
                'source_input_features_frames': 2,
                'source_speaker_embedding': [0.5] * 2048,
                'codes': [[1] * 16],
            },
            12: {
                'source_input_features': [0.75] * 128,
                'source_feature_attention_mask': [1],
                'source_input_features_frames': 1,
                'source_speaker_embedding': [1.5] * 2048,
                'codes': [[3] * 16],
            },
        })
        rows = [
            {
                'source_input_features': 'lance://12/features',
                'source_speaker_embedding': 'lance://12/speaker',
                'target_codec_codes': 'lance://12/codes',
            },
            {
                'source_input_features': 'lance://11/features',
                'source_speaker_embedding': 'lance://11/speaker',
                'target_codec_codes': 'lance://11/codes',
            },
        ]
        with mock.patch.object(wds_audio, '_get_lance_reader', return_value=reader):
            wds_audio.resolve_lance_training_rows_batch(rows)
        self.assertEqual(reader.calls[0][0], (11, 12))
        self.assertEqual(rows[0]['source_input_features']['frames'], 1)
        self.assertEqual(rows[1]['source_input_features']['frames'], 2)

    def test_arrow_take_table_stays_numpy_and_aligns_by_rowid(self):
        pa = __import__('pyarrow')
        table = pa.table({
            '_rowid': pa.array([12, 11], type=pa.int64()),
            'source_input_features': pa.array(
                [[0.75] * 128, [0.25] * 256],
                type=pa.large_list(pa.float32())),
            'source_feature_attention_mask': pa.array(
                [[1], [1, 1]], type=pa.large_list(pa.int8())),
            'source_input_features_frames': pa.array([1, 2], type=pa.int32()),
            'source_speaker_embedding': pa.array(
                [[1.5] * 2048, [0.5] * 2048],
                type=pa.list_(pa.float32(), 2048)),
            'codes': pa.array(
                [[[3] * 16], [[1] * 16, [2] * 16]],
                type=pa.large_list(pa.large_list(pa.int16()))),
        })
        by_id = wds_audio._rows_from_take_table(table)
        self.assertEqual(sorted(by_id), [11, 12])
        packed = wds_audio._pack_source_features(by_id[11])
        self.assertIsInstance(packed['flat'], np.ndarray)
        self.assertEqual(packed['flat'].dtype, np.float32)
        np.testing.assert_array_equal(packed['flat'], np.float32(0.25) * np.ones(256))
        np.testing.assert_array_equal(
            by_id[12]['source_speaker_embedding'],
            np.float32(1.5) * np.ones(2048))
        np.testing.assert_array_equal(
            by_id[11]['codes'], np.asarray([[1] * 16, [2] * 16]))

    def test_training_batch_keeps_already_materialized_values(self):
        rows = [{
            'source_input_features': {
                'flat': [0.0] * 128, 'mask': [1], 'frames': 1,
                'feature_size': 128,
            },
            'source_speaker_embedding': [0.0] * 2048,
            'target_codec_codes': [[4] * 16],
        }]
        with mock.patch.object(
                wds_audio, '_get_lance_reader',
                side_effect=AssertionError('reader must not be opened')):
            self.assertIs(wds_audio.resolve_lance_training_rows_batch(rows), rows)


class LancePrefetchNestedTest(unittest.TestCase):
    """Flatten helpers live in dataset.py (needs swift); re-check the contract here."""

    def test_flatten_unflatten_contract(self):
        # Mirror custom.dataset._flatten_audio_lists / _unflatten_audio_lists.
        def flatten(audio_lists):
            flat, lengths = [], []
            for cell in audio_lists:
                if cell is None:
                    lengths.append(0)
                    continue
                items = list(cell) if isinstance(cell, (list, tuple)) else [cell]
                lengths.append(len(items))
                flat.extend(items)
            return flat, lengths

        def unflatten(flat, lengths):
            out, idx = [], 0
            for n in lengths:
                out.append(flat[idx:idx + n])
                idx += n
            return out

        nested = [['lance://1/source'], ['lance://2/source', 'x'], None, []]
        flat, lengths = flatten(nested)
        self.assertEqual(flat, ['lance://1/source', 'lance://2/source', 'x'])
        self.assertEqual(lengths, [1, 2, 0, 0])
        self.assertEqual(unflatten(flat, lengths), [
            ['lance://1/source'], ['lance://2/source', 'x'], [], [],
        ])

    def test_get_lance_reader_reopens_when_pid_changes(self):
        inherited = object()
        wds_audio._LANCE_READER = inherited
        wds_audio._LANCE_READER_PID = -1
        opened = object()

        class _Ctor:
            def __new__(cls, *args, **kwargs):
                return opened

        with mock.patch.dict(os.environ, {'LANCE_TRAIN_SOURCE_URI': '/tmp/fake.lance'}), \
             mock.patch.object(wds_audio, '_import_lance', return_value=(_Ctor, None)):
            got = wds_audio._get_lance_reader()
        self.assertIs(got, opened)
        self.assertIs(wds_audio._FORK_ORPHANS[-1], inherited)
        self.assertEqual(wds_audio._LANCE_READER_PID, os.getpid())

    def test_get_lance_reader_reopens_when_pid_is_unset(self):
        inherited = object()
        wds_audio._LANCE_READER = inherited
        wds_audio._LANCE_READER_PID = None
        opened = object()

        class _Ctor:
            def __new__(cls, *args, **kwargs):
                return opened

        with mock.patch.dict(os.environ, {'LANCE_TRAIN_SOURCE_URI': '/tmp/fake.lance'}), \
             mock.patch.object(wds_audio, '_import_lance', return_value=(_Ctor, None)):
            got = wds_audio._get_lance_reader()
        self.assertIs(got, opened)
        self.assertIs(wds_audio._FORK_ORPHANS[-1], inherited)

    def test_reset_native_io_after_fork_orphans_reader(self):
        inherited = object()
        wds_audio._LANCE_READER = inherited
        wds_audio._LANCE_READER_PID = os.getpid()
        wds_audio.reset_native_io_after_fork()
        self.assertIsNone(wds_audio._LANCE_READER)
        self.assertIsNone(wds_audio._LANCE_READER_PID)
        self.assertIs(wds_audio._FORK_ORPHANS[-1], inherited)


class LanceMainProcessIsolationTest(unittest.TestCase):
    """pylance must stay out of the trainer so forked workers import it fresh."""

    def setUp(self):
        wds_audio._CHILD_RESOLVER = None
        wds_audio._CHILD_RESOLVER_PID = None
        wds_audio._CHILD_FALLBACK_WARNED = False
        wds_audio._FORK_ORPHANS.clear()

    def tearDown(self):
        live = [wds_audio._CHILD_RESOLVER] + list(wds_audio._FORK_ORPHANS)
        for resolver in live:
            proc = getattr(resolver, '_proc', None)
            if proc is None:
                continue
            proc.stdin.close()
            proc.stdout.close()
            proc.wait(timeout=30)
        wds_audio._CHILD_RESOLVER = None
        wds_audio._CHILD_RESOLVER_PID = None
        wds_audio._FORK_ORPHANS.clear()

    def test_disabled_by_env(self):
        with mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '0'}):
            self.assertFalse(wds_audio._should_isolate_lance())

    def test_inactive_inside_helper_child(self):
        with mock.patch.dict(os.environ, {wds_audio._CHILD_ENV_MARKER: '1'}):
            self.assertFalse(wds_audio._should_isolate_lance())

    def test_inactive_in_dataloader_worker(self):
        fake_tud = mock.Mock()
        fake_tud.get_worker_info.return_value = object()
        with mock.patch.dict(sys.modules, {'torch.utils.data': fake_tud}), \
             mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '1'}):
            self.assertFalse(wds_audio._should_isolate_lance())

    def test_resolver_dispatches_to_child_without_local_reader(self):
        calls = []

        class _Resolver:
            def call(self, name, args, kwargs):
                calls.append((name, args, kwargs))
                return ['from-child']

        with mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '1'}), \
             mock.patch.object(wds_audio, '_get_child_resolver', return_value=_Resolver()), \
             mock.patch.object(wds_audio, '_get_lance_reader',
                               side_effect=AssertionError('trainer must not open Lance')):
            out = wds_audio.resolve_lance_codec_codes_batch(['lance://7/codes'])

        self.assertEqual(out, ['from-child'])
        self.assertEqual(calls[0][0], 'resolve_lance_codec_codes_batch')
        self.assertEqual(calls[0][1], (['lance://7/codes'],))

    def test_row_resolver_merges_child_result_into_caller_rows(self):
        class _Resolver:
            def call(self, name, args, kwargs):
                return [{'source_speaker_embedding': 'resolved'}]

        rows = [{'source_speaker_embedding': 'lance://3/speaker'}]
        with mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '1'}), \
             mock.patch.object(wds_audio, '_get_child_resolver', return_value=_Resolver()):
            out = wds_audio.resolve_lance_training_rows_batch(rows)

        self.assertIs(out, rows)
        self.assertEqual(rows[0]['source_speaker_embedding'], 'resolved')

    def test_transport_failure_falls_back_in_process(self):
        def _boom():
            raise wds_audio._ChildResolverError('spawn failed')

        with mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '1'}), \
             mock.patch.object(wds_audio, '_get_child_resolver', side_effect=_boom):
            out = wds_audio.resolve_lance_codec_codes_batch(['/legacy.wav'])

        self.assertEqual(out, ['/legacy.wav'])
        self.assertTrue(wds_audio._CHILD_FALLBACK_WARNED)

    def test_callee_error_propagates_instead_of_retrying_locally(self):
        class _Resolver:
            def call(self, name, args, kwargs):
                raise ValueError('lance:// row 9 not found')

        with mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '1'}), \
             mock.patch.object(wds_audio, '_get_child_resolver', return_value=_Resolver()), \
             mock.patch.object(wds_audio, '_get_lance_reader',
                               side_effect=AssertionError('trainer must not open Lance')):
            with self.assertRaises(ValueError):
                wds_audio.resolve_lance_codec_codes_batch(['lance://9/codes'])

    def test_injected_byte_reader_stays_in_process(self):
        with mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '1'}), \
             mock.patch.object(wds_audio, '_get_child_resolver',
                               side_effect=AssertionError('must not spawn a child')):
            out = wds_audio.resolve_lance_audio_batch(
                ['/legacy.wav'], byte_reader=lambda *a: b'x')
        self.assertEqual(out, ['/legacy.wav'])

    def test_forked_child_resolver_is_orphaned_not_closed(self):
        inherited = object()
        wds_audio._CHILD_RESOLVER = inherited
        wds_audio._CHILD_RESOLVER_PID = -1
        wds_audio.discard_forked_child_resolver()
        self.assertIsNone(wds_audio._CHILD_RESOLVER)
        self.assertIs(wds_audio._FORK_ORPHANS[-1], inherited)

    def test_real_child_round_trip(self):
        """End-to-end spawn + pickle hop; passthrough args need no pylance."""
        with mock.patch.dict(os.environ, {'LANCE_MAIN_PROCESS_ISOLATION': '1'}), \
             mock.patch.object(wds_audio, '_get_lance_reader',
                               side_effect=AssertionError('trainer must not open Lance')):
            out = wds_audio.resolve_lance_codec_codes_batch(['/legacy.wav', None])
        self.assertEqual(out, ['/legacy.wav', None])
        self.assertFalse(wds_audio._CHILD_FALLBACK_WARNED)
        self.assertIsNotNone(wds_audio._CHILD_RESOLVER)


if __name__ == '__main__':
    unittest.main()
