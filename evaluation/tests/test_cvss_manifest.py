import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation import cvss_manifest


class CVSSManifestTest(unittest.TestCase):
    def test_read_translation_tsv(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.tsv"
            path.write_text("a.mp3\thello world\n", encoding="utf-8")
            self.assertEqual(cvss_manifest.read_cvss_translation_tsv(path), [("a.mp3", "hello world")])

    def test_requires_exact_paper_pair_count(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test.tsv").write_text("a.mp3\thello\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                cvss_manifest.build_cvss_manifest(root, root / "out")


if __name__ == "__main__":
    unittest.main()
