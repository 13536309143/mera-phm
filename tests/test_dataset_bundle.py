import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from prognostics.data import discover_data
from prognostics.dataset_bundle import (
    DATASET_NAME,
    bundle_paths,
    prepare_bundled_data,
    verify_bundled_data,
)


class DatasetBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def bundle(self, unsafe=None):
        archive, manifest, _ = bundle_paths(self.workspace)
        archive.parent.mkdir(parents=True, exist_ok=True)
        entries = {f"{DATASET_NAME}/1/time.txt": b"10\n", f"{DATASET_NAME}/Readme.txt": b"source"}
        if unsafe:
            entries[unsafe] = b"escape"
        with ZipFile(archive, "w") as zipped:
            for name, value in entries.items():
                zipped.writestr(name, value)
        manifest.write_text(
            json.dumps(
                {
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "file_count": len(entries),
                    "uncompressed_bytes": sum(len(value) for value in entries.values()),
                }
            ),
            encoding="utf-8",
        )
        return archive

    def test_extract_verify_and_repeat_without_overwrite(self):
        self.bundle()
        root = prepare_bundled_data(self.workspace)
        self.assertEqual(verify_bundled_data(self.workspace)["files_verified"], 2)
        (root / "Readme.txt").write_text("local modification", encoding="utf-8")
        self.assertEqual(prepare_bundled_data(self.workspace), root)
        self.assertEqual((root / "Readme.txt").read_text(), "local modification")
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_bundled_data(self.workspace)

    def test_bad_checksum_blocks_extraction(self):
        archive = self.bundle()
        with archive.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_bundled_data(self.workspace)
        self.assertFalse(bundle_paths(self.workspace)[2].exists())

    def test_zip_traversal_is_rejected_before_writing(self):
        self.bundle(f"{DATASET_NAME}/../../escape.txt")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            prepare_bundled_data(self.workspace)
        self.assertFalse((self.workspace / "data/raw/escape.txt").exists())
        self.assertFalse(bundle_paths(self.workspace)[2].exists())

    def test_incomplete_existing_data_is_preserved(self):
        self.bundle()
        destination = bundle_paths(self.workspace)[2]
        destination.mkdir()
        marker = destination / "local.txt"
        marker.write_text("retain", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            prepare_bundled_data(self.workspace)
        self.assertEqual(marker.read_text(), "retain")

    def test_discovery_prepares_default_and_explicit_bundled_path(self):
        self.bundle()
        with (
            patch("prognostics.data.PROJECT", self.workspace),
            patch.dict("os.environ", {"DATASET_PATH": ""}),
        ):
            root = discover_data()
        self.assertTrue((root / "1/time.txt").is_file())
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory)
            archive, manifest, _ = bundle_paths(self.workspace)
            other_archive, other_manifest, other_root = bundle_paths(other)
            other_archive.parent.mkdir(parents=True)
            other_archive.write_bytes(archive.read_bytes())
            other_manifest.write_bytes(manifest.read_bytes())
            with patch("prognostics.data.PROJECT", other):
                self.assertEqual(discover_data(str(other_root)), other_root.resolve())

    def test_checkout_without_archive_still_supports_external_data(self):
        self.assertIsNone(prepare_bundled_data(self.workspace))
        root = self.workspace / "external"
        (root / "1").mkdir(parents=True)
        (root / "1/time.txt").write_text("10", encoding="utf-8")
        with patch("prognostics.data.PROJECT", self.workspace):
            self.assertEqual(discover_data(str(root)), root.resolve())
