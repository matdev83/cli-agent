import unittest
import os
import shutil
import tempfile
import logging
from pathlib import Path

# Adjust import path if necessary, assuming src is a package or in PYTHONPATH
from file_cache import FileCache, DEFAULT_IGNORE_DIRS

# Suppress logging output during tests unless specifically testing for it
logging.basicConfig(level=logging.CRITICAL)


class TestFileCache(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for each test
        self.test_dir_root = Path(tempfile.mkdtemp())
        # print(f"Test directory created: {self.test_dir_root}")

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir_root)
        # print(f"Test directory removed: {self.test_dir_root}")

    def _create_files(self, file_list):
        """Helper to create files and directories in the test_dir_root."""
        for file_path_str in file_list:
            full_path = self.test_dir_root / file_path_str
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"content of {file_path_str}")

    def test_initial_scan_empty_directory(self):
        cache = FileCache(str(self.test_dir_root), initial_scan=True)
        self.assertEqual(cache.get_paths(), [])

    def test_initial_scan_with_files(self):
        files_to_create = [
            "file1.txt",
            "subdir/file2.py",
            "subdir/subsubdir/file3.md",
        ]
        self._create_files(files_to_create)

        cache = FileCache(str(self.test_dir_root), initial_scan=True)
        expected_paths = sorted(
            [
                "file1.txt",
                os.path.join("subdir", "file2.py"),
                os.path.join("subdir", "subsubdir", "file3.md"),
            ]
        )
        # Convert to OS-specific paths for comparison if needed, relpath should handle it
        self.assertEqual(cache.get_paths(), expected_paths)

    def test_scan_ignores_default_dirs_and_files(self):
        files_to_create = [
            "file_ok.txt",
            ".git/config",  # Ignored dir
            "subdir/.DS_Store",  # Ignored file
            "__pycache__/cachefile.pyc",  # Ignored dir
            "node_modules/module/package.json",  # Ignored dir
            "subdir/actual_file.py",
            ".vscode/settings.json",  # Ignored dir
            ".idea/workspace.xml",  # Ignored dir
        ]
        self._create_files(files_to_create)

        # Create the ignored directories explicitly as _create_files only makes parents
        for ignored_dir_name in DEFAULT_IGNORE_DIRS:
            if "/" not in ignored_dir_name:  # Simple dir name
                (self.test_dir_root / ignored_dir_name).mkdir(exist_ok=True)

        cache = FileCache(str(self.test_dir_root), initial_scan=True)
        expected_paths = sorted(
            [
                "file_ok.txt",
                os.path.join("subdir", "actual_file.py"),
            ]
        )
        self.assertEqual(cache.get_paths(), expected_paths)

    def test_scan_handles_os_specific_paths(self):
        # Test with paths that might look different on Win vs Unix if not handled well
        # os.path.join should normalize, and relpath should produce consistent format
        # relative to the root.
        files_to_create = [
            os.path.join("one", "two", "file_a.txt"),
            os.path.join("one", "file_b.txt"),
        ]
        self._create_files(files_to_create)
        cache = FileCache(str(self.test_dir_root))
        expected = sorted(
            [
                os.path.join("one", "two", "file_a.txt"),
                os.path.join("one", "file_b.txt"),
            ]
        )
        self.assertEqual(cache.get_paths(), expected)

    def test_refresh_updates_cache(self):
        self._create_files(["file1.txt"])
        cache = FileCache(str(self.test_dir_root), initial_scan=True)
        self.assertEqual(cache.get_paths(), ["file1.txt"])

        self._create_files(["file2.txt", os.path.join("subdir", "new_file.py")])
        cache.refresh()
        expected_paths = sorted(
            [
                "file1.txt",
                "file2.txt",
                os.path.join("subdir", "new_file.py"),
            ]
        )
        self.assertEqual(cache.get_paths(), expected_paths)

        # Test removal
        (self.test_dir_root / "file1.txt").unlink()
        cache.refresh()
        expected_paths_after_removal = sorted(
            [
                "file2.txt",
                os.path.join("subdir", "new_file.py"),
            ]
        )
        self.assertEqual(cache.get_paths(), expected_paths_after_removal)

    def test_non_existent_root_dir_initial_scan_false(self):
        non_existent_path = str(self.test_dir_root / "non_existent_dir")
        # Expect a warning to be logged, but not an exception here.
        # The logger in FileCache is logging.getLogger(__name__)
        # We can capture logs if needed, but for now, just check behavior.
        with self.assertLogs(logger="file_cache", level="WARNING") as cm:
            cache = FileCache(non_existent_path, initial_scan=False)
            self.assertTrue(
                any(
                    "Root directory for FileCache does not exist" in message
                    for message in cm.output
                )
            )
        self.assertEqual(cache.get_paths(), [])
        self.assertEqual(cache.root_dir, os.path.abspath(non_existent_path))

    def test_non_existent_root_dir_initial_scan_true(self):
        non_existent_path = str(self.test_dir_root / "non_existent_dir_scan")
        with self.assertLogs(logger="file_cache", level="WARNING") as cm:
            cache = FileCache(non_existent_path, initial_scan=True)  # Default is True
            self.assertTrue(
                any(
                    "Root directory for FileCache does not exist" in message
                    for message in cm.output
                )
            )
        self.assertEqual(cache.get_paths(), [])

    def test_refresh_non_existent_root_dir(self):
        non_existent_path = str(self.test_dir_root / "another_non_existent")
        cache = FileCache(non_existent_path, initial_scan=False)  # Create it without scan
        self.assertEqual(cache.get_paths(), [])

        with self.assertLogs(logger="file_cache", level="WARNING") as cm:
            result = cache.refresh()
            self.assertTrue(
                any(
                    "Cannot refresh: Root directory for FileCache does not exist" in message
                    for message in cm.output
                )
            )
        self.assertEqual(result, [])
        self.assertEqual(cache.get_paths(), [])

    def test_root_dir_is_a_file(self):
        file_as_root_path = self.test_dir_root / "i_am_a_file.txt"
        file_as_root_path.write_text("I am a file, not a directory.")

        with self.assertLogs(logger="file_cache", level="WARNING") as cm:
            cache = FileCache(str(file_as_root_path), initial_scan=True)
            self.assertTrue(
                any(
                    "Root directory for FileCache does not exist or is not a directory" in message
                    for message in cm.output
                )
            )
        self.assertEqual(cache.get_paths(), [])

        # Test refresh on this cache
        with self.assertLogs(logger="file_cache", level="WARNING") as cm:
            refresh_result = cache.refresh()
            self.assertTrue(
                any(
                    "Cannot refresh: Root directory for FileCache does not exist or is not a directory"
                    in message
                    for message in cm.output
                )
            )
        self.assertEqual(refresh_result, [])
        self.assertEqual(cache.get_paths(), [])

    def test_paths_are_relative_and_sorted_and_unique(self):
        files_to_create = [
            "z_file.txt",
            "a_file.txt",
            "subdir/b_file.py",
            "subdir/a_file.py",
            "a_file.txt",  # Duplicate
        ]
        self._create_files(files_to_create)

        cache = FileCache(str(self.test_dir_root))
        expected_paths = sorted(
            [
                "a_file.txt",
                os.path.join("subdir", "a_file.py"),
                os.path.join("subdir", "b_file.py"),
                "z_file.txt",
            ]
        )
        paths = cache.get_paths()
        self.assertEqual(paths, expected_paths)
        # Check all paths are relative (don't start with / or drive letter)
        for p in paths:
            self.assertFalse(os.path.isabs(p), f"Path '{p}' should be relative.")


if __name__ == "__main__":
    unittest.main()
