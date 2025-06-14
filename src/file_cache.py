import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_IGNORE_DIRS = ['.git', '.hg', '.svn', '__pycache__', 'node_modules', '.vscode', '.idea']
DEFAULT_IGNORE_FILES = ['.DS_Store']

class FileCache:
    """
    Scans a root directory for all files and stores their relative paths.
    Provides a method to refresh this cache.
    """
    def __init__(self, root_dir: str, initial_scan: bool = True):
        """
        Initializes the FileCache.

        Args:
            root_dir: The root directory to scan.
            initial_scan: Whether to perform an initial scan upon instantiation.
        """
        self.root_dir = os.path.abspath(root_dir)
        self.file_paths: list[str] = []
        self.ignore_dirs = set(DEFAULT_IGNORE_DIRS)
        self.ignore_files = set(DEFAULT_IGNORE_FILES)

        if not os.path.isdir(self.root_dir):
            logger.warning(f"Root directory for FileCache does not exist or is not a directory: {self.root_dir}")
            # Potentially raise an error, but for now, allow creation with an empty list.
            # This might be useful if the directory is expected to be created later.
            self.file_paths = []
            return

        if initial_scan:
            self.scan_directory()

    def _is_ignored(self, entry_name: str, is_dir: bool) -> bool:
        """Checks if a directory or file entry should be ignored."""
        if is_dir:
            return entry_name in self.ignore_dirs
        else:
            return entry_name in self.ignore_files

    def scan_directory(self) -> list[str]:
        """
        Scans the root directory (and its subdirectories) for all files.

        Updates `self.file_paths` with a list of relative file paths from the root_dir.
        Ignores common version control directories and specified ignore patterns.

        Returns:
            A list of relative file paths.
        """
        logger.info(f"Starting file scan in directory: {self.root_dir}")
        found_paths: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir, topdown=True):
            # Filter out ignored directories from further traversal
            dirnames[:] = [d for d in dirnames if not self._is_ignored(d, True)]

            for filename in filenames:
                if self._is_ignored(filename, False):
                    continue

                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, self.root_dir)
                found_paths.append(relative_path)

        self.file_paths = sorted(list(set(found_paths))) # Sort and remove duplicates
        logger.info(f"File scan completed. Found {len(self.file_paths)} files.")
        return self.file_paths

    def refresh(self) -> list[str]:
        """
        Re-scans the project directory to update the at-mention autocomplete cache.

        Returns:
            The updated list of relative file paths.
        """
        logger.info("Refreshing file cache...")
        if not os.path.isdir(self.root_dir):
            logger.warning(f"Cannot refresh: Root directory for FileCache does not exist or is not a directory: {self.root_dir}")
            self.file_paths = []
            return []
        return self.scan_directory()

    def get_paths(self) -> list[str]:
        """
        Returns the cached list of file paths.
        """
        return self.file_paths

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    logging.basicConfig(level=logging.INFO)
    # Create a dummy directory structure for testing
    if not os.path.exists("temp_test_dir"):
        os.makedirs("temp_test_dir/subdir1/.git") # .git folder
        os.makedirs("temp_test_dir/subdir1/subsubdir")
        os.makedirs("temp_test_dir/subdir2")

        with open("temp_test_dir/file1.txt", "w") as f: f.write("test")
        with open("temp_test_dir/.DS_Store", "w") as f: f.write("test") # ignored file
        with open("temp_test_dir/subdir1/file2.py", "w") as f: f.write("test")
        with open("temp_test_dir/subdir1/subsubdir/file3.md", "w") as f: f.write("test")
        with open("temp_test_dir/subdir2/file4.txt", "w") as f: f.write("test")
        with open("temp_test_dir/subdir1/.git/config", "w") as f: f.write("test") # ignored

    cache = FileCache("./temp_test_dir")
    print("\nInitial scan:")
    for path in cache.get_paths():
        print(path)

    # Simulate adding a new file
    with open("temp_test_dir/subdir2/new_file.txt", "w") as f: f.write("new")

    cache.refresh()
    print("\nAfter refresh:")
    for path in cache.get_paths():
        print(path)

    # Clean up dummy directory
    import shutil
    # shutil.rmtree("temp_test_dir") # Comment out to inspect

    # Test with a non-existent directory
    non_existent_cache = FileCache("./non_existent_dir", initial_scan=False)
    print(f"\nCache for non-existent_dir (no initial scan): {non_existent_cache.get_paths()}")
    non_existent_cache.refresh() # Should log a warning
    print(f"Cache for non-existent_dir after refresh: {non_existent_cache.get_paths()}")

    non_existent_cache_scan = FileCache("./non_existent_dir_scan_init") # Should log a warning
    print(f"\nCache for non_existent_dir_scan_init (initial scan): {non_existent_cache_scan.get_paths()}")

    # Test with a file instead of a directory
    file_as_dir_cache = FileCache("./temp_test_dir/file1.txt") # Should log a warning
    print(f"\nCache for file1.txt as root: {file_as_dir_cache.get_paths()}")

    # Clean up dummy directory if not already commented out
    if os.path.exists("temp_test_dir"):
         shutil.rmtree("temp_test_dir")
         print("\nCleaned up temp_test_dir.")
