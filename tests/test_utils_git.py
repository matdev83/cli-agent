import pytest
import subprocess
from pathlib import Path
import shutil
import os # For setting dummy git user

# Functions to test
from src.utils import (
    get_commit_history,
    revert_to_commit,
    revert_to_state_before_commit,
    get_initial_commit,
    commit_all_changes
)

@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """
    Creates a temporary directory, initializes a git repository in it,
    and creates an initial commit.
    Yields the Path object of the temporary repository.
    Cleans up the temporary directory after the test.
    """
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    try:
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        # Configure a dummy user for commits to avoid issues with global git config
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
        # Allow pushing to the current branch without setting upstream explicitly (for testing simplicity)
        subprocess.run(["git", "config", "push.default", "current"], cwd=repo_dir, check=True)


        # Create an initial commit
        (repo_dir / "README.md").write_text("Initial commit")
        subprocess.run(["git", "add", "README.md"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, check=True, capture_output=True)

        yield repo_dir
    finally:
        # shutil.rmtree(repo_dir) # tmp_path fixture handles cleanup
        pass


def _run_git_command(cwd: Path, *args) -> subprocess.CompletedProcess:
    """Helper to run git commands."""
    return subprocess.run(["git"] + list(args), cwd=cwd, check=True, capture_output=True, text=True)

def _get_current_commit_hash(repo_dir: Path) -> str:
    return _run_git_command(repo_dir, "rev-parse", "HEAD").stdout.strip()

# --- Tests for get_commit_history ---

def test_get_commit_history_single_commit(git_repo: Path):
    history = get_commit_history(str(git_repo))
    assert len(history) == 1
    assert history[0]['message'] == "Initial commit"
    assert len(history[0]['hash']) == 40 # Standard git hash length

def test_get_commit_history_multiple_commits(git_repo: Path):
    (git_repo / "file1.txt").write_text("content1")
    _run_git_command(git_repo, "add", "file1.txt")
    _run_git_command(git_repo, "commit", "-m", "Commit 2")

    (git_repo / "file2.txt").write_text("content2")
    _run_git_command(git_repo, "add", "file2.txt")
    _run_git_command(git_repo, "commit", "-m", "Commit 3")

    history = get_commit_history(str(git_repo))
    assert len(history) == 3
    assert history[0]['message'] == "Commit 3"
    assert history[1]['message'] == "Commit 2"
    assert history[2]['message'] == "Initial commit"

def test_get_commit_history_max_count(git_repo: Path):
    for i in range(5):
        (git_repo / f"file{i}.txt").write_text(f"content{i}")
        _run_git_command(git_repo, "add", f"file{i}.txt")
        _run_git_command(git_repo, "commit", "-m", f"Commit {i+2}")

    history = get_commit_history(str(git_repo), max_count=3)
    assert len(history) == 3
    assert history[0]['message'] == "Commit 6" # Most recent
    assert history[1]['message'] == "Commit 5"
    assert history[2]['message'] == "Commit 4"

def test_get_commit_history_empty_repo_after_init(tmp_path: Path):
    # Test with a repo that has been initialized but has no commits
    empty_repo_dir = tmp_path / "empty_repo"
    empty_repo_dir.mkdir()
    _run_git_command(empty_repo_dir, "init")
    # Configure user for this repo too, to avoid global config dependency
    _run_git_command(empty_repo_dir, "config", "user.name", "Test User")
    _run_git_command(empty_repo_dir, "config", "user.email", "test@example.com")

    history = get_commit_history(str(empty_repo_dir))
    # git log on a new repo (no commits) produces an error.
    # The function should catch this and return an empty list.
    assert history == []


def test_get_commit_history_non_git_directory(tmp_path: Path):
    non_git_dir = tmp_path / "not_a_repo"
    non_git_dir.mkdir()
    history = get_commit_history(str(non_git_dir))
    assert history == []

def test_get_commit_history_subprocess_error(git_repo: Path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "cmd", stderr="git error")))
    history = get_commit_history(str(git_repo))
    assert history == []
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(Exception("Some other error")))
    history = get_commit_history(str(git_repo))
    assert history == []

# --- Tests for commit_all_changes ---

def test_commit_all_changes_new_file(git_repo: Path):
    file_path = git_repo / "new_file.txt"
    file_path.write_text("Hello world")

    commit_hash = commit_all_changes(str(git_repo), "Commit new_file.txt")
    assert commit_hash is not None
    assert len(commit_hash) == 40

    status_output = _run_git_command(git_repo, "status", "--porcelain").stdout.strip()
    assert status_output == "" # Should be clean

    history = get_commit_history(str(git_repo), max_count=1)
    assert history[0]['hash'] == commit_hash
    assert history[0]['message'] == "Commit new_file.txt"

def test_commit_all_changes_modified_file(git_repo: Path):
    readme_path = git_repo / "README.md"
    readme_path.write_text("Updated README")

    commit_hash = commit_all_changes(str(git_repo), "Update README.md")
    assert commit_hash is not None
    history = get_commit_history(str(git_repo), max_count=1)
    assert history[0]['hash'] == commit_hash
    assert history[0]['message'] == "Update README.md"

def test_commit_all_changes_no_changes(git_repo: Path):
    commit_hash = commit_all_changes(str(git_repo), "Attempt commit with no changes")
    assert commit_hash is None

def test_commit_all_changes_custom_message(git_repo: Path):
    (git_repo / "another.txt").write_text("data")
    custom_msg = "My custom commit message"
    commit_hash = commit_all_changes(str(git_repo), custom_msg)
    assert commit_hash is not None
    history = get_commit_history(str(git_repo), max_count=1)
    assert history[0]['message'] == custom_msg

def test_commit_all_changes_non_git_directory(tmp_path: Path):
    non_git_dir = tmp_path / "not_a_repo_for_commit"
    non_git_dir.mkdir()
    (non_git_dir / "file.txt").write_text("some data")
    commit_hash = commit_all_changes(str(non_git_dir), "Test")
    assert commit_hash is None

# --- Tests for revert_to_commit ---

def test_revert_to_commit_valid(git_repo: Path):
    c1_hash = _get_current_commit_hash(git_repo) # Initial commit

    file1_path = git_repo / "file1.txt"
    file1_path.write_text("State for C2")
    _run_git_command(git_repo, "add", file1_path.name)
    _run_git_command(git_repo, "commit", "-m", "Commit C2")
    c2_hash = _get_current_commit_hash(git_repo)

    readme_path = git_repo / "README.md"
    readme_path.write_text("State for C3")
    _run_git_command(git_repo, "add", readme_path.name)
    _run_git_command(git_repo, "commit", "-m", "Commit C3")
    # c3_hash = _get_current_commit_hash(git_repo)

    assert revert_to_commit(str(git_repo), c2_hash) is True
    assert _get_current_commit_hash(git_repo) == c2_hash
    assert file1_path.exists()
    assert file1_path.read_text() == "State for C2"
    assert readme_path.read_text() == "Initial commit" # README was from C1, file1.txt added in C2

    assert revert_to_commit(str(git_repo), c1_hash) is True
    assert _get_current_commit_hash(git_repo) == c1_hash
    assert not file1_path.exists()
    assert readme_path.read_text() == "Initial commit"

def test_revert_to_commit_invalid_hash(git_repo: Path):
    assert revert_to_commit(str(git_repo), "invalidhash123") is False
    # Also test a valid-looking but non-existent hash
    assert revert_to_commit(str(git_repo), "a" * 40) is False

def test_revert_to_commit_non_git_directory(tmp_path: Path):
    non_git_dir = tmp_path / "no_repo_here"
    non_git_dir.mkdir()
    assert revert_to_commit(str(non_git_dir), "anyhash") is False

def test_revert_to_commit_invalid_hash_format(git_repo: Path):
    assert revert_to_commit(str(git_repo), "short") is False # Too short
    assert revert_to_commit(str(git_repo), "g" * 40) is False # Invalid characters

# --- Tests for revert_to_state_before_commit ---

def test_revert_to_state_before_commit_valid(git_repo: Path):
    # c1_hash = _get_current_commit_hash(git_repo) # Initial commit (README.md: "Initial commit")

    file1_path = git_repo / "file1.txt"
    file1_path.write_text("content for C2")
    _run_git_command(git_repo, "add", file1_path.name)
    _run_git_command(git_repo, "commit", "-m", "Commit C2")
    c2_hash = _get_current_commit_hash(git_repo)

    readme_path = git_repo / "README.md"
    readme_path.write_text("updated for C3")
    _run_git_command(git_repo, "add", readme_path.name)
    _run_git_command(git_repo, "commit", "-m", "Commit C3")
    c3_hash = _get_current_commit_hash(git_repo)

    # Revert to state before C3 (should be state of C2)
    assert revert_to_state_before_commit(str(git_repo), c3_hash) is True
    assert _get_current_commit_hash(git_repo) == c2_hash
    assert file1_path.read_text() == "content for C2"
    assert readme_path.read_text() == "Initial commit" # README was from C1, file1 was C2

def test_revert_to_state_before_initial_commit(git_repo: Path):
    initial_commit_hash = get_initial_commit(str(git_repo))
    assert initial_commit_hash is not None

    # Reverting to before the initial commit. `git reset --hard HASH^` will fail for the first commit.
    # The function should return False.
    assert revert_to_state_before_commit(str(git_repo), initial_commit_hash) is False
    # Ensure repo is still in a valid state (at initial commit)
    assert _get_current_commit_hash(git_repo) == initial_commit_hash

def test_revert_to_state_before_commit_invalid_hash(git_repo: Path):
    assert revert_to_state_before_commit(str(git_repo), "invalidhash123") is False
    assert revert_to_state_before_commit(str(git_repo), "b" * 40) is False # Valid format, non-existent

def test_revert_to_state_before_commit_non_git_directory(tmp_path: Path):
    non_git_dir = tmp_path / "no_repo_for_revert_before"
    non_git_dir.mkdir()
    assert revert_to_state_before_commit(str(non_git_dir), "anyhash") is False

# --- Tests for get_initial_commit ---

def test_get_initial_commit_multiple_commits(git_repo: Path):
    first_commit_hash = _get_current_commit_hash(git_repo)
    # Add a few more commits
    (git_repo / "fileA.txt").write_text("A")
    _run_git_command(git_repo, "add", "fileA.txt")
    _run_git_command(git_repo, "commit", "-m", "Commit A")
    (git_repo / "fileB.txt").write_text("B")
    _run_git_command(git_repo, "add", "fileB.txt")
    _run_git_command(git_repo, "commit", "-m", "Commit B")

    assert get_initial_commit(str(git_repo)) == first_commit_hash

def test_get_initial_commit_single_commit(git_repo: Path):
    first_commit_hash = _get_current_commit_hash(git_repo)
    assert get_initial_commit(str(git_repo)) == first_commit_hash

def test_get_initial_commit_empty_repo(tmp_path: Path):
    # Repo initialized but no commits
    empty_repo_dir = tmp_path / "empty_repo_for_initial"
    empty_repo_dir.mkdir()
    _run_git_command(empty_repo_dir, "init")
    _run_git_command(empty_repo_dir, "config", "user.name", "Test User") # Important for some git versions
    _run_git_command(empty_repo_dir, "config", "user.email", "test@example.com")
    assert get_initial_commit(str(empty_repo_dir)) is None

def test_get_initial_commit_non_git_directory(tmp_path: Path):
    non_git_dir = tmp_path / "no_repo_for_initial"
    non_git_dir.mkdir()
    assert get_initial_commit(str(non_git_dir)) is None

def test_get_initial_commit_subprocess_error(git_repo, monkeypatch):
    # Test CalledProcessError
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "cmd", stderr="git error")))
    assert get_initial_commit(str(git_repo)) is None

    # Test general Exception
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(Exception("Some other error")))
    assert get_initial_commit(str(git_repo)) is None

    # Test specific error message "does not have any commits yet"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "cmd", stderr="fatal: your current branch 'master' does not have any commits yet")))
    assert get_initial_commit(str(git_repo)) is None

    # Test specific error message "bad default revision 'HEAD'"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "cmd", stderr="fatal: bad default revision 'HEAD'")))
    assert get_initial_commit(str(git_repo)) is None

# Example of how to run pytest from python, can be useful for debugging
if __name__ == '__main__':
    # This allows running the tests with `python tests/test_utils_git.py`
    # You might need to adjust PYTHONPATH or sys.path if src is not found
    # For example, if tests/ is a sibling of src/:
    # import sys
    # sys.path.insert(0, str(Path(__file__).parent.parent))
    pytest.main([__file__])
