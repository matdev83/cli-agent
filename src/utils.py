from __future__ import annotations

from typing import Union # Union is used in the type hint
import subprocess
from pathlib import Path


def to_bool(value: Union[str, bool, None]) -> bool:
    """
    Converts a string or boolean value to a boolean, strictly.
    'true' (case-insensitive) -> True
    'false' (case-insensitive) -> False
    boolean -> itself
    None -> False (as a default for missing optional params)
    Other string values -> ValueError
    Other types -> TypeError
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        val_lower = value.lower()
        if val_lower == 'true':
            return True
        if val_lower == 'false':
            return False
        raise ValueError(f"Cannot convert string '{value}' to boolean. Expected 'true' or 'false'.")
    # If it's not bool, None, or str, it's an unexpected type for this conversion
    raise TypeError(f"Cannot convert value of type {type(value)} to boolean. Expected str, bool, or None.")


def commit_all_changes(cwd: str, message: str = "Auto-commit") -> str | None:
    """Commit all changes in the given directory using git and return the commit hash.

    If the directory is not a git repository or there are no changes to commit,
    the function silently returns ``None``.
    """
    repo_path = Path(cwd)
    if not (repo_path / ".git").exists():
        return None
    try:
        subprocess.run(["git", "add", "-A"], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        diff_proc = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd)
        if diff_proc.returncode == 0:
            return None
        subprocess.run(["git", "commit", "-m", message], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
        print(f"Auto-commit id: {commit_id}")
        return commit_id
    except Exception:
        return None

