from __future__ import annotations

import logging  # Added import
from typing import Union, List, Dict, Optional  # Union is used in the type hint
import subprocess
from pathlib import Path
import re  # For hex string validation


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
        if val_lower == "true":
            return True
        if val_lower == "false":
            return False
        raise ValueError(f"Cannot convert string '{value}' to boolean. Expected 'true' or 'false'.")
    # If it's not bool, None, or str, it's an unexpected type for this conversion
    raise TypeError(
        f"Cannot convert value of type {type(value)} to boolean. Expected str, bool, or None."
    )


def commit_all_changes(cwd: str, message: str = "Auto-commit") -> str | None:
    """Commit all changes in the given directory using git and return the commit hash.

    If the directory is not a git repository or there are no changes to commit,
    the function silently returns ``None``.
    """
    repo_path = Path(cwd)
    if not (repo_path / ".git").exists():
        return None
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        diff_proc = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd)
        if diff_proc.returncode == 0:
            return None
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
        logging.info(f"Auto-commit id: {commit_id}")
        return commit_id
    except Exception:
        return None


def get_commit_history(cwd: str, max_count: int = 50) -> List[Dict[str, str]]:
    """
    Retrieves the commit history for a Git repository.

    Args:
        cwd: The directory of the Git repository.
        max_count: The maximum number of commits to retrieve.

    Returns:
        A list of dictionaries, where each dictionary contains the 'hash' and 'message'
        of a commit. Returns an empty list if the directory is not a Git repository
        or if an error occurs.
    """
    repo_path = Path(cwd)
    if not (repo_path / ".git").exists():
        logging.error(f"{cwd} is not a git repository.")
        return []
    try:
        cmd = ["git", "log", "--pretty=format:%H %s", f"--max-count={max_count}"]
        result = subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.PIPE)
        history = []
        for line in result.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                history.append({"hash": parts[0], "message": parts[1]})
            else:
                # Handle cases where commit message might be missing (shouldn't happen with format)
                history.append({"hash": parts[0], "message": ""})
        return history
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting commit history for {cwd}: {e.stderr}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting commit history for {cwd}: {e}")
        return []


def revert_to_commit(cwd: str, commit_hash: str) -> bool:
    """
    Reverts the Git repository to a specific commit.

    Args:
        cwd: The directory of the Git repository.
        commit_hash: The hash of the commit to revert to.

    Returns:
        True if the revert was successful, False otherwise.
    """
    repo_path = Path(cwd)
    if not (repo_path / ".git").exists():
        logging.error(f"{cwd} is not a git repository.")
        return False
    if not re.fullmatch(r"[0-9a-f]{4,40}", commit_hash):  # Basic hex string check
        logging.error(f"Invalid commit hash format: {commit_hash}")
        return False
    try:
        cmd = ["git", "reset", "--hard", commit_hash]
        result = subprocess.run(
            cmd, cwd=cwd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            logging.info(f"Successfully reverted to commit {commit_hash} in {cwd}")
            return True
        else:
            logging.error(
                f"Failed to revert to commit {commit_hash} in {cwd}: {result.stderr.decode().strip()}"
            )
            return False
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while reverting to commit {commit_hash} in {cwd}: {e}"
        )
        return False


def revert_to_state_before_commit(cwd: str, commit_hash: str) -> bool:
    """
    Reverts the Git repository to the state before a specific commit.

    Args:
        cwd: The directory of the Git repository.
        commit_hash: The hash of the commit whose parent state to revert to.

    Returns:
        True if the revert was successful, False otherwise.
    """
    repo_path = Path(cwd)
    if not (repo_path / ".git").exists():
        logging.error(f"{cwd} is not a git repository.")
        return False
    if not re.fullmatch(r"[0-9a-f]{4,40}", commit_hash):  # Basic hex string check
        logging.error(f"Invalid commit hash format: {commit_hash}")
        return False
    try:
        commit_to_revert_to = f"{commit_hash}^"
        cmd = ["git", "reset", "--hard", commit_to_revert_to]
        result = subprocess.run(
            cmd, cwd=cwd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            logging.info(
                f"Successfully reverted to state before commit {commit_hash} (to {commit_to_revert_to}) in {cwd}"
            )
            return True
        else:
            logging.error(
                f"Failed to revert to state before commit {commit_hash} in {cwd}: {result.stderr.decode().strip()}"
            )
            return False
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while reverting to state before {commit_hash} in {cwd}: {e}"
        )
        return False


def get_initial_commit(cwd: str) -> Optional[str]:
    """
    Retrieves the hash of the initial commit in a Git repository.

    Args:
        cwd: The directory of the Git repository.

    Returns:
        The hash of the initial commit as a string, or None if not found or an error occurs.
    """
    repo_path = Path(cwd)
    if not (repo_path / ".git").exists():
        logging.error(f"{cwd} is not a git repository.")
        return None
    try:
        cmd = ["git", "rev-list", "--max-parents=0", "HEAD"]
        result = subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.PIPE)
        initial_commit_hash = result.strip().split("\n")[-1]  # Get the first commit
        if not initial_commit_hash:  # handle empty output case, though unlikely with HEAD
            logging.warning(f"No initial commit found for {cwd}")
            return None
        return initial_commit_hash
    except subprocess.CalledProcessError as e:
        # It's possible a repo exists but has no commits yet.
        if (
            "does not have any commits yet" in e.stderr.lower()
            or "bad default revision 'head'" in e.stderr.lower()
        ):
            logging.warning(f"Repository {cwd} has no commits yet.")
        else:
            logging.error(f"Error getting initial commit for {cwd}: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting initial commit for {cwd}: {e}")
        return None
