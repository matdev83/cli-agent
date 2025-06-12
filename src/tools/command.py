from __future__ import annotations

import subprocess
from typing import Tuple, Optional


def execute_command(
    command: str,
    requires_approval: bool,
    *,
    auto_approve: bool = False,
    timeout: Optional[int] = None,
) -> Tuple[bool, str]:
    """Execute a shell command and return success flag and output."""
    if requires_approval and not auto_approve:
        resp = input(f"Approve command '{command}'? [y/N]: ").strip().lower()
        if resp not in {"y", "yes"}:
            return False, "Command rejected by user"

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as exc:  # pragma: no cover - generic unexpected errors
        return False, f"Error executing command: {exc}"

    output = (completed.stdout or "") + (completed.stderr or "")
    success = completed.returncode == 0
    return success, output.strip()
