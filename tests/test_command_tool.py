import sys

from src.tools import execute_command


def test_execute_command_success():
    success, output = execute_command("echo hello", requires_approval=False)
    assert success
    assert "hello" in output


def test_execute_command_rejected(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    success, output = execute_command("echo hi", requires_approval=True)
    assert not success
    assert "rejected" in output.lower()


def test_execute_command_timeout():
    cmd = f"{sys.executable} -c \"import time; time.sleep(2)\""
    success, output = execute_command(cmd, requires_approval=False, timeout=0.5)
    assert not success
    assert "timed out" in output.lower()


def test_execute_command_error():
    success, output = execute_command("nonexistentcommand_xyz", requires_approval=False)
    assert not success
    assert output
