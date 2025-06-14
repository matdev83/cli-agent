import pytest
from unittest.mock import MagicMock, patch

from src.slash_commands import UndoCommand, UndoAllCommand
from src.cli import AgentCliContext # Corrected import
# Assuming DeveloperAgent can be mocked. If it's too complex,
# we'll mock its attributes directly on agent_context.agent.
# from src.agent import DeveloperAgent # Not strictly needed if we mock agent attributes

# We will mock these, so direct import for patching is good.
# from src import utils # Alternative: patch('src.slash_commands.utils.revert_to_commit')

# Helper to create a mock agent_context
def create_mock_agent_context(cwd: str, session_history: list[str], initial_hash: str | None):
    mock_agent = MagicMock()
    mock_agent.cwd = cwd
    mock_agent.session_commit_history = list(session_history) # Use a copy
    mock_agent.initial_session_head_commit_hash = initial_hash

    # Mock the .subprocess attribute on utils if the command tries to use it directly
    # (e.g. for the HEAD check in UndoAllCommand's "no commits" message)
    # This might be needed if 'from . import utils' was used and utils.subprocess is called.
    # For now, we assume direct calls to functions we patch.
    # mock_utils_module = MagicMock()
    # mock_utils_module.subprocess = MagicMock()
    # mock_utils_module.subprocess.check_output.return_value.strip.return_value = initial_hash or "dummy_head_for_check"

    # We are patching 'src.slash_commands.utils', so no need to mock utils module here
    # if commands correctly call e.g. utils.revert_to_commit.

    context = AgentCliContext(cli_args_namespace=MagicMock(), display_update_func=MagicMock(), agent_instance=mock_agent)
    return context

@pytest.fixture
def undo_command():
    return UndoCommand()

@pytest.fixture
def undo_all_command():
    return UndoAllCommand()

# --- Tests for UndoCommand ---

def test_undo_no_args_empty_history(undo_command: UndoCommand):
    context = create_mock_agent_context("/fake/cwd", [], "initial_hash")
    result = undo_command.execute([], agent_context=context)
    assert "No auto-commits in the current session to undo" in result

@patch('src.slash_commands.utils.revert_to_state_before_commit')
def test_undo_no_args_one_commit_success(mock_revert: MagicMock, undo_command: UndoCommand):
    mock_revert.return_value = True
    commit_hash = "commit1"
    context = create_mock_agent_context("/fake/cwd", [commit_hash], "initial_hash")

    result = undo_command.execute([], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", commit_hash)
    assert len(context.agent.session_commit_history) == 0
    assert f"Successfully reverted changes from the last auto-commit ({commit_hash[:7]})" in result

@patch('src.slash_commands.utils.revert_to_state_before_commit')
def test_undo_no_args_multiple_commits_success(mock_revert: MagicMock, undo_command: UndoCommand):
    mock_revert.return_value = True
    history = ["commit1", "commit2", "commit3"]
    context = create_mock_agent_context("/fake/cwd", history, "initial_hash")

    result = undo_command.execute([], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", "commit3")
    assert context.agent.session_commit_history == ["commit1", "commit2"]
    assert "Successfully reverted changes from the last auto-commit (commit3)" in result

@patch('src.slash_commands.utils.revert_to_state_before_commit')
def test_undo_no_args_revert_fails(mock_revert: MagicMock, undo_command: UndoCommand):
    mock_revert.return_value = False
    commit_hash = "commit1"
    context = create_mock_agent_context("/fake/cwd", [commit_hash], "initial_hash")

    result = undo_command.execute([], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", commit_hash)
    assert context.agent.session_commit_history == [commit_hash] # Unchanged
    assert f"Error: Failed to revert changes from commit {commit_hash[:7]}" in result

@patch('src.slash_commands.utils.revert_to_state_before_commit')
def test_undo_with_commit_id_success(mock_revert: MagicMock, undo_command: UndoCommand):
    mock_revert.return_value = True
    # Use valid hex strings for commit IDs
    commit1_hash = "a" * 40
    commit2_hash = "b" * 40
    commit3_hash = "c" * 40
    history = [commit1_hash, commit2_hash, commit3_hash]

    short_target_id = commit2_hash[:10] # e.g., "bbbbbbbbbb" - this is a valid hex substring

    context = create_mock_agent_context("/fake/cwd", history, "initial_hash")

    result = undo_command.execute([short_target_id], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", short_target_id)
    assert context.agent.session_commit_history == [commit1_hash] # commit2 and commit3 removed
    assert f"Successfully reverted to the state before commit {short_target_id[:7]}" in result
    assert "2 commit(s) removed from session history" in result


@patch('src.slash_commands.utils.revert_to_state_before_commit')
def test_undo_with_commit_id_not_in_session_history_git_success(mock_revert: MagicMock, undo_command: UndoCommand):
    mock_revert.return_value = True
    # Use valid hex strings
    commit1_hash = "a" * 40
    commit3_hash = "c" * 40
    history = [commit1_hash, commit3_hash]
    commit_to_undo = "b" * 10 # Valid hex, but not in history list (as a prefix)
    context = create_mock_agent_context("/fake/cwd", history, "initial_hash")

    result = undo_command.execute([commit_to_undo], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", commit_to_undo)
    # History should be unchanged because commit_to_undo ("bbbbbbbbbb") does not start with "a"*40 or "c"*40
    assert context.agent.session_commit_history == history
    assert f"Successfully reverted to the state before commit {commit_to_undo[:7]}" in result
    assert "(Commit not found in current session history" in result

@patch('src.slash_commands.utils.revert_to_state_before_commit')
def test_undo_with_commit_id_revert_fails(mock_revert: MagicMock, undo_command: UndoCommand):
    mock_revert.return_value = False
    # Use valid hex strings
    commit1_hash = "a" * 40
    commit2_hash = "b" * 40
    commit3_hash = "c" * 40
    history = [commit1_hash, commit2_hash, commit3_hash]
    commit_to_undo = commit2_hash[:7] # Valid hex, e.g. "bbbbbbb"
    context = create_mock_agent_context("/fake/cwd", history, "initial_hash")

    result = undo_command.execute([commit_to_undo], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", commit_to_undo)
    assert context.agent.session_commit_history == history # Unchanged
    assert f"Error: Failed to revert to the state before commit {commit_to_undo[:7]}" in result

def test_undo_with_invalid_commit_id_format(undo_command: UndoCommand):
    # Test depends on validation being present in UndoCommand itself
    history = ["commit1"]
    context = create_mock_agent_context("/fake/cwd", history, "initial_hash")

    result_short = undo_command.execute(["abc"], agent_context=context) # Too short
    assert "Error: Invalid commit ID format: abc" in result_short

    result_invalid_char = undo_command.execute(["commit1_xyz"], agent_context=context) # 'x' is invalid hex
    assert "Error: Invalid commit ID format: commit1_xyz" in result_invalid_char

def test_undo_too_many_arguments(undo_command: UndoCommand):
    context = create_mock_agent_context("/fake/cwd", ["c1"], "initial")
    result = undo_command.execute(["arg1", "arg2"], agent_context=context)
    assert "Error: /undo accepts 0 or 1 argument (commit_id)" in result

def test_undo_agent_context_missing(undo_command: UndoCommand):
    result_none_context = undo_command.execute([], agent_context=None)
    assert "Error: Agent context not available" in result_none_context

    mock_context_no_agent = MagicMock()
    mock_context_no_agent.agent = None
    result_none_agent = undo_command.execute([], agent_context=mock_context_no_agent)
    assert "Error: Agent context not available" in result_none_agent


# --- Tests for UndoAllCommand ---

@patch('src.slash_commands.utils.subprocess.check_output') # For the HEAD check
def test_undo_all_empty_history_already_at_initial(mock_git_check, undo_all_command: UndoAllCommand):
    initial_hash = "initial_hash_123"
    # Simulate git rev-parse HEAD returning the initial hash
    mock_git_check.return_value.strip.return_value = initial_hash

    context = create_mock_agent_context("/fake/cwd", [], initial_hash)
    result = undo_all_command.execute([], agent_context=context)
    assert "No auto-commits in the current session to undo. Repository is already at the initial session state." in result
    mock_git_check.assert_called_once_with(["git", "rev-parse", "HEAD"], cwd="/fake/cwd", text=True)


@patch('src.slash_commands.utils.subprocess.check_output') # For the HEAD check
def test_undo_all_empty_history_different_head(mock_git_check, undo_all_command: UndoAllCommand):
    initial_hash = "initial_hash_123"
    # Simulate git rev-parse HEAD returning something different
    mock_git_check.return_value.strip.return_value = "different_head_789"

    context = create_mock_agent_context("/fake/cwd", [], initial_hash)
    result = undo_all_command.execute([], agent_context=context)
    assert "No auto-commits recorded in the current session to undo." in result
    mock_git_check.assert_called_once_with(["git", "rev-parse", "HEAD"], cwd="/fake/cwd", text=True)


@patch('src.slash_commands.utils.revert_to_commit')
def test_undo_all_with_history_success(mock_revert: MagicMock, undo_all_command: UndoAllCommand):
    mock_revert.return_value = True
    history = ["commit1", "commit2"]
    initial_hash = "initial_hash_abc"
    context = create_mock_agent_context("/fake/cwd", history, initial_hash)

    result = undo_all_command.execute([], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", initial_hash)
    assert len(context.agent.session_commit_history) == 0
    assert f"Successfully reverted all 2 auto-committed changes made during this session to state {initial_hash[:7]}" in result

def test_undo_all_no_initial_hash(undo_all_command: UndoAllCommand):
    history = ["commit1", "commit2"]
    context = create_mock_agent_context("/fake/cwd", history, None) # initial_session_head_commit_hash is None

    result = undo_all_command.execute([], agent_context=context)
    assert "Error: Cannot determine the initial state of the repository for this session" in result

@patch('src.slash_commands.utils.revert_to_commit')
def test_undo_all_revert_fails(mock_revert: MagicMock, undo_all_command: UndoAllCommand):
    mock_revert.return_value = False
    history = ["commit1", "commit2"]
    initial_hash = "initial_hash_xyz"
    context = create_mock_agent_context("/fake/cwd", history, initial_hash)

    result = undo_all_command.execute([], agent_context=context)

    mock_revert.assert_called_once_with("/fake/cwd", initial_hash)
    assert context.agent.session_commit_history == history # Unchanged
    assert f"Error: Failed to revert all session changes to the initial session state {initial_hash[:7]}" in result

def test_undo_all_agent_context_missing(undo_all_command: UndoAllCommand):
    result_none_context = undo_all_command.execute([], agent_context=None)
    assert "Error: Agent context not available" in result_none_context

    mock_context_no_agent = MagicMock()
    mock_context_no_agent.agent = None
    result_none_agent = undo_all_command.execute([], agent_context=mock_context_no_agent)
    assert "Error: Agent context not available" in result_none_agent

# To run these tests: pytest tests/test_slash_commands_undo.py
# Ensure src directory is in PYTHONPATH or discoverable by pytest.
# Example: PYTHONPATH=. pytest tests/test_slash_commands_undo.py
if __name__ == '__main__':
    pytest.main([__file__])
