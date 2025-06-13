from src.cli import main
import json


def test_cli_basic(tmp_path, capsys):
    responses = [
        "<write_to_file><path>out.txt</path><content>hi</content></write_to_file>",
        "<attempt_completion><result>done</result></attempt_completion>",
    ]
    resp_file = tmp_path / "responses.json"
    resp_file.write_text(json.dumps(responses), encoding="utf-8")

    cwd = tmp_path / "work"
    cwd.mkdir()

    exit_code = main([
        "do something",
        "--responses-file",
        str(resp_file),
        "--auto-approve",
        "--cwd",
        str(cwd),
    ])

    assert exit_code == 0
    assert (cwd / "out.txt").read_text(encoding="utf-8") == "hi"
    out = capsys.readouterr().out
    assert "done" in out

from unittest.mock import patch, ANY

def test_cli_llm_timeout_argument_provided(tmp_path, capsys):
    """Test that --llm-timeout is correctly passed to OpenRouterLLM."""
    responses = ["<attempt_completion><result>done</result></attempt_completion>"]
    resp_file = tmp_path / "responses.json" # MockLLM will be used if --model is not OpenRouter
    resp_file.write_text(json.dumps(responses), encoding="utf-8")

    # We need to use a non-mock model to trigger OpenRouterLLM instantiation.
    # We also need to ensure OPENROUTER_API_KEY is set, or mock it.
    with patch('src.cli.OpenRouterLLM', autospec=True) as mock_llm_constructor, \
         patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):

        main([
            "some task",
            "--model", "some/model", # Use a non-mock model
            "--llm-timeout", "60.5",
            "--auto-approve",
            "--cwd", str(tmp_path)
        ])

        mock_llm_constructor.assert_called_once_with(
            model="some/model",
            api_key="test_key",
            timeout=60.5
        )

def test_cli_llm_timeout_argument_default(tmp_path, capsys):
    """Test that the default --llm-timeout (120.0) is used if not provided."""
    responses = ["<attempt_completion><result>done</result></attempt_completion>"]
    resp_file = tmp_path / "responses.json"
    resp_file.write_text(json.dumps(responses), encoding="utf-8")

    with patch('src.cli.OpenRouterLLM', autospec=True) as mock_llm_constructor, \
         patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):

        main([
            "some task",
            "--model", "some/model",
            # No --llm-timeout provided, should use default
            "--auto-approve",
            "--cwd", str(tmp_path)
        ])

        mock_llm_constructor.assert_called_once_with(
            model="some/model",
            api_key="test_key",
            timeout=120.0 # Default value specified in cli.py
        )

def test_cli_llm_timeout_with_mock_model(tmp_path, capsys):
    """Test that --llm-timeout is ignored (not passed) when using mock model."""
    responses = ["<attempt_completion><result>done</result></attempt_completion>"]
    resp_file = tmp_path / "responses.json"
    resp_file.write_text(json.dumps(responses), encoding="utf-8")

    # MockLLM is instantiated directly, OpenRouterLLM should not be called.
    with patch('src.cli.OpenRouterLLM', autospec=True) as mock_openrouter_constructor, \
         patch('src.cli.MockLLM', autospec=True) as mock_mockllm_constructor:

        main([
            "some task",
            "--model", "mock",
            "--responses-file", str(resp_file),
            "--llm-timeout", "90.0", # This should be ignored
            "--auto-approve",
            "--cwd", str(tmp_path)
        ])

        mock_openrouter_constructor.assert_not_called()
        mock_mockllm_constructor.from_file.assert_called_once_with(str(resp_file))
        # We don't check MockLLM's __init__ params regarding timeout, as it doesn't take one.

# --- Tests for new auto-approval flags ---

APPROVAL_ARGS_FLAGS = [
    "allow_read_files",
    "allow_edit_files",
    "allow_execute_safe_commands",
    "allow_execute_all_commands",
    "allow_use_browser",
    "allow_use_mcp",
]
OTHER_FLAGS = ["disable_git_auto_commits"]

@patch('src.cli.run_agent')
def test_cli_approval_flags_defaults(mock_run_agent, tmp_path):
    """Test that new approval flags default to False."""
    # Need to provide a minimal responses file for mock model
    resp_file = tmp_path / "responses.json"
    resp_file.write_text(json.dumps(["<attempt_completion><result>done</result></attempt_completion>"]), encoding="utf-8")

    main([
        "sample_task",
        "--model", "mock",
        "--responses-file", str(resp_file), # Added responses file for mock
        "--cwd", str(tmp_path)
    ])

    mock_run_agent.assert_called_once()
    # Args are passed as keyword arguments from main to run_agent
    called_kwargs = mock_run_agent.call_args[1]

    for arg_name in APPROVAL_ARGS_FLAGS:
        assert not called_kwargs.get(arg_name), f"Expected {arg_name} to default to False"
    # Also check the legacy auto_approve, though it's not in APPROVAL_ARGS_FLAGS
    assert not called_kwargs.get("auto_approve"), "Expected auto_approve to default to False"
    for flag in OTHER_FLAGS:
        assert not called_kwargs.get(flag), f"Expected {flag} to default to False"


@patch('src.cli.run_agent')
def test_cli_approval_flags_set(mock_run_agent, tmp_path):
    """Test that new approval flags can be set to True one by one."""
    resp_file = tmp_path / "responses.json"
    resp_file.write_text(json.dumps(["<attempt_completion><result>done</result></attempt_completion>"]), encoding="utf-8")

    base_cli_args = [
        "sample_task",
        "--model", "mock",
        "--responses-file", str(resp_file),
        "--cwd", str(tmp_path)
    ]

    for flag_to_set_true in APPROVAL_ARGS_FLAGS:
        # Construct CLI arguments for this specific test run
        current_cli_args = base_cli_args + [f"--{flag_to_set_true.replace('_', '-')}"]

        mock_run_agent.reset_mock() # Reset mock for each specific flag test

        main(current_cli_args)

        mock_run_agent.assert_called_once()
        called_kwargs = mock_run_agent.call_args[1]

        # Check that the current flag is True
        assert called_kwargs.get(flag_to_set_true), f"Expected {flag_to_set_true} to be True"

        # Check that all other new approval flags are False
        for other_flag in APPROVAL_ARGS_FLAGS:
            if other_flag != flag_to_set_true:
                assert not called_kwargs.get(other_flag), \
                    f"Expected {other_flag} to be False when {flag_to_set_true} is set"

        # Ensure legacy auto_approve is not affected unless explicitly set
        assert not called_kwargs.get("auto_approve"), \
            f"Expected auto_approve to be False when only {flag_to_set_true} is set"
        for flag in OTHER_FLAGS:
            assert not called_kwargs.get(flag), f"Expected {flag} to remain False"

@patch('src.cli.run_agent')
def test_cli_legacy_auto_approve_with_new_flags(mock_run_agent, tmp_path):
    """Test legacy --auto-approve in conjunction with a new flag."""
    resp_file = tmp_path / "responses.json"
    resp_file.write_text(json.dumps(["<attempt_completion><result>done</result></attempt_completion>"]), encoding="utf-8")

    cli_args_to_test = [
        "sample_task",
        "--model", "mock",
        "--responses-file", str(resp_file),
        "--cwd", str(tmp_path),
        "--auto-approve", # Legacy flag
        "--allow-read-files" # A new specific flag
    ]

    main(cli_args_to_test)

    mock_run_agent.assert_called_once()
    called_kwargs = mock_run_agent.call_args[1]

    assert called_kwargs.get("auto_approve"), "Expected auto_approve to be True"
    assert called_kwargs.get("allow_read_files"), "Expected allow_read_files to be True"

    # Check other new flags are False
    for flag_name in APPROVAL_ARGS_FLAGS:
        if flag_name != "allow_read_files":
            assert not called_kwargs.get(flag_name), f"Expected {flag_name} to be False"
    for flag in OTHER_FLAGS:
        assert not called_kwargs.get(flag), f"Expected {flag} to remain False"


@patch('src.cli.run_agent')
def test_cli_disable_git_auto_commits_flag(mock_run_agent, tmp_path):
    resp_file = tmp_path / "responses.json"
    resp_file.write_text(json.dumps(["<attempt_completion><result>done</result></attempt_completion>"]), encoding="utf-8")

    main([
        "task", "--model", "mock", "--responses-file", str(resp_file), "--cwd", str(tmp_path),
        "--disable-git-auto-commits"
    ])

    mock_run_agent.assert_called_once()
    called_kwargs = mock_run_agent.call_args[1]
    assert called_kwargs.get("disable_git_auto_commits")
