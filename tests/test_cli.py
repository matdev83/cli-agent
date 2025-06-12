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
