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
