from pathlib import Path

from src.tools import list_code_definition_names


def test_list_code_definition_names(tmp_path: Path):
    (tmp_path / "module.py").write_text(
        """class Foo:\n    pass\n\n\ndef bar():\n    return 1\n\nclass Baz:\n    pass\n""",
        encoding="utf-8",
    )
    (tmp_path / "ignore.txt").write_text("hello")

    result = list_code_definition_names(str(tmp_path))
    lines = result.splitlines()
    assert lines[0] == "module.py"
    assert "|----" in lines[1]
    assert any("class Foo" in l for l in lines)
    assert any("def bar" in l for l in lines)
    assert lines[-1] == "|----"


def test_no_definitions(tmp_path: Path):
    (tmp_path / "a.txt").write_text("no code")
    assert list_code_definition_names(str(tmp_path)) == "No source code definitions found."
