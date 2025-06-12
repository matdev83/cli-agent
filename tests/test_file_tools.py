from pathlib import Path

from src.tools import (
    read_file,
    write_to_file,
    replace_in_file,
    list_files,
    search_files,
)


def test_read_write_file(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    write_to_file(str(file_path), "hello world")
    assert file_path.exists()
    content = read_file(str(file_path))
    assert content == "hello world"


def test_replace_in_file(tmp_path: Path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("a\nfoo\nb\n", encoding="utf-8")
    diff = (
        "------- SEARCH\nfoo\n=======\nbar\n+++++++ REPLACE"
    )
    replace_in_file(str(file_path), diff)
    assert file_path.read_text(encoding="utf-8") == "a\nbar\nb\n"


def test_replace_in_file_not_found(tmp_path: Path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("hello", encoding="utf-8")
    diff = "------- SEARCH\nbye\n=======\nhi\n+++++++ REPLACE"
    try:
        replace_in_file(str(file_path), diff)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError"


def test_list_files(tmp_path: Path):
    (tmp_path / "a").write_text("x")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "b.txt").write_text("y")
    top = list_files(str(tmp_path))
    assert set(top) == {"a", "dir/"} # Added trailing slash for directory
    rec = list_files(str(tmp_path), recursive=True)
    assert "a" in rec and "dir/b.txt" in rec # Recursive listing should still be fine


def test_search_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello world\nbye", encoding="utf-8")
    (tmp_path / "b.md").write_text("nothing", encoding="utf-8")
    results = search_files(str(tmp_path), "hello", file_pattern="*.txt")
    assert len(results) == 1
    assert results[0]["file"] == "a.txt"
    assert results[0]["line"] == 1


def test_write_to_file_creates_directories(tmp_path: Path):
    nested = tmp_path / "dir1" / "dir2" / "file.txt"
    write_to_file(str(nested), "data")
    assert nested.exists()
    assert read_file(str(nested)) == "data"
