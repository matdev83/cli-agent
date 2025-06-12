import pytest
import json
import os
from pathlib import Path
# unittest.mock is not needed if not patching Python builtins directly,
# but can be useful. For now, direct file operations are tested.

# Tools to test
from src.tools.file import ReadFileTool, WriteToFileTool, ReplaceInFileTool, ListFilesTool, SearchFilesTool

# Dummy AgentMemory class for testing tools that expect it
class MockAgentMemory:
    def __init__(self, cwd: str, auto_approve: bool = True):
        self.cwd = cwd
        self.auto_approve = auto_approve
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

# --- ReadFileTool Tests ---
def test_read_file_tool_success(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReadFileTool()
    file_content = "Hello, world!"
    test_file = tmp_path / "test.txt"
    test_file.write_text(file_content)

    # Test with relative path
    result_rel = tool.execute({"path": "test.txt"}, agent_memory=mock_memory)
    assert result_rel == file_content

    # Test with absolute path
    result_abs = tool.execute({"path": str(test_file)}, agent_memory=mock_memory)
    assert result_abs == file_content

def test_read_file_tool_not_found(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReadFileTool()
    result = tool.execute({"path": "nonexistent.txt"}, agent_memory=mock_memory)
    assert "Error: File not found" in result

def test_read_file_tool_is_directory(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReadFileTool()
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    result = tool.execute({"path": "test_dir"}, agent_memory=mock_memory)
    assert "Error: Path" in result and "is a directory" in result

def test_read_file_tool_no_path_param(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReadFileTool()
    result = tool.execute({}, agent_memory=mock_memory)
    assert "Error: Missing required parameter 'path'" in result

# --- WriteToFileTool Tests ---
def test_write_to_file_tool_create_new(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = WriteToFileTool()
    file_content = "New content"
    file_path_str = "output.txt"

    result = tool.execute({"path": file_path_str, "content": file_content}, agent_memory=mock_memory)
    assert "File written successfully" in result
    full_path = tmp_path / file_path_str
    assert full_path.read_text() == file_content

def test_write_to_file_tool_overwrite(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = WriteToFileTool()
    initial_content = "Initial"
    new_content = "Overwritten"
    test_file_rel_path = "overwrite.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text(initial_content)

    tool.execute({"path": test_file_rel_path, "content": new_content}, agent_memory=mock_memory)
    assert test_file_abs_path.read_text() == new_content

def test_write_to_file_tool_create_dirs(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = WriteToFileTool()
    file_content = "Deep content"
    file_path_str = Path("deep") / "down" / "output.txt" # Use Path for os-agnostic paths

    tool.execute({"path": str(file_path_str), "content": file_content}, agent_memory=mock_memory)
    full_path = tmp_path / file_path_str
    assert full_path.read_text() == file_content

def test_write_to_file_tool_missing_params(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = WriteToFileTool()
    assert "Error: Missing required parameter 'path'" in tool.execute({"content": "abc"}, agent_memory=mock_memory)
    assert "Error: Missing required parameter 'content'" in tool.execute({"path": "abc"}, agent_memory=mock_memory)


# --- ReplaceInFileTool Tests ---
def test_replace_in_file_tool_success(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "replace_me.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text("Hello world, how are you world?")

    diff = "<<<<<<< SEARCH\nworld\n=======\nthere\n>>>>>>> REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff_blocks": diff}, agent_memory=mock_memory)
    assert "modified successfully with 1 block(s)" in result
    assert test_file_abs_path.read_text() == "Hello there, how are you world?"

def test_replace_in_file_tool_search_not_found(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "search_fail.txt"
    (tmp_path / test_file_rel_path).write_text("Hello world")
    diff = "<<<<<<< SEARCH\nbanana\n=======\napple\n>>>>>>> REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff_blocks": diff}, agent_memory=mock_memory)
    assert "Error: Search block 1" in result and "not found" in result

def test_replace_in_file_tool_invalid_diff_format(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "format_fail.txt"
    (tmp_path / test_file_rel_path).write_text("Hello world")
    diff = "this is not a valid diff" # This will trigger the ValueError in _parse_diff_blocks
    result = tool.execute({"path": test_file_rel_path, "diff_blocks": diff}, agent_memory=mock_memory)
    assert "Error processing diff_blocks" in result
    assert "No valid diff blocks found" in result # Specific error from _parse_diff_blocks

def test_replace_in_file_tool_empty_diff(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "empty_diff.txt"
    original_content = "No change"
    (tmp_path / test_file_rel_path).write_text(original_content)
    result = tool.execute({"path": test_file_rel_path, "diff_blocks": "  "}, agent_memory=mock_memory) # Whitespace only
    assert "Warning: 'diff_blocks' was empty" in result
    assert (tmp_path / test_file_rel_path).read_text() == original_content

def test_replace_in_file_tool_file_not_found(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    diff = "<<<<<<< SEARCH\nworld\n=======\nthere\n>>>>>>> REPLACE"
    result = tool.execute({"path": "nonexistent.txt", "diff_blocks": diff}, agent_memory=mock_memory)
    assert "Error: File not found" in result

# --- ListFilesTool Tests ---
def test_list_files_tool_non_recursive(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListFilesTool()
    (tmp_path / "file1.txt").write_text("1")
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.txt").write_text("2") # This should not be listed

    result_str = tool.execute({"path": "."}, agent_memory=mock_memory)
    result = json.loads(result_str)
    assert sorted(result) == sorted(["dir1/", "file1.txt"])

def test_list_files_tool_recursive(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListFilesTool()
    (tmp_path / "file1.txt").write_text("1")
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    (dir1 / "file2.txt").write_text("2")
    dir2 = dir1 / "dir2"
    dir2.mkdir()
    (dir2 / "file3.txt").write_text("3")

    # Test from root of tmp_path
    result_str = tool.execute({"path": ".", "recursive": True}, agent_memory=mock_memory)
    result = json.loads(result_str)
    # Expected paths are relative to the directory passed to the tool (tmp_path in this case)
    # Using set for comparison as order from os.walk can sometimes vary based on FS an OS.
    # The tool itself sorts the final list, so direct list comparison is also good.
    expected_files = sorted(["file1.txt", "dir1/", "dir1/file2.txt", "dir1/dir2/", "dir1/dir2/file3.txt"])
    assert sorted(result) == expected_files

    # Test from a sub-directory
    result_str_subdir = tool.execute({"path": "dir1", "recursive": True}, agent_memory=mock_memory)
    result_subdir = json.loads(result_str_subdir)
    expected_files_subdir = sorted(["file2.txt", "dir2/", "dir2/file3.txt"])
    assert sorted(result_subdir) == expected_files_subdir


def test_list_files_tool_empty_dir(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListFilesTool()
    empty_subdir = tmp_path / "empty_dir"
    empty_subdir.mkdir()

    result_str = tool.execute({"path": "empty_dir"}, agent_memory=mock_memory)
    result = json.loads(result_str)
    assert result == []

def test_list_files_tool_non_existent_dir(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = ListFilesTool()
    result = tool.execute({"path": "non_existent_dir"}, agent_memory=mock_memory)
    assert "Error: Directory not found" in result

# --- SearchFilesTool Tests ---
def test_search_files_tool_found_matches(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = SearchFilesTool()
    (tmp_path / "fileA.txt").write_text("hello world\nsearch me please\nworld again")
    (tmp_path / "fileB.log").write_text("another WORLD line\nno search here") # Case sensitive regex
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "fileC.txt").write_text("world in subdir")

    result_str = tool.execute({"directory": ".", "regex_pattern": "world"}, agent_memory=mock_memory)
    result = json.loads(result_str)

    assert len(result) == 3 # "hello world", "world again", "world in subdir"

    # Check some details
    files_matched = {r["file"] for r in result}
    assert "fileA.txt" in files_matched
    assert str(Path("sub") / "fileC.txt") in files_matched # Path("sub/fileC.txt").as_posix()

    # Check content from one of the matches
    file_a_matches = [r for r in result if r["file"] == "fileA.txt"]
    assert len(file_a_matches) == 2
    assert file_a_matches[0]["content"] == "hello world"
    assert file_a_matches[1]["content"] == "world again"


def test_search_files_tool_no_matches(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = SearchFilesTool()
    (tmp_path / "fileA.txt").write_text("hello there\nsearch me please")
    result_str = tool.execute({"directory": ".", "regex_pattern": "world"}, agent_memory=mock_memory)
    result = json.loads(result_str)
    assert len(result) == 0

def test_search_files_tool_with_file_pattern(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = SearchFilesTool()
    (tmp_path / "fileA.txt").write_text("hello world")
    (tmp_path / "fileB.log").write_text("world in log")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.txt").write_text("another world in txt")

    result_str = tool.execute({"directory": ".", "regex_pattern": "world", "file_pattern": "*.txt"}, agent_memory=mock_memory)
    result = json.loads(result_str)
    assert len(result) == 2
    files_found = {item['file'] for item in result}
    assert "fileA.txt" in files_found
    assert str(Path("docs") / "readme.txt") in files_found # Path("docs/readme.txt").as_posix()

def test_search_files_tool_invalid_regex(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = SearchFilesTool()
    # Test with an invalid regex pattern like an unclosed bracket
    result = tool.execute({"directory": ".", "regex_pattern": "["}, agent_memory=mock_memory)
    assert "Error: Invalid regex pattern" in result

def test_search_files_tool_dir_not_found(tmp_path: Path):
    mock_memory = MockAgentMemory(cwd=str(tmp_path))
    tool = SearchFilesTool()
    result = tool.execute({"directory": "nonexistent", "regex_pattern": "test"}, agent_memory=mock_memory)
    assert "Error: Directory not found" in result
```
