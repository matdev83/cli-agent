import pytest
import json
import os
from pathlib import Path
# unittest.mock is not needed if not patching Python builtins directly,
# but can be useful. For now, direct file operations are tested.

# Tools to test
from src.tools.file import ReadFileTool, WriteToFileTool, ReplaceInFileTool, ListFilesTool, SearchFilesTool

# Dummy AgentToolsInstance class for testing tools that expect it
class MockAgentToolsInstance:
    def __init__(self, cwd: str, auto_approve: bool = True, matching_strictness: int = 100):
        self.cwd = cwd
        self.auto_approve = auto_approve # Kept if any tool might use it, though file tools don't typically
        self.matching_strictness = matching_strictness
        self.messages = [] # Kept if any tool might use it

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

# --- ReadFileTool Tests ---
def test_read_file_tool_properties():
    tool = ReadFileTool()
    assert tool.name == "read_file"
    assert isinstance(tool.description, str)
    assert tool.parameters_schema == {
        "path": "The relative or absolute path to the file to be read."
    }

def test_read_file_tool_success(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReadFileTool()
    file_content = "Hello, world!"
    test_file = tmp_path / "test.txt"
    test_file.write_text(file_content)

    # Test with relative path
    result_rel = tool.execute({"path": "test.txt"}, agent_tools_instance=mock_agent_tools_instance)
    assert result_rel == file_content

    # Test with absolute path
    result_abs = tool.execute({"path": str(test_file)}, agent_tools_instance=mock_agent_tools_instance)
    assert result_abs == file_content

def test_read_file_tool_not_found(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReadFileTool()
    result = tool.execute({"path": "nonexistent.txt"}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: File not found" in result

def test_read_file_tool_is_directory(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReadFileTool()
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    result = tool.execute({"path": "test_dir"}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Path" in result and "is a directory" in result

def test_read_file_tool_no_path_param(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReadFileTool()
    result = tool.execute({}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Missing required parameter 'path'" in result

# --- WriteToFileTool Tests ---
def test_write_to_file_tool_properties():
    tool = WriteToFileTool()
    assert tool.name == "write_to_file"
    assert isinstance(tool.description, str)
    assert tool.parameters_schema == {
        "path": "The relative or absolute path to the file to be written.",
        "content": "The content to write to the file."
    }

def test_write_to_file_tool_create_new(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = WriteToFileTool()
    file_content = "New content"
    file_path_str = "output.txt"

    result = tool.execute({"path": file_path_str, "content": file_content}, agent_tools_instance=mock_agent_tools_instance)
    assert "File written successfully" in result
    full_path = tmp_path / file_path_str
    assert full_path.read_text() == file_content

def test_write_to_file_tool_overwrite(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = WriteToFileTool()
    initial_content = "Initial"
    new_content = "Overwritten"
    test_file_rel_path = "overwrite.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text(initial_content)

    tool.execute({"path": test_file_rel_path, "content": new_content}, agent_tools_instance=mock_agent_tools_instance)
    assert test_file_abs_path.read_text() == new_content

def test_write_to_file_tool_create_dirs(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = WriteToFileTool()
    file_content = "Deep content"
    file_path_str = Path("deep") / "down" / "output.txt" # Use Path for os-agnostic paths

    tool.execute({"path": str(file_path_str), "content": file_content}, agent_tools_instance=mock_agent_tools_instance)
    full_path = tmp_path / file_path_str
    assert full_path.read_text() == file_content

def test_write_to_file_tool_missing_params(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = WriteToFileTool()
    assert "Error: Missing required parameter 'path'" in tool.execute({"content": "abc"}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Missing required parameter 'content'" in tool.execute({"path": "abc"}, agent_tools_instance=mock_agent_tools_instance)


# --- ReplaceInFileTool Tests ---
def test_replace_in_file_tool_properties():
    tool = ReplaceInFileTool()
    assert tool.name == "replace_in_file"
    assert isinstance(tool.description, str)
    assert tool.parameters_schema == {
        "path": "The relative or absolute path to the file to be modified.",
        "diff": "A string containing one or more diff blocks in the specified format."
    }

def test_replace_in_file_tool_success(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "replace_me.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text("Hello world, how are you world?")

    diff = "------- SEARCH\nworld\n=======\nthere\n+++++++ REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "modified successfully with 1 block(s)" in result
    assert test_file_abs_path.read_text() == "Hello there, how are you world?"

def test_replace_in_file_tool_search_not_found(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "search_fail.txt"
    (tmp_path / test_file_rel_path).write_text("Hello world")
    diff = "------- SEARCH\nbanana\n=======\napple\n+++++++ REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Search block 1" in result and "not found" in result

def test_replace_in_file_tool_invalid_diff_format(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "format_fail.txt"
    (tmp_path / test_file_rel_path).write_text("Hello world")
    diff = "this is not a valid diff" # This will trigger the ValueError in _parse_diff_blocks
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error processing diff" in result
    # Specific error from _parse_diff_blocks, updated to the new format
    assert "No valid diff blocks found. Ensure the format is: ------- SEARCH...=======...+++++++ REPLACE" in result

def test_replace_in_file_tool_empty_diff(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "empty_diff.txt"
    original_content = "No change"
    (tmp_path / test_file_rel_path).write_text(original_content)
    result = tool.execute({"path": test_file_rel_path, "diff": "  "}, agent_tools_instance=mock_agent_tools_instance) # Whitespace only
    assert "Warning: 'diff' was empty" in result
    assert (tmp_path / test_file_rel_path).read_text() == original_content

def test_replace_in_file_tool_file_not_found(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    diff = "------- SEARCH\nworld\n=======\nthere\n+++++++ REPLACE"
    result = tool.execute({"path": "nonexistent.txt", "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: File not found" in result

# --- ReplaceInFileTool Matching Strictness Tests ---
def test_replace_in_file_exact_match_strictness_100(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), matching_strictness=100)
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_exact_100.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text("Hello World, this is a test.")

    diff = "------- SEARCH\nWorld\n=======\nUniverse\n+++++++ REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "modified successfully with 1 block(s)" in result
    assert test_file_abs_path.read_text() == "Hello Universe, this is a test."

def test_replace_in_file_exact_match_fails_due_to_case_strictness_100(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), matching_strictness=100)
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_exact_fail_100.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    initial_content = "Hello world, this is a test."
    test_file_abs_path.write_text(initial_content)

    diff = "------- SEARCH\nWorld\n=======\nUniverse\n+++++++ REPLACE" # "World" vs "world"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Search block 1" in result
    assert "(exact match)" in result
    assert "not found" in result
    assert test_file_abs_path.read_text() == initial_content # File should be unchanged

def test_replace_in_file_lenient_match_strictness_50_success(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), matching_strictness=50)
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_lenient_50.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text("Hello world, this is a test.") # content is "world"

    diff = "------- SEARCH\nWorld\n=======\nUniverse\n+++++++ REPLACE" # search is "World"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "modified successfully with 1 block(s)" in result
    assert test_file_abs_path.read_text() == "Hello Universe, this is a test."

def test_replace_in_file_lenient_match_different_casing_in_block_success(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), matching_strictness=50)
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_lenient_casing_block.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text("Hello World") # Content has "World"

    # Search block has mixed casing "hElLo wOrLd"
    diff = "------- SEARCH\nhElLo wOrLd\n=======\nGoodbye Universe\n+++++++ REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "modified successfully with 1 block(s)" in result
    assert test_file_abs_path.read_text() == "Goodbye Universe"

def test_replace_in_file_default_strictness_behaves_as_100_exact_match(tmp_path: Path):
    # Simulate agent_tools_instance without matching_strictness set (defaults to 100 in class)
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_default_exact.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text("Hello Default World.")

    diff = "------- SEARCH\nDefault World\n=======\nNew World\n+++++++ REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "modified successfully with 1 block(s)" in result
    assert test_file_abs_path.read_text() == "Hello New World."

def test_replace_in_file_default_strictness_fails_due_to_case(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path)) # Defaults to strictness 100
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_default_fail_case.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    initial_content = "hello default world."
    test_file_abs_path.write_text(initial_content)

    diff = "------- SEARCH\nDefault World\n=======\nNew World\n+++++++ REPLACE" # "Default World" vs "default world"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Search block 1" in result
    assert "(exact match)" in result # Default should be exact
    assert "not found" in result
    assert test_file_abs_path.read_text() == initial_content

def test_replace_in_file_lenient_match_search_not_found_strictness_50(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), matching_strictness=50)
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_lenient_not_found_50.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    initial_content = "Completely different text."
    test_file_abs_path.write_text(initial_content)

    diff = "------- SEARCH\nNonExistentPattern\n=======\nReplacement\n+++++++ REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Search block 1" in result
    assert "(case-insensitive match)" in result
    assert "not found" in result
    assert test_file_abs_path.read_text() == initial_content

def test_replace_in_file_lenient_match_invalid_regex_pattern(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path), matching_strictness=50)
    tool = ReplaceInFileTool()
    test_file_rel_path = "test_lenient_invalid_regex.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    initial_content = "Some text with Hello (World)."
    test_file_abs_path.write_text(initial_content)

    # Invalid regex: unclosed parenthesis in search block
    diff = "------- SEARCH\nHello (World\n=======\nReplacement\n+++++++ REPLACE"
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error compiling regex for search block 1" in result
    assert test_file_abs_path.read_text() == initial_content

def test_replace_in_file_tool_delete_block(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ReplaceInFileTool()
    test_file_rel_path = "delete_me.txt"
    test_file_abs_path = tmp_path / test_file_rel_path
    test_file_abs_path.write_text("Line one\ntext to delete\nLine three")

    diff = "------- SEARCH\ntext to delete\n=======\n\n+++++++ REPLACE" # Empty replace part
    result = tool.execute({"path": test_file_rel_path, "diff": diff}, agent_tools_instance=mock_agent_tools_instance)
    assert "modified successfully with 1 block(s)" in result
    assert test_file_abs_path.read_text() == "Line one\n\nLine three" # "text to delete" is replaced by empty string

def test_replace_in_file_tool_default_exact_matching(tmp_path: Path):
    # This test verifies default behavior is exact (strictness 100)
    # 1. Agent instance without matching_strictness explicitly set (should default to 100 in MockAgentToolsInstance)
    mock_agent_default = MockAgentToolsInstance(cwd=str(tmp_path))
    # 2. Agent instance with matching_strictness explicitly set to 100
    mock_agent_strict_100 = MockAgentToolsInstance(cwd=str(tmp_path), matching_strictness=100)

    tool = ReplaceInFileTool()
    test_file_rel_path = "default_exact_match_test.txt"
    test_file_abs_path = tmp_path / test_file_rel_path

    initial_content_exact = "Hello World, this is a test."
    diff_exact_match = "------- SEARCH\nWorld\n=======\nUniverse\n+++++++ REPLACE"
    diff_case_mismatch = "------- SEARCH\nworld\n=======\nUniverse\n+++++++ REPLACE"

    # Test with agent where matching_strictness defaults to 100
    test_file_abs_path.write_text(initial_content_exact)
    result_default_exact = tool.execute({"path": test_file_rel_path, "diff": diff_exact_match}, agent_tools_instance=mock_agent_default)
    assert "modified successfully with 1 block(s)" in result_default_exact
    assert test_file_abs_path.read_text() == "Hello Universe, this is a test."

    test_file_abs_path.write_text(initial_content_exact) # Reset content
    result_default_case_fail = tool.execute({"path": test_file_rel_path, "diff": diff_case_mismatch}, agent_tools_instance=mock_agent_default)
    assert "Error: Search block 1" in result_default_case_fail
    assert "(exact match)" in result_default_case_fail
    assert test_file_abs_path.read_text() == initial_content_exact # Should be unchanged

    # Test with agent where matching_strictness is explicitly 100
    test_file_abs_path.write_text(initial_content_exact) # Reset content
    result_strict_100_exact = tool.execute({"path": test_file_rel_path, "diff": diff_exact_match}, agent_tools_instance=mock_agent_strict_100)
    assert "modified successfully with 1 block(s)" in result_strict_100_exact
    assert test_file_abs_path.read_text() == "Hello Universe, this is a test."

    test_file_abs_path.write_text(initial_content_exact) # Reset content
    result_strict_100_case_fail = tool.execute({"path": test_file_rel_path, "diff": diff_case_mismatch}, agent_tools_instance=mock_agent_strict_100)
    assert "Error: Search block 1" in result_strict_100_case_fail
    assert "(exact match)" in result_strict_100_case_fail
    assert test_file_abs_path.read_text() == initial_content_exact # Should be unchanged

# --- ListFilesTool Tests ---
def test_list_files_tool_properties():
    tool = ListFilesTool()
    assert tool.name == "list_files"
    assert isinstance(tool.description, str)
    assert tool.parameters_schema == {
        "path": "The relative or absolute path to the directory to list.",
        "recursive": "Whether to list files recursively. Defaults to False."
    }

def test_list_files_tool_non_recursive(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListFilesTool()
    (tmp_path / "file1.txt").write_text("1")
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.txt").write_text("2") # This should not be listed

    result_str = tool.execute({"path": "."}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert sorted(result) == sorted(["dir1/", "file1.txt"])

def test_list_files_tool_recursive(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListFilesTool()
    (tmp_path / "file1.txt").write_text("1")
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    (dir1 / "file2.txt").write_text("2")
    dir2 = dir1 / "dir2"
    dir2.mkdir()
    (dir2 / "file3.txt").write_text("3")

    # Test from root of tmp_path
    result_str = tool.execute({"path": ".", "recursive": True}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    # Expected paths are relative to the directory passed to the tool (tmp_path in this case)
    # Using set for comparison as order from os.walk can sometimes vary based on FS an OS.
    # The tool itself sorts the final list, so direct list comparison is also good.
    expected_files = sorted(["file1.txt", "dir1/", "dir1/file2.txt", "dir1/dir2/", "dir1/dir2/file3.txt"])
    assert sorted(result) == expected_files

    # Test from a sub-directory
    result_str_subdir = tool.execute({"path": "dir1", "recursive": True}, agent_tools_instance=mock_agent_tools_instance)
    result_subdir = json.loads(result_str_subdir)
    expected_files_subdir = sorted(["file2.txt", "dir2/", "dir2/file3.txt"])
    assert sorted(result_subdir) == expected_files_subdir


def test_list_files_tool_empty_dir(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListFilesTool()
    empty_subdir = tmp_path / "empty_dir"
    empty_subdir.mkdir()

    result_str = tool.execute({"path": "empty_dir"}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert result == []

def test_list_files_tool_non_existent_dir(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = ListFilesTool()
    result = tool.execute({"path": "non_existent_dir"}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Directory not found" in result

# --- SearchFilesTool Tests ---
def test_search_files_tool_properties():
    tool = SearchFilesTool()
    assert tool.name == "search_files"
    assert isinstance(tool.description, str)
    assert tool.parameters_schema == {
        "path": "The relative or absolute path to the directory to search within.",
        "regex": "The Python regex to search for within file lines.",
        "file_pattern": "Optional glob pattern to filter files (e.g., '*.py', 'test_*.py'). Defaults to all files ('*')."
    }

def test_search_files_tool_found_matches(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = SearchFilesTool()
    (tmp_path / "fileA.txt").write_text("hello world\nsearch me please\nworld again")
    (tmp_path / "fileB.log").write_text("another WORLD line\nno search here") # Case sensitive regex
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "fileC.txt").write_text("world in subdir")

    result_str = tool.execute({"path": ".", "regex": "world"}, agent_tools_instance=mock_agent_tools_instance)
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
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = SearchFilesTool()
    (tmp_path / "fileA.txt").write_text("hello there\nsearch me please")
    result_str = tool.execute({"path": ".", "regex": "world"}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert len(result) == 0

def test_search_files_tool_with_file_pattern(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = SearchFilesTool()
    (tmp_path / "fileA.txt").write_text("hello world")
    (tmp_path / "fileB.log").write_text("world in log")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.txt").write_text("another world in txt")

    result_str = tool.execute({"path": ".", "regex": "world", "file_pattern": "*.txt"}, agent_tools_instance=mock_agent_tools_instance)
    result = json.loads(result_str)
    assert len(result) == 2
    files_found = {item['file'] for item in result}
    assert "fileA.txt" in files_found
    assert str(Path("docs") / "readme.txt") in files_found # Path("docs/readme.txt").as_posix()

def test_search_files_tool_invalid_regex(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = SearchFilesTool()
    # Test with an invalid regex pattern like an unclosed bracket
    result = tool.execute({"path": ".", "regex": "["}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Invalid regex" in result

def test_search_files_tool_dir_not_found(tmp_path: Path): # Renaming to path_not_found would be more consistent but problem asks for dir_not_found
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = SearchFilesTool()
    result = tool.execute({"path": "nonexistent", "regex": "test"}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Directory not found" in result

def test_search_files_tool_missing_params(tmp_path: Path):
    mock_agent_tools_instance = MockAgentToolsInstance(cwd=str(tmp_path))
    tool = SearchFilesTool()
    # Test missing 'path'
    result_no_path = tool.execute({"regex": "test"}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Missing required parameter 'path'" in result_no_path
    # Test missing 'regex'
    result_no_regex = tool.execute({"path": "."}, agent_tools_instance=mock_agent_tools_instance)
    assert "Error: Missing required parameter 'regex'" in result_no_regex
