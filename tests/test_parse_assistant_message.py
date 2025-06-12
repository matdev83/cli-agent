
from src.assistant_message import parse_assistant_message, TextContent, ToolUse


def test_parse_plain_text():
    msg = "Hello world"
    result = parse_assistant_message(msg)
    assert result == [TextContent(type="text", content="Hello world")]


def test_parse_single_tool():
    msg = "Do it <read_file><path>foo.txt</path></read_file> done"
    result = parse_assistant_message(msg)
    assert result[0].type == "text" and "Do it" in result[0].content
    tool = result[1]
    assert isinstance(tool, ToolUse)
    assert tool.name == "read_file"
    assert tool.params == {"path": "foo.txt"}
    assert not tool.partial
    assert result[2].type == "text" and "done" in result[2].content


def test_parse_multiple_tools():
    msg = (
        "Start <read_file><path>a.txt</path></read_file>"
        " between <write_to_file><path>b.txt</path><content>hi</content></write_to_file> end"
    )
    result = parse_assistant_message(msg)
    assert len(result) == 5
    assert isinstance(result[1], ToolUse)
    assert result[1].name == "read_file"
    assert isinstance(result[3], ToolUse)
    assert result[3].name == "write_to_file"


def test_partial_tool():
    msg = "unfinished <write_to_file><path>x.txt</path><content>hi"
    result = parse_assistant_message(msg)
    # Expect 1 because malformed XML is treated as a single TextContent block
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].content == msg # The whole message becomes the content
