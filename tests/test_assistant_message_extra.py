from src.assistant_message import parse_assistant_message, TextContent, ToolUse


def test_unknown_tag_ignored():
    msg = 'test <unknown>data</unknown> end'
    result = parse_assistant_message(msg)
    # Expecting unknown tags to be preserved as full text, and segments to be distinct
    # based on how ET.tostring and .text/.tail work.
    # root.text should be "test " -> .strip() -> "test"
    # element is <unknown>data</unknown>, ET.tostring(element) -> "<unknown>data</unknown>" (approx)
    # element.tail is " end" -> .strip() -> "end"
    # So, we expect three distinct TextContent objects.
    assert len(result) == 3
    assert result[0] == TextContent(content='test') # root.text.strip()
    assert result[1] == TextContent(content='<unknown>data</unknown>') # ET.tostring of the unknown element
    assert result[2] == TextContent(content='end') # element.tail.strip()


def test_adjacent_text_segments():
    msg = 'hello <read_file><path>a.txt</path></read_file>world'
    result = parse_assistant_message(msg)
    assert isinstance(result[1], ToolUse)
    assert result[0].content == 'hello'
    assert result[2].content == 'world'
