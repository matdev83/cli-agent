from src.assistant_message import _extract_param, parse_assistant_message, TextContent, ToolUse


def test_extract_param_full():
    block = '<path>foo.txt</path>'
    assert _extract_param(block, 'path') == 'foo.txt'


def test_extract_param_partial():
    block = '<path>foo.txt'
    assert _extract_param(block, 'path') == 'foo.txt'


def test_extract_param_missing():
    block = '<other>hi</other>'
    assert _extract_param(block, 'path') is None


def test_unknown_tag_ignored():
    msg = 'test <unknown>data</unknown> end'
    result = parse_assistant_message(msg)
    assert result == [TextContent(type='text', content='test'), TextContent(type='text', content='data'), TextContent(type='text', content='end')]


def test_adjacent_text_segments():
    msg = 'hello <read_file><path>a.txt</path></read_file>world'
    result = parse_assistant_message(msg)
    assert isinstance(result[1], ToolUse)
    assert result[0].content == 'hello'
    assert result[2].content == 'world'
