import re

# Regex to capture file paths starting with @.
# Handles Unix and Windows style paths, including spaces and special characters.
# It defines a path as one or more non-space path characters,
# optionally followed by groups of (a space, then more non-space path characters).
# This means the overall path cannot begin or end with a space.
# Regex for extracting file paths from @-mentions.
# It prioritizes precision for file reading, using non-greedy matching for path characters
# and lookaheads for common delimiters or stop words.
# Path characters: word characters, dot, hyphen, space, forward/backward slashes.
# Delimiters:
#   - Space followed by a common stop word (and, or, is, etc.) then a space.
#   - Optional spaces then common punctuation (comma, semicolon, parentheses).
#   - Optional spaces then end of string.
#   - Space then another @-mention.
FILE_MENTION_REGEX = re.compile(
    r"@([\w.\-\s\/\\]+?)(?=\s+(?:and|or|is|the|a|for|to|in|on|with|by|then)\s|\s*[,;()]|\s*$|\s+@)"
)


def extract_file_mentions(text: str) -> list[str]:
    """
    Extracts file path mentions from a given text.
    """
    return FILE_MENTION_REGEX.findall(text)
