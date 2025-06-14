import unittest
from unittest.mock import MagicMock

from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import to_plain_text # Import for checking display_meta

from src.autocompletion import AtMentionCompleter
from src.file_cache import FileCache # For type hinting and creating mock spec

class TestAtMentionCompleter(unittest.TestCase):

    def setUp(self):
        self.mock_file_cache = MagicMock(spec=FileCache)
        self.completer = AtMentionCompleter(file_cache=self.mock_file_cache)
        self.complete_event = CompleteEvent() # Dummy event

    def _get_completions_list(self, text_before_cursor: str) -> list[Completion]:
        document = Document(text=text_before_cursor, cursor_position=len(text_before_cursor))
        return list(self.completer.get_completions(document, self.complete_event))

    def test_no_file_cache(self):
        completer_no_cache = AtMentionCompleter(file_cache=None)
        completions = list(completer_no_cache.get_completions(
            Document(text="@some", cursor_position=5), self.complete_event
        ))
        self.assertEqual(completions, [])

    def test_no_at_symbol(self):
        self.mock_file_cache.get_paths.return_value = ["file1.txt", "file2.py"]
        completions = self._get_completions_list("file")
        self.assertEqual(completions, [])

    def test_at_symbol_no_partial_path(self):
        self.mock_file_cache.get_paths.return_value = ["file1.txt", "file2.py"]
        # Word before cursor logic in Document might return '@' or empty string depending on exact position.
        # If word_before_cursor is just "@", len(word_before_cursor) > 1 fails.
        completions = self._get_completions_list("@")
        self.assertEqual(completions, [])

        # Test with space after @ - get_word_before_cursor would be "@"
        completions = self._get_completions_list("@ ")
        self.assertEqual(completions, [])


    def test_empty_file_cache(self):
        self.mock_file_cache.get_paths.return_value = []
        completions = self._get_completions_list("@pa")
        self.assertEqual(completions, [])

    def test_simple_completion_match(self):
        self.mock_file_cache.get_paths.return_value = ["path/to/file1.txt", "path/another.py"]
        completions = self._get_completions_list("@path/to")

        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].text, "path/to/file1.txt")
        self.assertEqual(completions[0].start_position, -len("path/to")) # Replaces "path/to"
        self.assertEqual(to_plain_text(completions[0].display_meta), "file mention")

    def test_multiple_completion_matches(self):
        self.mock_file_cache.get_paths.return_value = [
            "src/file_cache.py",
            "src/cli.py",
            "docs/README.md"
        ]
        completions = self._get_completions_list("@src/")

        self.assertEqual(len(completions), 2)
        texts = sorted([c.text for c in completions])
        self.assertEqual(texts, ["src/cli.py", "src/file_cache.py"])

        # Check one completion for correct start_position
        # Assuming "src/file_cache.py" is one of them
        for comp in completions:
            if comp.text == "src/file_cache.py":
                self.assertEqual(comp.start_position, -len("src/"))
                break
        else:
            self.fail("Expected completion not found for detailed check.")

    def test_no_match(self):
        self.mock_file_cache.get_paths.return_value = ["file1.txt", "another/file2.py"]
        completions = self._get_completions_list("@nonexistent")
        self.assertEqual(completions, [])

    def test_case_insensitive_matching(self):
        self.mock_file_cache.get_paths.return_value = ["Path/To/File1.TXT", "path/ANOTHER.PY"]
        completions = self._get_completions_list("@path/to") # Lowercase partial

        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].text, "Path/To/File1.TXT") # Original casing
        self.assertEqual(completions[0].start_position, -len("path/to"))

        completions_upper = self._get_completions_list("@PATH/AN") # Uppercase partial
        self.assertEqual(len(completions_upper), 1)
        self.assertEqual(completions_upper[0].text, "path/ANOTHER.PY")
        self.assertEqual(completions_upper[0].start_position, -len("PATH/AN"))


    def test_completion_with_text_after_cursor(self):
        # The completer should only care about text before cursor for generating completions
        self.mock_file_cache.get_paths.return_value = ["src/important_file.py"]
        document = Document(text="@src/imp some_other_text", cursor_position=len("@src/imp"))
        completions = list(self.completer.get_completions(document, self.complete_event))

        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].text, "src/important_file.py")
        self.assertEqual(completions[0].start_position, -len("src/imp"))

    def test_word_boundary_logic(self):
        # If the @mention is not the current word, it shouldn't complete
        self.mock_file_cache.get_paths.return_value = ["some/path.txt"]

        # Case 1: Space after @mention part - word is just "@some/pa"
        completions_space_after = self._get_completions_list("@some/pa ")
        # Here, get_word_before_cursor(WORD=True) if cursor is at end would be empty or just space.
        # Let's refine how _get_completions_list calls Document to simulate typing.
        # If user typed "@some/pa " and then hits tab, word_before_cursor might be considered empty.
        # The current AtMentionCompleter relies on document.get_word_before_cursor.
        # prompt_toolkit's get_word_before_cursor usually stops at space.
        # So if text is "@some/pa ", word_before_cursor is "" if cursor is at end.
        # If text is "@some/pa", word_before_cursor is "@some/pa".

        document_at_end_of_word = Document(text="@some/pa", cursor_position=len("@some/pa"))
        completions_at_end = list(self.completer.get_completions(document_at_end_of_word, self.complete_event))
        self.assertEqual(len(completions_at_end), 1)
        self.assertEqual(completions_at_end[0].text, "some/path.txt")

        # If cursor is after a space following the word, get_word_before_cursor is empty.
        document_after_space = Document(text="@some/pa ", cursor_position=len("@some/pa "))
        completions_after_space = list(self.completer.get_completions(document_after_space, self.complete_event))
        self.assertEqual(completions_after_space, [])


    def test_special_characters_in_partial_path(self):
        # Test if partial path with characters like '-' or '.' works
        self.mock_file_cache.get_paths.return_value = ["project-alpha/src/my-file.v1.0.py"]
        completions = self._get_completions_list("@project-a")
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].text, "project-alpha/src/my-file.v1.0.py")
        self.assertEqual(completions[0].start_position, -len("project-a"))

        completions_dot = self._get_completions_list("@project-alpha/src/my-file.v1")
        self.assertEqual(len(completions_dot), 1)
        self.assertEqual(completions_dot[0].text, "project-alpha/src/my-file.v1.0.py")
        self.assertEqual(completions_dot[0].start_position, -len("project-alpha/src/my-file.v1"))

if __name__ == '__main__':
    unittest.main()
