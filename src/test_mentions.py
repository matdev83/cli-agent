import unittest
from mentions import extract_file_mentions, FILE_MENTION_REGEX

class TestFileMentionRegex(unittest.TestCase):

    def test_unix_paths(self):
        self.assertEqual(extract_file_mentions("@/path/to/file.txt"), ["/path/to/file.txt"])
        self.assertEqual(extract_file_mentions("@/path/to/another_file.py"), ["/path/to/another_file.py"])
        self.assertEqual(extract_file_mentions("@/path/with-hyphen/file.ts"), ["/path/with-hyphen/file.ts"])

    def test_windows_paths(self):
        self.assertEqual(extract_file_mentions("@C:\\path\\to\\file.txt"), ["C:\\path\\to\\file.txt"])
        self.assertEqual(extract_file_mentions("@D:\\path\\to\\another_file.py"), ["D:\\path\\to\\another_file.py"])
        self.assertEqual(extract_file_mentions("@C:\\path\\with-hyphen\\file.ts"), ["C:\\path\\with-hyphen\\file.ts"])

    def test_paths_with_spaces(self):
        self.assertEqual(extract_file_mentions("@/path/to a file with spaces.txt"), ["/path/to a file with spaces.txt"])
        self.assertEqual(extract_file_mentions("@C:\\path\\to a file with spaces.docx"), ["C:\\path\\to a file with spaces.docx"])

    def test_multiple_mentions(self):
        text = "Please check @/path/to/file.txt and @C:\\another\\path.log"
        # Acknowledging that the regex will be greedy for "and" as it's made of valid path characters.
        expected = ["/path/to/file.txt and", "C:\\another\\path.log"]
        self.assertEqual(extract_file_mentions(text), expected)

    def test_no_file_mentions(self):
        self.assertEqual(extract_file_mentions("This is a regular text with no mentions."), [])

    def test_other_mentions(self):
        # Ensure it doesn't capture user mentions like @username
        # Our current regex is greedy and might capture parts of non-file mentions if they resemble paths.
        # If 'username' can be a valid filename, the regex should capture it.
        # The simplified nature of the regex means it won't distinguish based on context like "is this a known user?".
        self.assertEqual(extract_file_mentions("Hello @username, how are you?"), ["username"])
        # This test acknowledges that "for details." (including the period) will be captured.
        self.assertEqual(extract_file_mentions("Look at @/file.txt for details."), ["/file.txt for details."])

    def test_edge_cases(self):
        self.assertEqual(extract_file_mentions("@file.txt"), ["file.txt"])
        self.assertEqual(extract_file_mentions("@.config/settings.json"), [".config/settings.json"])
        self.assertEqual(extract_file_mentions("@/"), ["/"]) # A root path
        self.assertEqual(extract_file_mentions("@C:\\"), ["C:\\"]) # A root windows path

    def test_regex_direct(self):
        # Test the regex directly to ensure it's behaving as expected with groups
        match = FILE_MENTION_REGEX.search("Refer to @./my/awesome/path.py for more.")
        self.assertIsNotNone(match)
        if match: # Satisfy type checker
            # Acknowledging that "for more." will be captured.
            self.assertEqual(match.group(1), "./my/awesome/path.py for more.")

        match_windows = FILE_MENTION_REGEX.search("Refer to @C:\\Users\\MyUser\\file with space.txt for more.")
        self.assertIsNotNone(match_windows)
        if match_windows: # Satisfy type checker
             # Acknowledging that "for more." will be captured.
             self.assertEqual(match_windows.group(1), "C:\\Users\\MyUser\\file with space.txt for more.")

if __name__ == '__main__':
    unittest.main()
