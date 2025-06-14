from __future__ import annotations

import logging
from typing import Iterable, Optional, TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from src.file_cache import FileCache  # Use type checking import for FileCache

logger = logging.getLogger(__name__)


class AtMentionCompleter(Completer):
    """
    Custom completer for at-mentions (@file_path) based on a FileCache.
    """

    def __init__(self, file_cache: Optional[FileCache]):
        self.file_cache = file_cache

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """
        Yields completions for at-mentions.
        """
        if self.file_cache is None:
            # logger.debug("AtMentionCompleter: FileCache is not available.")
            return []

        # text_before_cursor = document.text_before_cursor # F841: Unused
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        # logger.debug(f"Text before cursor: '{text_before_cursor}'")
        # logger.debug(f"Word before cursor: '{word_before_cursor}'")

        if word_before_cursor.startswith("@") and len(word_before_cursor) > 1:
            # User has typed "@" followed by some characters
            partial_path = word_before_cursor[1:]
            # logger.debug(f"Partial path for completion: '{partial_path}'")

            cached_paths = self.file_cache.get_paths()
            if not cached_paths:
                # logger.debug("No paths in FileCache.")
                return []

            # logger.debug(f"Searching in {len(cached_paths)} cached paths.")
            for file_path in cached_paths:
                if partial_path.lower() in file_path.lower():  # Case-insensitive matching
                    # logger.debug(f"Found match: '{file_path}'")
                    yield Completion(
                        text=file_path,
                        start_position=-len(partial_path),  # Replace the typed partial path
                        display=file_path,
                        display_meta="file mention",
                    )
        # else:
        # logger.debug("No '@' prefix or not enough characters for completion.")
        # return [] # No relevant completions if no @-mention is being typed
        return []  # Ensure it returns an iterable if no conditions met
