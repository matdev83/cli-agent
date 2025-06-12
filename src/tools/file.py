from __future__ import annotations

import os
import re
import fnmatch
from pathlib import Path
from typing import List, Dict, Any


def read_file(path: str) -> str:
    """Return the entire contents of *path* as a string."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return f.read()


def write_to_file(path: str, content: str) -> None:
    """Write *content* to *path*, creating directories as needed."""
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(content)


def _parse_diff_blocks(diff: str) -> List[tuple[str, str]]:
    pattern = re.compile(
        r"------- SEARCH\n(?P<search>.*?)\n=======\n(?P<replace>.*?)\n\+\+\+\+\+\+\+ REPLACE",
        re.DOTALL,
    )
    blocks = []
    for m in pattern.finditer(diff):
        blocks.append((m.group("search"), m.group("replace")))
    if not blocks:
        raise ValueError("No diff blocks found")
    return blocks


def replace_in_file(path: str, diff: str) -> None:
    """Apply SEARCH/REPLACE diff blocks to a file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    blocks = _parse_diff_blocks(diff)
    for search, replace in blocks:
        if search not in text:
            raise ValueError("search block not found in file")
        text = text.replace(search, replace, 1)
    p.write_text(text, encoding="utf-8")


def list_files(path: str, recursive: bool = False) -> List[str]:
    p = Path(path)
    if not recursive:
        return sorted([entry.name for entry in p.iterdir()])
    results: List[str] = []
    for root, dirs, files in os.walk(p):
        for name in files:
            rel = str(Path(root, name).relative_to(p)).replace("\\", "/")
            results.append(rel)
        for name in dirs:
            rel = str(Path(root, name).relative_to(p)).replace("\\", "/")
            results.append(rel + "/")
    return sorted(results)


def search_files(path: str, regex: str, file_pattern: str | None = None) -> List[Dict[str, Any]]:
    pattern = re.compile(regex)
    root = Path(path)
    file_glob = file_pattern or "*"
    matches: List[Dict[str, Any]] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not fnmatch.fnmatch(filename, file_glob):
                continue
            file_path = Path(dirpath) / filename
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if pattern.search(line):
                        matches.append(
                            {
                                "file": str(file_path.relative_to(root)),
                                "line": lineno,
                                "content": line.rstrip("\n"),
                            }
                        )
    return matches
