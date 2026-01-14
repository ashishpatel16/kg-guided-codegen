from __future__ import annotations

from pathlib import Path


def tool_ls(directory: str | Path) -> list[str]:
    """
    List the entries in a directory (like `ls`).

    Returns a sorted list of entry names (files and subdirectories).
    """
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_dir():
        raise NotADirectoryError(str(path))

    return sorted(p.name for p in path.iterdir())


def tool_grep(filename: str | Path, query: str = "") -> str | list[str]:
    """
    Search a file for a substring (like a minimal `grep`).

    - If query is empty, returns the full file contents.
    - Else, returns a list of matching lines (case-sensitive literal match),
      with trailing newline characters removed.
    """
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file():
        raise IsADirectoryError(str(path))

    text = path.read_text(encoding="utf-8")
    if query == "":
        return text

    return [line for line in text.splitlines() if query in line]


if __name__ == "__main__":
    print(tool_ls("."))
    print(tool_grep("src/program_analysis/example.py", "def"))
