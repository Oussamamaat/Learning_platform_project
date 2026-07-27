"""
Index tracker for the knowledge base corpus.
"""

from pathlib import Path

INDEX_PATH = Path("raw/CORPUS_INDEX.csv")


def init_index():
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text("path;title;rights_status;domain;scope\n", encoding="utf-8")


def append_to_index(path: Path, title: str, rights: str, domain: str, scope: str):
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(f"{path};{title};{rights};{domain};{scope}\n")
