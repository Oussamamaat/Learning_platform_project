"""
Index tracker + shared file-write helper for the knowledge base corpus.
"""

import csv
from pathlib import Path

INDEX_PATH = Path("raw/CORPUS_INDEX.csv")
_INDEX_FIELDS = ("path", "title", "rights_status", "domain", "scope")


def init_index():
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=";").writerow(_INDEX_FIELDS)


def append_to_index(path: Path, title: str, rights: str, domain: str, scope: str):
    with open(INDEX_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=";").writerow([path, title, rights, domain, scope])


def write_corpus_doc(path: Path, content: str) -> None:
    """Write one generated corpus document to `path`, creating parent dirs
    as needed. Shared by generate_corpus.py / generate_corpus_arabic.py /
    generate_corpus_generalization.py, which otherwise each hand-rolled the
    same mkdir + write_text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
