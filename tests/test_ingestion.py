"""
Tests for the Stage-2 ingestion fixes: heading-aware chunking, per-file
language detection, and path-based domain detection. Extended 2026-08-11
with ingest_directory's domain resolution (--domain flag, tenant-default
fallback, never-NULL) -- see the "Automatic Domain Routing" plan, section 3.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.ingestion import (
    chunk_document,
    detect_document_domain,
    detect_document_language,
    ingest_directory,
    parse_document_to_markdown,
    split_by_headings,
)


# --- split_by_headings -------------------------------------------------

def test_splits_on_headings_of_any_level():
    text = "# Title\n\nintro\n\n## Section A\n\nbody a\n\n### Sub B\n\nbody b\n"
    sections = split_by_headings(text)
    assert [h for h, _ in sections] == ["Title", "Section A", "Sub B"]


def test_preamble_before_first_heading_kept_as_empty_heading_section():
    text = "some intro text\n\n## Section A\n\nbody\n"
    sections = split_by_headings(text)
    assert sections[0] == ("", "some intro text")
    assert sections[1][0] == "Section A"


def test_no_headings_returns_single_section():
    assert split_by_headings("just plain text") == [("", "just plain text")]


# --- chunk_document: the citation-preservation guarantee ----------------

def test_heading_travels_with_every_chunk_of_its_section():
    """The core bug fix: a heading carrying an article reference must never
    be separable from its own body text, regardless of chunk boundaries."""
    text = (
        "## Obligations du travailleur (Art. 283)\n\n"
        "Chaque travailleur doit prendre soin de sa securite."
    )
    chunks = chunk_document(text)
    assert len(chunks) == 1
    assert "Art. 283" in chunks[0]["content"]
    assert "Chaque travailleur" in chunks[0]["content"]


def test_short_adjacent_sections_are_packed_together():
    """Many one-sentence sections (this corpus's 'المادة 1'..'المادة 5'
    style articles) should be packed into shared chunks up to chunk_size,
    not isolated one-per-chunk -- isolating them measurably hurt recall
    (docs/architecture/data-and-retrieval.md)."""
    text = "\n\n".join(
        f"## Article {i}\n\nShort body text {i}." for i in range(1, 6)
    )
    chunks = chunk_document(text, chunk_size=400)
    assert len(chunks) < 5, "short sections should be packed, not one chunk each"
    # Every article's own heading must still be inline in *some* chunk's content.
    for i in range(1, 6):
        assert any(f"Article {i}" in c["content"] for c in chunks)


def test_long_section_is_still_split_with_heading_reprefixed():
    long_body = " ".join(["word"] * 200)  # comfortably over chunk_size
    text = f"## Long Section (Art. 99)\n\n{long_body}"
    chunks = chunk_document(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert "Art. 99" in c["content"], "every split piece must keep the heading"


def test_empty_document_returns_no_chunks():
    assert chunk_document("") == []


def test_no_heading_document_still_chunks():
    chunks = chunk_document("Just a paragraph with no markdown headings at all.")
    assert len(chunks) == 1
    assert chunks[0]["heading"] == ""


# --- detect_document_language -------------------------------------------

def test_detects_arabic_script_document():
    text = "المادة 283: يجب على المشغل أن يوفر معدات الوقاية الشخصية."
    assert detect_document_language(text) == "ar"


def test_detects_french_document():
    text = "L'employeur doit fournir des equipements de protection individuelle adaptes."
    assert detect_document_language(text) == "fr"


def test_mostly_latin_with_some_arabic_terms_stays_french():
    text = "Le contrat prevoit la reference suivante: Loi N 42-25, ISO 45001."
    assert detect_document_language(text) == "fr"


# --- detect_document_domain ----------------------------------------------

def test_derives_domain_from_shared_convention():
    source_dir = Path("raw/shared")
    file_path = Path("raw/shared/industrial/text/1.1_code_du_travail.md")
    assert detect_document_domain(file_path, source_dir) == "industrial"


def test_normalizes_securite_physique_alias():
    source_dir = Path("raw/shared")
    file_path = Path("raw/shared/securite_physique/text/2.1_loi_27_06.md")
    assert detect_document_domain(file_path, source_dir) == "securite"


def test_returns_none_for_path_not_matching_convention():
    source_dir = Path("raw/tenant_placeholder")
    file_path = Path("raw/tenant_placeholder/random_file.md")
    assert detect_document_domain(file_path, source_dir) is None


# --- parse_document_to_markdown: OCR pluggable-but-disabled --------------

def test_md_passes_through_unchanged(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nBody text.", encoding="utf-8")
    assert parse_document_to_markdown(f) == "# Title\n\nBody text."


def test_txt_passes_through_unchanged(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("plain text content", encoding="utf-8")
    assert parse_document_to_markdown(f) == "plain text content"


def test_pdf_raises_clear_not_implemented_error(tmp_path):
    """Must fail loudly and specifically -- not silently, and not with an
    opaque UnicodeDecodeError from trying to read binary bytes as text --
    until the OCR spike's Arabic-script fidelity is actually verified."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake binary content")
    with pytest.raises(NotImplementedError, match="OCR"):
        parse_document_to_markdown(f)


def test_unsupported_extension_raises_value_error(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_bytes(b"not really a docx")
    with pytest.raises(ValueError, match="unsupported"):
        parse_document_to_markdown(f)


# --- ingest_directory: domain resolution, never NULL ----------------------
# ingest_file (which touches Postgres via ingest_text/insert_documents) is
# mocked so these test only the domain-resolution order: path convention ->
# --domain flag -> settings.default_domain (logged, never silently guessed).

def _fake_ingest_file(**kwargs):
    return {"chunks_created": 1, **kwargs}


def test_path_convention_wins_over_explicit_domain_flag(tmp_path, capsys):
    (tmp_path / "industrial" / "text").mkdir(parents=True)
    f = tmp_path / "industrial" / "text" / "doc.md"
    f.write_text("# Heading\n\nBody", encoding="utf-8")

    with patch("app.services.ingestion.ingest_file", side_effect=_fake_ingest_file) as mock_ingest:
        ingest_directory(source_dir=str(tmp_path), tenant_id="t1", domain="blockchain")

    assert mock_ingest.call_args.kwargs["domain"] == "industrial"
    assert "[INGEST WARNING]" not in capsys.readouterr().out


def test_explicit_domain_flag_used_when_path_does_not_match_convention(tmp_path, capsys):
    f = tmp_path / "doc.md"
    f.write_text("# Heading\n\nBody", encoding="utf-8")

    with patch("app.services.ingestion.ingest_file", side_effect=_fake_ingest_file) as mock_ingest:
        ingest_directory(source_dir=str(tmp_path), tenant_id="t1", domain="securite")

    assert mock_ingest.call_args.kwargs["domain"] == "securite"
    assert "[INGEST WARNING]" not in capsys.readouterr().out


def test_falls_back_to_tenant_default_and_warns_when_neither_available(tmp_path, capsys):
    f = tmp_path / "doc.md"
    f.write_text("# Heading\n\nBody", encoding="utf-8")

    with patch("app.services.ingestion.ingest_file", side_effect=_fake_ingest_file) as mock_ingest:
        ingest_directory(source_dir=str(tmp_path), tenant_id="t1", domain=None)

    from app.config import get_settings
    assert mock_ingest.call_args.kwargs["domain"] == get_settings().default_domain
    out = capsys.readouterr().out
    assert "[INGEST WARNING]" in out
    assert "doc.md" in out


def test_domain_is_never_none_on_the_ingest_file_call(tmp_path):
    """The never-NULL guarantee itself: regardless of path/flag/default,
    ingest_file must never be called with domain=None -- a NULL domain row
    is invisible to every domain-filtered query (WHERE domain = %s never
    matches NULL), which is the exact bug this closes (2026-08-11 E2E audit,
    two untagged files silently unreachable until backfilled by hand)."""
    f = tmp_path / "doc.md"
    f.write_text("# Heading\n\nBody", encoding="utf-8")

    with patch("app.services.ingestion.ingest_file", side_effect=_fake_ingest_file) as mock_ingest:
        ingest_directory(source_dir=str(tmp_path), tenant_id="t1", domain=None)

    assert mock_ingest.call_args.kwargs["domain"] is not None
