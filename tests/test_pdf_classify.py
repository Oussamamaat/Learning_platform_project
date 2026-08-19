"""
Tests for app.services.pdf_classify: the per-page NATIVE/EMPTY/OCR_REQUIRED/
OCR_PREFERRED decision that replaced ingestion.py's old
`len(text) < PDF_TEXT_MIN_CHARS` floor.

Two layers:
  - Unit tests against synthetic fake pages, one per branch.
  - Regression tests against the real Moroccan PDFs committed at
    tests/data/real_pdfs/, pinning the exact per-page classification
    measured when this module was written -- see pdf_classify.py's module
    docstring for how those numbers were obtained. This is what stops a
    future threshold tweak from silently reintroducing the old floor's
    silent-data-loss failure mode.
"""
from pathlib import Path

import pytest

from app.services.pdf_classify import (
    BAD_CHAR_MAX,
    DENSITY_EMPTY_MAX,
    DENSITY_UNUSABLE_MAX,
    PageStrategy,
    classify_page,
    extract_signals,
)

REAL_PDF_DIR = Path(__file__).parent / "data" / "real_pdfs"


class _FakeDict(dict):
    def get_object(self):
        return self


class _FakeBox:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class _FakePage:
    """US Letter (612x792pt = 8.5x11in, area 93.5in^2) unless overridden."""

    def __init__(self, *, fonts=None, images=None, width=612, height=792):
        self.mediabox = _FakeBox(width, height)
        self._resources = {}
        if fonts is not None:
            self._resources["/Font"] = _FakeDict(
                {f"/F{i}": _FakeDict({}) for i in range(fonts)}
            )
        if images is not None:
            self._resources["/XObject"] = _FakeDict(
                {f"/Im{i}": _FakeDict(img) for i, img in enumerate(images)}
            )

    def get(self, key):
        if key == "/Resources":
            return _FakeDict(self._resources)
        return None


def _image_xobject(px_w, px_h):
    return {"/Subtype": "/Image", "/Width": px_w, "/Height": px_h}


# --- unit: one case per branch --------------------------------------------

def test_no_fonts_no_images_is_empty():
    page = _FakePage(fonts=0, images=None)
    decision = classify_page(page, text="")
    assert decision.strategy == PageStrategy.EMPTY


def test_no_fonts_with_image_is_ocr_required():
    """No font resources at all -- there is no text layer to trust,
    regardless of density. This is the "genuinely blank scanned page,
    no OCR engine" case the original test_ingestion.py fixture covered."""
    page = _FakePage(fonts=0, images=[_image_xobject(100, 100)])
    decision = classify_page(page, text="")
    assert decision.strategy == PageStrategy.OCR_REQUIRED


def test_full_page_scan_with_minimal_text_is_ocr_required():
    """guide_rh_sante_ar.pdf page 27's shape: a full-page raster at scan
    resolution with only a handful of stray characters (85 chars on the
    real page -- five over the OLD len(text) < 80 floor, which wrongly
    accepted it as NATIVE). Density this low (< DENSITY_UNUSABLE_MAX) means
    OCR is not just preferred but required -- there's nothing usable to
    fall back to."""
    # 1241x1754 @ 150 DPI on an 8.27x11.69in (A4) page, matching the real
    # page's aspect ratio closely.
    page = _FakePage(fonts=3, images=[_image_xobject(1241, 1754)], width=595, height=842)
    text = "Stamp. " * 12  # ~84 chars, density well under 3.0/in^2 on this page
    decision = classify_page(page, text=text)
    assert decision.strategy == PageStrategy.OCR_REQUIRED
    assert decision.signals.has_full_page_raster
    assert decision.signals.char_density < DENSITY_UNUSABLE_MAX


def test_full_page_scan_with_moderate_stamp_text_is_ocr_preferred():
    """guide_rh_sante_ar.pdf page 1's shape: real fonts present, a
    full-page raster at scan resolution, and a moderate amount of header-
    stamp text (349 chars on the real page, 3.6 chars/in^2) -- above
    DENSITY_UNUSABLE_MAX but well under DENSITY_SCAN_MAX, so OCR would
    improve it but the embedded text is still a usable fallback rather
    than a hard failure."""
    page = _FakePage(fonts=3, images=[_image_xobject(1241, 1754)], width=595, height=842)
    text = "Stamp header text only. " * 14  # ~336 chars, density ~3.5/in^2
    decision = classify_page(page, text=text)
    assert decision.strategy == PageStrategy.OCR_PREFERRED
    assert decision.signals.has_full_page_raster


def test_full_page_raster_over_real_text_is_ocr_preferred():
    """profil_sst_maroc_fr.pdf page 50's shape: a full-page diagram
    (organigramme) with real, substantial text extracted from box labels
    -- OCR would preserve the layout better, but native text is a usable
    fallback, not a failure. A4 page (595x842pt), matching the real
    page's dimensions and the image's 74-implied-DPI/aspect-ratio fit
    measured against it -- a Letter-sized fake page here fails to
    reproduce either (image DPI drops under RASTER_MIN_DPI)."""
    page = _FakePage(fonts=3, images=[_image_xobject(610, 1014)], width=595, height=842)
    text = "DIRECTION DE L'EMPLOI DIRECTION DU TRAVAIL " * 20  # dense
    decision = classify_page(page, text=text)
    assert decision.strategy == PageStrategy.OCR_PREFERRED


def test_small_logo_image_does_not_trigger_full_page_raster():
    """avis_cese_sst_fr.pdf page 1's shape: a small logo (wrong aspect
    ratio and/or low implied DPI for the page) alongside normal body text
    must classify NATIVE, not be mistaken for a scan."""
    page = _FakePage(fonts=6, images=[_image_xobject(513, 299)], width=612, height=792)
    text = "Avis du CESE sur la sante et securite au travail. " * 8
    decision = classify_page(page, text=text)
    assert decision.strategy == PageStrategy.NATIVE
    assert not decision.signals.has_full_page_raster


def test_normal_body_text_page_is_native():
    page = _FakePage(fonts=6, images=None)
    text = "Chaque travailleur doit prendre soin de sa securite. " * 40
    decision = classify_page(page, text=text)
    assert decision.strategy == PageStrategy.NATIVE


def test_blank_trailing_page_is_empty_not_ocr_required():
    """profil_sst_maroc_fr.pdf page 138's shape: negligible text, no
    images at all -- must be EMPTY (never sent to OCR), which is what
    stops a single blank trailing page from failing an otherwise fully
    readable 138-page document."""
    page = _FakePage(fonts=1, images=None)
    decision = classify_page(page, text="1")
    assert decision.strategy == PageStrategy.EMPTY
    assert decision.signals.char_density < DENSITY_EMPTY_MAX


def test_broken_encoding_text_is_ocr_required():
    page = _FakePage(fonts=2, images=None)
    text = "�" * 200 + "readable " * 20
    decision = classify_page(page, text=text)
    assert decision.strategy == PageStrategy.OCR_REQUIRED
    assert decision.signals.bad_char_ratio > BAD_CHAR_MAX


def test_extract_signals_reuses_provided_text_without_recalling_extract():
    """Callers that already have page.extract_text() shouldn't pay for it
    twice -- extract_signals(text=...) must not call page.extract_text()."""
    class _NoExtract(_FakePage):
        def extract_text(self):
            raise AssertionError("extract_text() should not be called when text= is provided")

    page = _NoExtract(fonts=1, images=None)
    signals = extract_signals(page, text="hello")
    assert signals.char_count == 5


# --- regression: real Moroccan PDFs ----------------------------------------

pytestmark_real = pytest.mark.skipif(
    not REAL_PDF_DIR.exists(), reason="tests/data/real_pdfs/ fixtures not present"
)


def _classify_all_pages(pdf_path: Path) -> list[PageStrategy]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [classify_page(page).strategy for page in reader.pages]


@pytestmark_real
def test_avis_cese_all_pages_native():
    """40-page, fully digital-native document -- every page must classify
    NATIVE. This is the "don't cry wolf on a normal PDF" regression: the
    small logo on page 1 must not be mistaken for a scan."""
    strategies = _classify_all_pages(REAL_PDF_DIR / "avis_cese_sst_fr.pdf")
    assert len(strategies) == 40
    assert all(s == PageStrategy.NATIVE for s in strategies)


@pytestmark_real
def test_guide_rh_sante_ar_catches_both_real_scans():
    """27-page Arabic document with two genuinely scanned pages that the
    OLD len(text) < 80 floor silently accepted as NATIVE:
      - page 1: 349 chars of stamp text over a full-page scan (3.6
        chars/in^2) -- above the unusable floor, so OCR_PREFERRED (native
        text is a usable fallback, but degraded).
      - page 27: 85 chars (just 5 over the old 80-char floor) over a
        full-page scan (0.9 chars/in^2) -- below the unusable floor, so
        OCR_REQUIRED (nothing usable to fall back to)."""
    from pypdf import PdfReader

    reader = PdfReader(str(REAL_PDF_DIR / "guide_rh_sante_ar.pdf"))
    decisions = [classify_page(p) for p in reader.pages]
    assert len(decisions) == 27
    assert decisions[0].strategy == PageStrategy.OCR_PREFERRED  # page 1
    assert decisions[26].strategy == PageStrategy.OCR_REQUIRED  # page 27
    native_count = sum(1 for d in decisions if d.strategy == PageStrategy.NATIVE)
    assert native_count == 25


@pytestmark_real
def test_profil_sst_maroc_catches_diagram_scan_and_blank_trailing_page():
    """138-page document:
      - page 50 (organigramme, full-page raster over real text) must be
        OCR_PREFERRED, not silently treated as either a clean NATIVE page
        or a hard OCR failure.
      - page 136 (a genuine full-page scan, 0.0 chars/in^2 -- only a page-
        number stamp survives extraction) must be OCR_REQUIRED. Not caught
        by this plan's original page-range spot check (which only sampled
        pages 1-6 and the last 2); found by running the classifier over
        every page, which is the point of testing the full document
        instead of a hand-picked sample.
      - page 138 (blank trailing page, no text and no image) must be
        EMPTY, not OCR_REQUIRED -- under the OLD len(text) < 80 floor this
        single blank page failed the entire otherwise-readable document."""
    from pypdf import PdfReader

    reader = PdfReader(str(REAL_PDF_DIR / "profil_sst_maroc_fr.pdf"))
    decisions = [classify_page(p) for p in reader.pages]
    assert len(decisions) == 138
    assert decisions[49].strategy == PageStrategy.OCR_PREFERRED  # page 50
    assert decisions[135].strategy == PageStrategy.OCR_REQUIRED  # page 136
    assert decisions[137].strategy == PageStrategy.EMPTY  # page 138
    ocr_required_count = sum(1 for d in decisions if d.strategy == PageStrategy.OCR_REQUIRED)
    assert ocr_required_count == 1
