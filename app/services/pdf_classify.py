"""
PDF Page Classification
────────────────────────
Decides, per page, whether app.services.ingestion._parse_pdf should trust the
PDF's embedded text layer or send the page to OCR (app.services.ocr). Pure and
dependency-light: takes a pypdf PageObject in, returns a PageDecision, does no
rendering/OCR/file I/O itself, so it is unit-testable without an OCR stack and
without opening a real PDF.

Replaces a single `len(text) < 80` threshold (see ingestion.py's git history),
which was area-blind and content-blind and, measured against the real
Moroccan PDFs at tests/data/real_pdfs/, was wrong in both directions:

  - guide_rh_sante_ar.pdf page 1: 349 chars of stamp/header text overlaying a
    full-page 1241x1754 @150 DPI scan cleared the old threshold outright,
    silently ingesting near-nothing from a fully scanned page.
  - guide_rh_sante_ar.pdf page 27: 85 chars, five over the old threshold --
    same failure, worse margin.
  - profil_sst_maroc_fr.pdf page 138: a genuinely blank trailing page (3
    stray chars, no images) failed the old threshold and, because
    OcrUnavailableError propagates uncaught, failed the ENTIRE 138-page
    document under the default ocr_engine="none" -- wasted OCR compute on a
    blank page, for a cost paid by unrelated readable pages.
  - profil_sst_maroc_fr.pdf page 50: an organigramme (org chart) -- 886
    chars of real box-label text extract, but the layout/hierarchy that
    carries the actual meaning is lost. Neither "trust the text" nor "this
    page has no text" describes it; OCR would improve it but its absence
    should not fail the document the way a true scan's absence should.

Four strategies, not two, follow directly from those four cases:

  NATIVE        -- embedded text is good, use it.
  EMPTY         -- nothing readable (no fonts, no images, negligible text).
                   Never calls OCR -- this is what stops a blank page from
                   failing an otherwise-fully-readable document.
  OCR_REQUIRED  -- native text is unusable (no fonts, or a full-page raster
                   over near-zero text). OcrUnavailableError propagates when
                   no engine is configured -- the document-failure signal is
                   the correct behaviour here, not a bug to work around.
  OCR_PREFERRED -- native text is real but degraded by a full-page raster
                   (the organigramme case). OCR improves it; when
                   unavailable, ingestion.py falls back to the native text
                   instead of failing the document.

Thresholds below are set from that measurement, not chosen a priori -- see
each constant's comment for the specific pages that fixed its value.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# chars/in² (NOT raw char count -- see PDF_TEXT_MIN_CHARS's retirement above).
# Real body-text pages across all three ground-truth PDFs measure 15-78
# chars/in²; the two true-scan pages (with stamp/header text laid over a
# full-page raster) measure 3.6 and 0.9. 12.0 sits with wide margin between
# the lowest real body page (avis_cese_sst_fr.pdf page 1, 3.8 -- itself
# borderline text-only, no raster) and typical scans.
DENSITY_SCAN_MAX = 12.0
# A full-page raster below this density has essentially no usable text of
# its own (the true scans measure 3.6 and 0.9) -- OCR_REQUIRED.
DENSITY_UNUSABLE_MAX = 3.0
# Below this with NO full-page raster and NO fonts, there is nothing to
# read or render -- EMPTY. profil_sst_maroc_fr.pdf page 138 measures 0.03.
DENSITY_EMPTY_MAX = 1.0
# A raster is "full-page" (candidate scan) only if its implied resolution
# at the page's physical size clears this floor. Distinguishes the two real
# scans (150 DPI) from decorative/logo images (avis_cese_sst_fr.pdf page 1's
# 513x299 logo implies ~62 DPI at that page's width -- correctly excluded).
RASTER_MIN_DPI = 72
# A raster's aspect ratio must match the page's within this tolerance to
# count as "full-page" rather than an inset figure. The two real scans
# match within 0.01%; the logo on avis_cese_sst_fr.pdf page 1 is off by 72%.
RASTER_ASPECT_TOLERANCE = 0.15
# Replacement chars (U+FFFD), control chars, and private-use-area codepoints
# over total length. The worst legitimate page across all 205 measures 1.8%
# -- wide margin below this floor.
BAD_CHAR_MAX = 0.10


class PageStrategy(str, Enum):
    NATIVE = "native"
    OCR_REQUIRED = "ocr_required"
    OCR_PREFERRED = "ocr_preferred"
    EMPTY = "empty"


@dataclass(frozen=True)
class PageSignals:
    char_count: int
    page_area_in2: float
    char_density: float  # chars per in^2
    font_count: int
    image_count: int
    has_full_page_raster: bool
    full_raster_dpi: Optional[int]
    bad_char_ratio: float


@dataclass(frozen=True)
class PageDecision:
    strategy: PageStrategy
    signals: PageSignals
    reason: str


def _bad_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(
        1
        for c in text
        if c == "�" or (ord(c) < 32 and c not in "\n\r\t") or 0xE000 <= ord(c) <= 0xF8FF
    )
    return bad / len(text)


def _full_page_raster(page, page_w_in: float, page_h_in: float) -> tuple[bool, Optional[int]]:
    """Largest image XObject on the page, if any looks like a full-page
    scan: aspect ratio close to the page's own, and implied DPI at the
    page's physical size at or above RASTER_MIN_DPI. Returns
    (is_full_page, dpi_if_full_page).
    """
    resources = page.get("/Resources")
    resources = resources.get_object() if resources else {}
    xobjects = resources.get("/XObject")
    if not xobjects:
        return False, None

    page_aspect = page_w_in / page_h_in if page_h_in else 0
    best_dpi: Optional[int] = None
    for _, ref in xobjects.get_object().items():
        obj = ref.get_object()
        if obj.get("/Subtype") != "/Image":
            continue
        px_w, px_h = int(obj.get("/Width", 0)), int(obj.get("/Height", 0))
        if not px_w or not px_h or not page_aspect:
            continue
        img_aspect = px_w / px_h
        if abs(img_aspect - page_aspect) / page_aspect > RASTER_ASPECT_TOLERANCE:
            continue
        dpi = px_w / page_w_in if page_w_in else 0
        if dpi < RASTER_MIN_DPI:
            continue
        if best_dpi is None or dpi > best_dpi:
            best_dpi = round(dpi)

    return best_dpi is not None, best_dpi


def extract_signals(page, text: Optional[str] = None) -> PageSignals:
    """Reads the metadata already on a pypdf PageObject -- no rendering, no
    OCR call. `text` should be the caller's already-extracted
    `page.extract_text()` result (extraction is the expensive part of this
    probe; callers that already have it should not pay for it twice).
    """
    if text is None:
        text = page.extract_text() or ""
    text = text.strip()

    box = page.mediabox
    page_w_in = float(box.width) / 72
    page_h_in = float(box.height) / 72
    area = page_w_in * page_h_in or 1.0

    resources = page.get("/Resources")
    resources = resources.get_object() if resources else {}
    fonts = resources.get("/Font")
    font_count = len(fonts.get_object()) if fonts else 0
    xobjects = resources.get("/XObject")
    image_count = 0
    if xobjects:
        image_count = sum(
            1 for _, ref in xobjects.get_object().items() if ref.get_object().get("/Subtype") == "/Image"
        )

    has_full_page_raster, full_raster_dpi = _full_page_raster(page, page_w_in, page_h_in)

    return PageSignals(
        char_count=len(text),
        page_area_in2=area,
        char_density=len(text) / area,
        font_count=font_count,
        image_count=image_count,
        has_full_page_raster=has_full_page_raster,
        full_raster_dpi=full_raster_dpi,
        bad_char_ratio=_bad_char_ratio(text),
    )


def classify_page(page, text: Optional[str] = None) -> PageDecision:
    """The dispatch app.services.ingestion._parse_pdf calls per page. See
    this module's docstring for what each strategy means and the specific
    real pages that set each threshold.
    """
    s = extract_signals(page, text=text)

    if s.font_count == 0 and s.image_count == 0:
        return PageDecision(PageStrategy.EMPTY, s, "no fonts and no images on the page")

    if s.font_count == 0:
        return PageDecision(
            PageStrategy.OCR_REQUIRED, s, "no font resources -- page has no extractable text layer at all"
        )

    if s.has_full_page_raster and s.char_density < DENSITY_UNUSABLE_MAX:
        return PageDecision(
            PageStrategy.OCR_REQUIRED,
            s,
            f"full-page raster ({s.full_raster_dpi} DPI) with negligible text "
            f"({s.char_density:.1f} chars/in^2) -- looks like a scan",
        )

    if s.has_full_page_raster and s.char_density < DENSITY_SCAN_MAX:
        return PageDecision(
            PageStrategy.OCR_PREFERRED,
            s,
            f"full-page raster ({s.full_raster_dpi} DPI) over sparse text "
            f"({s.char_density:.1f} chars/in^2) -- text likely a stamp/header "
            f"overlaying a scan",
        )

    if s.has_full_page_raster:
        return PageDecision(
            PageStrategy.OCR_PREFERRED,
            s,
            f"full-page raster ({s.full_raster_dpi} DPI) alongside real text "
            f"({s.char_density:.1f} chars/in^2) -- likely a diagram/org chart "
            f"whose layout OCR would preserve better than raw text order",
        )

    if s.bad_char_ratio > BAD_CHAR_MAX:
        return PageDecision(
            PageStrategy.OCR_REQUIRED,
            s,
            f"text layer {s.bad_char_ratio:.0%} replacement/control characters "
            f"-- embedded encoding looks broken",
        )

    if s.char_density < DENSITY_EMPTY_MAX:
        return PageDecision(PageStrategy.EMPTY, s, f"negligible text ({s.char_density:.1f} chars/in^2), no raster")

    return PageDecision(PageStrategy.NATIVE, s, f"text layer looks usable ({s.char_density:.1f} chars/in^2)")
