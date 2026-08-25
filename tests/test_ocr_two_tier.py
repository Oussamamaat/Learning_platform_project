"""
Tests for the two-tier OCR routing in app.services.ingestion._ocr_pdf_page
and its escalation predicate _classic_ocr_looks_incomplete.

Why this exists: PaddleOCR-VL measures ~52.5s/page warm against classic
PP-OCRv5's ~5.5s (~10x), but classic scored 0/4 on arabic_test.pdf p51's
formula constants while scoring 4/5 on p15's CAD-layer table. So the light
engine is safe for pages whose native text is being merged anyway, and
unsafe for formula pages -- routing must reflect that, and must escalate
rather than silently keep an under-read result.

These tests pin the ROUTING DECISIONS, with the OCR engine mocked. The
engines' actual fidelity is measured by scripts/ocr_bakeoff.py, not here;
this suite must stay runnable in .gguf_venv with no OCR stack installed.
"""
import pytest

from app.services.ingestion import _classic_ocr_looks_incomplete


# --- the escalation predicate ---------------------------------------------

@pytest.mark.parametrize(
    "text,expected,why",
    [
        # The two real measured cases this rule was calibrated against.
        ("حساب معدل استهلاك الفرد لتر للفرد", True,
         "p51-shaped: classic returned Arabic prose but no numbers -- it "
         "could not read the formula table (measured 0/4)"),
        ("UTM Zone 39 - الإسقاط ... 1970 المسند", False,
         "p15-shaped: the 4-digit datum year is numeric evidence classic "
         "really did read (measured 4/5)"),
        # Boundary behaviour of each half of the rule.
        ("valeur 1.5765 du facteur", False, "a decimal alone is evidence"),
        ("total 86400 secondes", False, "a >=4-digit integer alone is evidence"),
        ("il y a 3 cas et 25 pages", True,
         "short integers are NOT evidence -- page/section numbers appear on "
         "every page and would defeat the check"),
        ("", True, "empty output must escalate, not be accepted"),
        ("   \n  ", True, "whitespace-only output must escalate"),
    ],
)
def test_escalation_predicate(text, expected, why):
    assert _classic_ocr_looks_incomplete(text) is expected, why


def test_escalation_predicate_handles_none():
    """_ocr_pdf_page passes the engine's return value straight in; an engine
    returning None must not raise here."""
    assert _classic_ocr_looks_incomplete(None) is True


# --- routing: which tier each page strategy asks for -----------------------

class _RecordingEngine:
    """Records the `tier` of every call and replays a scripted result per
    call, so a test can assert both WHICH engine was asked and how many
    times."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def image_to_markdown(self, image_bytes, *, lang_hint="ar+fr", tier=None):
        self.calls.append(tier)
        return self._results.pop(0)


@pytest.fixture
def fake_page_render(monkeypatch):
    """Stub out pypdfium2 rendering -- these tests are about routing, not
    rasterisation, and this keeps them independent of any real PDF."""
    class _FakeBitmap:
        def to_pil(self):
            from PIL import Image
            return Image.new("RGB", (8, 8), "white")

    class _FakePage:
        def render(self, scale):
            return _FakeBitmap()

    class _FakeDoc:
        def __init__(self, path):
            pass

        def __getitem__(self, i):
            return _FakePage()

        def close(self):
            pass

    import pypdfium2
    monkeypatch.setattr(pypdfium2, "PdfDocument", _FakeDoc)


def _run(monkeypatch, fake_page_render, engine, tier, two_tier=True):
    from pathlib import Path
    from app.services import ingestion

    monkeypatch.setattr("app.services.ocr.get_ocr_engine", lambda: engine)

    settings = ingestion.get_settings()
    monkeypatch.setattr(settings, "ocr_two_tier", two_tier, raising=False)

    return ingestion._ocr_pdf_page(Path("doc.pdf"), 0, tier=tier)


def test_heavy_tier_never_calls_the_light_engine(monkeypatch, fake_page_render):
    """An OCR_REQUIRED page has no native text to fall back on, so it must
    go straight to the heavy engine -- no light attempt, no escalation."""
    engine = _RecordingEngine(["formule 1.5765 et 86400"])
    out = _run(monkeypatch, fake_page_render, engine, tier="heavy")
    assert engine.calls == ["heavy"]
    assert "1.5765" in out


def test_light_tier_keeps_a_numerically_plausible_result(monkeypatch, fake_page_render):
    """p15-shaped: the light engine found a 4-digit number, so its result
    stands and the expensive engine is never invoked."""
    engine = _RecordingEngine(["UTM Zone 39 ... 1970"])
    out = _run(monkeypatch, fake_page_render, engine, tier="light")
    assert engine.calls == ["light"]
    assert "1970" in out


def test_light_tier_escalates_when_no_numeric_evidence(monkeypatch, fake_page_render):
    """p51-shaped: the light engine returned prose but no numbers, so the
    page is re-run on the heavy engine and the HEAVY result is returned --
    this is the case that would otherwise silently lose a formula."""
    engine = _RecordingEngine(["نص بدون أرقام", "DPF = 1.5765 / 86400"])
    out = _run(monkeypatch, fake_page_render, engine, tier="light")
    assert engine.calls == ["light", "heavy"]
    assert "1.5765" in out


def test_empty_light_result_escalates(monkeypatch, fake_page_render):
    engine = _RecordingEngine(["", "recovered 2024 content"])
    out = _run(monkeypatch, fake_page_render, engine, tier="light")
    assert engine.calls == ["light", "heavy"]
    assert "recovered" in out


def test_two_tier_disabled_sends_light_pages_to_the_heavy_engine(monkeypatch, fake_page_render):
    """settings.ocr_two_tier=False is the documented escape hatch back to
    pre-2026-08-19 behaviour: every OCR page on the heavy engine."""
    engine = _RecordingEngine(["heavy result 1234"])
    out = _run(monkeypatch, fake_page_render, engine, tier="light", two_tier=False)
    assert engine.calls == ["heavy"]
    assert "1234" in out
