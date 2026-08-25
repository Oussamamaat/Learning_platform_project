"""
OCR Arabic-fidelity verification harness.

Closes the question the original OCR spike left open: is any candidate
engine's output actually usable for a citation-grounded, Arabic-script
Darija/French corpus? Not "does it produce readable text" -- specifically
whether article references (المادة N) survive in the right ORDER and the
right SCRIPT, since extract_citations (app/services/citations.py) matches
on exact patterns like that.

Ground truth: tests/data/ocr/1.12_ar_procedure_consignation.png, a
rendered screenshot of raw/shared/industrial/text/1.12_ar_procedure_
consignation.md (real corpus text, 5 numbered المادة articles). Committed
to the repo rather than rendered live at test time -- the live-render
approach (headless Edge) is Windows-only and non-deterministic across
machines; re-render only when the source .md changes, via --regenerate.

Four gates, all must pass before an engine is enabled outside a test run
(settings.ocr_engine stays "none" until then):

  1. Fidelity      -- SequenceMatcher ratio on strip_markdown'd text >= 0.90
  2. Logical order  -- every المادة N marker present, in ascending order
  3. Not reversed   -- ratio(gold, ocr) > ratio(gold, ocr[::-1])
  4. Digit integrity -- no Arabic-Indic <-> ASCII digit substitution in
                        article numbers

Gate 2/3 are the ones that actually matter here: a model can produce every
correct glyph but emit RTL runs in VISUAL order rather than LOGICAL order
and still score fine on plain similarity while being useless for citation
extraction (extract_citations would never match "المادة 4" if the digits
or ordering came out scrambled).

Usage:
    .gguf_venv/Scripts/python.exe scripts/verify_ocr_arabic.py --engine paddleocr
    .gguf_venv/Scripts/python.exe scripts/verify_ocr_arabic.py --engine tesseract
    .gguf_venv/Scripts/python.exe scripts/verify_ocr_arabic.py --engine unlimited_ocr
    .gguf_venv/Scripts/python.exe scripts/verify_ocr_arabic.py --engine paddleocr --regenerate
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GROUND_TRUTH_MD = REPO_ROOT / "raw" / "shared" / "industrial" / "text" / "1.12_ar_procedure_consignation.md"
GROUND_TRUTH_PNG = REPO_ROOT / "tests" / "data" / "ocr" / "1.12_ar_procedure_consignation.png"
GROUND_TRUTH_HTML = REPO_ROOT / "tests" / "data" / "ocr" / "1.12_ar_procedure_consignation.html"

# Marker widened via app.services.citations.arabic_variant_pattern (teh
# marbuta<->heh, alef variants, ya/waw-hamza) so a page that OCR's the
# measured teh-marbuta->heh defect ("الماده" for "المادة") still counts as the
# marker being PRESENT for gates 2/3 -- this is the one deliberate
# exception to "leave gates 1/4 on raw text": gate 2 asks "is the marker
# there, in order", not "is every glyph pixel-correct", so tolerating
# this one measured OCR failure mode here is the whole point of adding
# it. Gate 1 (fidelity) and gate 4 (_has_arabic_indic_digit_leak) below
# still run on the untouched raw text -- folding digits there would
# destroy the exact signal gate 4 exists to catch. citations.py has zero
# heavy dependencies (stdlib re + unicodedata only), so it's importable
# even in a lightweight .ocr_venv that doesn't carry
# sentence-transformers/SQLAlchemy.
from app.services.citations import arabic_variant_pattern

_ARTICLE_RE = re.compile(arabic_variant_pattern("المادة") + r"\s*([0-9\u0660-\u0669]+)")
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ASCII_DIGITS = "0123456789"
_TO_ASCII = str.maketrans(_ARABIC_INDIC_DIGITS, _ASCII_DIGITS)


def regenerate_ground_truth_png() -> None:
    """Re-render GROUND_TRUTH_HTML to GROUND_TRUTH_PNG via headless Edge.
    Windows-only, and only needed if raw/shared/industrial/text/
    1.12_ar_procedure_consignation.md (and therefore the committed HTML)
    changes -- normal runs use the already-committed PNG.
    """
    import subprocess

    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge = next((p for p in edge_candidates if Path(p).exists()), None)
    if edge is None:
        raise SystemExit("msedge.exe not found -- cannot regenerate the ground-truth PNG on this machine.")

    subprocess.run(
        [
            edge, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--force-device-scale-factor=2", "--window-size=1100,1600",
            f"--screenshot={GROUND_TRUTH_PNG}", "--virtual-time-budget=5000",
            f"file:///{GROUND_TRUTH_HTML.as_posix()}",
        ],
        check=True, timeout=30,
    )
    print(f"Regenerated {GROUND_TRUTH_PNG} ({GROUND_TRUTH_PNG.stat().st_size} bytes)")


def _extract_articles(text: str) -> list[str]:
    """Article numbers in the order they appear, normalized to ASCII
    digits so gate 2 (order) and gate 4 (digit integrity) can be checked
    independently -- order comparison shouldn't fail just because a
    digit's SCRIPT changed, and vice versa.
    """
    return [m.translate(_TO_ASCII) for m in _ARTICLE_RE.findall(text)]


def _has_arabic_indic_digit_leak(ocr_text: str) -> bool:
    """True if any المادة marker in the OCR output uses Arabic-Indic
    digits when the gold source (this ground truth) uses ASCII digits for
    its article numbers -- a script-substitution defect, not a fidelity
    one; SequenceMatcher's aggregate ratio can stay high while still
    silently flipping every digit's script.
    """
    for m in _ARTICLE_RE.finditer(ocr_text):
        raw = m.group(1)
        if any(c in _ARABIC_INDIC_DIGITS for c in raw):
            return True
    return False


_HTML_TAG_NAMES_STANDALONE = (
    "table", "thead", "tbody", "tfoot", "tr", "td", "th",
    "p", "div", "span", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "b", "i", "em", "strong", "u",
    "a", "img",
)
_HTML_TAG_RE_STANDALONE = re.compile(
    r"</?(?:" + "|".join(_HTML_TAG_NAMES_STANDALONE) + r")\b[^<>]*/?>", re.IGNORECASE
)


def _strip_markdown_standalone(text: str) -> str:
    """Copy of app.services.ingestion.strip_markdown, kept byte-for-byte in
    sync (including app.services.ingestion._HTML_TAG_NAMES/_HTML_TAG_RE
    above), for venvs (e.g. .ocr_venv) that don't carry the full app
    dependency stack ingestion.py's other top-level imports require."""
    import html as _html_mod

    text = re.sub(r"</\s*(?:td|th)\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</\s*(?:tr|p|div|li|h[1-6])\s*>|<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE
    )
    text = _HTML_TAG_RE_STANDALONE.sub("", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!\w)\*{1,3}(?!\s)([^*]+)(?<!\s)\*{1,3}(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_{1,3}(?!\s)([^_]+)(?<!\s)_{1,3}(?!\w)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^>\s+(?!\d)", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!\\)\|\s*", " ", text)
    text = text.replace("\\|", "|")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _html_mod.unescape(text).replace("\xa0", " ")
    return text.strip()


def run_gates(gold_text: str, ocr_text: str) -> dict:
    try:
        from app.services.ingestion import strip_markdown
    except ImportError:
        # OCR-only venvs (e.g. .ocr_venv) don't carry the full app
        # dependency stack (sentence-transformers, SQLAlchemy) that
        # ingestion.py's other top-level imports require, so duplicate
        # this one pure-regex function rather than pulling all of that in.
        strip_markdown = _strip_markdown_standalone


    gold_clean = strip_markdown(gold_text)
    ocr_clean = strip_markdown(ocr_text)

    fidelity = difflib.SequenceMatcher(None, gold_clean, ocr_clean).ratio()

    gold_articles = _extract_articles(gold_clean)
    ocr_articles = _extract_articles(ocr_clean)
    all_present = set(gold_articles) <= set(ocr_articles)
    order_matches = ocr_articles == gold_articles if all_present else False

    forward_ratio = difflib.SequenceMatcher(None, gold_clean, ocr_clean).ratio()
    reversed_ratio = difflib.SequenceMatcher(None, gold_clean, ocr_clean[::-1]).ratio()
    not_reversed = forward_ratio > reversed_ratio

    digit_leak = _has_arabic_indic_digit_leak(ocr_clean)

    return {
        "fidelity": fidelity,
        "fidelity_pass": fidelity >= 0.90,
        "gold_articles": gold_articles,
        "ocr_articles": ocr_articles,
        "order_pass": order_matches,
        "not_reversed_pass": not_reversed,
        "forward_ratio": forward_ratio,
        "reversed_ratio": reversed_ratio,
        "digit_integrity_pass": not digit_leak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, choices=["tesseract", "paddleocr", "unlimited_ocr"])
    parser.add_argument("--regenerate", action="store_true", help="Re-render the ground-truth PNG first (Windows/Edge only).")
    args = parser.parse_args()

    # A Windows console defaults to cp1252, which cannot encode Arabic at
    # all -- any diagnostic touching the corpus text raises
    # UnicodeEncodeError and kills the run *after* the multi-minute model
    # load. Reconfigure once here rather than tiptoeing around every print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # already UTF-8, or a stream that doesn't support it

    if args.regenerate:
        regenerate_ground_truth_png()

    if not GROUND_TRUTH_PNG.exists():
        raise SystemExit(f"Ground truth PNG missing: {GROUND_TRUTH_PNG}. Run with --regenerate.")

    from app.config import get_settings
    get_settings.cache_clear()
    import os
    os.environ["OCR_ENGINE"] = args.engine

    from app.services.ocr import get_ocr_engine
    get_ocr_engine.cache_clear()

    gold_text = GROUND_TRUTH_MD.read_text(encoding="utf-8")
    image_bytes = GROUND_TRUTH_PNG.read_bytes()

    print(f"Running engine={args.engine} against {GROUND_TRUTH_PNG.name} ...")
    engine = get_ocr_engine()
    ocr_text = engine.image_to_markdown(image_bytes, lang_hint="ar+fr")

    # Always persist the raw OCR text, before gating. Loading a VL model
    # costs minutes, so a run that reports only PASS/FAIL forces a full
    # reload just to see WHY it failed -- and the failure that matters here
    # (a marker present but in an unmatched Unicode form) is invisible
    # without the raw bytes. Written as UTF-8 rather than printed: a Windows
    # console is cp1252 and raises UnicodeEncodeError on Arabic.
    dump_path = GROUND_TRUTH_PNG.with_suffix(f".{args.engine}.out.txt")
    dump_path.write_text(ocr_text, encoding="utf-8")
    print(f"Raw OCR output written to {dump_path}")

    results = run_gates(gold_text, ocr_text)

    print(f"\n=== OCR Arabic-fidelity verification: engine={args.engine} ===")
    print(f"  1. Fidelity        = {results['fidelity']:.3f}  (>=0.90 required)  {'PASS' if results['fidelity_pass'] else 'FAIL'}")
    print(f"  2. Logical order    gold={results['gold_articles']}")
    print(f"                       ocr ={results['ocr_articles']}  {'PASS' if results['order_pass'] else 'FAIL'}")
    print(f"  3. Not reversed     forward={results['forward_ratio']:.3f} reversed={results['reversed_ratio']:.3f}  {'PASS' if results['not_reversed_pass'] else 'FAIL'}")
    print(f"  4. Digit integrity  {'PASS' if results['digit_integrity_pass'] else 'FAIL'}")

    all_pass = all([
        results["fidelity_pass"], results["order_pass"],
        results["not_reversed_pass"], results["digit_integrity_pass"],
    ])
    print(f"\n{'ALL GATES PASSED' if all_pass else 'FAILED -- do not enable this engine outside a test run'}")

    if not all_pass:
        # Diagnose the gap between "text is right" and "markers didn't match":
        # a high fidelity score with missing article numbers means the marker
        # is on the page but not in the form extract_citations expects.
        missing = [a for a in results["gold_articles"] if a not in results["ocr_articles"]]
        if missing:
            marker = "المادة"  # المادة
            print(f"\n  missing article numbers: {missing}")
            print(f"  bare marker occurrences -- gold: {gold_text.count(marker)}, "
                  f"ocr: {ocr_text.count(marker)}")
            if ocr_text.count(marker) < gold_text.count(marker):
                # Distinguishes "heading lost" from "heading present but
                # spelled differently" -- the 2026-08-14 paddleocr result was
                # the latter (teh marbuta U+0629 read as heh U+0647), which
                # keeps fidelity high while breaking exact-match extraction.
                variant = "الماده"  # الماده
                print(f"  heh-for-teh-marbuta variant in ocr: {ocr_text.count(variant)}")
        print(f"\nInspect the full OCR text at: {dump_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
