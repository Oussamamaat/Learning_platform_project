"""
OCR engine bake-off: vl vs structure vs classic, on real ground-truth pages.

Answers the question app/services/ocr.py's PaddleOcrEngine docstring
points back to this file for: which of the three pipelines
scripts/ocr_worker_resident.py can run should settings.ocr_paddle_engine
default to? Chosen from MEASUREMENT, not assumption -- the three trade
wall-clock against fidelity differently enough that guessing would be
wrong in either direction:

  vl        -- PaddleOCR-VL (3B vision-language). The engine already
               proven (by direct manual verification against
               arabic_test.pdf pages 15/51) to recover a real Arabic CAD
               layer table and a real hydraulic-formula table as an HTML
               <table>, including the exact formula constants. Slowest,
               and its model load dominates a cold call.
  structure -- PPStructureV3 (layout+table-structure recognition). Much
               lighter than the 3B VLM; the candidate for "does it still
               get tables right without paying the VLM's cost."
  classic   -- PaddleOCR (PP-OCRv5 detect+recognize). Lightest by a wide
               margin; no layout/table structure at all -- every
               recognized text line comes back flat, in reading order,
               with no row/column association. The candidate for "is
               structure recognition even necessary for THIS content."

Run (from repo root, .gguf_venv):
    .gguf_venv/Scripts/python.exe scripts/ocr_bakeoff.py

Renders arabic_test.pdf pages 15 and 51 at the same 200 DPI
app.services.ingestion.OCR_RENDER_DPI uses, sends each through all three
engines via the resident worker (app.services.ocr._ResidentOcrWorker,
scripts/ocr_worker_resident.py) so the model-load-once behaviour under
test is the SAME code path production uses, and scores wall-clock plus
survival of known ground-truth tokens:
  p51 (hydraulic DPF formulas, classified EMPTY before the pdf_classify.py
       fix in this same change): 1.5765, 0.158, 86400, 240, 80
  p15 (CAD layer table, classified NATIVE with the table silently
       dropped before the fix): the 5 Arabic layer names, 39 (UTM zone),
       1970 (datum year)

Writes ocr_bakeoff_results.json (structured scores) and
ocr_bakeoff_<engine>_p<page>.txt (each engine's raw output, for reading)
next to this script. Never asserts / exits nonzero -- this is a
measurement tool, not a test; scripts/verify_ocr_arabic.py is the gate.
"""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.services.ocr import _ResidentOcrWorker  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PDF_PATH = REPO_ROOT / "arabic_test.pdf"
RENDER_DPI = 200  # matches app.services.ingestion.OCR_RENDER_DPI

CHECKS = {
    15: ["حدود القسيمة", "التقسيمات", "نص التقسيمات", "طريق", "نص الطريق", "39", "1970"],
    51: ["1.5765", "0.158", "86400", "240", "80"],
}
PAGES = sorted(CHECKS)
ENGINES = ("classic", "structure", "vl")  # cheapest first


def _render_pages() -> dict[int, Path]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(PDF_PATH))
    paths = {}
    for page_num in PAGES:
        page = pdf[page_num - 1]
        bitmap = page.render(scale=RENDER_DPI / 72)
        img_path = OUT_DIR / f"ocr_bakeoff_p{page_num}.png"
        bitmap.to_pil().save(img_path)
        paths[page_num] = img_path
    pdf.close()
    return paths


def main() -> None:
    if not PDF_PATH.exists():
        print(f"Missing {PDF_PATH} -- this bake-off is scoped to arabic_test.pdf's known pages.")
        return

    settings = get_settings()
    venv_python = str(REPO_ROOT / ".ocr_venv" / "Scripts" / "python.exe")
    worker_script = str(REPO_ROOT / "scripts" / "ocr_worker_resident.py")
    if not Path(venv_python).exists():
        print(f"No .ocr_venv at {venv_python} -- see app/services/ocr.py's PaddleOcrEngine docstring to set it up.")
        return

    print("Rendering ground-truth pages...")
    page_images = _render_pages()

    worker = _ResidentOcrWorker(venv_python, worker_script)
    results: dict[str, dict] = {}

    for engine in ENGINES:
        for page_num in PAGES:
            image_path = str(page_images[page_num])
            t0 = time.time()
            try:
                markdown = worker.ocr(image_path, engine=engine, timeout=600)
                elapsed = time.time() - t0
                ok = True
                error = None
            except Exception as e:
                elapsed = time.time() - t0
                markdown = ""
                ok = False
                error = f"{type(e).__name__}: {e}"

            (OUT_DIR / f"ocr_bakeoff_{engine}_p{page_num}.txt").write_text(markdown, encoding="utf-8")
            found = {tok: (tok in markdown) for tok in CHECKS[page_num]}
            key = f"{engine}_p{page_num}"
            results[key] = {
                "seconds": round(elapsed, 1),
                "ok": ok,
                "error": error,
                "chars": len(markdown),
                "hits": sum(found.values()),
                "total": len(found),
                "found": found,
            }
            status = f"hits={sum(found.values())}/{len(found)}" if ok else f"ERROR: {error}"
            print(f"{engine:<10} p{page_num:<4} {elapsed:>7.1f}s  chars={len(markdown):<6} {status}", flush=True)

    (OUT_DIR / "ocr_bakeoff_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'ocr_bakeoff_results.json'}")
    print(
        "\nNOTE: the FIRST call to each engine includes that engine's own cold model load "
        "(and, for structure/classic on a fresh machine, a one-time download of models not yet "
        "cached under ~/.paddlex) -- this is the real per-run cost under the resident-worker "
        "design (paid once per ingestion run, not once per page), not a steady-state number."
    )


if __name__ == "__main__":
    main()
