"""
OCR throughput benchmark (heavy PaddleOCR-VL).
    /app/.gguf_venv/bin/python scripts/benchmark/bench_ocr.py --image page.png
    /app/.gguf_venv/bin/python scripts/benchmark/bench_ocr.py --pdf raw/.../doc.pdf --page 15

Times one page through the same OCR path ingestion uses
(app.services.ocr.get_ocr_engine().image_to_markdown, which for paddleocr talks
to the resident .ocr_venv worker). Baseline on the 4060 8 GB laptop: heavy
PaddleOCR-VL ≈52.5 s/page warm (docs/architecture/cloud-scaling-plan.md §1);
on the 5090, expect a multiple faster and no VRAM juggling.

Runs the same page twice: cold (includes model load) and warm. Exits cleanly
with a note if OCR is unavailable (OCR_ENGINE=none, or paddle didn't build on
Blackwell) — that's the accepted best-effort fallback, not a failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _render_pdf_page(pdf_path: str, page: int, dpi: int) -> bytes:
    import io
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    bitmap = doc[page].render(scale=dpi / 72)
    pil = bitmap.to_pil()
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="Path to a page image (PNG/JPG)")
    ap.add_argument("--pdf", help="Path to a PDF (rendered with --page)")
    ap.add_argument("--page", type=int, default=0, help="0-based page index for --pdf")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--tier", default="heavy", choices=["heavy", "light"])
    ap.add_argument("--out", default="benchmark_ocr.json")
    args = ap.parse_args()

    if not args.image and not args.pdf:
        sys.exit("Provide --image or --pdf.")

    from app.config import get_settings
    from app.services.ocr import get_ocr_engine, OcrUnavailableError

    settings = get_settings()
    print(f"OCR_ENGINE={settings.ocr_engine}  paddle_engine={settings.ocr_paddle_engine}")

    if args.image:
        image_bytes = Path(args.image).read_bytes()
        source = args.image
    else:
        print(f"rendering {args.pdf} page {args.page} @ {args.dpi} DPI ...")
        image_bytes = _render_pdf_page(args.pdf, args.page, args.dpi)
        source = f"{args.pdf}#p{args.page}"

    try:
        engine = get_ocr_engine()
    except OcrUnavailableError as e:
        print(f"OCR unavailable: {e}")
        Path(args.out).write_text(json.dumps({"ocr_available": False, "reason": str(e)}, indent=2))
        return 0

    report = {"source": source, "engine": engine.name, "tier": args.tier, "runs": []}
    for label in ("cold", "warm"):
        try:
            t0 = time.perf_counter()
            md = engine.image_to_markdown(image_bytes, tier=args.tier)
            dt = time.perf_counter() - t0
            print(f"  {label}: {dt:6.2f}s  ({len(md)} chars extracted)")
            report["runs"].append({"label": label, "seconds": round(dt, 2), "chars": len(md)})
        except OcrUnavailableError as e:
            print(f"OCR unavailable ({label}): {e}")
            report.update({"ocr_available": False, "reason": str(e)})
            break
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {e}")
            report["runs"].append({"label": label, "error": f"{type(e).__name__}: {e}"})
            break

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
