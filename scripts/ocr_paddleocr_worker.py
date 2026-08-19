"""
PaddleOCR-VL subprocess worker.

Runs inside .ocr_venv, never inside .gguf_venv (the app's own venv) --
paddleocr/paddlepaddle pin CUDA/torch versions that conflict with the app's
own torch pin (see app/services/ocr.py's PaddleOcrEngine docstring). The app
process invokes this script via subprocess with .ocr_venv's interpreter
rather than importing paddleocr directly, so the two dependency stacks never
have to coexist in one process.

Contract: read an image from --image, write extracted markdown text to
--out as UTF-8. All diagnostics go to stderr, never stdout -- stdout is
reserved in case a future version of this script needs to emit structured
output, and this keeps the caller's subprocess.run(capture_output=True)
straightforward to reason about (exit code + a file, not a parsed stream).

Usage (from app/services/ocr.py, not run by hand normally):
    .ocr_venv/Scripts/python.exe scripts/ocr_paddleocr_worker.py \\
        --image page.png --out page.md --lang-hint ar+fr
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the input page image.")
    parser.add_argument("--out", required=True, help="Path to write extracted markdown to (UTF-8).")
    # Accepted for interface symmetry with app.services.ocr.OcrEngine.image_to_markdown
    # and forward-compatibility (e.g. per-language model selection later) --
    # PaddleOCR-VL's language handling is automatic per current usage in
    # PaddleOcrEngine, so this isn't passed through to the pipeline today.
    parser.add_argument("--lang-hint", default="ar+fr")
    args = parser.parse_args()

    try:
        from paddleocr import PaddleOCRVL
    except ImportError as e:
        print(f"paddleocr is not installed in this interpreter ({sys.executable}): {e}", file=sys.stderr)
        sys.exit(2)

    pipeline = PaddleOCRVL()
    result = pipeline.predict(args.image)
    # page.markdown is a dict (PaddleOCRVLResult), not an object with a
    # .text attribute -- verified against the real 3.7.0 pipeline output
    # (page.markdown.keys() == {'markdown_texts', 'markdown_images',
    # 'page_index', 'input_path'}), same as app.services.ocr.PaddleOcrEngine.
    markdown_parts = [page.markdown["markdown_texts"] for page in result]
    markdown = "\n\n".join(markdown_parts)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)


if __name__ == "__main__":
    main()
