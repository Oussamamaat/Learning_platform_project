"""
Resident PaddleOCR IPC worker.

Runs inside .ocr_venv, never inside .gguf_venv (the app's own venv) --
paddleocr/paddlepaddle pin CUDA/torch versions that conflict with the app's
own torch pin (see app/services/ocr.py's PaddleOcrEngine docstring).

Unlike scripts/ocr_paddleocr_worker.py (one process per page, reloading the
model every call -- measured 193-200s/page, almost entirely model-load
cost), this process is started ONCE per ingestion run and stays alive:
app/services/ocr.py's _ResidentOcrWorker spawns it, then sends one JSON
job per line on stdin for every page and reads one JSON response per line
from stdout, for as many pages as the run has. Each requested `engine` is
constructed lazily on first use and kept in _PIPELINES for the rest of
this process's life, so a bulk run pays the load cost once per engine, not
once per page.

Protocol -- one JSON object per line, both directions, flushed immediately:
  in:  {"cmd": "ping"}
  out: {"ok": true, "cmd": "pong"}

  in:  {"cmd": "ocr", "id": "<echoed back>", "image": "<path to a PNG>",
        "engine": "vl" | "structure" | "classic"}
  out: {"ok": true,  "id": "...", "markdown": "..."}
  out: {"ok": false, "id": "...", "error": "..."}

Three engines, all already present in .ocr_venv's paddleocr==3.7.0 install
(see scripts/ocr_bakeoff.py for why three, and how the choice among them is
made from measurement rather than assumption):
  vl        -- PaddleOCRVL, the 3B vision-language pipeline already in
               production (app/services/ocr.py's PaddleOcrEngine). Slowest,
               highest fidelity; the only one with a proven ground-truth
               recovery on this corpus.
  structure -- PPStructureV3, layout+table-structure recognition. Much
               lighter than the 3B VLM; candidate for pages that need table
               structure but not full document understanding.
  classic   -- PaddleOCR (PP-OCRv5 detect+recognize). Lightest by a wide
               margin; candidate for plain running text with no table.

stdout carries ONLY JSON response lines -- all diagnostics go to stderr, so
the caller's line-oriented stdout reader never has to distinguish a log
line from a response.
"""
import json
import sys
import traceback

_PIPELINES: dict = {}


def _get_pipeline(engine: str):
    if engine in _PIPELINES:
        return _PIPELINES[engine]

    if engine == "vl":
        from paddleocr import PaddleOCRVL
        # Preprocessing disabled deliberately, and measured rather than
        # assumed: app.services.ingestion._ocr_pdf_page renders every page
        # itself from the PDF at a known scale and orientation, so document
        # orientation classification (PP-LCNet_x1_0_doc_ori) and dewarping
        # (UVDoc) have nothing to correct; seal and chart recognition are
        # irrelevant to this corpus. Measured on arabic_test.pdf p15/p51:
        # construct 30.8s -> 9.2s, warm per-page average 62.0s -> 52.5s
        # (page-dependent: p15 52.9s -> 34.7s, p51 essentially unchanged),
        # with ground-truth recovery IDENTICAL at 12/13 -- i.e. a real but
        # modest speedup at zero fidelity cost. It is NOT the big lever;
        # two-tier routing in _ocr_pdf_page is (see this module's `classic`
        # entry and ingestion._classic_ocr_looks_incomplete).
        pipeline = PaddleOCRVL(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
        )
    elif engine == "structure":
        from paddleocr import PPStructureV3
        pipeline = PPStructureV3()
    elif engine == "classic":
        from paddleocr import PaddleOCR
        pipeline = PaddleOCR(lang="ar", use_textline_orientation=True)
    else:
        raise ValueError(f"unknown engine {engine!r} (expected 'vl', 'structure', or 'classic')")

    _PIPELINES[engine] = pipeline
    print(f"[ocr_worker_resident] loaded engine={engine!r}", file=sys.stderr, flush=True)
    return pipeline


def _markdown_from_vl_or_structure(pipeline, image_path: str) -> str:
    result = pipeline.predict(image_path)
    # page.markdown is a dict (PaddleOCRVLResult / PPStructureV3 result),
    # not an object with a .text attribute -- verified against the real
    # 3.7.0 pipeline output for PaddleOCRVL; PPStructureV3 shares the same
    # .markdown["markdown_texts"] shape in this version.
    parts = [page.markdown["markdown_texts"] for page in result]
    return "\n\n".join(parts)


def _markdown_from_classic(pipeline, image_path: str) -> str:
    result = pipeline.predict(image_path)
    # Classic PaddleOCR (PP-OCRv5) returns one OCRResult per image with
    # parallel rec_texts/rec_scores/rec_polys lists, not a markdown field --
    # there is no layout/table structure to preserve, so lines are joined
    # in reading order (top-to-bottom as returned) with no table markup.
    lines = []
    for page in result:
        texts = page.get("rec_texts") if hasattr(page, "get") else getattr(page, "rec_texts", None)
        if texts:
            lines.extend(texts)
    return "\n".join(lines)


def _handle_ocr(req: dict) -> dict:
    rid = req.get("id")
    engine = req.get("engine", "vl")
    image_path = req.get("image")
    try:
        pipeline = _get_pipeline(engine)
        if engine == "classic":
            markdown = _markdown_from_classic(pipeline, image_path)
        else:
            markdown = _markdown_from_vl_or_structure(pipeline, image_path)
        return {"ok": True, "id": rid, "markdown": markdown}
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"ok": False, "id": rid, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    print(f"[ocr_worker_resident] ready, pid={__import__('os').getpid()}", file=sys.stderr, flush=True)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"bad json: {e}"}), flush=True)
            continue

        cmd = req.get("cmd")
        if cmd == "ping":
            print(json.dumps({"ok": True, "cmd": "pong"}), flush=True)
        elif cmd == "ocr":
            print(json.dumps(_handle_ocr(req)), flush=True)
        else:
            print(json.dumps({"ok": False, "error": f"unknown cmd {cmd!r}"}), flush=True)


if __name__ == "__main__":
    main()
