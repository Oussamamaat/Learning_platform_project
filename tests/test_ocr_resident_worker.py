"""
Tests for app.services.ocr._ResidentOcrWorker -- the persistent-subprocess
IPC layer PaddleOcrEngine now talks to (scripts/ocr_worker_resident.py),
replacing a fresh-subprocess-per-page design that measured ~193-200s/page
almost entirely as PaddleOCR-VL's own model-load cost, repeated per page.

These tests exercise ONLY the IPC/process-management layer -- request/
response round-trip, timeout handling, error propagation, and restart
after a process death -- against a small fake worker script implementing
the same JSON-lines protocol (see scripts/ocr_worker_resident.py's
docstring), NOT against real PaddleOCR or .ocr_venv. That keeps this
suite runnable in .gguf_venv with no OCR stack installed, same posture as
the rest of this module's tests (test_pdf_scanned_page_with_ocr_disabled_
raises_actionable_error pins ocr_engine="none" for the same reason).
scripts/ocr_bakeoff.py is the tool that exercises the REAL engines.
"""
import json
import sys
import textwrap

import pytest

from app.services.ocr import OcrUnavailableError, _ResidentOcrWorker


@pytest.fixture
def fake_worker_script(tmp_path):
    """A minimal stand-in for scripts/ocr_worker_resident.py: same
    JSON-lines protocol, no paddleocr import. Behaviour is driven by the
    `image` path's basename so a single fixture script covers every case
    below without extra fixture files:
      "ok.png"      -> normal success response echoing the image path
      "slow.png"    -> sleeps 5s before responding (timeout test)
      "crash.png"   -> the worker process exits immediately, no response
      anything else -> an {"ok": false, ...} error response
    """
    script = tmp_path / "fake_worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys, time
            print("[fake_worker] ready", file=sys.stderr, flush=True)
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                req = json.loads(line)
                if req.get("cmd") == "ping":
                    print(json.dumps({"ok": True, "cmd": "pong"}), flush=True)
                    continue
                image = req.get("image", "")
                rid = req.get("id")
                if image.endswith("crash.png"):
                    sys.exit(1)
                if image.endswith("slow.png"):
                    time.sleep(5)
                    print(json.dumps({"ok": True, "id": rid, "markdown": "slow-done"}), flush=True)
                elif image.endswith("ok.png"):
                    print(json.dumps({"ok": True, "id": rid, "markdown": f"recognized:{image}"}), flush=True)
                else:
                    print(json.dumps({"ok": False, "id": rid, "error": "unrecognized image"}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    return str(script)


def test_ocr_round_trip_returns_markdown(fake_worker_script):
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script)
    result = worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    assert result == "recognized:C:/pages/ok.png"


def test_ocr_reuses_the_same_process_across_calls(fake_worker_script):
    """The whole point of the resident design: one subprocess serves every
    page, not one subprocess per page."""
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script)
    worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    pid_after_first = worker._proc.pid
    worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    assert worker._proc.pid == pid_after_first


def test_ocr_error_response_raises_ocr_unavailable(fake_worker_script):
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script)
    with pytest.raises(OcrUnavailableError, match="unrecognized image"):
        worker.ocr("C:/pages/unknown.png", engine="vl", timeout=10)


def test_ocr_timeout_kills_process_and_raises(fake_worker_script):
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script)
    with pytest.raises(OcrUnavailableError, match="did not respond within"):
        worker.ocr("C:/pages/slow.png", engine="vl", timeout=1)
    assert worker._proc is None  # killed, ready to restart on the next call


def test_ocr_restarts_after_the_worker_process_dies(fake_worker_script):
    """Self-healing, not fault-tolerant (see _ResidentOcrWorker's
    docstring): a crashed worker costs at most the one in-flight page, and
    the NEXT call transparently starts a fresh process."""
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script)
    with pytest.raises(OcrUnavailableError, match="exited unexpectedly"):
        worker.ocr("C:/pages/crash.png", engine="vl", timeout=10)
    assert worker._proc is None

    # A subsequent call must succeed against a freshly spawned process.
    result = worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    assert result == "recognized:C:/pages/ok.png"
