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
import time

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
      "oom.png"     -> an {"ok": false, ...} error shaped like a real CUDA
                       OOM (see scripts/ocr_worker_resident.py's
                       _handle_ocr, which catches this exception class
                       exactly like any other and keeps the process alive)
      anything else -> an ordinary {"ok": false, ...} error response
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
                elif image.endswith("oom.png"):
                    err = "RuntimeError: Exception from the 'cv' worker: (External) CUDA error(2), out of memory."
                    print(json.dumps({"ok": False, "id": rid, "error": err}), flush=True)
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
    # An ordinary per-page application error is NOT a reason to discard a
    # perfectly healthy process -- only a GPU-memory error is (see below).
    assert worker._proc is not None


def test_ocr_gpu_memory_error_kills_the_process(fake_worker_script):
    """Reproduces a live incident: one page's CUDA OOM left the worker's
    allocator poisoned, and because the worker process itself survives an
    application-level {"ok": false} response, 5 more unrelated pages in
    the same still-alive process failed identically before anything
    restarted it. _ResidentOcrWorker.ocr must treat a GPU-memory-shaped
    error as fatal to the PROCESS, not just the page, so the next call
    gets a fresh CUDA context instead of reusing the poisoned one."""
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script)
    with pytest.raises(OcrUnavailableError, match="GPU-memory error"):
        worker.ocr("C:/pages/oom.png", engine="vl", timeout=10)
    assert worker._proc is None  # killed, ready to restart on the next call


def test_ocr_restarts_with_a_fresh_process_after_a_gpu_memory_error(fake_worker_script):
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script)
    with pytest.raises(OcrUnavailableError):
        worker.ocr("C:/pages/oom.png", engine="vl", timeout=10)
    pid_before = worker._proc
    assert pid_before is None
    result = worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    assert result == "recognized:C:/pages/ok.png"
    assert worker._proc is not None  # respawned


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


# --- 2026-08-23: idle-release, so the worker's VRAM isn't held for this
# app process's entire remaining lifetime once an ingestion run finishes --

def test_idle_worker_is_released_after_the_timeout(fake_worker_script):
    """Measured motive: with this worker resident indefinitely, the
    7.5GB Darija tutor model could not fit alongside it on an 8GB card and
    loaded 31% CPU / 69% GPU, slow enough to blow a live chat request's
    180s client timeout. A short idle window (0.2s here) must free the
    subprocess -- verified by pid disappearing, not by timing alone."""
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script, idle_release_seconds=0.2)
    worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    assert worker._proc is not None

    time.sleep(0.6)  # comfortably past the 0.2s idle window
    assert worker._proc is None


def test_a_call_within_the_idle_window_resets_it(fake_worker_script):
    """Back-to-back pages within one document's OCR pass must NOT each pay
    a cold reload -- every call re-arms the idle timer, so a worker used
    more often than idle_release_seconds stays resident indefinitely."""
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script, idle_release_seconds=0.5)
    worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    pid_first = worker._proc.pid

    time.sleep(0.3)  # inside the 0.5s window
    worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)  # re-arms the timer
    assert worker._proc is not None
    assert worker._proc.pid == pid_first  # same process, never released

    time.sleep(0.3)  # inside the window again, measured from the 2nd call
    assert worker._proc is not None


def test_idle_release_disabled_when_zero(fake_worker_script):
    """settings.ocr_worker_idle_release_seconds=0 is the documented escape
    hatch back to pre-2026-08-23 behaviour: hold the worker resident for
    this app process's whole remaining lifetime."""
    worker = _ResidentOcrWorker(sys.executable, fake_worker_script, idle_release_seconds=0)
    worker.ocr("C:/pages/ok.png", engine="vl", timeout=10)
    assert worker._idle_timer is None  # nothing was ever armed

    time.sleep(0.3)
    assert worker._proc is not None  # still resident
