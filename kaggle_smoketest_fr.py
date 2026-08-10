"""
Kaggle kernel: French-mode Phase 2 smoke test (single GPU, 100 rows).

Verifies four gate fixes found after the Phase 2 dual-GPU run
(darija-tutor-fr-generation-full) was stopped mid-flight:
  1. row_is_french_clean had no English-prose detection (structured_explanation
     slipped English past it on a couple of French headings).
  2. _REFUSAL_MARKERS was Darija-only, so no_context_refusal's correct French
     refusals were rejected outright.
  3. _CONFUSION_MARKERS (learner_adaptation) was Darija-only -- 100% rejection,
     read as a hang in the live log (STALL fired repeatedly) until the raw
     generator log showed attempts climbing normally the whole time; the
     watchdog's wording has also been fixed to distinguish "still working,
     every attempt just failing a gate" from an actual hang.
  4. _DISCLOSURE_MARKERS (general_knowledge_disclosed) was ALSO Darija-only --
     found by audit, not by another live failure, since this component kept
     getting skipped by an unrelated scale_component_targets rounding bug in
     every earlier check and so was never actually exercised live before now.

--target-rows 100 is sized so every one of the 8 FRENCH_COMPONENT_CONFIG
components gets at least a few real attempts, including
general_knowledge_disclosed and learner_adaptation specifically -- this is
the first live check either of those two has had against the fixed gates.

Scaffolding (Ollama install/launch-with-retry, python -u unbuffered
subprocess, flush=True blocking monitor, /proc-based liveness check) is
adapted from the proven darija-tutor-gen-v4 kernel -- see that kernel's
module docstring for why each piece exists (idle-timeout survival, the
poll() phantom-exit bug, the P100-instead-of-T4 accelerator incident).
Single GPU here, not dual: a smoke test doesn't need dual-T4 throughput, so
this skips the CUDA_VISIBLE_DEVICES 0/1 split and the cross-shard
merge_shards step the full run needs.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

WORK = Path("/kaggle/working")
print("STATUS: setup_start", flush=True)

MARKER = os.path.join("app", "services", "generate_training_data.py")
SRC = None
for root, _d, _f in os.walk("/kaggle/input"):
    if (Path(root) / MARKER).is_file():
        SRC = Path(root)
        break
if SRC is None:
    print("STATUS: error dataset_not_found", flush=True)
    sys.exit(1)
print(f"STATUS: dataset_mount_found path={SRC}", flush=True)

for d in ("app", "data", "raw", "tests"):
    p = SRC / d
    if p.exists():
        shutil.copytree(p, WORK / d, dirs_exist_ok=True)
os.chdir(WORK)

print("STATUS: preflight_start", flush=True)
compile_ok = subprocess.run(
    [sys.executable, "-m", "py_compile", "app/services/generate_training_data.py"]
).returncode == 0
if not compile_ok:
    print("STATUS: error generator_does_not_compile", flush=True)
    sys.exit(1)
if Path("tests").is_dir():
    if subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"]).returncode != 0:
        print("STATUS: error pytest_failed", flush=True)
        sys.exit(1)
sys.path.insert(0, str(WORK))
try:
    import sentence_transformers  # noqa: F401
except ImportError as e:
    print(f"STATUS: error sentence_transformers_missing {e}", flush=True)
    sys.exit(1)

import inspect
from app.services.generate_training_data import (
    generate_component, FRENCH_COMPONENT_CONFIG, scale_component_targets,
    PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR, english_marker_count,
    row_is_learner_adaptation, row_discloses_general_knowledge,
    _CONFUSION_MARKERS_FR, _DISCLOSURE_MARKERS_FR,
)
sig = inspect.signature(generate_component)
assert "language" in sig.parameters, "Phase 1 French-mode fix MISSING -- stale app/ upload"
assert set(scale_component_targets(100, "fr")) == set(FRENCH_COMPONENT_CONFIG), \
    "FRENCH_COMPONENT_CONFIG / scale_component_targets mismatch -- stale app/ upload"
assert all(v["target"] >= 0 for v in scale_component_targets(100, "fr").values()), \
    "scale_component_targets negative-target bug MISSING its fix -- stale app/ upload"
assert callable(english_marker_count), \
    "Phase 2 fix #1 (English-detection gate) MISSING -- stale app/ upload"
assert "language" in inspect.signature(row_is_learner_adaptation).parameters, \
    "Phase 2 fix #3 (_CONFUSION_MARKERS_FR) MISSING -- stale app/ upload"
assert "language" in inspect.signature(row_discloses_general_knowledge).parameters, \
    "Phase 2 fix #4 (_DISCLOSURE_MARKERS_FR) MISSING -- stale app/ upload"
assert _CONFUSION_MARKERS_FR.search("Je ne comprends toujours pas."), \
    "_CONFUSION_MARKERS_FR does not match its own documented example"
assert _DISCLOSURE_MARKERS_FR.search("Information generale : ceci est vrai."), \
    "_DISCLOSURE_MARKERS_FR does not match its own documented example"
try:
    from app.services import llm as _llm
    assert PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR == _llm.SYSTEM_PROMPT_TEMPLATE_FR, (
        "train/serve parity broken -- PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR != "
        "app.services.llm.SYSTEM_PROMPT_TEMPLATE_FR"
    )
    print("STATUS: template_parity_ok", flush=True)
except ImportError as e:
    # app.services.llm pulls in app.config -> pydantic_settings, which the
    # Kaggle base image is not guaranteed to have (this generation script
    # deliberately avoids that dependency itself -- see
    # generate_training_data.py's PRODUCTION_SYSTEM_PROMPT_TEMPLATE_FR
    # comment). Not fatal to the smoke test: the parity check already runs
    # in tests/test_generation_gates.py, which just passed above.
    print(f"STATUS: template_parity_check_skipped {e}", flush=True)
print("STATUS: preflight_ok", flush=True)

print("STATUS: gpu_check_start", flush=True)
gpu_list = subprocess.run(
    ["nvidia-smi", "-L"], capture_output=True, text=True, check=True,
).stdout.strip().splitlines()
print(f"STATUS: gpu_detected count={len(gpu_list)} names={gpu_list}", flush=True)
if len(gpu_list) < 1:
    print("STATUS: error no_gpu_detected", flush=True)
    sys.exit(1)
print("STATUS: gpu_check_ok", flush=True)

print("STATUS: apt_zstd_start", flush=True)
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)
subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh"], check=True,
               stdout=open(WORK / "install.sh", "wb"))
subprocess.run(["sh", "install.sh"], check=True)

MODEL = "gemma2:9b"
PORT = 11434

base_env = dict(
    os.environ,
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    OLLAMA_NUM_PARALLEL="2", OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="60m",
)
os.makedirs("/root/.ollama/models/manifests", exist_ok=True)
os.makedirs("/root/.ollama/models/blobs", exist_ok=True)


def wait_for_server(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def launch_ollama(port, attempts=3):
    env = dict(base_env, CUDA_VISIBLE_DEVICES="0", OLLAMA_HOST=f"127.0.0.1:{port}")
    for attempt in range(1, attempts + 1):
        log = open(WORK / "ollama_gpu0.log", "a")
        proc = subprocess.Popen(
            ["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if wait_for_server(port):
            print(f"STATUS: ollama_up port={port} attempt={attempt} pid={proc.pid}", flush=True)
            return proc
        print(f"STATUS: ollama_retry attempt={attempt}", flush=True)
        if proc.poll() is None:
            proc.terminate()
        time.sleep(5)
    print("STATUS: error ollama_never_came_up", flush=True)
    sys.exit(1)


ollama_proc = launch_ollama(PORT)

print("STATUS: model_pull_start", flush=True)
subprocess.run(["ollama", "pull", MODEL],
               env=dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{PORT}"), check=True)
tags = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/tags", timeout=10).read().decode()
assert MODEL in tags, f"{MODEL} not visible on port {PORT} after pull"
print("STATUS: model_confirmed", flush=True)

OUT_DIR = WORK / "out_fr_smoke"


def launch_generator():
    cmd = [
        sys.executable, "-u", "-m", "app.services.generate_training_data",
        "--language", "fr",
        "--target-rows", "100",
        "--concurrency", "2",
        "--model", MODEL,
        "--ollama-url", f"http://127.0.0.1:{PORT}",
        "--script-policy", "allow",
        "--log-level", "INFO",
        "--output-dir", str(OUT_DIR),
    ]
    log_fp = open(WORK / "gen_fr_smoke.log", "a")
    return subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, cwd=str(WORK),
    )


gen_proc = launch_generator()
print(f"STATUS: generator_launched pid={gen_proc.pid}", flush=True)

# ---------------------------------------------------------------------------
# Blocking monitor -- required, not optional. If this script returned right
# after launching the generator, Kaggle would tear the container down and
# the generator would die seconds later (see module docstring / v4 kernel).
# flush=True on every write, every POLL_SECONDS, per the explicit ask that
# the Kaggle run-log view update live rather than sitting in a buffer.
# ---------------------------------------------------------------------------
TARGETS = scale_component_targets(100, "fr")
TARGETS = {k: v["target"] for k, v in TARGETS.items()}
POLL_SECONDS = 300
MAX_HOURS = 2.0
STALL_MINUTES = 15

status_path = WORK / "monitor_status.txt"
status_file = open(status_path, "a")


def log(line=""):
    print(line, flush=True)
    status_file.write(line + "\n")
    status_file.flush()
    os.fsync(status_file.fileno())


def proc_alive(proc):
    """PID-exists check via /proc, not Popen.poll() -- poll() can report a
    phantom clean exit (returncode 0) after the child is reparented."""
    try:
        with open(f"/proc/{proc.pid}/cmdline", "rb") as fh:
            return b"generate_training_data" in fh.read()
    except FileNotFoundError:
        return False
    except PermissionError:
        return True
    except OSError:
        pass
    try:
        os.kill(proc.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def snapshot():
    parts = []
    for comp, target in TARGETS.items():
        raw = OUT_DIR / f"{comp}_raw.jsonl"
        total = sum(1 for _ in open(raw, encoding="utf-8")) if raw.exists() else 0
        idle = None
        prog = OUT_DIR / f"{comp}_raw.progress.jsonl"
        if prog.exists() and prog.stat().st_size:
            lines = prog.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                idle = (time.time() - json.loads(lines[-1])["ts"]) / 60
        parts.append((comp, total, target, idle))
    return proc_alive(gen_proc), parts


log(f"=== fr-smoke monitor started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
    f"every {POLL_SECONDS}s, budget {MAX_HOURS}h ===")

started = time.time()
while True:
    elapsed_h = (time.time() - started) / 3600
    alive, parts = snapshot()
    grand = sum(t for _, t, _, _ in parts)
    grand_target = sum(TARGETS.values())
    log(f"[t+{elapsed_h:5.2f}h] {time.strftime('%H:%M:%S')}  "
        f"grand total: {grand}/{grand_target} rows on disk  "
        f"generator {'RUNNING' if alive else 'EXITED'}")
    for comp, total, target, idle in parts:
        idle_s = "  n/a" if idle is None else f"{idle:5.1f}"
        note = ""
        if total >= target:
            note = "  (target met)"
        elif idle is not None and idle > STALL_MINUTES and alive:
            note = "  <-- STALLED"
        log(f"    {comp:24s} {total:4d}/{target:<4d} idle {idle_s} min{note}")
    if not alive:
        log("Generator has exited.")
        break
    if elapsed_h > MAX_HOURS:
        log("Hit MAX_HOURS budget - stopping monitor so the tally below still runs.")
        break
    time.sleep(POLL_SECONDS)

status_file.close()

print("STATUS: final_tally", flush=True)
grand = 0
for comp, target in TARGETS.items():
    raw = OUT_DIR / f"{comp}_raw.jsonl"
    n = sum(1 for _ in open(raw, encoding="utf-8")) if raw.exists() else 0
    grand += n
    print(f"  {comp:24s} {n:4d}/{target:<4d} {'OK' if n >= target else 'SHORT'}", flush=True)
train_path, eval_path, stats_path = OUT_DIR / "train.jsonl", OUT_DIR / "eval.jsonl", OUT_DIR / "component_stats.json"
if stats_path.exists():
    print("STATUS: component_stats " + stats_path.read_text(encoding="utf-8"), flush=True)
if train_path.exists() and eval_path.exists():
    shutil.make_archive(str(WORK / "fr_smoke_export"), "zip", str(OUT_DIR))
    print("STATUS: packaged fr_smoke_export.zip", flush=True)
else:
    print("STATUS: no train/eval written -- packaging raw per-component files instead", flush=True)
    shutil.make_archive(str(WORK / "fr_smoke_export"), "zip", str(OUT_DIR))

print(f"STATUS: done grand_total={grand}/{grand_target}", flush=True)
