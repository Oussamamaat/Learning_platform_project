"""
Kaggle kernel: French Gemma-2-9B full generation run, dual-GPU (~1,800 rows).

Runs app/services/generate_training_data.py --language fr as two independent
full processes, one per physical T4 (--target-rows 900 each), then merges
and cross-shard dedups with merge_shards.py. Each --language fr process
already does its own component-proportional scaling
(FRENCH_COMPONENT_CONFIG, 8 components) and its own dedup/train-eval split,
so this script's only job beyond launching both is combining the two shards.

Scaffolding (Ollama install/launch-with-retry, python -u unbuffered
subprocess, flush=True blocking monitor every 5 min, /proc-based liveness,
fail-fast 2-GPU assertion before the apt/ollama install wastes any time)
is the proven darija-tutor-gen-v4 pattern, adapted: that kernel calls a
specialised gen_worker.py for a fixed component subset (a delta top-up);
this one calls the general-purpose generate_training_data.py CLI directly,
because a fresh French run needs proportional scaling across all 8
components, not a subset.

Preceded by a Phase 1 Kaggle smoke test (--target-rows 50, single GPU) that
found the per-component generator prompts were still Darija-instructed --
now fixed with French-language prompt builders (build_socratic_prompt_fr
etc.) and two gate gaps (English prose slipping past the French-quality
check, a Darija-shaped refusal-phrase regex) closed. Verified against a
small local Ollama run before this kernel was pushed.
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
    build_socratic_prompt_fr, row_is_french_clean, english_marker_count,
)
sig = inspect.signature(generate_component)
assert "language" in sig.parameters, "Phase 1 French-mode fix MISSING -- stale app/ upload"
assert set(scale_component_targets(1800, "fr")) == set(FRENCH_COMPONENT_CONFIG), \
    "FRENCH_COMPONENT_CONFIG / scale_component_targets mismatch -- stale app/ upload"
assert callable(build_socratic_prompt_fr), \
    "Phase 2 French prompt builders MISSING -- stale app/ upload"
assert callable(english_marker_count), \
    "Phase 2 English-detection gate MISSING -- stale app/ upload"
print("STATUS: preflight_ok", flush=True)

# 2026-08-01 incident (darija-tutor-gen-v4): a pilot push landed on a
# single-GPU P100 instead of the T4 x2 this script assumes, and nothing
# caught it until a human eyeballed nvidia-smi. Automated here, before the
# apt/ollama install below wastes any time on a shape this pipeline can't use.
print("STATUS: gpu_check_start", flush=True)
gpu_list = subprocess.run(
    ["nvidia-smi", "-L"], capture_output=True, text=True, check=True,
).stdout.strip().splitlines()
print(f"STATUS: gpu_detected count={len(gpu_list)} names={gpu_list}", flush=True)
if len(gpu_list) != 2:
    print(
        f"STATUS: error wrong_gpu_count expected=2 got={len(gpu_list)} "
        "-- this script hardcodes CUDA_VISIBLE_DEVICES 0/1 for two physical "
        "GPUs; re-push with --accelerator NvidiaTeslaT4 rather than let this "
        "run limp along on half its workers", flush=True,
    )
    sys.exit(1)
print("STATUS: gpu_check_ok", flush=True)

print("STATUS: apt_zstd_start", flush=True)
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)
subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh"], check=True,
               stdout=open(WORK / "install.sh", "wb"))
subprocess.run(["sh", "install.sh"], check=True)

MODEL = "gemma2:9b"
PORTS = {0: 11434, 1: 11435}

base_env = dict(
    os.environ,
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    OLLAMA_NUM_PARALLEL="4", OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="60m",
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


def launch_ollama(gpu, port, attempts=3):
    env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu), OLLAMA_HOST=f"127.0.0.1:{port}")
    for attempt in range(1, attempts + 1):
        log = open(WORK / f"ollama_gpu{gpu}.log", "a")
        proc = subprocess.Popen(
            ["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if wait_for_server(port):
            print(f"STATUS: ollama_up gpu={gpu} port={port} attempt={attempt} pid={proc.pid}", flush=True)
            return proc
        print(f"STATUS: ollama_retry gpu={gpu} attempt={attempt}", flush=True)
        if proc.poll() is None:
            proc.terminate()
        time.sleep(5)
    print(f"STATUS: error ollama_never_came_up gpu={gpu}", flush=True)
    sys.exit(1)


ollama_procs = {gpu: launch_ollama(gpu, port) for gpu, port in PORTS.items()}

print("STATUS: model_pull_start", flush=True)
for gpu, port in PORTS.items():
    subprocess.run(["ollama", "pull", MODEL],
                   env=dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}"), check=True)
for gpu, port in PORTS.items():
    tags = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=10).read().decode()
    assert MODEL in tags, f"GPU{gpu}: {MODEL} not visible on port {port} after pull"
print("STATUS: model_confirmed both_gpus", flush=True)

TARGET_PER_GPU = 900  # 1800 total, split evenly


def launch_generator(gpu):
    out_dir = WORK / f"out_fr_gpu{gpu}"
    cmd = [
        sys.executable, "-u", "-m", "app.services.generate_training_data",
        "--language", "fr",
        "--target-rows", str(TARGET_PER_GPU),
        "--concurrency", "4",
        "--model", MODEL,
        "--ollama-url", f"http://127.0.0.1:{PORTS[gpu]}",
        "--script-policy", "allow",
        "--log-level", "INFO",
        "--output-dir", str(out_dir),
    ]
    log_fp = open(WORK / f"gen_fr_gpu{gpu}.log", "a")
    return subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, cwd=str(WORK),
    )


gen_procs = {gpu: launch_generator(gpu) for gpu in PORTS}
print("STATUS: generators_launched " + str({g: p.pid for g, p in gen_procs.items()}), flush=True)

# ---------------------------------------------------------------------------
# Blocking monitor -- required, not optional (see module docstring: Kaggle
# tears the container down when the script's last cell/statement returns).
# flush=True on every write, every 5 min, per the explicit ask that the
# Kaggle run-log view update live rather than sitting in a buffer.
# ---------------------------------------------------------------------------
TARGETS = {k: v["target"] for k, v in scale_component_targets(TARGET_PER_GPU, "fr").items()}
POLL_SECONDS = 300
MAX_HOURS = 10.0
STALL_MINUTES = 20

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
    out = []
    for gpu in PORTS:
        d = WORK / f"out_fr_gpu{gpu}"
        parts = []
        for comp, target in TARGETS.items():
            raw = d / f"{comp}_raw.jsonl"
            total = sum(1 for _ in open(raw, encoding="utf-8")) if raw.exists() else 0
            idle = None
            prog = d / f"{comp}_raw.progress.jsonl"
            if prog.exists() and prog.stat().st_size:
                lines = prog.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    idle = (time.time() - json.loads(lines[-1])["ts"]) / 60
            parts.append((comp, total, target, idle))
        out.append((gpu, proc_alive(gen_procs[gpu]), parts))
    return out


log(f"=== fr-full monitor started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
    f"every {POLL_SECONDS}s, budget {MAX_HOURS}h, target {TARGET_PER_GPU}/GPU ===")

started = time.time()
while True:
    elapsed_h = (time.time() - started) / 3600
    states = snapshot()
    grand = sum(t for _, _, parts in states for _, t, _, _ in parts)
    grand_target = sum(TARGETS.values()) * 2
    log(f"[t+{elapsed_h:5.2f}h] {time.strftime('%H:%M:%S')}  "
        f"grand total: {grand}/{grand_target} rows on disk")
    for gpu, alive, parts in states:
        log(f"  GPU{gpu} {'RUNNING' if alive else 'EXITED '}")
        for comp, total, target, idle in parts:
            idle_s = "  n/a" if idle is None else f"{idle:5.1f}"
            note = ""
            if total >= target:
                note = "  (target met)"
            elif idle is not None and idle > STALL_MINUTES and alive:
                note = "  <-- STALLED"
            log(f"    {comp:24s} {total:4d}/{target:<4d} idle {idle_s} min{note}")
    if not any(alive for _, alive, _ in states):
        log("Both generators have exited.")
        break
    if elapsed_h > MAX_HOURS:
        log("Hit MAX_HOURS budget - stopping monitor so merge/packaging still runs.")
        break
    time.sleep(POLL_SECONDS)

status_file.close()

print("STATUS: merge_start", flush=True)
merge_result = subprocess.run(
    [sys.executable, "-m", "app.services.merge_shards",
     str(WORK / "out_fr_gpu0"), str(WORK / "out_fr_gpu1"),
     "--output", str(WORK / "final_export_fr")],
    cwd=str(WORK),
)
if merge_result.returncode != 0:
    print("STATUS: error merge_shards_failed -- raw per-GPU output still on disk, "
          "not lost", flush=True)
else:
    print("STATUS: merge_done", flush=True)
    shutil.make_archive(str(WORK / "fr_dataset_export"), "zip", str(WORK / "final_export_fr"))
    print("STATUS: packaged fr_dataset_export.zip", flush=True)

print("STATUS: done", flush=True)
