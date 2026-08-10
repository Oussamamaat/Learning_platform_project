"""
Kaggle kernel: French Gemma-2-9B resume + top-up run, dual-GPU.

Continues from the Phase 2 dual-GPU run (darija-tutor-fr-generation-full)
that was stopped mid-flight after learner_adaptation deadlocked (root cause
found and fixed: a Darija-only confusion-marker regex, 100% rejection --
not a hang. Three more Darija-only gate bugs found and fixed the same way:
English-detection, refusal markers, disclosure markers). Verified against a
100-row single-GPU smoke test (darija-tutor-fr-smoke-test v2) before this
kernel was written: all 8/8 components passed cleanly.

What got recovered from the stopped run, after cross-shard dedup:
  socratic                466 (of 550 target -- 15% dedup loss, close enough)
  quiz_generation         162 (of 300 target -- 46% dedup loss)
  structured_explanation  123 (of 300 target -- 59% dedup loss)
  (the other 5 components: 0 -- never reached before the stop)

Strategy: seed each GPU's raw per-component files with half the banked rows
(gpu0/gpu1 splits shipped in the dataset's fr_resume_seed/ folder) and run
--resume with --target-rows 750/GPU. At that size, scale_component_targets
gives socratic a 229/GPU target -- already met by the 233/GPU seed, so it
generates ~0 new socratic rows -- while structured_explanation (125/GPU
target vs ~61/GPU seeded) and quiz_generation (125/GPU vs ~81/GPU seeded)
top up, and the 5 never-run components generate fully fresh. This uses the
existing proportional --target-rows scaling as-is rather than adding new
per-component-override CLI plumbing overnight -- lower risk for unattended
execution than shipping new argument-parsing code with no human review.

Same scaffolding as the other Kaggle kernels in this project (Ollama
install/launch-with-retry, python -u unbuffered subprocess, flush=True
blocking monitor every 5 min, /proc-based liveness, fail-fast 2-GPU
assertion) -- see darija-tutor-gen-v4's module docstring for why each
piece exists.
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

for d in ("app", "data", "raw", "tests", "fr_resume_seed"):
    p = SRC / d
    if p.exists():
        shutil.copytree(p, WORK / d, dirs_exist_ok=True)
    elif d == "fr_resume_seed":
        print("STATUS: error seed_data_missing_from_dataset", flush=True)
        sys.exit(1)
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
    english_marker_count, row_is_learner_adaptation, row_discloses_general_knowledge,
    _CONFUSION_MARKERS_FR, _DISCLOSURE_MARKERS_FR,
)
sig = inspect.signature(generate_component)
assert "language" in sig.parameters, "Phase 1 French-mode fix MISSING -- stale app/ upload"
assert set(scale_component_targets(750, "fr")) == set(FRENCH_COMPONENT_CONFIG), \
    "FRENCH_COMPONENT_CONFIG / scale_component_targets mismatch -- stale app/ upload"
assert all(v["target"] >= 0 for v in scale_component_targets(16, "fr").values()), \
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
print("STATUS: preflight_ok", flush=True)

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

# Seed each GPU's output dir with its half of the banked rows BEFORE
# launching --resume. main() only skips the stale-file cleanup when
# --resume is passed; it never creates the raw files itself from nothing.
print("STATUS: seeding_start", flush=True)
seed_counts = {}
for gpu in (0, 1):
    out_dir = WORK / f"out_fr_gpu{gpu}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_src = WORK / "fr_resume_seed" / f"gpu{gpu}"
    for f in seed_src.glob("*_raw.jsonl"):
        dst = out_dir / f.name
        shutil.copy(f, dst)
        n = sum(1 for _ in open(dst, encoding="utf-8"))
        seed_counts[f"gpu{gpu}/{f.stem}"] = n
print(f"STATUS: seeded {seed_counts}", flush=True)

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

TARGET_PER_GPU = 750


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
        "--resume",
    ]
    log_fp = open(WORK / f"gen_fr_gpu{gpu}.log", "a")
    return subprocess.Popen(
        cmd, stdout=log_fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, cwd=str(WORK),
    )


gen_procs = {gpu: launch_generator(gpu) for gpu in PORTS}
print("STATUS: generators_launched " + str({g: p.pid for g, p in gen_procs.items()}), flush=True)

# ---------------------------------------------------------------------------
# Blocking monitor -- required, not optional. flush=True every 5 min.
# MAX_HOURS is the self-terminating safety net this project relies on since
# there is no CLI mechanism to cancel a running Kaggle kernel remotely --
# whatever is on disk when this fires gets merged and packaged regardless.
# ---------------------------------------------------------------------------
TARGETS = {k: v["target"] for k, v in scale_component_targets(TARGET_PER_GPU, "fr").items()}
POLL_SECONDS = 300
MAX_HOURS = 5.0
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


log(f"=== fr-resume monitor started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
    f"every {POLL_SECONDS}s, budget {MAX_HOURS}h, target {TARGET_PER_GPU}/GPU, "
    f"seeded {seed_counts} ===")

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
                note = "  <-- STALLED (check gate rejection rate, not just liveness)"
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
