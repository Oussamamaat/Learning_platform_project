"""
Kaggle kernel: French Gemma-2-9B targeted regeneration, dual-GPU.

Post-audit fix run. A systematic audit of data/fr_v1_merged (1256 rows, the
Stage 2 output) found five defects, three of them new (found live, not by
smoke test):

  F1  normalize_row() rewrote correct French ("l'article 2") into
      ungrammatical French ("l'المادة 2") -- 24 shipped rows corrupted.
      Fixed: citation injection is now skipped entirely for language="fr".
  F2  build_quiz_prompt_fr had no citation_anchor_rule despite
      quiz_generation being hard-gated on it -- only 21.3% of shipped quiz
      rows cited anything. Fixed: anchor_rule added, matching the other
      four grounded French builders.
  F3  socratic's turn_count_reject_budget (target*1) exhausted before the
      already-correct scripted-exchange design could push compliance past
      38.5% (target 75%; even Darija's own scripted design only reaches
      57.5% against the same 1x budget). Fixed: 3x budget for socratic.
      (grounded_refusal's 0% multi-turn was investigated and found to be a
      PRE-EXISTING, language-symmetric characteristic -- Darija's own
      grounded_refusal delivers 0.2% too, and green_light_model.md's own
      acceptance gate never checks it. Not touched.)
  F4  QUIZ_USER_FALLBACKS was Darija-only, reachable from French
      quiz_generation whenever the model's own `request` field came back
      under 15 chars. Latent (0/316 hit) but fixed: QUIZ_USER_FALLBACKS_FR
      added, language-branched.
  F5  Systemic: every *_reject_budget gate self-disables silently on
      exhaustion (a logger.warning at the exact moment, nothing after).
      Fixed: generate_component now returns gates_exhausted and logs a
      loud STATUS: gate_exhausted line.

Only 3 of the 8 shipped components carry these defects: socratic (F1, F3),
quiz_generation (F2, F4), grounded_refusal (part of F1's corruption via
its own citation-bearing sample). The other 5 -- learner_adaptation,
structured_explanation, injection_resistance, no_context_refusal,
general_knowledge_disclosed -- are unaffected and are KEPT, not
regenerated (structured_explanation has 9 of its 156 rows carrying F1's
corruption too; those are filtered out locally after download, not here).

Mechanism: a new --components CLI flag (this session) restricts
scale_component_targets' apportionment to just the requested subset, so
--target-rows scales ONLY among socratic/quiz_generation/grounded_refusal
instead of being diluted across all 8. TARGET_PER_GPU=525 with these three
components' combined weight (1050) makes each GPU's per-component target
exactly half its FRENCH_COMPONENT_CONFIG weight, so both GPUs combined
reproduce each component's original weight exactly (550/300/200) -- see
scale_component_targets(525, "fr", ["socratic","quiz_generation",
"grounded_refusal"]) => {socratic: 275, quiz_generation: 150,
grounded_refusal: 100} per GPU, 550/300/200 combined. This is a FRESH run,
not a --resume: no seeding needed, because the untouched 5 components are
simply never in `component_config` at all and no raw file is ever created
for them.

Same scaffolding as kaggle_fr_generation_resume.py (Ollama install/launch-
with-retry, python -u unbuffered subprocess, flush=True blocking monitor
every 5 min, /proc-based liveness, fail-fast 2-GPU assertion) -- see that
script's docstring for why each piece exists. Simplified here: no seeding
step (nothing to resume from).
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
    generate_component, scale_component_targets, FRENCH_COMPONENT_CONFIG,
    normalize_row, build_quiz_prompt_fr, QUIZ_USER_FALLBACKS_FR,
    citation_anchor_rule, extract_citations,
)

# --- this session's fixes, all five, asserted present before spending GPU time ---
assert "components" in inspect.signature(scale_component_targets).parameters, \
    "--components filter MISSING -- stale app/ upload (needed for targeted regen)"
COMPONENTS = ("socratic", "quiz_generation", "grounded_refusal")
_subset_targets = scale_component_targets(525, "fr", COMPONENTS)
assert set(_subset_targets) == set(COMPONENTS), \
    f"components filter not restricting apportionment: got {set(_subset_targets)}"
assert _subset_targets["socratic"]["target"] == 275, \
    f"unexpected per-GPU socratic target: {_subset_targets['socratic']['target']}"
assert "language" in inspect.signature(normalize_row).parameters, \
    "F1 fix MISSING (normalize_row has no language param) -- stale app/ upload"
_fr_row = normalize_row(
    {"messages": [{"role": "user", "content": "Q"},
                  {"role": "assistant", "content": "Selon l'article 2, ok."}]},
    "CONTEXTE:\nL'article 2 de la Loi N. 1 impose une regle.",
    "socratic", "industrial", "fr",
)
_fr_text = _fr_row["messages"][-1]["content"]
assert "المادة" not in _fr_text and "l'article 2" in _fr_text, \
    f"F1 regression: citation injection still corrupting French text: {_fr_text!r}"
assert "citation_anchor_rule" in inspect.getsource(build_quiz_prompt_fr), \
    "F2 fix MISSING (build_quiz_prompt_fr has no anchor_rule) -- stale app/ upload"
_anchor_check = citation_anchor_rule(
    extract_citations("L'article 281 impose une regle."), paired_refusal=False,
)
assert "281" in _anchor_check, "citation_anchor_rule not naming a real reference"
_gc_src = inspect.getsource(generate_component)
assert 'turn_count_reject_budget = target * 3 if component == "socratic"' in _gc_src, \
    "F3 fix MISSING (socratic budget not widened) -- stale app/ upload"
assert len(QUIZ_USER_FALLBACKS_FR) >= 1 and all(
    "؀" > c or c > "ۿ" for s in QUIZ_USER_FALLBACKS_FR for c in s
), "F4 fix MISSING or QUIZ_USER_FALLBACKS_FR contains Arabic script"
assert "gates_exhausted" in _gc_src and "STATUS: gate_exhausted" in _gc_src, \
    "F5 fix MISSING (gate exhaustion not reported) -- stale app/ upload"
print("STATUS: preflight_ok all_five_fixes_confirmed", flush=True)

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

TARGET_PER_GPU = 525
COMPONENTS_ARG = ",".join(COMPONENTS)


def launch_generator(gpu):
    out_dir = WORK / f"out_fr_gpu{gpu}"
    cmd = [
        sys.executable, "-u", "-m", "app.services.generate_training_data",
        "--language", "fr",
        "--components", COMPONENTS_ARG,
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
# Blocking monitor -- required, not optional. flush=True every 5 min.
# MAX_HOURS is the self-terminating safety net this project relies on since
# there is no CLI mechanism to cancel a running Kaggle kernel remotely --
# whatever is on disk when this fires gets merged and packaged regardless.
# ---------------------------------------------------------------------------
TARGETS = {k: v["target"] for k, v in scale_component_targets(TARGET_PER_GPU, "fr", COMPONENTS).items()}
POLL_SECONDS = 300
MAX_HOURS = 3.0
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


log(f"=== fr-regen-targeted monitor started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
    f"every {POLL_SECONDS}s, budget {MAX_HOURS}h, target {TARGET_PER_GPU}/GPU, "
    f"components {COMPONENTS_ARG} ===")

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
     "--output", str(WORK / "final_export_fr_regen")],
    cwd=str(WORK),
)
if merge_result.returncode != 0:
    print("STATUS: error merge_shards_failed -- raw per-GPU output still on disk, "
          "not lost", flush=True)
else:
    print("STATUS: merge_done", flush=True)
    shutil.make_archive(str(WORK / "fr_regen_export"), "zip", str(WORK / "final_export_fr_regen"))
    print("STATUS: packaged fr_regen_export.zip", flush=True)

print("STATUS: done", flush=True)
