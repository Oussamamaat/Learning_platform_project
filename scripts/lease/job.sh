#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The Phase B lease's own orchestrator. Plan rev 2, Phase A9 (written) /
# Phase B (run). Runs as the container's foreground process (invoked by
# deploy/akash-vllm-bench.yaml's command/args, after a `hf download` of
# the kit this script itself is part of).
#
# Exists to satisfy plan rule R1: no interactive shell work. Everything
# below runs unattended -- status/results are pushed to HF and to a
# locally-served status.json, not read back through a Console shell
# session (the 2026-09-10 lease lost two runs of work to exactly that: a
# manually-run `ollama serve` that correlated with a container restart a
# few minutes later, twice, and a dropped Console exec session that had
# nothing to do with the container's actual health).
#
# Phases (each idempotent -- a marker file skips a completed phase, so a
# restart resumes instead of re-running from zero, rule/assumption 4):
#   B0  preflight            GPU model check, record versions
#   B1  ollama_setup         fetch GGUFs, start 2 Ollama servers (NUM_PARALLEL 1 and 4)
#   B2  ollama_bench         concurrency sweeps against both
#   B3  awq_build            merge + AWQ quantize both languages, upload
#   B4  vllm_setup           serve_pair.sh, parity probe, quality sample
#   B5  vllm_bench           concurrency sweeps against vLLM
#   B6  vllm_bench_fp8       restart the pair with --kv-cache-dtype fp8, re-bench
#   B7  done                 final status, everything published
#
# A phase that fails publishes status=failed with its log tail, then
# SLEEPS instead of exiting -- rule/assumption 4: no restart loop, the box
# stays inspectable (shell access, when it's up, can still look around;
# the point is nothing is LOST if it isn't).
#
# JOB_PROFILE=smoke (plan Phase A11): proves THIS SCRIPT's own orchestration
# -- phase transitions, idempotency markers, status.json, HF publish, error
# handling -- runs correctly, using tiny public stand-in models instead of
# the real 9B GGUFs/AWQ builds, cheaply and locally (a laptop GPU, not a
# lease). It does NOT re-validate the individual scripts' own logic (that's
# already covered: build_merged_awq.py by scripts/vllm/dry_run_awq.py,
# bench/parity/quality by the Phase A6/A8 smoke tests against real Ollama).
# B3 (the real AWQ build) and B6 (the fp8 restart) are skipped outright in
# this profile -- they don't exercise job.sh's own control flow any
# differently than B1/B2/B4/B5 already do, and skipping them is what makes
# this profile fast enough to run before every lease instead of never.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

WORK_DIR="${WORK_DIR:-/work}"
PUBLIC_DIR="$WORK_DIR/public"
LOG_DIR="$WORK_DIR/logs"
KIT_DIR="$WORK_DIR/kit"
mkdir -p "$PUBLIC_DIR" "$LOG_DIR" "$WORK_DIR/build"

JOB_PROFILE="${JOB_PROFILE:-full}"  # full | smoke
SMOKE_MODEL_OLLAMA="${SMOKE_MODEL_OLLAMA:-gemma2:2b}"     # real Ollama library tag
SMOKE_MODEL_HF="${SMOKE_MODEL_HF:-Qwen/Qwen2.5-0.5B-Instruct}"  # HF repo id, vLLM downloads it itself
# NOT unsloth/gemma-2-2b-it (rev 2's original choice) -- proven live during
# plan Phase A11's Docker smoke test to be both too large in bf16 (~5.2 GB)
# for two instances to share an 8GB card at vLLM's default
# --gpu-memory-utilization, AND (separately) this app's real ~2000-token
# system prompts leave no room in a small stand-in's context window unless
# that window is genuinely large. Qwen2.5-0.5B-Instruct fixes both: ~1 GB
# in bf16, native max_position_embeddings=32768. See
# config/docker-compose.yml's `vllm` service comment for the full story.
# A smoke run publishes under a DIFFERENT prefix than a real Phase B run --
# both can share one RESULTS_REPO without a smoke run's tiny-model numbers
# colliding with (or being mistaken for) a real lease's results.
RESULTS_PREFIX="results"; [ "$JOB_PROFILE" = "smoke" ] && RESULTS_PREFIX="results-smoke"
# Max GPU memory (MiB) still in use before a vLLM start; a laptop smoke run shares the card with the desktop.
GPU_IDLE_MIB="${GPU_IDLE_MIB:-1024}"; [ "$JOB_PROFILE" = "smoke" ] && GPU_IDLE_MIB="${GPU_IDLE_MIB_SMOKE:-3072}"

: "${KIT_REPO:?set KIT_REPO to the HF dataset repo from publish_kit.py}"
: "${HF_TOKEN:?set HF_TOKEN (read on GGUF/adapters/kit repos, write on the AWQ+results repos)}"
: "${RESULTS_REPO:?set RESULTS_REPO, e.g. Oussamamaat/iblog-vllm-lease (results/ prefix used within it)}"
if [ "$JOB_PROFILE" = "full" ]; then
    : "${GGUF_REPO:?set GGUF_REPO, e.g. Oussamamaat/iblog-tutor-gguf}"
    : "${ADAPTER_REPO:?set ADAPTER_REPO, e.g. Oussamamaat/iblog-tutor-adapters}"
    : "${AWQ_REPO:?set AWQ_REPO, e.g. Oussamamaat/iblog-tutor-awq}"
fi
KIT_SHA="${KIT_SHA:-}"  # empty = read kit/latest.txt

log() { echo "[job] $(date -u +%H:%M:%SZ) $*"; }

# ── status.json: written on every phase transition AND every 30s from the
# heartbeat below, so a viewer watching http://<ingress>/status.json never
# sees anything older than 30s, even mid-phase. ─────────────────────────────
#
# Phase is tracked in a FILE, not just the $CURRENT_PHASE shell variable --
# found for real running this exact script (plan Phase A11): the heartbeat
# below runs in a background `( ... ) &` SUBSHELL, which forks a separate
# process with its OWN copy of every variable at fork time. Every later
# `CURRENT_PHASE="X"` assignment in the main script is invisible to that
# already-forked subshell, so its periodic write_status call kept reporting
# "boot" forever, overwriting the correct phase the main script itself had
# written moments earlier -- status.json was right for a moment after each
# phase transition, then wrong again within 30s. A file both processes
# actually read fixes it for both.
PHASE_FILE="$WORK_DIR/.current_phase"
set_phase() { CURRENT_PHASE="$1"; echo "$1" > "$PHASE_FILE"; }
set_phase "boot"
write_status() {  # write_status <state> [extra_json_fragment]
    local state=$1 extra=${2:-}
    python3 - "$state" "$(cat "$PHASE_FILE" 2>/dev/null || echo boot)" "$extra" <<'PYEOF'
import json, sys, time
state, phase, extra = sys.argv[1], sys.argv[2], sys.argv[3]
doc = {"phase": phase, "state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
if extra:
    try:
        doc.update(json.loads(extra))
    except Exception:
        doc["extra_raw"] = extra
with open("/work/public/status.json", "w") as f:
    json.dump(doc, f, indent=2)
PYEOF
}

start_heartbeat() {
    ( while true; do write_status "running"; sleep 30; done ) &
    HEARTBEAT_PID=$!
    log "heartbeat started (pid $HEARTBEAT_PID)"
}

fail_and_sleep() {  # fail_and_sleep <phase> <log_file>
    local phase=$1 log_file=$2
    log "PHASE FAILED: $phase -- see $log_file"
    # Kill the heartbeat FIRST -- found for real running this exact script
    # (plan Phase A11): `sleep infinity` below never lets the main script
    # reach its own EXIT trap, so `trap '... EXIT'`'s heartbeat-kill never
    # fires. Left running, the heartbeat's own periodic write_status
    # "running" call keeps silently overwriting the "failed" status this
    # function is about to write, every 30s, forever -- status.json looked
    # permanently "running" even though the job had already stopped and
    # was correctly parked in sleep infinity the whole time.
    kill "${HEARTBEAT_PID:-}" 2>/dev/null || true
    local tail_text
    tail_text=$(tail -c 4000 "$log_file" 2>/dev/null | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")
    write_status "failed" "{\"failed_phase\": \"$phase\", \"log_tail\": $tail_text}"
    publish_public
    log "Sleeping (not exiting) so the box stays inspectable. Fix + redeploy, or shell in if it's up."
    sleep infinity
}

_done_marker() { echo "$WORK_DIR/.done.$1"; }
phase_done() { [ -f "$(_done_marker "$1")" ]; }
mark_done() { date -u +%Y-%m-%dT%H:%M:%SZ > "$(_done_marker "$1")"; }

publish_public() {
    python3 - "$RESULTS_REPO" "$HF_TOKEN" "$PUBLIC_DIR" "$RESULTS_PREFIX" <<'PYEOF'
import sys
from huggingface_hub import HfApi
repo, token, local_dir, prefix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
api = HfApi(token=token)
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=local_dir, path_in_repo=f"{prefix}/public", repo_id=repo, repo_type="dataset")
PYEOF
}

publish_file() {  # publish_file <local_path> <repo_relative_path>
    local local_path=$1 repo_path=$2
    python3 - "$RESULTS_REPO" "$HF_TOKEN" "$local_path" "$repo_path" <<'PYEOF'
import sys
from huggingface_hub import HfApi
repo, token, local_path, repo_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
HfApi(token=token).upload_file(path_or_fileobj=local_path, path_in_repo=repo_path,
                                repo_id=repo, repo_type="dataset")
PYEOF
}

hf_has() {  # hf_has <repo> <repo_type> <path> -- lets a rerun skip work whose output is already on HF
    python3 - "$1" "$2" "$3" "$HF_TOKEN" <<'PYEOF'
import sys
from huggingface_hub import HfApi
repo, repo_type, path, token = sys.argv[1:5]
sys.exit(0 if HfApi(token=token).file_exists(repo, path, repo_type=repo_type) else 1)
PYEOF
}

# Ollama and vLLM must never share the card: vLLM's startup memory check fails, and Ollama falls back to CPU.
ollama_unload() {
    for port in 11434 11435; do
        for model in IBLOG_TUTOR:latest iblog-tutor-fr:latest; do
            curl -sf "http://127.0.0.1:$port/api/generate" \
                 -d "{\"model\":\"$model\",\"keep_alive\":0}" >/dev/null 2>&1 || true
        done
    done
}

wait_gpu_idle() {
    local used=""
    for i in $(seq 1 90); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ -n "$used" ] && [ "$used" -le "$GPU_IDLE_MIB" ]; then echo "GPU idle: ${used} MiB used"; return 0; fi
        sleep 2
    done
    echo "ERROR: GPU still has ${used} MiB in use (limit $GPU_IDLE_MIB)"; return 1
}

# ── Tiny status file server on :8000 (SDL maps this to global port 80) ──────
( cd "$PUBLIC_DIR" && python3 -m http.server 8000 > "$LOG_DIR/fileserver.log" 2>&1 ) &
log "status file server started on :8000, serving $PUBLIC_DIR"

start_heartbeat
trap 'kill $HEARTBEAT_PID 2>/dev/null' EXIT

# ── Fetch the kit (idempotent -- cheap to re-run on a restart) ─────────────
set_phase "fetch_kit"; write_status "running"
if [ -z "$KIT_SHA" ]; then
    KIT_SHA=$(python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='$KIT_REPO', repo_type='dataset', filename='kit/latest.txt', token='$HF_TOKEN')
print(open(p).read().strip())
")
fi
log "Kit sha: $KIT_SHA"
if [ ! -d "$KIT_DIR/kit_marker_$KIT_SHA" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$KIT_REPO', repo_type='dataset', allow_patterns='kit/$KIT_SHA/*',
                   local_dir='$KIT_DIR', token='$HF_TOKEN')
"
    mkdir -p "$KIT_DIR/kit_marker_$KIT_SHA"
fi
# publish_kit.py uploads the CONTENTS of its local "scripts_and_fixtures"
# staging folder (not the folder itself) to path_in_repo=f"kit/{sha}" --
# found for real running this exact script (plan Phase A11): the extra
# "/scripts_and_fixtures" segment here doesn't exist in the repo, and
# every downstream $KIT_ROOT/... reference 404'd until this was removed.
KIT_ROOT="$KIT_DIR/kit/$KIT_SHA"
# publish_kit.py's FIXTURE_FILES list uses paths like
# "scripts/vllm/fixtures/bench_prompts.json" (relative to the repo root),
# not a top-level "fixtures/" folder -- same class of bug as KIT_ROOT
# above, caught the same way (a real run, not a re-read of the other script).
FIXTURES_DIR="$KIT_ROOT/scripts/vllm/fixtures"

# ═══════════════════════════ B0 -- preflight ════════════════════════════════
set_phase "B0_preflight"
if phase_done "B0"; then
    log "B0 already done -- skipping"
else
    write_status "running"
    {
        echo "=== nvidia-smi ==="; nvidia-smi
        echo "=== GPU name check ==="
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "GPU: $GPU_NAME"
        if [ "$JOB_PROFILE" = "full" ] && ! echo "$GPU_NAME" | grep -qi "5090"; then
            echo "ERROR: expected an rtx5090 (Q1's chosen tier for a representative vLLM number), got: $GPU_NAME"
            exit 1
        fi
        [ "$JOB_PROFILE" = "smoke" ] && echo "JOB_PROFILE=smoke -- GPU tier check skipped (any CUDA GPU is fine for orchestration validation)"
        echo "=== disk ==="; df -h "$WORK_DIR"
        echo "=== driver ==="; nvidia-smi --query-gpu=driver_version --format=csv,noheader
    } > "$LOG_DIR/B0_preflight.log" 2>&1
    if [ $? -ne 0 ]; then fail_and_sleep "B0_preflight" "$LOG_DIR/B0_preflight.log"; fi
    mark_done "B0"
    publish_file "$LOG_DIR/B0_preflight.log" "$RESULTS_PREFIX/logs/B0_preflight.log"
fi

# ═══════════════════════════ B1 -- Ollama setup ═════════════════════════════
set_phase "B1_ollama_setup"
if phase_done "B1"; then
    log "B1 already done -- skipping"
else
    write_status "running"
    (
        set -e
        # vllm/vllm-openai:v0.29.0 does NOT ship zstd -- found for real
        # running this exact script (plan Phase A11's smoke test); `tar -I
        # zstd` below fails outright without it. Same base image as the
        # real Phase B lease (deploy/akash-vllm-bench.yaml), so this isn't
        # smoke-test-only -- it would have failed there too.
        if ! command -v zstd >/dev/null 2>&1; then
            apt-get update -qq && apt-get install -y -qq zstd
        fi
        OLLAMA_VERSION="v0.33.2"
        curl -fL -sS --connect-timeout 20 --speed-limit 10240 --speed-time 60 \
             --retry 20 --retry-delay 5 --retry-all-errors -C - \
             -o "$WORK_DIR/ollama.tar.zst" \
             "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
        tar -I zstd -C /usr/local -xf "$WORK_DIR/ollama.tar.zst"
        test -x /usr/local/bin/ollama

        mkdir -p "$WORK_DIR/ollama_models"
        export OLLAMA_MODELS="$WORK_DIR/ollama_models"

        OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1 nohup ollama serve > "$LOG_DIR/ollama-11434.log" 2>&1 &
        echo $! > "$WORK_DIR/ollama_11434.pid"
        for i in $(seq 1 60); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 2; done
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null || { echo "ERROR: ollama :11434 never came up"; exit 1; }

        if [ "$JOB_PROFILE" = "smoke" ]; then
            # One small public pull, then two cheap tag copies (no
            # re-download) so the rest of the script's model NAMES
            # (IBLOG_TUTOR:latest / iblog-tutor-fr:latest) stay identical
            # between profiles -- B2/B4 never need their own branch.
            OLLAMA_HOST=127.0.0.1:11434 ollama pull "$SMOKE_MODEL_OLLAMA"
            OLLAMA_HOST=127.0.0.1:11434 ollama cp "$SMOKE_MODEL_OLLAMA" IBLOG_TUTOR:latest
            OLLAMA_HOST=127.0.0.1:11434 ollama cp "$SMOKE_MODEL_OLLAMA" iblog-tutor-fr:latest
        else
            for pair in "IBLOG_TUTOR:$GGUF_REPO/IBLOG_TUTOR" "iblog-tutor-fr:$GGUF_REPO/iblog-tutor-fr"; do
                name="${pair%%:*}"; hf_stem="${pair#*:}"
                python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$GGUF_REPO', repo_type='model', filename='${hf_stem##*/}.gguf',
                 local_dir='$WORK_DIR/gguf', token='$HF_TOKEN')
hf_hub_download(repo_id='$GGUF_REPO', repo_type='model', filename='${hf_stem##*/}.Modelfile',
                 local_dir='$WORK_DIR/gguf', token='$HF_TOKEN')
"
                ( cd "$WORK_DIR/gguf" && OLLAMA_HOST=127.0.0.1:11434 ollama create "$name" -f "${hf_stem##*/}.Modelfile" )
            done
        fi
        echo "Registered models:"; OLLAMA_HOST=127.0.0.1:11434 ollama list

        # Second instance, SAME models dir, different port + NUM_PARALLEL --
        # registration already happened via the first instance above, this
        # one only serves (rule R2: never kill/restart a serving process by
        # hand once up -- both stay up for the rest of the lease).
        OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 nohup ollama serve > "$LOG_DIR/ollama-11435.log" 2>&1 &
        echo $! > "$WORK_DIR/ollama_11435.pid"
        for i in $(seq 1 60); do curl -sf http://127.0.0.1:11435/api/tags >/dev/null 2>&1 && break; sleep 2; done
        curl -sf http://127.0.0.1:11435/api/tags >/dev/null || { echo "ERROR: ollama :11435 never came up"; exit 1; }
    ) > "$LOG_DIR/B1_ollama_setup.log" 2>&1
    if [ $? -ne 0 ]; then fail_and_sleep "B1_ollama_setup" "$LOG_DIR/B1_ollama_setup.log"; fi
    mark_done "B1"
    publish_file "$LOG_DIR/B1_ollama_setup.log" "$RESULTS_PREFIX/logs/B1_ollama_setup.log"
fi

# ═══════════════════════════ B2 -- Ollama benches ═══════════════════════════
set_phase "B2_ollama_bench"
if phase_done "B2"; then
    log "B2 already done -- skipping"
else
    write_status "running"
    APP_PY="python3"  # this image has no .gguf_venv; bench_concurrency.py is stdlib-only
    N_VALUES="1,2,4,8,16,32"; [ "$JOB_PROFILE" = "smoke" ] && N_VALUES="1,2"
    (
        set -e
        for cfg in "11434:unset" "11435:4"; do
            port="${cfg%%:*}"; note="${cfg#*:}"
            for lang in darija fr; do
                model="IBLOG_TUTOR:latest"; [ "$lang" = "fr" ] && model="iblog-tutor-fr:latest"
                if hf_has "$RESULTS_REPO" dataset "$RESULTS_PREFIX/public/bench_ollama_${note}_${lang}.json"; then
                    echo "bench_ollama_${note}_${lang}.json already on HF -- skipping"; continue
                fi
                "$APP_PY" "$KIT_ROOT/scripts/benchmark/bench_concurrency.py" \
                    --backend ollama --base-url "http://127.0.0.1:$port" --model "$model" \
                    --prompts "$FIXTURES_DIR/bench_prompts.json" --language "$lang" \
                    --n-values "$N_VALUES" --num-parallel-note "$note" \
                    --out "$PUBLIC_DIR/bench_ollama_${note}_${lang}.json"
            done
        done
        # Ollama's side of the quality comparison, collected while it has the card to itself.
        "$APP_PY" "$KIT_ROOT/scripts/vllm/quality_sample.py" \
            --prompts "$FIXTURES_DIR/quality_prompts.json" --backends ollama \
            --ollama-url http://127.0.0.1:11434 \
            --ollama-model-darija IBLOG_TUTOR:latest --ollama-model-fr iblog-tutor-fr:latest \
            --out "$PUBLIC_DIR/quality_ollama.md" --out-json "$PUBLIC_DIR/quality_ollama.json"

        # Explicitly release Ollama's GPU memory now, rather than waiting
        # on its default 5-minute idle timeout -- found for real running
        # this exact script (plan Phase A11): with that natural timeout,
        # B4's vLLM startup began WHILE Ollama's model was still resident,
        # and something (VRAM contention) silently killed both Ollama
        # SERVER PROCESSES outright, not just their loaded models -- B4's
        # later parity check then got "Connection refused", not a cold-load
        # delay. keep_alive:0 unloads the model but leaves the server
        # listening, so B4's parity check against Ollama still works (with
        # one cold-load's worth of extra latency on its first call),
        # instead of failing outright. Rule R2's own principle (unload via
        # API, never kill the process) applied to a resource problem, not
        # just the process-liveness problem it was written for.
        ollama_unload
    ) > "$LOG_DIR/B2_ollama_bench.log" 2>&1
    if [ $? -ne 0 ]; then fail_and_sleep "B2_ollama_bench" "$LOG_DIR/B2_ollama_bench.log"; fi
    mark_done "B2"
    publish_public
    publish_file "$LOG_DIR/B2_ollama_bench.log" "$RESULTS_PREFIX/logs/B2_ollama_bench.log"
fi

# ═══════════════════════════ B3 -- AWQ builds ═══════════════════════════════
set_phase "B3_awq_build"
if phase_done "B3"; then
    log "B3 already done -- skipping"
elif [ "$JOB_PROFILE" = "smoke" ]; then
    log "JOB_PROFILE=smoke -- B3 (real AWQ build) skipped. Covered separately by "
    log "scripts/vllm/dry_run_awq.py, not by this orchestration smoke test."
    mark_done "B3"
    write_status "running" "{\"note\": \"B3 skipped in smoke profile\"}"
elif hf_has "$AWQ_REPO" model darija/model.safetensors && hf_has "$AWQ_REPO" model french/model.safetensors; then
    log "B3: both AWQ models already on $AWQ_REPO -- skipping the build"
    mark_done "B3"
else
    write_status "running"
    (
        set -e
        python3 -m venv "$WORK_DIR/awq_venv"
        "$WORK_DIR/awq_venv/bin/pip" install --upgrade pip -q
        "$WORK_DIR/awq_venv/bin/pip" install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 -q
        "$WORK_DIR/awq_venv/bin/pip" install -r "$KIT_ROOT/scripts/vllm/requirements-build.txt" -q

        "$WORK_DIR/awq_venv/bin/python" "$KIT_ROOT/scripts/vllm/build_merged_awq.py" \
            --adapter-repo "$ADAPTER_REPO" --adapter-subdir darija_v11/lora_model \
            --base MBZUAI-Paris/Atlas-Chat-9B \
            --calib-local-file "$FIXTURES_DIR/calib_darija.jsonl" \
            --out-repo "$AWQ_REPO" --out-subdir darija \
            --hf-token "$HF_TOKEN" --work-dir "$WORK_DIR/build/darija" --delete-base-cache

        "$WORK_DIR/awq_venv/bin/python" "$KIT_ROOT/scripts/vllm/build_merged_awq.py" \
            --adapter-repo "$ADAPTER_REPO" --adapter-subdir french_v1/lora_model \
            --base unsloth/gemma-2-9b \
            --calib-local-file "$FIXTURES_DIR/calib_fr.jsonl" \
            --out-repo "$AWQ_REPO" --out-subdir french \
            --hf-token "$HF_TOKEN" --work-dir "$WORK_DIR/build/french" --delete-base-cache

        cp "$WORK_DIR/build/darija/awq_manifest_darija.json" "$PUBLIC_DIR/" 2>/dev/null || true
        cp "$WORK_DIR/build/french/awq_manifest_french.json" "$PUBLIC_DIR/" 2>/dev/null || true
    ) > "$LOG_DIR/B3_awq_build.log" 2>&1
    if [ $? -ne 0 ]; then fail_and_sleep "B3_awq_build" "$LOG_DIR/B3_awq_build.log"; fi
    mark_done "B3"
    publish_public
    publish_file "$LOG_DIR/B3_awq_build.log" "$RESULTS_PREFIX/logs/B3_awq_build.log"
fi

# ═══════════════════════════ B4 -- vLLM setup + checks ══════════════════════
set_phase "B4_vllm_setup"
if phase_done "B4"; then
    log "B4 already done -- skipping"
else
    write_status "running"
    (
        set -e
        if [ "$JOB_PROFILE" = "smoke" ]; then
            # vLLM accepts a bare HF repo id as its model argument and
            # downloads it itself -- no snapshot_download step needed, and
            # no dependency on a real AWQ_REPO ever having been built.
            export DARIJA_MODEL_DIR="$SMOKE_MODEL_HF"
            export FRENCH_MODEL_DIR="$SMOKE_MODEL_HF"
        else
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$AWQ_REPO', repo_type='model', allow_patterns='darija/*', local_dir='$WORK_DIR/awq/darija', token='$HF_TOKEN')
snapshot_download(repo_id='$AWQ_REPO', repo_type='model', allow_patterns='french/*', local_dir='$WORK_DIR/awq/french', token='$HF_TOKEN')
"
            export DARIJA_MODEL_DIR="$WORK_DIR/awq/darija/darija"
            export FRENCH_MODEL_DIR="$WORK_DIR/awq/french/french"
        fi
        export LOG_DIR="$LOG_DIR"
        # KV byte split left unset here deliberately (Q1: the real numbers
        # come from THIS run's own startup log, not a guess) -- runs
        # auto-inferred once to observe it, real Phase B5 numbers get
        # whatever serve_pair.sh's log shows worked.
        export VLLM_AUTO_KV=1
        ollama_unload
        wait_gpu_idle
        bash "$KIT_ROOT/scripts/vllm/serve_pair.sh" > "$LOG_DIR/serve_pair_startup.log" 2>&1 &
        SERVE_PAIR_PID=$!
        echo "$SERVE_PAIR_PID" > "$WORK_DIR/serve_pair.pid"
        for i in $(seq 1 150); do curl -sf http://127.0.0.1:8101/health >/dev/null 2>&1 && curl -sf http://127.0.0.1:8102/health >/dev/null 2>&1 && break; sleep 4; done
        curl -sf http://127.0.0.1:8101/health >/dev/null && curl -sf http://127.0.0.1:8102/health >/dev/null \
            || { echo "ERROR: vLLM pair never became healthy -- see serve_pair_startup.log / vllm-*.log"; exit 1; }

        set +e  # parity_probe.py deliberately exits 1 on a mismatch (Verification V3) --
                # capture that without letting `set -e` abort this block before quality_sample.py runs.
        python3 "$KIT_ROOT/scripts/vllm/parity_probe.py" \
            --prompts "$FIXTURES_DIR/parity_prompts.json" \
            --ollama-url http://127.0.0.1:11434 \
            --ollama-model-darija IBLOG_TUTOR:latest --ollama-model-fr iblog-tutor-fr:latest \
            --vllm-url-darija http://127.0.0.1:8101 --vllm-url-fr http://127.0.0.1:8102 \
            --vllm-model-darija iblog-tutor-darija-awq --vllm-model-fr iblog-tutor-fr-awq \
            --out "$PUBLIC_DIR/parity_report.json"
        PARITY_RC=$?
        set -e

        ollama_unload
        python3 "$KIT_ROOT/scripts/vllm/quality_sample.py" \
            --prompts "$FIXTURES_DIR/quality_prompts.json" \
            --backends vllm --ollama-answers "$PUBLIC_DIR/quality_ollama.json" \
            --vllm-url-darija http://127.0.0.1:8101 --vllm-url-fr http://127.0.0.1:8102 \
            --vllm-model-darija iblog-tutor-darija-awq --vllm-model-fr iblog-tutor-fr-awq \
            --out "$PUBLIC_DIR/quality_transcripts.md" --out-json "$PUBLIC_DIR/quality_transcripts.json"

        # Parity is a hard gate (Verification V3) -- a mismatch means
        # nothing downstream (bench, quality) should be trusted either.
        # Record it either way, but only proceed automatically if it held.
        if [ "$PARITY_RC" -ne 0 ]; then
            echo "PARITY FAILED -- see parity_report.json. Stopping here rather than"
            echo "spending more lease time on numbers that can't be trusted."
            exit 1
        fi
    ) > "$LOG_DIR/B4_vllm_setup.log" 2>&1
    if [ $? -ne 0 ]; then fail_and_sleep "B4_vllm_setup" "$LOG_DIR/B4_vllm_setup.log"; fi
    mark_done "B4"
    publish_public
    publish_file "$LOG_DIR/B4_vllm_setup.log" "$RESULTS_PREFIX/logs/B4_vllm_setup.log"
fi

# ═══════════════════════════ B5 -- vLLM benches ═════════════════════════════
set_phase "B5_vllm_bench"
if phase_done "B5"; then
    log "B5 already done -- skipping"
else
    write_status "running"
    N_VALUES="1,2,4,8,16,32"; [ "$JOB_PROFILE" = "smoke" ] && N_VALUES="1,2"
    (
        set -e
        for lang in darija fr; do
            url="http://127.0.0.1:8101"; model="iblog-tutor-darija-awq"
            [ "$lang" = "fr" ] && url="http://127.0.0.1:8102" && model="iblog-tutor-fr-awq"
            python3 "$KIT_ROOT/scripts/benchmark/bench_concurrency.py" \
                --backend vllm --base-url "$url" --model "$model" \
                --prompts "$FIXTURES_DIR/bench_prompts.json" --language "$lang" \
                --n-values "$N_VALUES" --out "$PUBLIC_DIR/bench_vllm_${lang}.json"
        done
    ) > "$LOG_DIR/B5_vllm_bench.log" 2>&1
    if [ $? -ne 0 ]; then fail_and_sleep "B5_vllm_bench" "$LOG_DIR/B5_vllm_bench.log"; fi
    mark_done "B5"
    publish_public
    publish_file "$LOG_DIR/B5_vllm_bench.log" "$RESULTS_PREFIX/logs/B5_vllm_bench.log"
fi

# ═══════════════════════════ B6 -- FP8 KV variant ═══════════════════════════
# The ONE step that stops a serving process (rule R2's exception -- this is
# a deliberate, scripted restart with a new flag, not an ad hoc `pkill`;
# serve_pair.sh's own supervision loop is what's being restarted, not
# fought against).
set_phase "B6_vllm_bench_fp8"
if phase_done "B6"; then
    log "B6 already done -- skipping"
elif [ "$JOB_PROFILE" = "smoke" ]; then
    log "JOB_PROFILE=smoke -- B6 (fp8 KV restart) skipped, not needed to validate orchestration."
    mark_done "B6"
    write_status "running" "{\"note\": \"B6 skipped in smoke profile\"}"
else
    write_status "running"
    (
        set -e
        if [ -f "$WORK_DIR/serve_pair.pid" ]; then
            kill "$(cat "$WORK_DIR/serve_pair.pid")" 2>/dev/null || true
            pkill -f "vllm serve" 2>/dev/null || true
        fi
        ollama_unload
        wait_gpu_idle
        export DARIJA_MODEL_DIR="$WORK_DIR/awq/darija/darija"
        export FRENCH_MODEL_DIR="$WORK_DIR/awq/french/french"
        export VLLM_AUTO_KV=1
        export VLLM_EXTRA_ARGS="--kv-cache-dtype fp8"
        export LOG_DIR="$LOG_DIR/fp8"
        mkdir -p "$LOG_DIR"
        bash "$KIT_ROOT/scripts/vllm/serve_pair.sh" > "$LOG_DIR/serve_pair_startup_fp8.log" 2>&1 &
        echo $! > "$WORK_DIR/serve_pair_fp8.pid"
        for i in $(seq 1 150); do curl -sf http://127.0.0.1:8101/health >/dev/null 2>&1 && curl -sf http://127.0.0.1:8102/health >/dev/null 2>&1 && break; sleep 4; done
        curl -sf http://127.0.0.1:8101/health >/dev/null && curl -sf http://127.0.0.1:8102/health >/dev/null \
            || { echo "ERROR: fp8 vLLM pair never became healthy"; exit 1; }

        python3 "$KIT_ROOT/scripts/benchmark/bench_concurrency.py" \
            --backend vllm --base-url http://127.0.0.1:8101 --model iblog-tutor-darija-awq \
            --prompts "$FIXTURES_DIR/bench_prompts.json" --language darija --n-values 16,32 \
            --num-parallel-note "fp8-kv" --out "$PUBLIC_DIR/bench_vllm_fp8_darija.json"

        python3 "$KIT_ROOT/scripts/vllm/quality_sample.py" \
            --prompts "$FIXTURES_DIR/quality_prompts.json" \
            --backends vllm --ollama-answers "$PUBLIC_DIR/quality_ollama.json" \
            --vllm-url-darija http://127.0.0.1:8101 --vllm-url-fr http://127.0.0.1:8102 \
            --vllm-model-darija iblog-tutor-darija-awq --vllm-model-fr iblog-tutor-fr-awq \
            --out "$PUBLIC_DIR/quality_transcripts_fp8.md" --out-json "$PUBLIC_DIR/quality_transcripts_fp8.json"
    ) > "$LOG_DIR/B6_vllm_bench_fp8.log" 2>&1
    if [ $? -ne 0 ]; then fail_and_sleep "B6_vllm_bench_fp8" "$LOG_DIR/B6_vllm_bench_fp8.log"; fi
    mark_done "B6"
    publish_public
    publish_file "$LOG_DIR/B6_vllm_bench_fp8.log" "$RESULTS_PREFIX/logs/B6_vllm_bench_fp8.log"
fi

# ═══════════════════════════ B7 -- done ═════════════════════════════════════
set_phase "B7_done"
mark_done "B7"
kill "${HEARTBEAT_PID:-}" 2>/dev/null || true  # same reasoning as fail_and_sleep -- nothing may overwrite "done" afterward
write_status "done"
publish_public
log "All phases complete. Safe to close the lease -- everything is on $RESULTS_REPO."
sleep infinity
