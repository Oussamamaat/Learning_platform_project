#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch BOTH vLLM instances (Darija + French -- different base models, so
# always two processes, never one shared server) and supervise them.
# Plan rev 2, Phase A9. Shared by the benchmark/build lease (scripts/lease/
# job.sh's Phase B4/B5) and production (scripts/docker/entrypoint.sh's
# LLM_BACKEND=vllm branch, Phase D2).
#
# Replaces rev 1's "start it by hand from a shell" approach entirely --
# the 2026-09-10 lease lost work twice to a dropped Console shell session
# right after a manually-run `ollama serve`. This script is meant to run
# as a container's own foreground process (or job.sh's own child), so its
# lifetime does not depend on any shell staying connected (plan rule R1).
#
# On EITHER instance exiting (crash or otherwise), logs which one and its
# exit code to $LOG_DIR/serve_pair.exit (rule R5 -- forensics, not a
# guess), kills the other so the container doesn't limp along half-served,
# and exits 1 -- Kubernetes/the lease's own restart policy takes it from
# there.
#
# Env vars (required unless noted):
#   DARIJA_MODEL_DIR, FRENCH_MODEL_DIR   local paths to the AWQ weights
#                                         (already downloaded by the caller
#                                         -- this script does no fetching)
#   DARIJA_SERVED_NAME  (default: iblog-tutor-darija-awq, matches
#                        app/config.py's llm_model_darija default)
#   FRENCH_SERVED_NAME  (default: iblog-tutor-fr-awq, matches llm_model_fr)
#   DARIJA_PORT (default: 8101, matches llm_base_url)
#   FRENCH_PORT (default: 8102, matches llm_base_url_fr)
#   DARIJA_KV_BYTES, FRENCH_KV_BYTES     e.g. "6G" (GiB; vLLM reads "6g" as 10^9 bytes) -- REQUIRED unless
#                                         VLLM_AUTO_KV=1. Plan rev 2
#                                         deliberately rejected
#                                         --gpu-memory-utilization
#                                         fractions ("Rejected alternatives":
#                                         less deterministic than explicit
#                                         KV bytes) -- the real numbers come
#                                         from Phase B0's `GPU KV cache
#                                         size: N tokens` log line, not a
#                                         guess baked into this script.
#   VLLM_AUTO_KV=1      Skip --kv-cache-memory-bytes and let vLLM infer via
#                        its own --gpu-memory-utilization default instead.
#                        SMOKE-TEST ONLY (plan Phase A11's local Docker
#                        check, two tiny models on one 8-32GB card) -- never
#                        for Phase B or production.
#   LOG_DIR             (default: /work/logs, falls back to ./logs)
#   VLLM_MAX_MODEL_LEN  (default: 8192, matches settings.ollama_num_ctx)
#   VLLM_EXTRA_ARGS     appended verbatim to both `vllm serve` invocations
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail  # NOT -e: a failed health-wait must be handled, not crash the script

log() { echo "[serve_pair] $(date -u +%H:%M:%SZ) $*"; }

: "${DARIJA_MODEL_DIR:?set DARIJA_MODEL_DIR to the local AWQ weights path}"
: "${FRENCH_MODEL_DIR:?set FRENCH_MODEL_DIR to the local AWQ weights path}"
DARIJA_SERVED_NAME="${DARIJA_SERVED_NAME:-iblog-tutor-darija-awq}"
FRENCH_SERVED_NAME="${FRENCH_SERVED_NAME:-iblog-tutor-fr-awq}"
DARIJA_PORT="${DARIJA_PORT:-8101}"
FRENCH_PORT="${FRENCH_PORT:-8102}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
LOG_DIR="${LOG_DIR:-/work/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="./logs"; mkdir -p "$LOG_DIR"

if [ "${VLLM_AUTO_KV:-0}" != "1" ]; then
    : "${DARIJA_KV_BYTES:?set DARIJA_KV_BYTES (e.g. 6g) or VLLM_AUTO_KV=1 for a smoke test}"
    : "${FRENCH_KV_BYTES:?set FRENCH_KV_BYTES (e.g. 6g) or VLLM_AUTO_KV=1 for a smoke test}"
fi

log "Darija: $DARIJA_MODEL_DIR -> :$DARIJA_PORT as '$DARIJA_SERVED_NAME'"
log "French: $FRENCH_MODEL_DIR -> :$FRENCH_PORT as '$FRENCH_SERVED_NAME'"

_serve_args() {  # _serve_args <model_dir> <served_name> <port> <kv_bytes_or_empty>
    local model_dir=$1 served_name=$2 port=$3 kv_bytes=$4
    local args=(serve "$model_dir" --served-model-name "$served_name" --port "$port"
                --max-model-len "$VLLM_MAX_MODEL_LEN" --enable-prefix-caching)
    if [ -n "$kv_bytes" ]; then
        args+=(--kv-cache-memory-bytes "$kv_bytes")
    else
        # VLLM_AUTO_KV=1 path (smoke test only): vLLM's own
        # --gpu-memory-utilization default (~0.9) assumes ONE instance owns
        # the whole card. Two instances both defaulting to ~0.9 crash with
        # `ValueError: No available memory for the cache blocks` -- found
        # for real running this exact script locally (plan Phase A11).
        # --max-num-seqs bounds how many concurrent sequences vLLM
        # pre-reserves KV cache for; left at its own default (much higher
        # than a smoke test needs), the KV cache reservation alone can
        # exceed the budget even with gpu-memory-utilization capped.
        args+=(--gpu-memory-utilization "${VLLM_AUTO_KV_MEM_UTIL:-0.45}"
               --max-num-seqs "${VLLM_AUTO_KV_MAX_NUM_SEQS:-8}")
    fi
    # shellcheck disable=SC2206
    [ -n "${VLLM_EXTRA_ARGS:-}" ] && args+=(${VLLM_EXTRA_ARGS})
    printf '%s\n' "${args[@]}"
}

mapfile -t darija_args < <(_serve_args "$DARIJA_MODEL_DIR" "$DARIJA_SERVED_NAME" "$DARIJA_PORT" "${DARIJA_KV_BYTES:-}")
mapfile -t french_args < <(_serve_args "$FRENCH_MODEL_DIR" "$FRENCH_SERVED_NAME" "$FRENCH_PORT" "${FRENCH_KV_BYTES:-}")

log "Starting Darija: vllm ${darija_args[*]}"
vllm "${darija_args[@]}" > "$LOG_DIR/vllm-darija.log" 2>&1 &
DARIJA_PID=$!

wait_for() {  # wait_for <name> <url> <timeout_s> <pid_to_watch>
    local name=$1 url=$2 timeout=$3 pid=$4 waited=0
    until curl -sf "$url" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: $name's process (pid $pid) died while waiting for /health -- see $LOG_DIR/vllm-$name.log"
            return 1
        fi
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -ge "$timeout" ]; then
            log "ERROR: $name not reachable at $url after ${timeout}s"; return 1
        fi
    done
    log "$name is up (${waited}s)"
}

if ! wait_for "darija" "http://127.0.0.1:$DARIJA_PORT/health" 600 "$DARIJA_PID"; then
    log "Darija instance failed to become healthy -- see $LOG_DIR/vllm-darija.log"
    kill "$DARIJA_PID" 2>/dev/null
    echo "darija_startup_failed" > "$LOG_DIR/serve_pair.exit"
    exit 1
fi

log "Starting French: vllm ${french_args[*]}"
vllm "${french_args[@]}" > "$LOG_DIR/vllm-french.log" 2>&1 &
FRENCH_PID=$!

if ! wait_for "french" "http://127.0.0.1:$FRENCH_PORT/health" 600 "$FRENCH_PID"; then
    log "French instance failed to become healthy -- see $LOG_DIR/vllm-french.log"
    kill "$DARIJA_PID" "$FRENCH_PID" 2>/dev/null
    echo "french_startup_failed" > "$LOG_DIR/serve_pair.exit"
    exit 1
fi

log "Both instances healthy. Recording KV cache sizes for Phase B0/Q1 (grep the logs) ..."
grep -i "kv cache" "$LOG_DIR/vllm-darija.log" | tail -3 | sed 's/^/  [darija] /'
grep -i "kv cache" "$LOG_DIR/vllm-french.log" | tail -3 | sed 's/^/  [french] /'

log "Supervising both PIDs (darija=$DARIJA_PID french=$FRENCH_PID). Waiting on either to exit ..."
wait -n "$DARIJA_PID" "$FRENCH_PID"
EXIT_CODE=$?

if ! kill -0 "$DARIJA_PID" 2>/dev/null; then
    DIED="darija"; SURVIVOR_PID=$FRENCH_PID
else
    DIED="french"; SURVIVOR_PID=$DARIJA_PID
fi

log "Instance '$DIED' exited (code=$EXIT_CODE) -- see $LOG_DIR/vllm-$DIED.log. Stopping the survivor."
{
    echo "died=$DIED"
    echo "exit_code=$EXIT_CODE"
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$LOG_DIR/serve_pair.exit"
kill "$SURVIVOR_PID" 2>/dev/null
exit 1
