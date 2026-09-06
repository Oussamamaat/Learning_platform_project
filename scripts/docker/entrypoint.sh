#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Container boot orchestration for the GPU image (config/Dockerfile.gpu).
# Ordered to satisfy app/main.py's startup needs: Ollama + both models ready,
# Postgres reachable + schema initialized, THEN uvicorn.
# Idempotent: on a warm restart (models already registered, schema present)
# every step is a fast no-op.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# ── 1. Environment: Linux venv paths + 32 GB relaxations ─────────────────────
# app/config.py defaults these to Windows "Scripts/python.exe" paths; override
# to the image's Linux venvs. `:=` means an SDL/akash.env value still wins.
export OCR_VENV_PYTHON="${OCR_VENV_PYTHON:-/app/.ocr_venv/bin/python}"
export STT_VENV_PYTHON="${STT_VENV_PYTHON:-/app/.speech_venv/bin/python}"
export TTS_XTTS_VENV_PYTHON="${TTS_XTTS_VENV_PYTHON:-/app/.tts_venv/bin/python}"
# XTTS checkpoint lives on the persistent /models volume, not in the image:
# it is 5.6 GB, and baking it in would bloat every image pull (same reasoning
# as the GGUFs above). Downloaded once per fresh volume in step 3b.
export TTS_XTTS_MODEL_DIR="${TTS_XTTS_MODEL_DIR:-/models/darija_xtts}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
# Relaxations the 8 GB card could not afford (see app/config.py comments). A
# 32 GB card holds both tutors + embeddings + OCR/STT resident at once.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OCR_KEEP_RESIDENT="${OCR_KEEP_RESIDENT:-true}"
export OCR_WORKER_IDLE_RELEASE_SECONDS="${OCR_WORKER_IDLE_RELEASE_SECONDS:-0}"
export SPEECH_WORKER_IDLE_RELEASE_SECONDS="${SPEECH_WORKER_IDLE_RELEASE_SECONDS:-0}"
# Missing here until 2026-09-06's incident review: the app defaults this to
# 300s (app/config.py), so 5 idle minutes on a lease -- easy during a live
# demo/meeting -- silently released the XTTS worker and made the NEXT
# spoken sentence pay the full ~51s checkpoint reload. Looked exactly like
# "it doesn't speak" / "latency increased". A 32GB+ card has no VRAM
# pressure reason to ever release it.
export TTS_WORKER_IDLE_RELEASE_SECONDS="${TTS_WORKER_IDLE_RELEASE_SECONDS:-0}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-128}"
export UPLOADS_READ_ONLY="${UPLOADS_READ_ONLY:-true}"
# Where `ollama serve` binds (in-container only; not exposed by the SDL).
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

APP_PY=/app/.gguf_venv/bin/python
cd /app

# ── 2. Start Ollama in the background ────────────────────────────────────────
log "starting ollama serve ..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

wait_for() {  # wait_for <name> <url> <timeout_s>
    local name=$1 url=$2 timeout=$3 waited=0
    until curl -sf "$url" >/dev/null 2>&1; do
        sleep 1; waited=$((waited + 1))
        if [ "$waited" -ge "$timeout" ]; then
            log "ERROR: $name not reachable at $url after ${timeout}s"; return 1
        fi
    done
    log "$name is up (${waited}s)"
}
wait_for "ollama" "http://127.0.0.1:11434/api/tags" 120

# ── 3. Fetch GGUFs (once) + register the tutor models ────────────────────────
# The .Modelfile files ship in the image; the multi-GB .gguf blobs are fetched
# at boot onto the persistent /models volume and cached there, so this is a
# one-time download per fresh volume. For each model 'stem', the download URL
# comes from ${STEM}_GGUF_URL (stem uppercased, '-'→'_'), e.g.
#   IBLOG_TUTOR      -> IBLOG_TUTOR_GGUF_URL
#   iblog-tutor-fr   -> IBLOG_TUTOR_FR_GGUF_URL
# `ollama create` is idempotent: skipped when `ollama list` already shows the tag.
shopt -s nullglob
registered=$(ollama list 2>/dev/null || true)
for modelfile in /models/*.Modelfile; do
    name=$(basename "$modelfile" .Modelfile)
    if echo "$registered" | grep -q "^${name}\b"; then
        log "model '$name' already registered — skipping"
        continue
    fi
    gguf="/models/${name}.gguf"
    if [ ! -f "$gguf" ]; then
        var="$(echo "$name" | tr '[:lower:]-' '[:upper:]_')_GGUF_URL"
        url="${!var:-}"
        if [ -z "$url" ]; then
            log "ERROR: $gguf is missing and \$$var is not set — cannot register '$name'."
            log "       Host the GGUF (object storage / HF) and set $var in the SDL env."
            continue
        fi
        # Private HuggingFace repos need a Bearer token; HF_TOKEN (or a generic
        # GGUF_AUTH_BEARER) is sent as an Authorization header when set. Public
        # URLs need neither.
        auth=()
        bearer="${GGUF_AUTH_BEARER:-${HF_TOKEN:-}}"
        [ -n "$bearer" ] && auth=(-H "Authorization: Bearer $bearer")
        log "downloading '$name' GGUF from \$$var (one-time; cached on the /models volume) ..."
        if curl -fL --retry 3 --retry-delay 5 "${auth[@]}" -o "$gguf.part" "$url"; then
            mv "$gguf.part" "$gguf"
        else
            log "ERROR: download of '$name' failed from $url"; rm -f "$gguf.part"; continue
        fi
    fi
    log "registering model '$name' ..."
    ( cd /models && ollama create "$name" -f "$(basename "$modelfile")" )
done
log "ollama models: $(ollama list | awk 'NR>1{print $1}' | tr '\n' ' ')"

# ── 3a. Pull the web-search-fallback generation model (opt-in feature) ──────
# ADR 0010: app.services.llm.generate_web_fallback_response deliberately
# uses settings.web_search_fallback_model ("gemma2:9b"), NOT either
# fine-tuned tutor -- live-tested 2026-09-06, the tutors false-refused 4/5
# answerable web-snippet questions the base model got right 3/3. That model
# was never added to entrypoint.sh's own provisioning (ADR 0010's own "Open,
# for later"): enabling WEB_SEARCH_ENGINE=tavily on a lease without this
# calls a model that was never pulled. Gated on the same env var so a lease
# running with the fallback off doesn't pay a ~5.4GB pull for nothing.
if [ "${WEB_SEARCH_ENGINE:-}" = "tavily" ]; then
    if ollama list 2>/dev/null | grep -q "^gemma2:9b\b"; then
        log "web-search-fallback model 'gemma2:9b' already present — skipping"
    else
        log "pulling web-search-fallback model 'gemma2:9b' (~5.4GB, one-time) ..."
        ollama pull gemma2:9b || log "ERROR: 'ollama pull gemma2:9b' failed — web-search fallback will fail loudly"
    fi
fi

# ── 3b. Fetch the XTTS Darija checkpoint (only when that engine is selected) ─
# medmac01/darija_xtt_2.0 -- see app/services/tts.py's XttsDarijaEngine and
# ADR 0006 (including why it is NOT the commercially-safe default). ~5.6 GB,
# cached on the /models volume, so this is one download per fresh volume.
if [ "${TTS_ENGINE:-}" = "xtts_darija" ]; then
    XTTS_BASE="${TTS_XTTS_BASE_URL:-https://huggingface.co/medmac01/darija_xtt_2.0/resolve/main}"
    mkdir -p "$TTS_XTTS_MODEL_DIR"
    # local name -> remote name (the checkpoint is published as model_2.1.pth)
    for pair in "config.json:config.json" "vocab.json:vocab.json" \
                "speaker_ref.wav:speaker_ref.wav" "model.pth:model_2.1.pth"; do
        local_name="${pair%%:*}"; remote_name="${pair##*:}"
        dest="$TTS_XTTS_MODEL_DIR/$local_name"
        [ -f "$dest" ] && { log "XTTS asset '$local_name' already present — skipping"; continue; }
        log "downloading XTTS asset '$remote_name' (one-time; cached on /models) ..."
        # --speed-limit/--speed-time turn a stalled transfer into an error the
        # retry can act on, rather than a silent hang: the 5.6 GB file dying
        # mid-transfer is exactly what happened pulling it on a laptop.
        if curl -fL -C - --retry 20 --retry-delay 5 --retry-all-errors \
                --speed-limit 2048 --speed-time 60 \
                -o "$dest.part" "$XTTS_BASE/$remote_name"; then
            mv "$dest.part" "$dest"
        else
            log "ERROR: XTTS asset '$remote_name' failed to download — voice will fail loudly"
            rm -f "$dest.part"
        fi
    done
fi

# ── 4. Wait for Postgres, then initialize schema (idempotent) ────────────────
if [ -n "${DATABASE_URL:-}" ]; then
    log "waiting for Postgres ..."
    waited=0
    until pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; do
        sleep 1; waited=$((waited + 1))
        if [ "$waited" -ge 120 ]; then log "ERROR: Postgres not ready after 120s"; break; fi
    done
    log "initializing DB schema (pgvector + tables; no-op if present) ..."
    "$APP_PY" -m app.models.db_init || log "WARNING: db_init failed — check DATABASE_URL"
else
    log "DATABASE_URL unset — skipping DB init (retrieval will fall back to disk)"
fi

# ── 5. Optionally warm both tutor models so keep_alive=-1 pins them ───────────
if [ "${WARM_MODELS:-1}" = "1" ]; then
    for name in $(ollama list | awk 'NR>1{print $1}'); do
        case "$name" in
            IBLOG_TUTOR*|iblog-tutor*)
                log "warming '$name' ..."
                curl -sf http://127.0.0.1:11434/api/generate \
                     -d "{\"model\":\"$name\",\"prompt\":\"مرحبا\",\"stream\":false,\"keep_alive\":-1}" \
                     >/dev/null 2>&1 || true ;;
        esac
    done
fi

# ── 6. Launch the API (replaces this shell as PID 1's child) ─────────────────
log "starting uvicorn on 0.0.0.0:8000 ..."
exec "$APP_PY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
