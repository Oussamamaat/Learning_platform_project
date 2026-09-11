#!/usr/bin/env bash
# Boot script for the production `llm` service (config/Dockerfile.vllm, ADR 0011).
# Downloads the Darija and French AWQ weights once, then hands the container to
# serve_pair.sh, which starts both vLLM instances and exits if either dies.
#
# Env:
#   HF_TOKEN            read token for AWQ_REPO (needed only when weights are missing)
#   AWQ_REPO            default Oussamamaat/iblog-tutor-awq
#   MODELS_DIR          default /models
#   DARIJA_KV_BYTES, FRENCH_KV_BYTES, VLLM_EXTRA_ARGS, ...  passed through to serve_pair.sh
set -euo pipefail

log() { echo "[llm-entrypoint] $(date -u +%H:%M:%SZ) $*"; }

AWQ_REPO="${AWQ_REPO:-Oussamamaat/iblog-tutor-awq}"
MODELS_DIR="${MODELS_DIR:-/models}"
export LOG_DIR="${LOG_DIR:-$MODELS_DIR/logs}"
mkdir -p "$LOG_DIR" "$MODELS_DIR/awq"

log "boot"
echo "boot $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/boots.log"
( while true; do date -u +%Y-%m-%dT%H:%M:%SZ > "$LOG_DIR/heartbeat"; sleep 30; done ) &

for lang in darija french; do
    if [ -f "$MODELS_DIR/awq/$lang/config.json" ] && [ -f "$MODELS_DIR/awq/$lang/model.safetensors" ]; then
        log "$lang weights already present -- skipping download"
        continue
    fi
    log "downloading $lang weights from $AWQ_REPO ..."
    # snapshot_download keeps the repo-relative path, so files land in $MODELS_DIR/awq/<lang>/.
    python3 - "$AWQ_REPO" "$lang" "$MODELS_DIR/awq" <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

repo, lang, local_dir = sys.argv[1:4]
snapshot_download(repo_id=repo, allow_patterns=[f"{lang}/*"], local_dir=local_dir,
                  token=os.environ.get("HF_TOKEN") or None)
PYEOF
done

export DARIJA_MODEL_DIR="$MODELS_DIR/awq/darija"
export FRENCH_MODEL_DIR="$MODELS_DIR/awq/french"
exec bash /opt/iblog/serve_pair.sh
