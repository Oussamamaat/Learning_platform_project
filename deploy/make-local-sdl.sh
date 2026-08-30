#!/usr/bin/env bash
# Renders the committed template into deploy/akash-deploy.local.yaml (gitignored)
# with a real Postgres password and your HuggingFace READ token.
#
#   HF_READ_TOKEN=hf_xxx bash deploy/make-local-sdl.sh
#
# Deploy the .local.yaml — never the template. Akash providers can read SDL env,
# so HF_READ_TOKEN must be a fine-grained READ token scoped to the GGUF repo.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_READ_TOKEN:?set HF_READ_TOKEN to a fine-grained READ token for Oussamamaat/iblog-tutor-gguf}"

case "$HF_READ_TOKEN" in
  hf_*) ;;
  *) echo "ERROR: HF_READ_TOKEN does not look like a HuggingFace token" >&2; exit 1 ;;
esac

# `head -c 32` closes the pipe, tr dies of SIGPIPE (141), and with `pipefail`
# that aborts the whole script before it writes anything. Read a bounded chunk
# up front so nothing has to be killed by a closed pipe.
PW=$(LC_ALL=C tr -dc 'A-Za-z0-9' < <(head -c 4096 /dev/urandom) | cut -c1-32)

sed -e "s|CHANGE_ME_strong_password|${PW}|g" \
    -e "s|PASTE_A_READ_ONLY_HF_TOKEN|${HF_READ_TOKEN}|" \
    deploy/akash-deploy.yaml > deploy/akash-deploy.local.yaml

echo "wrote deploy/akash-deploy.local.yaml (gitignored)"
echo "postgres password: ${PW}"
grep -n "PASTE_A_READ_ONLY_HF_TOKEN\|CHANGE_ME" deploy/akash-deploy.local.yaml \
  && { echo "ERROR: placeholders remain"; exit 1; } || echo "no placeholders remain — ready to deploy"
