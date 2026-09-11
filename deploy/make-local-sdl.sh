#!/usr/bin/env bash
# Renders a committed SDL template into deploy/<name>.local.yaml (gitignored)
# with real secrets substituted in. Plan rev 2, Phase A10: generalized from a
# single hardcoded template to any of this repo's SDLs, so the same script
# renders both the always-on production deploy and the one-shot Phase B
# benchmark/build lease without duplicating this logic.
#
#   HF_READ_TOKEN=hf_xxx bash deploy/make-local-sdl.sh deploy/akash-deploy.yaml
#   HF_TOKEN=hf_yyy      bash deploy/make-local-sdl.sh deploy/akash-vllm-bench.yaml
#
# Template arg defaults to deploy/akash-deploy.yaml (the production SDL) if
# omitted, so the original one-arg invocation from before this generalization
# still works unchanged.
#
# Deploy the .local.yaml — never the template. Akash providers can read SDL
# env, so every token substituted here must already be scoped narrowly (a
# fine-grained READ token for the production GGUF repo; a fine-grained
# token scoped to exactly the repos the bench lease needs -- see
# akash-vllm-bench.yaml's own header note on what that token needs).
set -euo pipefail
cd "$(dirname "$0")/.."

TEMPLATE="${1:-deploy/akash-deploy.yaml}"
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: template not found: $TEMPLATE" >&2
    exit 1
fi

BASENAME=$(basename "$TEMPLATE" .yaml)
OUT="deploy/${BASENAME}.local.yaml"

SED_ARGS=()

if grep -q "CHANGE_ME_strong_password" "$TEMPLATE"; then
    # `head -c 32` closes the pipe, tr dies of SIGPIPE (141), and with
    # `pipefail` that aborts the whole script before it writes anything.
    # Read a bounded chunk up front so nothing has to be killed by a closed pipe.
    PW=$(LC_ALL=C tr -dc 'A-Za-z0-9' < <(head -c 4096 /dev/urandom) | cut -c1-32)
    SED_ARGS+=(-e "s|CHANGE_ME_strong_password|${PW}|g")
    echo "postgres password: ${PW}"
fi

if grep -q "PASTE_A_READ_ONLY_HF_TOKEN" "$TEMPLATE"; then
    : "${HF_READ_TOKEN:?set HF_READ_TOKEN to a fine-grained READ token for Oussamamaat/iblog-tutor-gguf}"
    case "$HF_READ_TOKEN" in
        hf_*) ;;
        *) echo "ERROR: HF_READ_TOKEN does not look like a HuggingFace token" >&2; exit 1 ;;
    esac
    SED_ARGS+=(-e "s|PASTE_A_READ_ONLY_HF_TOKEN|${HF_READ_TOKEN}|")
fi

if grep -q "PASTE_A_SCOPED_HF_TOKEN" "$TEMPLATE"; then
    : "${HF_TOKEN:?set HF_TOKEN to a fine-grained token, scoped per the header note in akash-vllm-bench.yaml}"
    case "$HF_TOKEN" in
        hf_*) ;;
        *) echo "ERROR: HF_TOKEN does not look like a HuggingFace token" >&2; exit 1 ;;
    esac
    SED_ARGS+=(-e "s|PASTE_A_SCOPED_HF_TOKEN|${HF_TOKEN}|")
fi

if [ "${#SED_ARGS[@]}" -eq 0 ]; then
    echo "ERROR: no known placeholders found in $TEMPLATE -- nothing to substitute." >&2
    exit 1
fi

sed "${SED_ARGS[@]}" "$TEMPLATE" > "$OUT"

echo "wrote $OUT (gitignored)"
if grep -qE "PASTE_A_READ_ONLY_HF_TOKEN|PASTE_A_SCOPED_HF_TOKEN|CHANGE_ME" "$OUT"; then
    echo "ERROR: placeholders remain"
    exit 1
else
    echo "no placeholders remain — ready to deploy"
fi
