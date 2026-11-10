#!/usr/bin/env bash
# Stage a converted model and upload it to Hugging Face with the hf CLI.
# Usage: tools/publish/publish.sh <0.8b|2b> [owner]     (owner defaults to the logged-in hf user)
# MSG="..." sets the commit message.
# Afterwards, put the printed commit hash into MODELS[...].revision in extension/src/shared/common.ts
# (and the sizes/sha256 from manifest.json if the files changed).
set -euo pipefail
cd "$(dirname "$0")"
SIZE=$1
OWNER=${2:-$(hf auth whoami | sed -n 's/^user=//p')}
REPO=$OWNER/plamo-3-translate-$SIZE-onnx-webgpu
STAGE=$(mktemp -d)/$SIZE
python3 stage_hf.py "$SIZE" "$REPO" "$STAGE"
hf upload "$REPO" "$STAGE" . --repo-type model --commit-message "${MSG:-Add q4f16 ONNX (WebGPU) conversion of pfnet/plamo-3-translate-$SIZE}"
echo "revision: $(curl -fsSL "https://huggingface.co/api/models/$REPO" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')"
