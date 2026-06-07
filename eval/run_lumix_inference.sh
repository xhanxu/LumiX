#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Inference"


# Unconditional text-to-multimodal inference:
python eval/eval_lumix_inference.py \
    --config eval/configs/lumix_inference.yaml

# Conditional inference:
# python eval/eval_lumix_conditional_inference.py \
#     --config eval/configs/lumix_conditional_inference.yaml
