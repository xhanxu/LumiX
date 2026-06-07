#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Test
echo "Inference"


# Unconditional text-to-multimodal inference:
python eval/eval_lumix_hypersim_inference.py \
    --config eval/configs/lumix_hypersim_inference.yaml

# Conditional inference:
# python eval/eval_lumix_hypersim_inference_condition_no_metrics.py \
#     --config eval/configs/conditional_inference_config_no_metrics_lumix.yaml
