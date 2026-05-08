#!/bin/bash
set -e
source /root/llm-detection/.venv/bin/activate
cd /root/llm-detection/itsai-detection
exec python -u scripts/run_inference_server_hssd.py \
    --model-dir models/seam_detector_v4_rebalanced/best \
    --base-model models/deberta-v3-large \
    --host 0.0.0.0 \
    --port 20000 \
    --device cuda \
    --max-batch-size 32 \
    --token '7f3f09a57330e877a0a59cfc7868d2bc3bf4cb677c92fd31b4d62f0d71fcf0c0'
