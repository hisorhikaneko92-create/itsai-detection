#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llmdetect
cd /root/llm-detection

exec python neurons/miner.py \
    --netuid 32 \
    --subtensor.network finney \
    --wallet.name term-wallet \
    --wallet.hotkey itsai-1 \
    --axon.port 8091 \
    --axon.external_ip 74.119.194.67 \
    --axon.external_port 8091 \
    --neuron.remote_inference_url http://45.20.65.0:20000 \
    --neuron.remote_inference_timeout 18 \
    --blacklist.minimum_stake_requirement 30000 \
    --logging.debug \
    --neuron.remote_inference_token 7f3f09a57330e877a0a59cfc7868d2bc3bf4cb677c92fd31b4d62f0d71fcf0c0
