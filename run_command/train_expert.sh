#!/usr/bin/env bash
set -euo pipefail

python train.py \
  -a Boulware Linear Conceder Atlas3 \
  -i Laptop \
  --model-type expert \
  --total-timesteps 1100000 \
  --device auto
