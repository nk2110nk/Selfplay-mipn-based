#!/usr/bin/env bash
set -euo pipefail

python train.py \
  -a Boulware Linear Conceder Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A \
  --model-type general \
  --general-domain EnergySmall_A \
  --total-timesteps 1100000 \
  --device auto
