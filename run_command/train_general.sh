#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DOMAINS=(
  Laptop
  ItexvsCypress
  IS_BT_Acquisition
  Grocery
  thompson
  Car
  EnergySmall_A
)

AGENTS=(
  Boulware
  Conceder
  Linear
  Atlas3
)

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_ROOT="${SAVE_ROOT:-results}"
TIMESTEPS="${TIMESTEPS:-300000}"
N_ENVS="${N_ENVS:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
TARGET_ENTROPY_RATIO="${TARGET_ENTROPY_RATIO:-0.8}"
TARGET_ENTROPY_FINAL_RATIO="${TARGET_ENTROPY_FINAL_RATIO:-0.3}"
ENTROPY_ANNEAL_STEPS="${ENTROPY_ANNEAL_STEPS:-$TIMESTEPS}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "$PYTHON_BIN" train.py
  -a "${AGENTS[@]}"
  -i "${DOMAINS[@]}"
  -sp "$SAVE_ROOT"
  --model-type general
  --general-domain EnergySmall_A
  --total-timesteps "$TIMESTEPS"
  --num-envs "$N_ENVS"
  --device "$DEVICE"
  --seed "$SEED"
  --auto-entropy
  --target-entropy-ratio "$TARGET_ENTROPY_RATIO"
  --target-entropy-final-ratio "$TARGET_ENTROPY_FINAL_RATIO"
  --entropy-anneal-steps "$ENTROPY_ANNEAL_STEPS"
  --allow-duplicate-opponents
)

echo "General training: ${#DOMAINS[@]} domains, ${#AGENTS[@]} scripted opponents, $TIMESTEPS steps"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '  %q' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
