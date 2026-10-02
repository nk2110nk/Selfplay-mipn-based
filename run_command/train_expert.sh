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
TIMESTEPS="${TIMESTEPS:-100000}"
N_ENVS="${N_ENVS:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
TARGET_ENTROPY_RATIO="${TARGET_ENTROPY_RATIO:-0.8}"
TARGET_ENTROPY_FINAL_RATIO="${TARGET_ENTROPY_FINAL_RATIO:-0.3}"
ENTROPY_ANNEAL_STEPS="${ENTROPY_ANNEAL_STEPS:-$TIMESTEPS}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"

total=$(( ${#DOMAINS[@]} * (${#AGENTS[@]} * (${#AGENTS[@]} + 1) / 2) ))
current=0
ran=0

for domain in "${DOMAINS[@]}"; do
  for ((i = 0; i < ${#AGENTS[@]}; i++)); do
    for ((j = i; j < ${#AGENTS[@]}; j++)); do
      current=$((current + 1))
      if [[ "$LIMIT" -gt 0 && "$ran" -ge "$LIMIT" ]]; then
        echo "Reached LIMIT=$LIMIT"
        exit 0
      fi

      agent0="${AGENTS[$i]}"
      agent1="${AGENTS[$j]}"
      pool_agents=("$agent0")
      duplicate_option=(--allow-duplicate-opponents)
      if [[ "$agent0" != "$agent1" ]]; then
        pool_agents+=("$agent1")
        duplicate_option=(--no-allow-duplicate-opponents)
      fi

      cmd=(
        "$PYTHON_BIN" train.py
        -a "${pool_agents[@]}"
        -i "$domain"
        -sp "$SAVE_ROOT"
        --model-type expert
        --total-timesteps "$TIMESTEPS"
        --num-envs "$N_ENVS"
        --device "$DEVICE"
        --seed "$SEED"
        --auto-entropy
        --target-entropy-ratio "$TARGET_ENTROPY_RATIO"
        --target-entropy-final-ratio "$TARGET_ENTROPY_FINAL_RATIO"
        --entropy-anneal-steps "$ENTROPY_ANNEAL_STEPS"
        "${duplicate_option[@]}"
      )

      echo "[$current/$total] domain=$domain opponents=$agent0-$agent1"
      if [[ "$DRY_RUN" == "1" ]]; then
        printf '  %q' "${cmd[@]}"
        printf '\n'
      else
        "${cmd[@]}"
      fi
      ran=$((ran + 1))
    done
  done
done
