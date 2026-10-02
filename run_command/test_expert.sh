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
RESULTS_ROOT="${RESULTS_ROOT:-results}"
EPISODES="${EPISODES:-100}"
STYLE="${STYLE:-neutral}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"
STRICT="${STRICT:-1}"

total=$(( ${#DOMAINS[@]} * (${#AGENTS[@]} * (${#AGENTS[@]} + 1) / 2) ))
current=0
ran=0
missing=0

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
      pool_name="$agent0"
      if [[ "$agent0" != "$agent1" ]]; then
        pool_name="$agent0-$agent1"
      fi
      experiment_dir="${RESULTS_ROOT%/}/${domain}_${pool_name}"

      mapfile -t model_dirs < <(
        find "$experiment_dir" -mindepth 2 -maxdepth 2 -type d \
          -name AlphaNego_Negotiator 2>/dev/null | sort
      )
      if [[ "${#model_dirs[@]}" -eq 0 ]]; then
        echo "[$current/$total] missing: domain=$domain opponents=$agent0-$agent1" >&2
        missing=$((missing + 1))
        continue
      fi
      model_dir="${model_dirs[$(( ${#model_dirs[@]} - 1 ))]}"
      if [[ ! -f "$model_dir/checkpoint.pt" ]]; then
        echo "[$current/$total] missing checkpoint: $model_dir" >&2
        missing=$((missing + 1))
        continue
      fi

      cmd=(
        "$PYTHON_BIN" test_negotiator.py
        --model-path "$model_dir"
        --agents "$agent0" "$agent1"
        --issues "$domain"
        --episodes "$EPISODES"
        --style "$STYLE"
        --device "$DEVICE"
        --seed "$SEED"
      )

      echo "[$current/$total] domain=$domain opponents=$agent0-$agent1 model=$model_dir"
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

echo "Expert evaluation complete: ran=$ran missing=$missing"
if [[ "$STRICT" == "1" && "$missing" -gt 0 ]]; then
  exit 1
fi
if [[ "$ran" -eq 0 ]]; then
  exit 1
fi
