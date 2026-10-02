#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

KNOWN_DOMAINS=(
  Laptop
  ItexvsCypress
  IS_BT_Acquisition
  Grocery
  thompson
  Car
  EnergySmall_A
)

# Unseen during training, but compatible with the checkpoint's six heads and
# maximum of five values per head. Coffee and SmartPhone exceed that limit.
UNKNOWN_DOMAINS=(
  Camera
  Lunch
  Kitchen
)

DOMAINS=("${KNOWN_DOMAINS[@]}" "${UNKNOWN_DOMAINS[@]}")

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
MODEL_PATH="${MODEL_PATH:-}"

if [[ -z "$MODEL_PATH" ]]; then
  domain_name="$(IFS=-; echo "${KNOWN_DOMAINS[*]}")"
  agent_name="$(IFS=-; echo "${AGENTS[*]}")"
  experiment_dir="${RESULTS_ROOT%/}/${domain_name}_${agent_name}"
  mapfile -t model_dirs < <(
    find "$experiment_dir" -mindepth 2 -maxdepth 2 -type d \
      -name AlphaNego_Negotiator 2>/dev/null | sort
  )
  if [[ "${#model_dirs[@]}" -eq 0 ]]; then
    echo "General model not found under: $experiment_dir" >&2
    exit 1
  fi
  MODEL_PATH="${model_dirs[$(( ${#model_dirs[@]} - 1 ))]}"
fi

if [[ ! -f "${MODEL_PATH%/}/checkpoint.pt" ]]; then
  echo "checkpoint.pt not found: $MODEL_PATH" >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN" test_negotiator.py
  --model-path "$MODEL_PATH"
  --agents "${AGENTS[@]}"
  --issues "${DOMAINS[@]}"
  --episodes "$EPISODES"
  --style "$STYLE"
  --device "$DEVICE"
  --seed "$SEED"
)

echo "General evaluation: 10 opponent pairs x ${#DOMAINS[@]} domains = $((10 * ${#DOMAINS[@]})) cases"
echo "Known domains: ${KNOWN_DOMAINS[*]}"
echo "Unseen compatible domains: ${UNKNOWN_DOMAINS[*]}"
echo "Model: $MODEL_PATH"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '  %q' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
