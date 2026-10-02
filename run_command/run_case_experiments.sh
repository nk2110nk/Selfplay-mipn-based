#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CASES=(case1 case2 case3)
KNOWN_DOMAINS=(Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A)
UNKNOWN_DOMAINS=(Coffee Camera Lunch SmartPhone Kitchen)
AGENTS=(Boulware Conceder Linear Atlas3)

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
EXPERT_TIMESTEPS="${EXPERT_TIMESTEPS:-100000}"
GENERAL_TIMESTEPS="${GENERAL_TIMESTEPS:-300000}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"

if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: index=$SHARD_INDEX count=$SHARD_COUNT" >&2
  exit 2
fi

checkpoint_step() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import torch

try:
    checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(checkpoint.get("global_step", -1)))
except Exception:
    print(-1)
PY
}

run_training() {
  local job_index="$1" model_dir="$2" target="$3"
  shift 3
  if (( job_index % SHARD_COUNT != SHARD_INDEX )); then
    return
  fi
  if (( LIMIT > 0 && completed_jobs >= LIMIT )); then
    return
  fi

  local checkpoint="$model_dir/checkpoint.pt" step=-1
  if [[ -f "$checkpoint" ]]; then
    step="$(checkpoint_step "$checkpoint")"
  fi
  if (( step >= target )); then
    echo "[job $job_index] complete ($step/$target): $model_dir"
    completed_jobs=$((completed_jobs + 1))
    return
  fi

  local -a command=("$PYTHON_BIN" train.py "$@" --total-timesteps "$target" --device "$DEVICE")
  if (( step >= 0 )); then
    command+=(--resume "$model_dir")
    echo "[job $job_index] resume ($step/$target): $model_dir"
  else
    command+=(-sp "$model_dir")
    echo "[job $job_index] start (0/$target): $model_dir"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
  completed_jobs=$((completed_jobs + 1))
}

common_args=(
  --num-envs 1
  --seed "$SEED"
  --style neutral
  --auto-entropy
  --target-entropy-ratio 0.8
  --target-entropy-final-ratio 0.3
  --pool-eval-freq 10000
  --pool-eval-episodes 4
  --snapshot-freq 10000
  --max-pool-size 16
  --self-play-probability 0.0
  --scripted-probability 0.5
  --snapshot-probability 0.5
  --pfsp-alpha 1.0
  --uniform-mix 0.1
  --canonical-accept-action
  --hierarchical-entropy
)

job_index=0
completed_jobs=0

for case_name in "${CASES[@]}"; do
  for domain in "${KNOWN_DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENTS[@]}; i++)); do
      for ((j = i; j < ${#AGENTS[@]}; j++)); do
        agent0="${AGENTS[$i]}"
        agent1="${AGENTS[$j]}"
        pair="$agent0-$agent1"
        model_dir="$RESULTS_ROOT/$case_name/models/expert/$pair/$domain/AlphaNego_Negotiator"
        pool_agents=("$agent0")
        duplicate_option=(--allow-duplicate-opponents)
        if [[ "$agent0" != "$agent1" ]]; then
          pool_agents+=("$agent1")
          duplicate_option=(--no-allow-duplicate-opponents)
        fi
        run_training "$job_index" "$model_dir" "$EXPERT_TIMESTEPS" \
          -a "${pool_agents[@]}" -i "$domain" --model-type expert --case "$case_name" \
          --entropy-anneal-steps "$EXPERT_TIMESTEPS" "${duplicate_option[@]}" \
          "${common_args[@]}"
        job_index=$((job_index + 1))
      done
    done
  done

  model_dir="$RESULTS_ROOT/$case_name/models/general/AlphaNego_Negotiator"
  run_training "$job_index" "$model_dir" "$GENERAL_TIMESTEPS" \
    -a "${AGENTS[@]}" -i "${KNOWN_DOMAINS[@]}" --model-type general --case "$case_name" \
    --general-domain EnergySmall_A --compatible-domains "${UNKNOWN_DOMAINS[@]}" \
    --entropy-anneal-steps "$GENERAL_TIMESTEPS" --allow-duplicate-opponents \
    "${common_args[@]}"
  job_index=$((job_index + 1))
done

mkdir -p "$RESULTS_ROOT/experiment_status"
if (( LIMIT == 0 )) && [[ "$DRY_RUN" != "1" ]]; then
  touch "$RESULTS_ROOT/experiment_status/train-shard-${SHARD_INDEX}-of-${SHARD_COUNT}.done"
fi
echo "Training shard complete: shard=$SHARD_INDEX/$SHARD_COUNT jobs=$completed_jobs"
