#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CASES=(case1 case2 case3)
KNOWN_DOMAINS=(Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A)
UNKNOWN_DOMAINS=(Coffee Camera Lunch SmartPhone Kitchen)
ALL_DOMAINS=("${KNOWN_DOMAINS[@]}" "${UNKNOWN_DOMAINS[@]}")
AGENTS=(Boulware Conceder Linear Atlas3)

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-100}"
STYLE="${STYLE:-neutral}"
SEED="${SEED:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"

if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: index=$SHARD_INDEX count=$SHARD_COUNT" >&2
  exit 2
fi

run_evaluation() {
  local job_index="$1"
  shift
  if (( job_index % SHARD_COUNT != SHARD_INDEX )); then
    return
  fi
  if (( LIMIT > 0 && completed_jobs >= LIMIT )); then
    return
  fi
  echo "[job $job_index] evaluate: $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "$PYTHON_BIN" test_negotiator.py "$@"
    printf '\n'
  else
    "$PYTHON_BIN" test_negotiator.py "$@"
  fi
  completed_jobs=$((completed_jobs + 1))
}

job_index=0
completed_jobs=0
for case_name in "${CASES[@]}"; do
  export_root="$RESULTS_ROOT/$case_name/evaluation"
  for domain in "${KNOWN_DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENTS[@]}; i++)); do
      for ((j = i; j < ${#AGENTS[@]}; j++)); do
        agent0="${AGENTS[$i]}"
        agent1="${AGENTS[$j]}"
        pair="$agent0-$agent1"
        model_dir="$RESULTS_ROOT/$case_name/models/expert/$pair/$domain/AlphaNego_Negotiator"
        if [[ ! -f "$model_dir/checkpoint.pt" && "$DRY_RUN" != "1" ]]; then
          echo "Missing expert checkpoint: $model_dir" >&2
          exit 1
        fi
        run_evaluation "$job_index" --model-path "$model_dir" \
          --agents "$agent0" "$agent1" --issues "$domain" --episodes "$EPISODES" \
          --style "$STYLE" --seed "$SEED" --device "$DEVICE" --case "$case_name" \
          --export-root "$export_root"
        job_index=$((job_index + 1))
      done
    done
  done

  model_dir="$RESULTS_ROOT/$case_name/models/general/AlphaNego_Negotiator"
  if [[ ! -f "$model_dir/checkpoint.pt" && "$DRY_RUN" != "1" ]]; then
    echo "Missing general checkpoint: $model_dir" >&2
    exit 1
  fi
  run_evaluation "$job_index" --model-path "$model_dir" \
    --agents "${AGENTS[@]}" --issues "${ALL_DOMAINS[@]}" --episodes "$EPISODES" \
    --style "$STYLE" --seed "$SEED" --device "$DEVICE" --case "$case_name" \
    --export-root "$export_root"
  job_index=$((job_index + 1))
done

mkdir -p "$RESULTS_ROOT/experiment_status"
if (( LIMIT == 0 )) && [[ "$DRY_RUN" != "1" ]]; then
  touch "$RESULTS_ROOT/experiment_status/eval-shard-${SHARD_INDEX}-of-${SHARD_COUNT}.done"
fi
echo "Evaluation shard complete: shard=$SHARD_INDEX/$SHARD_COUNT jobs=$completed_jobs"
