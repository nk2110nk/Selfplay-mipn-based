#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-selfplay-mipn-cases}"
CONTAINER_GPU0="${CONTAINER_GPU0:-alpha-nego-training}"
CONTAINER_GPU1="${CONTAINER_GPU1:-alpha-nego-training-gpu1}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/experiment_logs"
containers=("$CONTAINER_GPU0" "$CONTAINER_GPU1")
for gpu in 0 1; do
  log="$RESULTS_ROOT/experiment_logs/train-gpu${gpu}.log"
  container="${containers[$gpu]}"
  command="docker exec --workdir '$PROJECT_DIR' -e CUDA_VISIBLE_DEVICES=0 -e SHARD_INDEX=$gpu -e SHARD_COUNT=2 -e RESULTS_ROOT='$RESULTS_ROOT' '$container' bash run_command/run_case_experiments.sh 2>&1 | tee -a '$log'"
  if [[ "$gpu" == "0" ]]; then
    tmux new-session -d -s "$SESSION_NAME" -n "gpu0" "$command"
  else
    tmux new-window -t "$SESSION_NAME" -n "gpu1" "$command"
  fi
done

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
