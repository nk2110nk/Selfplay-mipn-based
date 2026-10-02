#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_DIR/results}"
TARGET_ROOT="${TARGET_ROOT:-$PROJECT_DIR/../Results_MultiNego/results_α-Nego-based}"
CASE_NAME="${CASE_NAME:-case1}"
EPISODES="${EPISODES:-100}"
EXPECTED_EXPERT="${EXPECTED_EXPERT:-70}"
EXPECTED_GENERAL="${EXPECTED_GENERAL:-100}"

declare -a expert_files=()
declare -a general_files=()

while IFS= read -r file; do
  model_dir="${file%%/csv/*}"
  config="$model_dir/config.json"
  if [[ ! -f "$config" ]]; then
    echo "Missing config for result: $file" >&2
    exit 1
  fi
  if grep -q '"model_type": "general"' "$config"; then
    general_files+=("$file")
  elif grep -q '"model_type": "expert"' "$config"; then
    expert_files+=("$file")
  else
    echo "Unknown model type in: $config" >&2
    exit 1
  fi
done < <(find "$SOURCE_ROOT" -type f -path '*/csv/*/det=False_noise=False/*.tsv' | sort)

if [[ "${#expert_files[@]}" -ne "$EXPECTED_EXPERT" ]]; then
  echo "Expected $EXPECTED_EXPERT expert TSVs, found ${#expert_files[@]}" >&2
  exit 1
fi
if [[ "${#general_files[@]}" -ne "$EXPECTED_GENERAL" ]]; then
  echo "Expected $EXPECTED_GENERAL general TSVs, found ${#general_files[@]}" >&2
  exit 1
fi

expected_lines=$((EPISODES + 1))
for file in "${expert_files[@]}" "${general_files[@]}"; do
  lines=$(wc -l < "$file")
  if [[ "$lines" -ne "$expected_lines" ]]; then
    echo "Expected $expected_lines lines, found $lines: $file" >&2
    exit 1
  fi
done

copy_group() {
  local mode="$1"
  shift
  local file relative pair remainder domain target
  for file in "$@"; do
    relative="${file#*/csv/}"
    pair="${relative%%/*}"
    remainder="${relative#*/}"
    domain="${remainder%%/*}"
    target="$TARGET_ROOT/$mode/$pair/$domain/$CASE_NAME/$(basename "$file")"
    mkdir -p "$(dirname "$target")"
    cp -p "$file" "$target"
  done
}

copy_group expert "${expert_files[@]}"
copy_group general "${general_files[@]}"

echo "Exported ${#expert_files[@]} expert TSVs and ${#general_files[@]} general TSVs"
echo "Destination: $TARGET_ROOT (case: $CASE_NAME)"
