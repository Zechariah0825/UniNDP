#!/usr/bin/env bash

# Run compile_e2e_pax.py over all models defined in script/models_llm.csv.
# Extra arguments are forwarded to compile_e2e_pax.py so you can override
# architecture, topk, output-dir, etc.
#
# Example:
#   bash script/run_all_llm_e2e_pax.sh \
#     --architecture hbm-pim \
#     --topk 1 \
#     --output-dir pruning_and_breakdown/e2e_pax

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "${PROJECT_ROOT}"

python3 "${SCRIPT_DIR}/compile_e2e_pax.py" \
  --models-csv "${SCRIPT_DIR}/models_llm.csv" \
  "$@"


