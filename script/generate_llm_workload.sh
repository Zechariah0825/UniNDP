#!/usr/bin/env bash

# Thin wrapper around generate_llm_workload.py
# Usage examples:
#   ./generate_llm_workload.sh --model Qwen3-8B --phase decode --tklen 32 --attn gqa
#   ./generate_llm_workload.sh --model-csv models.csv --phase decode --attn gqa

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/generate_llm_workload.py" "$@"




