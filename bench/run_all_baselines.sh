#!/usr/bin/env bash
# Run the full baseline set through bench/run_benchmark.py and rebuild the workbook.
# Usage: bench/run_all_baselines.sh [SEED]   (default seed 42)
# Each run lands in docs/baselines/<model>__<tag>__<timestamp>/ with provenance.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${1:-42}"
PY="${ATALLA_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PY="${PY:-$HOME/.venvs/atalla/bin/python}"
[ -x "$PY" ] || PY=python3
cd "$REPO"

run() { echo "== $*"; "$PY" bench/run_benchmark.py --seed "$SEED" "$@"; }

run --model basic           --tag strict-chained        --extra-args="--strict-end"
run --model layernorm_smoke --tag strict-chained        --extra-args="--strict-end"
run --model alexnet_small   --tag scale0.01-chained     --scale 0.01
run --model vit_micro       --tag default-sdpa-chained
run --model vit_micro       --tag flash-trial-chained   --env ATALLA_SDPA_FLASH=1
run --model gpt2_mlp        --tag chained
run --model gpt2_mlp        --tag gelu-emulator-chained --env ATALLA_GELU_KERNEL=tanh
run --model gpt2_attn       --tag chained
run --model gpt2_block      --tag chained
run --model gpt2_block      --tag gelu-emulator-chained --env ATALLA_GELU_KERNEL=tanh

"$PY" bench/build_workbook.py
echo "Workbook: output/Atalla_Benchmarking_Workbook.xlsx ; table: docs/baselines/README.md"
