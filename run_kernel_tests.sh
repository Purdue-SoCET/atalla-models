#!/usr/bin/env bash
# run_kernel_tests.sh — Unified Atalla-C Kernel Verification Runner
#
# Runs automated 5-step validation (Compile -> Spill Check -> Assemble -> Simulate -> Golden Diff):
#   ./run_kernel_tests.sh                # Runs all kernel tests
#   ./run_kernel_tests.sh avgpool        # Runs specific kernel test (test_avgpool.py)
#   ./run_kernel_tests.sh mse_loss       # Runs mse_loss tests

set -e

# Resolve real directory even if called through symlink
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python3"
if [ ! -f "${VENV_PYTHON}" ]; then
    VENV_PYTHON="python3"
fi

TESTS_DIR="${SCRIPT_DIR}/tests/kernels"

if [ -z "$1" ] || [ "$1" = "--all" ]; then
    echo "================================================================="
    echo "  Running All Atalla-C Standalone Kernel Verification Tests      "
    echo "================================================================="
    ${VENV_PYTHON} -m pytest "${TESTS_DIR}" -v
else
    KERNEL_NAME="$1"
    # Strip test_ prefix or .py suffix if user provided it
    KERNEL_NAME="$(echo "${KERNEL_NAME}" | sed 's/^test_//' | sed 's/\.py$//' | sed 's/\.c$//')"
    TARGET_TEST="${TESTS_DIR}/test_${KERNEL_NAME}.py"

    if [ ! -f "${TARGET_TEST}" ]; then
        echo "Error: Test file not found for kernel '${KERNEL_NAME}': ${TARGET_TEST}"
        echo "Available kernels:"
        ls "${TESTS_DIR}"/test_*.py | sed 's/.*test_//' | sed 's/\.py$//' | awk '{print "  - " $0}'
        exit 1
    fi

    echo "================================================================="
    echo "  Running Verification for Kernel: ${KERNEL_NAME}               "
    echo "================================================================="
    ${VENV_PYTHON} "${TARGET_TEST}"
fi
