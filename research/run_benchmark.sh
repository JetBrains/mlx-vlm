#!/bin/bash
#
# Run the mlx-vlm int8 NAX prefill benchmark
#
# This script:
# 1. Activates the project's .venv (or creates it if missing)
# 2. Installs any missing dependencies
# 3. Runs the benchmark script (baseline vs --int8-prefill)
#
# Usage:
#   ./research/run_benchmark.sh                    # Default (3 iterations)
#   ./research/run_benchmark.sh --iterations 5     # Custom iterations
#   ./research/run_benchmark.sh --port 9000        # Custom port
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  mlx-vlm Performance Benchmark"
echo "=========================================="
echo ""

# Step 1: Set up virtual environment
echo "[1/3] Setting up virtual environment..."

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "  Creating .venv..."
    python3 -m venv "$PROJECT_ROOT/.venv"
    echo "  Created."
fi

# Activate the virtual environment
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"
echo "  Using Python: $(which python)"
echo "  Python version: $(python --version)"

# Step 2: Install dependencies
echo ""
echo "[2/3] Checking dependencies..."

MISSING=""
for pkg in openai huggingface_hub mlx-vlm; do
    if ! python -c "import ${pkg//-/_}" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo "  Installing missing packages:$MISSING"
    pip install -q $MISSING
else
    echo "  All dependencies satisfied."
fi

# Step 3: Run the benchmark
echo ""
echo "[3/3] Starting benchmark..."
echo ""

python "$SCRIPT_DIR/benchmark.py" "$@"