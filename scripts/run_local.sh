#!/bin/bash
# Local development run script

set -e

echo "=== Power Fair Value Pipeline ==="
echo ""

# Check Python version
python --version

# Setup directories
echo "Creating directories..."
make setup-dirs

# Install dependencies (if needed)
if [ "$1" == "install" ]; then
    echo "Installing dependencies..."
    make install
fi

# Run pipeline
echo "Running pipeline..."
DATE=${2:-$(date +%Y-%m-%d)}

echo "Target date: $DATE"

python -m pipeline.cli run --date $DATE

echo ""
echo "✓ Pipeline completed!"
echo "Check outputs/ and reports/ directories for results"
