#!/bin/bash

# Script to parse model predictions and calculate metrics
# Usage: ./run_metrics.sh [options] <results_file.json> [<results_file2.json> ...]
#
# Options:
#   --tracking     Enable detailed extraction method tracking
#   --strict       Use strict academic evaluation (JSON-only)
#   --compare      Compare multiple result files (requires at least 2 files)
#   --help         Show this help message
#
# Examples:
#   ./run_metrics.sh results/abl2_1500_iters.json
#   ./run_metrics.sh --tracking results/abl2_1500_iters.json
#   ./run_metrics.sh --strict results/abl2_1500_iters.json
#   ./run_metrics.sh --compare results/abl0_2250_iters_2.json results/abl1.json results/abl2_1750_iters.json

# Function to display help message
show_help() {
  echo "Usage: ./run_metrics.sh [options] <results_file.json> [<results_file2.json> ...]"
  echo ""
  echo "Options:"
  echo "  --tracking     Enable detailed extraction method tracking"
  echo "  --strict       Use strict academic evaluation (JSON-only)"
  echo "  --compare      Compare multiple result files (requires at least 2 files)"
  echo "  --help         Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run_metrics.sh results/abl2_1500_iters.json"
  echo "  ./run_metrics.sh --tracking results/abl2_1500_iters.json"
  echo "  ./run_metrics.sh --strict results/abl2_1500_iters.json"
  echo "  ./run_metrics.sh --compare results/abl0_2250_iters_2.json results/abl1.json results/abl2_1750_iters.json"
  exit 0
}

# Parse command line options
OPTIONS=""
FILES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --tracking|--strict|--compare)
      OPTIONS="$OPTIONS $1"
      shift
      ;;
    --help)
      show_help
      ;;
    --*)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
    *)
      FILES+=("$1")
      shift
      ;;
  esac
done

# Check if files are provided
if [ ${#FILES[@]} -eq 0 ]; then
  echo "Error: No results file specified"
  show_help
fi

# If using compare, make sure we have at least 2 files
if [[ "$OPTIONS" == *"--compare"* ]] && [ ${#FILES[@]} -lt 2 ]; then
  echo "Error: --compare option requires at least 2 result files"
  exit 1
fi

# Prepare file arguments for parse_predictions.py
FILE_ARGS=$(printf " %s" "${FILES[@]}")

# Run the prediction parser in Docker container
echo "Running with options: $OPTIONS"
echo "Processing files: $FILE_ARGS"

docker run --rm \
  --gpus device=0 \
  -v $(pwd):/app \
  -v $(pwd)/results:/app/results \
  -w /app \
  mistral-nli-ft \
  bash -c "pip install scikit-learn && python evaluate/parse_predictions.py$OPTIONS$FILE_ARGS"

echo "Metrics processing complete!"