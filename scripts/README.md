# Data Processing Scripts

This directory contains the scripts used in the data preparation pipeline for NLI fine-tuning.

## Key Components

- `generate_thoughts.py`: Generates Chain-of-Thought reasoning for original examples
- `generate_thoughts_reflected.py`: Creates improved reasoning for initially incorrect examples
- `prepare_ft_data.py`: Combines correct examples and reflected examples into training data

## Usage

For detailed documentation on the data preparation process, including quickstart guides and technical details, please refer to the [DATA.md](../DATA.md) document in the project root.

Basic usage examples:

```bash
# Generate thoughts for training data
docker run --rm --gpus all -v $(pwd):/app -w /app mistral-nli-ft python3 scripts/generate_thoughts.py \
  --api mistral \
  --input-csv data/original_data/train.csv \
  --output-json data/original_thoughts/train_thoughts.json \
  --workers 6
```

See the main [DATA.md](../DATA.md) for the complete pipeline and detailed explanations.