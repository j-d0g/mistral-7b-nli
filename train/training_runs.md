# Training Run Records

## Run 1: Initial Test Run

- **Date**: May 1, 2025
- **Model**: Mistral-7B-v0.3
- **Device**: RTX 4090 (GPU 1)
- **Batch Size**: 8
- **Gradient Accumulation**: 2
- **Effective Batch Size**: 16
- **Sequence Length**: 512
- **Learning Rate**: 2e-4
- **Learning Rate Schedule**: Cosine with 3% warmup
- **Weight Decay**: 0.01
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Training Time**: ~2.5-3 hours (estimated)
- **Output Directory**: models/mistral-7b-nli-cot

## Ablation 1: Standard (Planned)

- **Date**: TBD
- **Model**: Mistral-7B-v0.3
- **Device**: RTX 4090 (GPU 1)
- **Batch Size**: 16
- **Gradient Accumulation**: 2
- **Effective Batch Size**: 32
- **Sequence Length**: 512
- **Learning Rate**: 2e-4
- **Learning Rate Schedule**: Cosine with 3% warmup
- **Weight Decay**: 0.01
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Output Directory**: models/mistral-7b-nli-cot-ablation1

## Ablation 2: Mixed Data Optimization (Planned)

- **Date**: TBD
- **Model**: Mistral-7B-v0.3
- **Device**: RTX 4090 (GPU 1)
- **Batch Size**: 16
- **Gradient Accumulation**: 4
- **Effective Batch Size**: 64
- **Sequence Length**: 512
- **Learning Rate**: 1e-4
- **Learning Rate Schedule**: Cosine with 10% warmup
- **Weight Decay**: 0.01
- **Gradient Clipping**: 1.0
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Output Directory**: models/mistral-7b-nli-cot-ablation2 