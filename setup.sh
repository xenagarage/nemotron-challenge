#!/bin/bash
# Run once on a fresh GCP VM (Deep Learning VM with CUDA 12 pre-installed)
set -e

echo "=== NVIDIA Nemotron Challenge — GCP Setup ==="

pip install --upgrade pip

# Core stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.45.0"
pip install "peft>=0.12.0"
pip install "trl>=0.12.0"
pip install "accelerate>=0.30.0"
pip install "bitsandbytes>=0.43.0"
pip install datasets huggingface_hub kaggle wandb scipy

# Flash Attention for speed (takes a few minutes to compile)
pip install flash-attn --no-build-isolation || echo "Flash-attn failed — optional, skipping"

echo "=== GPU check ==="
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
echo "=== Setup complete ==="
