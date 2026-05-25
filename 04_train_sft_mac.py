"""
Mac-native SFT training using Apple MLX.
Wraps mlx_lm.lora which runs natively on M-series chips (no CUDA needed).

Usage:
    python3 04_train_sft_mac.py

Or let the tmux launcher (run_training.sh) handle everything.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Verify exact model ID at: huggingface.co/collections/nvidia/nvidia-nemotron-v3
MODEL_ID       = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SFT_DATA_PATH  = "sft_training_data.jsonl"
OUTPUT_DIR     = "./checkpoints/sft-mac"
LORA_RANK      = 32     # Competition max
LORA_LAYERS    = 32     # Number of transformer layers to apply LoRA to
BATCH_SIZE     = 2      # M4 Pro 48GB can handle 2
LEARNING_RATE  = 2e-4
ITERS          = 2000   # ~2 epochs over 9500 examples with packing
SAVE_EVERY     = 200    # Save adapter checkpoint every N steps
# ──────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. Set it with: export HF_TOKEN=your_token")
    print("Get your token at: huggingface.co/settings/tokens")


def convert_to_mlx_format(jsonl_path: str, out_path: str):
    """
    MLX lora expects a JSONL where each line is:
        {"text": "<full conversation as string>"}
    We re-format our SFT data into that shape using a simple chat template.
    """
    print(f"Converting {jsonl_path} → {out_path}")
    count = 0
    with open(jsonl_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])

            # Nemotron / ChatML style template
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"<|system|>\n{content}")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}")
            parts.append("")  # trailing newline
            text = "\n".join(parts)
            fout.write(json.dumps({"text": text}) + "\n")
            count += 1
    print(f"Converted {count} examples → {out_path}")


def main():
    # Step 1: Convert data
    mlx_data_path = "sft_mlx_format.jsonl"
    if not Path(mlx_data_path).exists():
        if not Path(SFT_DATA_PATH).exists():
            print(f"ERROR: {SFT_DATA_PATH} not found.")
            print("Run python3 03_generate_sft_data.py first.")
            sys.exit(1)
        convert_to_mlx_format(SFT_DATA_PATH, mlx_data_path)
    else:
        print(f"Using existing {mlx_data_path}")

    # Count lines for info
    n = sum(1 for _ in open(mlx_data_path))
    print(f"Training examples: {n}")
    print(f"Model            : {MODEL_ID}")
    print(f"LoRA rank        : {LORA_RANK}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Iterations       : {ITERS}")
    print(f"Output           : {OUTPUT_DIR}")
    print()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Step 2: Build mlx_lm.lora command
    # mlx_lm.lora is the Mac-native LoRA trainer
    mlx_bin = Path.home() / "Library/Python/3.14/bin/mlx_lm.lora"
    if not mlx_bin.exists():
        mlx_bin = "mlx_lm.lora"  # try PATH

    cmd = [
        str(mlx_bin),
        "--model", MODEL_ID,
        "--train",
        "--data", mlx_data_path,
        "--adapter-path", OUTPUT_DIR,
        "--lora-layers", str(LORA_LAYERS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--iters", str(ITERS),
        "--save-every", str(SAVE_EVERY),
        "--steps-per-report", "10",
        "--grad-checkpoint",   # saves memory during backprop
    ]

    if HF_TOKEN:
        env = os.environ.copy()
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    else:
        env = os.environ.copy()

    print("=== Starting MLX LoRA SFT training ===")
    print("Command:", " ".join(cmd))
    print()
    print("TIP: Check memory pressure in Activity Monitor → Memory tab.")
    print("     Keep Mac plugged in. Do not close the tmux session.")
    print()

    result = subprocess.run(cmd, env=env)

    if result.returncode == 0:
        print("\n=== SFT Training complete! ===")
        print(f"Adapter saved at: {OUTPUT_DIR}")
        print("Next: run python3 05_fuse_and_submit_mac.py")
    else:
        print(f"\nTraining exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
