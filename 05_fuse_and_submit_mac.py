"""
Step 5 (Mac): Fuse LoRA adapter into base model (MLX format) and
package as submission.zip compatible with the competition's vLLM evaluator.

MLX saves adapters in its own format. The competition loads via vLLM + PEFT.
This script:
  1. Fuses the MLX LoRA weights into a merged model
  2. Converts to safetensors HuggingFace format
  3. Extracts ONLY the delta (LoRA adapter) as adapter_config.json + adapter weights
  4. Packages into submission.zip

Run: python3 05_fuse_and_submit_mac.py
"""
import os
import sys
import json
import zipfile
import subprocess
from pathlib import Path

MLX_ADAPTER_DIR  = "./checkpoints/sft-mac"
FUSED_DIR        = "./checkpoints/fused-mlx"
ADAPTER_OUT_DIR  = "./checkpoints/final-adapter"
MODEL_ID         = "nvidia/Nemotron-3-Nano-30B-Instruct"
LORA_RANK        = 32

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def fuse_mlx_adapter():
    """Fuse MLX adapter weights into merged model using mlx_lm.fuse."""
    mlx_fuse = Path.home() / "Library/Python/3.14/bin/mlx_lm.fuse"
    if not mlx_fuse.exists():
        mlx_fuse = "mlx_lm.fuse"

    Path(FUSED_DIR).mkdir(parents=True, exist_ok=True)

    cmd = [
        str(mlx_fuse),
        "--model", MODEL_ID,
        "--adapter-path", MLX_ADAPTER_DIR,
        "--save-path", FUSED_DIR,
        "--de-quantize",  # convert back to bf16 for vLLM compatibility
    ]

    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    print("Fusing LoRA adapter into model...")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("Fusion failed!")
        sys.exit(1)
    print(f"Fused model saved to: {FUSED_DIR}")


def extract_lora_delta():
    """
    The competition requires a LoRA adapter (not a full merged model).
    After fusing with MLX, we need to re-extract the LoRA delta in PEFT format
    so vLLM can load it with the base model.

    We use the mlx_lm checkpointed adapter directly — MLX saves
    adapter_config.json + adapters.safetensors which are PEFT-compatible.
    """
    Path(ADAPTER_OUT_DIR).mkdir(parents=True, exist_ok=True)
    src = Path(MLX_ADAPTER_DIR)

    # Copy MLX adapter files to final output
    files_to_copy = list(src.glob("*.safetensors")) + list(src.glob("*.json"))
    if not files_to_copy:
        print(f"ERROR: No adapter files found in {MLX_ADAPTER_DIR}")
        print("Make sure training completed successfully.")
        sys.exit(1)

    for f in files_to_copy:
        dest = Path(ADAPTER_OUT_DIR) / f.name
        dest.write_bytes(f.read_bytes())
        print(f"  Copied: {f.name}")

    # Write a PEFT-compatible adapter_config.json if not present
    peft_config_path = Path(ADAPTER_OUT_DIR) / "adapter_config.json"
    if not peft_config_path.exists():
        peft_config = {
            "base_model_name_or_path": MODEL_ID,
            "bias": "none",
            "fan_in_fan_out": False,
            "lora_alpha": LORA_RANK * 2,
            "lora_dropout": 0.05,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": LORA_RANK,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "task_type": "CAUSAL_LM",
        }
        with open(peft_config_path, "w") as f:
            json.dump(peft_config, f, indent=2)
        print("  Wrote adapter_config.json (PEFT format)")

    # Validate rank
    with open(peft_config_path) as f:
        cfg = json.load(f)
    rank = cfg.get("r", cfg.get("rank", "?"))
    print(f"\n  LoRA rank: {rank}")
    if isinstance(rank, int) and rank > 32:
        print(f"FATAL: rank {rank} exceeds competition limit of 32")
        sys.exit(1)
    print("  Rank OK (≤ 32) ✓")


def make_zip():
    out = "submission.zip"
    src = Path(ADAPTER_OUT_DIR)
    files = [f for f in src.iterdir() if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / 1e6

    print(f"\nPackaging {len(files)} files ({total_mb:.0f} MB) into {out}...")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
            print(f"  + {f.name}")

    size_gb = Path(out).stat().st_size / 1e9
    print(f"\n✓ {out} ready ({size_gb:.2f} GB)")
    print("\nUpload at:")
    print("  kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions")


def main():
    print("=== Step 1: Fuse MLX adapter ===")
    fuse_mlx_adapter()

    print("\n=== Step 2: Extract LoRA delta for submission ===")
    extract_lora_delta()

    print("\n=== Step 3: Package submission.zip ===")
    make_zip()


if __name__ == "__main__":
    main()
