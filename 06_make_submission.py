"""
Step 6: Package the final LoRA adapter into submission.zip for Kaggle.
Prefers GRPO checkpoint over SFT if both exist.
Validates rank ≤ 32 before packaging.

Run: python 06_make_submission.py
"""
import json
import zipfile
from pathlib import Path

ADAPTER_CANDIDATES = [
    "./checkpoints/grpo/final",
    "./checkpoints/grpo",
    "./checkpoints/sft/final",
    "./checkpoints/sft",
]

# Find adapter
adapter_dir = None
for path in ADAPTER_CANDIDATES:
    p = Path(path)
    if p.exists() and (p / "adapter_config.json").exists():
        adapter_dir = p
        break

if adapter_dir is None:
    print("ERROR: No adapter_config.json found. Run 04_train_sft.py first.")
    exit(1)

print(f"Using adapter: {adapter_dir}")

# Validate
with open(adapter_dir / "adapter_config.json") as f:
    cfg = json.load(f)

rank = cfg.get("r", cfg.get("rank", "?"))
base  = cfg.get("base_model_name_or_path", "?")
print(f"  base_model : {base}")
print(f"  lora_rank  : {rank}")

if isinstance(rank, int) and rank > 32:
    print(f"\nFATAL: rank={rank} exceeds competition limit of 32. Aborting.")
    exit(1)

print("  rank check : OK (≤ 32) ✓")

# List files
files = [f for f in adapter_dir.iterdir() if f.is_file()]
total_mb = sum(f.stat().st_size for f in files) / 1e6
print(f"\nFiles to zip ({total_mb:.0f} MB total):")
for f in files:
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

# Package
out = "submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        zf.write(f, f.name)

size_gb = Path(out).stat().st_size / 1e9
print(f"\n✓ {out} created ({size_gb:.2f} GB)")
print("\nNext: upload submission.zip at")
print("  kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions")
