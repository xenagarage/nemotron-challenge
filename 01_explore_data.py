"""
Step 1: Download and explore the training data.
Run: python 01_explore_data.py
Requires: KAGGLE_USERNAME and KAGGLE_KEY env vars (from kaggle.com/settings/account)
"""
import os
import json
import subprocess
from collections import Counter
from pathlib import Path


def download_data():
    if Path("train.jsonl").exists() or list(Path(".").glob("train*.jsonl")):
        print("Data already downloaded.")
        return
    print("Downloading competition data...")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "nvidia-nemotron-model-reasoning-challenge",
        "--path", "."
    ], check=True)
    for zf in Path(".").glob("*.zip"):
        subprocess.run(["unzip", "-o", str(zf)], check=True)
    print("Data downloaded!")


download_data()


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# Find training file
train_files = list(Path(".").glob("train*.jsonl")) + list(Path(".").glob("train*.json"))
print(f"\nAll files in directory:")
for f in sorted(Path(".").iterdir()):
    print(f"  {f.name}")

if not train_files:
    print("\nERROR: No training file found. Check the filenames above.")
    exit(1)

train_data = load_jsonl(str(train_files[0]))
print(f"\nLoaded {len(train_data)} training examples from {train_files[0]}")
print(f"Fields: {list(train_data[0].keys())}")

# Count by category
categories = Counter(
    d.get("category") or d.get("type") or "unknown"
    for d in train_data
)
print("\n=== Category distribution ===")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat:35s} {count:5d}")

# Show 1 example per category
print("\n=== One example per category ===")
seen = set()
for d in train_data:
    cat = d.get("category") or d.get("type") or "unknown"
    if cat not in seen:
        seen.add(cat)
        print(f"\n{'─'*60}")
        print(f"CATEGORY: {cat}")
        print(json.dumps(d, indent=2)[:800])

with open("category_summary.json", "w") as f:
    json.dump(dict(categories), f, indent=2)
print("\nSaved category_summary.json")
