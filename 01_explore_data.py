"""
Step 1: Download and explore the training data.
Run: python 01_explore_data.py
Requires: KAGGLE_USERNAME and KAGGLE_KEY env vars (from kaggle.com/settings/account)
"""
import os
import json
from collections import Counter
from pathlib import Path


def download_data():
    # Check for CSV (competition uses CSV format)
    if Path("train.csv").exists():
        print("train.csv already present.")
        return
    if list(Path(".").glob("train*.jsonl")) or list(Path(".").glob("train*.json")):
        print("Training data already present.")
        return
    # Try kagglehub first
    try:
        import kagglehub
        print("Downloading via kagglehub...")
        path = kagglehub.competition_download("nvidia-nemotron-model-reasoning-challenge")
        print(f"Downloaded to: {path}")
        # Copy files here
        import shutil
        for f in Path(path).iterdir():
            shutil.copy(f, Path(".") / f.name)
        return
    except Exception as e:
        print(f"kagglehub failed: {e}")
    print("ERROR: Please manually download from:")
    print("  kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data")
    print("  and unzip into:", Path(".").resolve())
    exit(1)


download_data()


def load_data(path):
    """Load CSV or JSONL training data into a list of dicts."""
    p = Path(path)
    if p.suffix == ".csv":
        import csv
        data = []
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    else:
        data = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data


# Find training file — prefer CSV since that's what the competition provides
train_files = (
    list(Path(".").glob("train*.csv")) +
    list(Path(".").glob("train*.jsonl")) +
    list(Path(".").glob("train*.json"))
)
print(f"\nAll files in directory:")
for f in sorted(Path(".").iterdir()):
    print(f"  {f.name}")

if not train_files:
    print("\nERROR: No training file found.")
    exit(1)

train_data = load_data(str(train_files[0]))
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
