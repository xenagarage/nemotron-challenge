# NVIDIA Nemotron Model Reasoning Challenge

**Target: 0.95+ accuracy** | Current leaderboard ceiling: ~0.87

## Strategy Overview

The benchmark has 9 deterministic puzzle categories. Most competitors are stuck at 0.86–0.87 because:
1. They train on the **official training traces which are ~50% wrong** (especially `bit_manipulation`)
2. They treat it as a general reasoning task instead of writing **category-specific solvers**

Our edge:
- Generate **100% verified clean reasoning traces** using programmatic solvers
- **SFT** on correct data → **GRPO** RL to directly optimize the competition metric
- Hybrid: deterministic solver for easy categories, neural model for hard ones

## Category Breakdown

| Category | Examples | Deterministic Solvable | Our Target |
|---|---|---|---|
| `cipher` | 1576 | 100% | 99%+ |
| `gravity` | 1597 | 100% | 99%+ |
| `numeral` | 1576 | 100% | 99%+ |
| `unit_conversion` | 1594 | 100% | 99%+ |
| `bit_manipulation` | 1602 | 99.4% | 98%+ |
| `equation_numeric_deduce` | 596 | 90.6% | 92%+ |
| `cryptarithm_deduce` | 659 | 43% | 70%+ via GRPO |
| `cryptarithm_guess` | 164 | 16% | 60%+ via GRPO |
| `equation_numeric_guess` | 136 | 16% | 60%+ via GRPO |

## Setup

### 1. Local Mac (exploration only)
```bash
pip install kaggle datasets transformers
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
python 01_explore_data.py
```

### 2. Google Cloud VM (actual training)
```bash
# Create a G4 VM (RTX PRO 6000 Blackwell, 96GB VRAM) on GCP console
# Use Deep Learning VM image (Ubuntu 22.04 + CUDA 12 pre-installed)
# Then SSH in and:
bash setup.sh
export HF_TOKEN=your_huggingface_token
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Run pipeline
python 01_explore_data.py       # ~5 min
python 02_build_solvers.py      # instant (unit tests)
python 03_generate_sft_data.py  # ~15 min
python 04_train_sft.py          # ~4–6 hours
python 05_train_grpo.py         # ~2–3 hours
python 06_make_submission.py    # instant
# Upload submission.zip on Kaggle
```

### VS Code → GCP Connection
1. Install **Remote - SSH** extension in VS Code
2. Add GCP VM IP to `~/.ssh/config`:
   ```
   Host nemotron-gcp
       HostName YOUR_GCP_EXTERNAL_IP
       User your_gcp_username
       IdentityFile ~/.ssh/id_ed25519
   ```
3. `Cmd+Shift+P` → `Remote-SSH: Connect to Host` → `nemotron-gcp`

## Training Time Estimates (RTX PRO 6000, 96GB)

| Step | Time |
|---|---|
| Data generation | ~15 min |
| SFT (2 epochs, 9500 examples) | ~4–6 hours |
| GRPO (1 epoch, RL) | ~2–3 hours |
| **Total** | **~7–10 hours** |

## Notes on 0.86 Ceiling (what others are hitting)

Based on public discussion:
- Everyone using `huikang`'s CoT notebook hits ~0.86 because they share the same flawed data
- More synthetic data without fixing the **wrong bit_manipulation traces** doesn't help
- GRPO on top of bad SFT doesn't help either
- **Our fix**: programmatic verification of every trace before training

## File Structure

```
nemotron-challenge/
├── setup.sh                    # One-time GCP VM setup
├── 01_explore_data.py          # Download & analyze competition data
├── 02_build_solvers.py         # Deterministic solvers per category
├── 03_generate_sft_data.py     # Generate clean verified training traces
├── 04_train_sft.py             # QLoRA SFT training
├── 05_train_grpo.py            # GRPO reinforcement learning
├── 06_make_submission.py       # Package adapter → submission.zip
└── README.md
```
