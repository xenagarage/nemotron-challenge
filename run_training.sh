#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_training.sh — Full training pipeline in tmux
#
# Usage:
#   1. Set your HuggingFace token:
#      export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#      (get it at huggingface.co/settings/tokens)
#
#   2. Set your Kaggle credentials:
#      export KAGGLE_USERNAME=your_kaggle_username
#      export KAGGLE_KEY=your_kaggle_api_key
#      (get it at kaggle.com/settings/account → API → Create New Token)
#
#   3. Run:
#      bash run_training.sh
#
# This creates a tmux session called "nemotron".
# To check progress later:   tmux attach -t nemotron
# To detach (keep running):  Ctrl+B, then D
# To kill:                   tmux kill-session -t nemotron
# ─────────────────────────────────────────────────────────────────────────────

SESSION="nemotron"

# Kill any existing session with same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Auto-load .env if present
if [[ -f "$(dirname "$0")/.env" ]]; then
  set -a
  source "$(dirname "$0")/.env"
  set +a
  echo "✓ Loaded credentials from .env"
fi

# Check HF_TOKEN (required — needed to download the model)
if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Run:  export HF_TOKEN=hf_your_token_here"
  echo "Get token at: huggingface.co/settings/tokens"
  exit 1
fi

# Kaggle credentials — only required if train.jsonl not already downloaded
HAS_DATA=$(ls "$(dirname "$0")"/train*.jsonl 2>/dev/null | head -1)
if [[ -z "$HAS_DATA" ]] && [[ -z "$KAGGLE_USERNAME" || -z "$KAGGLE_KEY" ]]; then
  echo "ERROR: No training data found and KAGGLE credentials not set."
  echo "Either:"
  echo "  a) Set KAGGLE_USERNAME + KAGGLE_KEY to auto-download, or"
  echo "  b) Manually download train.jsonl from kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data"
  echo "     and place it in $(dirname "$0")/"
  exit 1
fi

cd "$(dirname "$0")"
WORKDIR="$(pwd)"

echo "╔══════════════════════════════════════════════════╗"
echo "║   Nemotron Challenge — Starting Training         ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  tmux session: $SESSION                          "
echo "║  workdir: $WORKDIR                               "
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t $SESSION"
echo "To detach and leave running:"
echo "  Ctrl+B, then D"
echo ""

# Build the full pipeline command that runs inside tmux
PIPELINE=$(cat <<'SCRIPT'
set -e
set -o pipefail
export PATH="$HOME/Library/Python/3.14/bin:/opt/homebrew/bin:$PATH"

LOG="training_$(date +%Y%m%d_%H%M%S).log"
echo "=== Logging to $LOG ===" | tee -a "$LOG"

step() {
  echo "" | tee -a "$LOG"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
  echo "STEP: $1  [$(date)]" | tee -a "$LOG"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
}

step "1/5 — Download competition data"
python3 01_explore_data.py 2>&1 | tee -a "$LOG"

step "2/5 — Test solvers"
python3 02_build_solvers.py 2>&1 | tee -a "$LOG"

step "3/5 — Generate clean SFT training traces"
python3 03_generate_sft_data.py 2>&1 | tee -a "$LOG"

step "4/5 — SFT training on Mac (MLX)"
python3 04_train_sft_mac.py 2>&1 | tee -a "$LOG"

step "5/5 — Package submission.zip"
python3 05_fuse_and_submit_mac.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "╔══════════════════════════════════════════╗" | tee -a "$LOG"
echo "║   ALL DONE! Upload submission.zip        ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════╝" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/submissions" | tee -a "$LOG"

SCRIPT
)

# Create tmux session and run the pipeline
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "cd '$WORKDIR'" Enter
tmux send-keys -t "$SESSION" "export HF_TOKEN='$HF_TOKEN'" Enter
tmux send-keys -t "$SESSION" "export KAGGLE_USERNAME='$KAGGLE_USERNAME'" Enter
tmux send-keys -t "$SESSION" "export KAGGLE_KEY='$KAGGLE_KEY'" Enter
tmux send-keys -t "$SESSION" "export HUGGING_FACE_HUB_TOKEN='$HF_TOKEN'" Enter

# Write pipeline to a temp script and run it
TMPSCRIPT="/tmp/nemotron_run_$$.sh"
printf '%s\n' "$PIPELINE" > "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"

tmux send-keys -t "$SESSION" "bash '$TMPSCRIPT'; echo 'Session complete. Press any key to exit.'; read" Enter

echo "✓ Pipeline started in tmux session '$SESSION'"
echo ""
echo "  Attach to watch:  tmux attach -t $SESSION"
echo "  Detach to leave:  Ctrl+B → D"
echo "  Check log later:  ls training_*.log"
