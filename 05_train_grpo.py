"""
Step 5: GRPO reinforcement learning on top of the SFT checkpoint.
Directly optimizes the competition metric (correct \\boxed{} answer).

GRPO: for each question, generate N candidates → reward correct ones
relative to the group → gradient update. No separate critic needed.

Run: HF_TOKEN=xxx python 05_train_grpo.py
"""
import os
import re
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from huggingface_hub import login

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID        = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
SFT_ADAPTER     = "./checkpoints/sft/final"
OUTPUT_DIR      = "./checkpoints/grpo"
LORA_RANK       = 32
NUM_GENERATIONS = 6      # Candidate answers per question during GRPO
MAX_NEW_TOKENS  = 4096
# ───────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert puzzle solver. Think step by step, showing your full "
    "reasoning clearly. Always place your final answer inside \\boxed{} at the end."
)

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])


# ─── REWARD FUNCTIONS ──────────────────────────────────────────────────────────

def extract_boxed(text: str):
    """Extract the last \\boxed{} content (handles nested braces)."""
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        nums = re.findall(r'-?\d+(?:\.\d+)?', text)
        return nums[-1] if nums else None
    # Use the LAST \boxed{} (reasoning traces can have intermediate boxes)
    m = matches[-1]
    start = m.end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        i += 1
    return text[start:i-1].strip() if depth == 0 else None


def numerically_close(pred: str, gold: str, tol: float = 1e-2) -> bool:
    try:
        p = float(pred.replace(",", "").replace(" ", ""))
        g = float(gold.replace(",", "").replace(" ", ""))
        return abs(p - g) / (abs(g) + 1e-8) < tol
    except ValueError:
        return False


def correctness_reward(completions: list, answer: list, **_) -> list:
    """
    +1.0 if answer is correct (exact string or within 1% numeric tolerance).
    This mirrors the exact competition metric.
    """
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_boxed(completion)
        gold_s = str(gold).strip()
        if pred is None:
            rewards.append(0.0)
        elif pred.strip() == gold_s or numerically_close(pred, gold_s):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list, **_) -> list:
    """Small bonus for using \\boxed{} and showing reasoning (not just guessing)."""
    rewards = []
    for c in completions:
        score = 0.0
        if r'\boxed{' in c:
            score += 0.1
        if len(c) > 100:
            score += 0.05
        rewards.append(score)
    return rewards


def combined_reward(completions: list, answer: list, **kwargs) -> list:
    c = correctness_reward(completions, answer)
    f = format_reward(completions)
    return [ci + fi for ci, fi in zip(c, f)]


# ─── DATA ──────────────────────────────────────────────────────────────────────

def load_grpo_dataset(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            answer = item.get("answer", "")
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            if user_msgs and answer:
                data.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msgs[0]},
                    ],
                    "answer": str(answer),
                })
    print(f"GRPO dataset: {len(data)} examples")
    return Dataset.from_list(data)


print("Loading GRPO dataset...")
grpo_dataset = load_grpo_dataset("sft_training_data.jsonl")

# ─── MODEL ─────────────────────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load SFT adapter as starting point for GRPO
if os.path.exists(SFT_ADAPTER):
    print(f"Loading SFT adapter from {SFT_ADAPTER}")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
    print("SFT adapter loaded — GRPO continues from this checkpoint ✓")
else:
    print("WARNING: SFT adapter not found — GRPO starting from base model (suboptimal)")

# ─── TRAINING ──────────────────────────────────────────────────────────────────

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,               # RL needs much lower LR than SFT
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=20,
    bf16=True,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_NEW_TOKENS,
    temperature=0.8,
    top_p=0.95,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    run_name="nemotron-grpo",
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[combined_reward],
    args=grpo_config,
    train_dataset=grpo_dataset,
)

print("\n=== Starting GRPO training ===")
trainer.train()

final = OUTPUT_DIR + "/final"
trainer.save_model(final)
print(f"\nGRPO complete! Final adapter → {final}")
