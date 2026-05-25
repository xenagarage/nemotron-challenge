"""
Step 4: QLoRA SFT training on clean reasoning traces.

Model: nvidia/Nemotron-3-Nano-30B-Instruct  (verify exact ID on HuggingFace)
LoRA rank ≤ 32 (competition limit).

Run: HF_TOKEN=xxx python 04_train_sft.py
"""
import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Verify the exact model ID at:
# huggingface.co/collections/nvidia/nvidia-nemotron-v3
MODEL_ID        = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
SFT_DATA_PATH   = "sft_training_data.jsonl"
OUTPUT_DIR      = "./checkpoints/sft"
MAX_SEQ_LENGTH  = 4096    # Training length (eval uses 8192)
LORA_RANK       = 32      # Competition max
LORA_ALPHA      = 64      # 2× rank is standard
BATCH_SIZE      = 1
GRAD_ACCUM      = 8       # Effective batch size = 8
LR              = 2e-4
EPOCHS          = 2
# ───────────────────────────────────────────────────────────────────────────────

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
    print("Logged in to HuggingFace ✓")


def load_sft_dataset(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {path}")
    return Dataset.from_list(data)


print(f"\nLoading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_sft_dataset(SFT_DATA_PATH)


def format_example(ex):
    text = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


dataset = dataset.map(format_example, remove_columns=dataset.column_names)
# Filter out very long examples to avoid OOM
before = len(dataset)
dataset = dataset.filter(lambda x: len(x["text"]) < MAX_SEQ_LENGTH * 4)
print(f"After length filter: {len(dataset)}/{before} examples")

print(f"\nLoading model: {MODEL_ID} [4-bit QLoRA]")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # remove if flash-attn not installed
)
model = prepare_model_for_kbit_training(model)
print(f"Model loaded — VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    # Target all attention + FFN projection layers
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,
    dataloader_num_workers=4,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    run_name="nemotron-sft-clean",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

print("\n=== Starting SFT training ===")
trainer.train()

final = OUTPUT_DIR + "/final"
trainer.save_model(final)
tokenizer.save_pretrained(final)
print(f"\nSFT complete! Adapter → {final}")
