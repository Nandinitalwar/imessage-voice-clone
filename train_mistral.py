"""LoRA SFT of Mistral-7B-Instruct-v0.3 on the iMessage voice-clone dataset.

Runs on a real GPU (H100). bf16 throughout -- unlike the CPU run, bf16 is
the FAST path on GPU (the CPU run's slowness came from bf16 autocast on
CPU, which has no fast kernels there; that constraint doesn't apply here).

Mistral's own chat template has no {% generation %} markers (same problem
as Qwen's), so this uses a custom minimal template matching Mistral's real
[INST]/[/INST] format, with generation markers around each assistant turn
so trl's assistant_only_loss can mask the loss correctly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = Path(__file__).parent / "data" / "voice_clone_train.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "voice_clone_mistral"

# Mistral's own template has no {% generation %} markers, and it doesn't
# have a dedicated system-turn token -- system content gets folded into the
# first [INST] block. Alternation after that must be strict user/assistant.
CHAT_TEMPLATE = (
    "{%- set system_message = messages[0].content if messages[0].role == 'system' else '' %}"
    "{%- set loop_messages = messages[1:] if messages[0].role == 'system' else messages %}"
    "{{- bos_token }}"
    "{%- for message in loop_messages %}"
    "{%- if message.role == 'user' %}"
    "{%- if loop.first and system_message %}"
    "{{- '[INST] ' + system_message + '\\n\\n' + message.content + ' [/INST]' }}"
    "{%- else %}"
    "{{- '[INST] ' + message.content + ' [/INST]' }}"
    "{%- endif %}"
    "{%- elif message.role == 'assistant' %}"
    "{% generation %}{{- message.content + eos_token }}{% endgeneration %}"
    "{%- endif %}"
    "{%- endfor %}"
)


def load_data() -> list[dict]:
    rows = []
    with DATA_PATH.open() as f:
        for line in f:
            rows.append({"messages": json.loads(line)["messages"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = load_data()
    print(f"Loaded {len(rows)} training examples")

    ds = Dataset.from_list(rows)
    if args.smoke:
        ds = ds.select(range(min(80, len(ds))))
        print(f"Smoke test: using {len(ds)} examples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(OUT_DIR),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            logging_steps=5,
            save_strategy="no" if args.smoke else "epoch",
            report_to=[],
            max_length=args.max_seq_len,
            assistant_only_loss=True,
            bf16=True,
            gradient_checkpointing=False,  # 80GB VRAM is plenty for a 7B model + LoRA
        ),
        train_dataset=ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    n_steps = trainer.state.global_step
    print(f"\nDone: {n_steps} steps in {elapsed:.1f}s ({elapsed/max(n_steps,1):.2f}s/step)")

    if args.smoke:
        full_steps = (len(load_data()) // args.batch_size) * args.epochs
        est_total = full_steps * (elapsed / max(n_steps, 1))
        print(f"Full run estimate: {full_steps} steps -> ~{est_total/60:.1f} min ({est_total/3600:.2f} hr)")
    else:
        model.save_pretrained(str(OUT_DIR / "lora_adapter"))
        tokenizer.save_pretrained(str(OUT_DIR / "lora_adapter"))
        print(f"Saved LoRA adapter to {OUT_DIR / 'lora_adapter'}")
        print("TRAINING_COMPLETE_MARKER")


if __name__ == "__main__":
    main()
