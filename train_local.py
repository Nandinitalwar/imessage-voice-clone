"""Local CPU LoRA SFT for the iMessage voice-clone dataset.

Plain transformers + peft + trl -- no Unsloth, since Unsloth's kernels are
CUDA/Triton-only and this machine has no usable GPU backend (x86_64 under
Rosetta, no MPS). Runs fully on CPU. Slower per-step than a GPU, but $0 and
correct for a 500M-param model.

Usage:
    python3 train_local.py --smoke        # ~20 steps, to time throughput
    python3 train_local.py                # full run
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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "voice_clone_train.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "voice_clone_qwen_local"

# Qwen's stock chat template has no {% generation %} markers, which trl's
# assistant_only_loss needs to know which tokens to train on. Our data only
# ever uses system/user/assistant roles (no tool calls), so a minimal
# template covering just those -- with generation markers around the
# assistant's content + its closing <|im_end|> -- is safer than patching
# Qwen's full tool-call template.
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'system' %}"
    "{{- '<|im_start|>system\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}{{- message.content + '<|im_end|>\\n' }}{% endgeneration %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def load_data() -> list[dict]:
    rows = []
    with DATA_PATH.open() as f:
        for line in f:
            rows.append({"messages": json.loads(line)["messages"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run ~20 steps to time throughput, then exit")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE
    # fp32, not "auto" (which resolves to the model's native bf16): PyTorch's
    # CPU backward kernels for bf16 are dramatically slower than fp32 (~40x,
    # measured directly on this node) -- forward is fine either way, but
    # autograd on CPU falls back to a much slower path for bf16.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

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

    ds = Dataset.from_list(rows)  # keeps the raw "messages" column -- SFTTrainer
    # applies the chat template itself when assistant_only_loss is set, since
    # it needs the template's own turn boundaries to mask non-assistant loss.

    if args.smoke:
        ds = ds.select(range(min(80, len(ds))))  # ~20 optimizer steps at batch 4
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
            use_cpu=True,
            # SFTConfig defaults to gradient_checkpointing=True and bf16=True.
            # Both are expensive-to-pointless on this CPU node: checkpointing
            # recomputes forward activations during backward (trading compute
            # for memory we don't need on a 500M model with 32GB budget), and
            # bf16=True wraps the step in a bf16 autocast context regardless
            # of the model's own dtype -- which is why loading the model in
            # fp32 alone didn't fix the ~390s/step rate measured earlier.
            gradient_checkpointing=False,
            bf16=False,
            fp16=False,
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


if __name__ == "__main__":
    main()
