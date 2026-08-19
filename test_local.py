"""Compare base Qwen2.5-0.5B-Instruct vs the LoRA-tuned voice-clone model on
held-out contexts (never seen during training), against what Nandini
actually replied.

This is the real test: not a metric, a side-by-side read. Voice-cloning
doesn't have an objective linter the way Pinch's astro-voice does -- "does
this sound like me" is inherently a human judgment call.

Usage:
    python3 test_local.py --n 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
HELDOUT_PATH = Path(__file__).parent / "data" / "voice_clone_heldout.jsonl"
ADAPTER_PATH = Path(__file__).parent / "runs" / "voice_clone_qwen_local" / "lora_adapter"

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


def generate(model, tokenizer, messages: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base_model.eval()

    print("Loading fine-tuned model (base + LoRA adapter)...")
    tuned_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    tuned_model = PeftModel.from_pretrained(tuned_base, str(ADAPTER_PATH))
    tuned_model.eval()

    rows = [json.loads(l) for l in HELDOUT_PATH.open()][: args.n]

    for i, row in enumerate(rows):
        messages = row["messages"]
        context = messages[:-1]  # everything up to the real reply
        real_reply = messages[-1]["content"]

        base_reply = generate(base_model, tokenizer, context)
        tuned_reply = generate(tuned_model, tokenizer, context)

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(rows)}] thread: {row.get('thread', '?')}")
        last_incoming = next(
            (m["content"] for m in reversed(context) if m["role"] == "user"), ""
        )
        print(f"Last incoming: {last_incoming[:150]!r}")
        print(f"REAL reply:   {real_reply!r}")
        print(f"BASE model:   {base_reply!r}")
        print(f"TUNED model:  {tuned_reply!r}")


if __name__ == "__main__":
    main()
