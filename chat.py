"""Interactive local chat with the iMessage voice-clone model.

Runs entirely on this laptop, CPU-only (fine for inference on a 500M model --
this is not training, no backward pass, much lighter). Type a message as if
you were the OTHER person in a text thread; the model replies as you would.

Usage:
    venv/bin/python chat.py
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
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

SYSTEM_PROMPT = (
    "You're texting as a real person. Reply the way they naturally "
    "would in a casual iMessage conversation -- their own tone, length, "
    "and phrasing, not a generic assistant voice."
)


def main() -> None:
    print("Loading model (first run may take ~10-20s on CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    model.eval()

    print("Ready. Type as the other person in the conversation. Ctrl+C to quit.\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user_input = input("them> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
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
        reply = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        print(f"you>  {reply}\n")
        history.append({"role": "assistant", "content": reply})

        # keep context bounded so the model doesn't drift off tiny-model rails
        if len(history) > 13:
            history = [history[0]] + history[-12:]


if __name__ == "__main__":
    main()
