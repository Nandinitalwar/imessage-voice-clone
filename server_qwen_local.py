"""Local web chat UI for the iMessage voice-clone model.

Runs entirely on this laptop, CPU-only inference. Open http://localhost:5057
after starting this.

Usage:
    venv/bin/python server.py
"""
from __future__ import annotations

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "runs" / "voice_clone_qwen_local" / "lora_adapter"
PORT = 5057

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

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.chat_template = CHAT_TEMPLATE
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model = PeftModel.from_pretrained(_base, str(ADAPTER_PATH))
model.eval()
print("Model ready.")

app = Flask(__name__)

# Single-user local server: conversation state lives in memory, per session id.
sessions: dict[str, list[dict]] = {}


def generate(history: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.55,
            top_p=0.85,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>voice clone chat</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 560px;
         margin: 40px auto; padding: 0 16px; background: Canvas; color: CanvasText; }
  h3 { font-weight: 600; opacity: 0.7; font-size: 14px; text-transform: uppercase;
       letter-spacing: 0.04em; margin-bottom: 16px; }
  #log { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
  .msg { padding: 8px 12px; border-radius: 14px; max-width: 75%; line-height: 1.35;
         font-size: 15px; white-space: pre-wrap; }
  .them { align-self: flex-start; background: #e5e5ea; color: #000; }
  .you { align-self: flex-end; background: #007aff; color: #fff; }
  form { display: flex; gap: 8px; }
  input { flex: 1; padding: 10px 12px; border-radius: 10px; border: 1px solid #ccc;
          font-size: 15px; background: Field; color: FieldText; }
  button { padding: 10px 16px; border-radius: 10px; border: none; background: #007aff;
           color: #fff; font-size: 15px; cursor: pointer; }
  button:disabled { opacity: 0.5; }
  #reset { background: none; color: #888; font-size: 12px; text-decoration: underline;
           border: none; cursor: pointer; padding: 0; margin-top: 12px; }
</style>
</head>
<body>
<h3>voice clone chat &mdash; type as the other person</h3>
<div id="log"></div>
<form id="f">
  <input id="i" autocomplete="off" placeholder="type a message..." autofocus>
  <button id="send">send</button>
</form>
<button id="reset">reset conversation</button>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('i');
const sendBtn = document.getElementById('send');

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  window.scrollTo(0, document.body.scrollHeight);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'them');
  input.value = '';
  sendBtn.disabled = true;
  const resp = await fetch('/api/message', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  const data = await resp.json();
  addMsg(data.reply, 'you');
  sendBtn.disabled = false;
  input.focus();
});

document.getElementById('reset').addEventListener('click', async () => {
  await fetch('/api/reset', {method: 'POST'});
  log.innerHTML = '';
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/message", methods=["POST"])
def message():
    text = request.json.get("text", "").strip()
    history = sessions.setdefault("default", [{"role": "system", "content": SYSTEM_PROMPT}])
    history.append({"role": "user", "content": text})
    reply = generate(history)
    history.append({"role": "assistant", "content": reply})
    if len(history) > 13:
        sessions["default"] = [history[0]] + history[-12:]
    return jsonify({"reply": reply})


@app.route("/api/reset", methods=["POST"])
def reset():
    sessions["default"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"\nOpen http://localhost:{PORT} in your browser.\n")
    app.run(host="127.0.0.1", port=PORT, debug=False)
