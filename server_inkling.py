"""Local chat UI backed by Inkling through OpenRouter.

This is an inference-only alternative to ``server_mistral.py``. It does not
load the Mistral LoRA adapter and it never reads the local iMessage dataset.
Only messages typed into this UI are sent to OpenRouter.
"""
from __future__ import annotations

import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


MODEL = os.getenv("OPENROUTER_MODEL", "thinkingmachines/inkling")
PORT = int(os.getenv("PORT", "5058"))
TIMEOUT_MS = int(os.getenv("OPENROUTER_TIMEOUT_MS", "30000"))
MAX_BODY_BYTES = 64 * 1024
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = os.getenv(
    "VOICE_SYSTEM_PROMPT",
    (
        "You are Nandini texting a friend. Reply like a real person in a casual "
        "iMessage conversation: warm, expressive, concise, and natural. Do not "
        "sound like a generic assistant. Do not invent autobiographical facts."
    ),
)

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>voice clone chat (Inkling)</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 560px;
         margin: 40px auto; padding: 0 16px; background: Canvas; color: CanvasText; }
  h3 { font-weight: 600; opacity: 0.75; font-size: 14px; text-transform: uppercase;
       letter-spacing: 0.04em; margin-bottom: 6px; }
  .note { color: #888; font-size: 12px; margin: 0 0 16px; }
  #log { display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px; }
  .msg { padding: 8px 12px; border-radius: 14px; max-width: 75%; line-height: 1.35;
         font-size: 15px; white-space: pre-wrap; }
  .them { align-self: flex-start; background: #fff; color: #000;
          border: 1px solid #e0e0e0; }
  .you { align-self: flex-end; background: #007aff; color: #fff; }
  .error { align-self: flex-start; background: #ff3b30; color: #fff; }
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
<h3>voice clone chat &mdash; Inkling via OpenRouter</h3>
<p class="note">Type as the other person. Only this live conversation is sent to OpenRouter.</p>
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

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'you');
  input.value = '';
  sendBtn.disabled = true;
  try {
    const response = await fetch('/api/message', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'request failed');
    addMsg(data.reply, 'them');
  } catch (error) {
    addMsg(error.message, 'error');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

document.getElementById('reset').addEventListener('click', async () => {
  await fetch('/api/reset', {method: 'POST'});
  log.innerHTML = '';
});
</script>
</body>
</html>
"""

_client: Any = None
_client_lock = threading.Lock()
_history_lock = threading.RLock()
_history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]


def _enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    if MODEL.endswith(":free"):
        raise RuntimeError(
            "The Inkling :free endpoint is not suitable for private message content; "
            "use thinkingmachines/inkling"
        )

    try:
        from openrouter import OpenRouter
    except ImportError as exc:
        raise RuntimeError(
            "OpenRouter SDK is not installed; run: "
            "uv pip install --python .venv/bin/python -r requirements-openrouter.txt"
        ) from exc

    with _client_lock:
        if _client is None:
            _client = OpenRouter(api_key=api_key)
    return _client


def _response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif getattr(item, "type", None) == "text":
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts).strip()
    return str(content).strip()


def generate(history: list[dict[str, str]]) -> str:
    provider: dict[str, Any] = {}
    if _enabled("OPENROUTER_ZDR"):
        provider["zdr"] = True
    if _enabled("OPENROUTER_DENY_DATA_COLLECTION"):
        provider["data_collection"] = "deny"

    response = _get_client().chat.send(
        model=MODEL,
        messages=history,
        provider=provider or None,
        max_tokens=80,
        temperature=0.7,
        timeout_ms=TIMEOUT_MS,
    )
    text = _response_text(response)
    if not text:
        raise RuntimeError("Inkling returned an empty response")
    return text


class Handler(BaseHTTPRequestHandler):
    server_version = "InklingVoiceClone/1.0"

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/api/reset":
            with _history_lock:
                _history[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
            self._json(HTTPStatus.OK, {"ok": True})
            return
        if self.path != "/api/message":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("message is empty")

            with _history_lock:
                _history.append({"role": "user", "content": text})
                try:
                    reply = generate(list(_history))
                except Exception:
                    _history.pop()
                    raise
                _history.append({"role": "assistant", "content": reply})
                if len(_history) > MAX_HISTORY_MESSAGES + 1:
                    _history[:] = [_history[0]] + _history[-MAX_HISTORY_MESSAGES:]
            self._json(HTTPStatus.OK, {"reply": reply, "model": MODEL})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self.log_error("OpenRouter request failed: %s", exc)
            self._json(
                HTTPStatus.BAD_GATEWAY,
                {"error": f"OpenRouter request failed: {exc}"},
            )


def main() -> None:
    _get_client()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Inkling chat ready at http://127.0.0.1:{PORT}")
    print(f"Model: {MODEL}; ZDR: {_enabled('OPENROUTER_ZDR')}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
