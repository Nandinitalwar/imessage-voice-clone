# imessage-voice-clone

A LoRA fine-tune of Mistral-7B-Instruct on your own iMessage history, so it
replies the way you actually text. Standalone experiment, not part of any
other project.

## Pipeline

1. `build_voice_dataset_v2.py` -- reads `~/Library/Messages/chat.db`
   directly, groups messages into real conversation sessions (time-gap
   based), and produces SFT training pairs where each example's context is
   the actual chronological history that preceded your reply -- not an
   arbitrary slice of the full thread. Anonymizes the other party in each
   thread to `Contact_N`. Filters out URL/junk/coursework-boilerplate
   replies and caps duplicate filler replies (e.g. "ok", "lol") so they
   don't dominate training.
2. `train_mistral.py` -- LoRA SFT of Mistral-7B-Instruct-v0.3 on the
   resulting dataset (`assistant_only_loss`, custom chat template with
   `{% generation %}` markers so loss masks correctly). Meant to run on a
   real GPU.
3. `server_mistral.py` -- Flask chat UI serving the fine-tuned model.
4. `viewer/index.html` -- local dataset browser (thread selector, chat
   bubbles, real session timestamps) for eyeballing example quality before
   training. Point it at a `data.json` built from your own
   `voice_clone_train.jsonl` / `voice_clone_heldout.jsonl` (not included in
   this repo -- see below).

`train_local.py`, `test_local.py`, `chat.py`, `server_qwen_local.py` are
an earlier CPU-only pass using Qwen2.5-0.5B, kept for reference.

## Not included

`data/`, `venv/`, `runs/` (trained adapter weights), and any `.jsonl` files
are gitignored. The dataset contains real (if anonymized) personal text
message content and shouldn't live in version control; the venv and model
weights are just large and reproducible from the scripts above.

## Known issues fixed along the way

Two rounds of the model regurgitating verbatim memorized strings (a CS
coursework snippet, then a recurring Zoom invite link) on vague prompts
like "hi" turned out to be a training data problem, not a model or
sampling problem -- specific boilerplate strings were overrepresented in
the training set. Fixed by filtering URL/junk-boilerplate replies and
capping exact-duplicate replies. A separate, structural issue (unrelated
message chunks presented as if they were one continuous conversation) was
fixed by `build_voice_dataset_v2.py`'s session-aware windowing.
