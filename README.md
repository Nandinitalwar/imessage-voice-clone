# imessage-voice-clone

A LoRA fine-tune of Mistral-7B-Instruct on your own iMessage history, so it
replies the way you actually text. Standalone experiment, not part of any
other project.

## Pipeline

1. `build_voice_dataset_v3.py` -- reads `~/Library/Messages/chat.db`
   directly, groups messages into real conversation sessions (time-gap
   based), and produces SFT training pairs where each example's context is
   the actual chronological history that preceded your reply -- not an
   arbitrary slice of the full thread. Anonymizes the other party in each
   thread to `Contact_N`. Filters out URL/junk/coursework-boilerplate
   replies and document-length pastes, caps duplicate filler replies (e.g.
   "ok", "lol") and per-thread contribution so one relationship can't
   dominate, splits train/heldout by session (no leakage), and appends a
   short measured style profile (punctuation/emoji habits) to the system
   prompt.
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

- **Verbatim memorization** (two rounds: a CS coursework snippet, then a
  recurring Zoom invite link, regurgitated on vague prompts like "hi")
  turned out to be a training data problem, not a model or sampling
  problem -- specific boilerplate strings were overrepresented. Fixed by
  filtering URL/junk-boilerplate replies and capping exact-duplicate
  replies.
- **Unrelated message chunks presented as one continuous conversation** --
  the original naive slicer chopped each thread into fixed-size windows
  regardless of real elapsed time, so two adjacent "examples" could be
  months apart. Fixed by `build_voice_dataset_v3.py`'s session-aware
  windowing (time-gap-based session boundaries, real chronological context
  per example).
- **One relationship dominating the dataset** -- the top 3 threads made up
  65.6% of all examples in the v2 rebuild (one thread alone was 37.9%),
  which would teach the model that one specific relationship's register is
  "your voice" generally. Fixed with a per-thread cap.
- **Train/heldout leakage** -- splitting by flattened example (rather than
  by conversation session) let 70 of 211 heldout sessions also appear in
  train, so the "held-out" coherence check wasn't actually testing
  generalization. Fixed by splitting on `(thread, session)` as the atomic
  unit.
- **Document-paste outliers** -- a few individual messages were entire
  pasted documents (a college essay, source code, a work task) reaching
  6,000+ characters in one bubble, which isn't representative texting
  voice. Filtered out at the source (messages over `MAX_RAW_MSG_CHARS`).
