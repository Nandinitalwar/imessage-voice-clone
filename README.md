# iMessage Voice Clone

LoRA fine-tuning pipeline for learning a person's texting style from their local
iMessage history. It converts `chat.db` into session-aware supervised fine-tuning
(SFT) examples, trains a Mistral adapter, and serves the result through a local
chat UI.

## Why this is an evaluation problem

Texting style is relational and context-dependent. A random message split can
reward memorization while overstating generalization. This pipeline therefore:

- reads the Messages database in read-only mode and keeps only one-to-one chats;
- merges short message bursts into turns and separates sessions after long gaps;
- builds each target from only the conversation that chronologically preceded it;
- caps duplicate replies and per-thread contribution to reduce register collapse;
- filters links, pasted documents, coursework, and malformed attachment text;
- anonymizes contacts and splits train/heldout data by session to prevent leakage;
- derives a small style profile from measured punctuation and emoji behavior.

The training objective masks non-assistant tokens, so loss is computed only on
the reply being learned.

## Pipeline

```text
~/Library/Messages/chat.db (read-only)
  -> session reconstruction + filtering
  -> train/heldout JSONL (local only)
  -> Mistral-7B-Instruct + LoRA SFT
  -> local Flask inference UI
```

| File | Role |
| --- | --- |
| `build_voice_dataset_v4.py` | Builds leakage-resistant SFT examples from iMessage. |
| `train_mistral.py` | Trains a bf16 LoRA adapter on Mistral-7B-Instruct-v0.3. |
| `server_mistral.py` | Serves the adapter in a local chat interface. |
| `viewer/index.html` | Reviews reconstructed sessions before training. |
| `server_inkling.py` | Optional hosted-model baseline through OpenRouter; not a fine-tune. |

## Run

Grant your terminal Full Disk Access so the dataset builder can read Messages,
then create the private dataset:

```bash
python build_voice_dataset_v4.py
```

Train on a CUDA GPU:

```bash
python train_mistral.py --smoke
python train_mistral.py --epochs 3 --batch-size 4
python server_mistral.py
```

For the OpenRouter inference baseline:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-openrouter.txt
export OPENROUTER_API_KEY="..."
.venv/bin/python server_inkling.py
```

Open `http://127.0.0.1:5058`. The baseline enables Zero Data Retention and
disables provider data collection. It sends only text entered in the live UI,
never the local training dataset.

## Privacy

Message data, generated JSONL, virtual environments, and model weights are
gitignored. Contact anonymization reduces direct identifiers but does not make a
personal corpus safe to publish; generated datasets should remain local.

## Failure analysis

Early versions exposed four useful failure modes:

- fixed-size slicing joined conversations separated by weeks;
- example-level splitting leaked 70 of 211 heldout sessions into training;
- the top three threads contributed 65.6% of examples;
- repeated coursework and meeting links caused verbatim memorization.

Session-level splitting, full-window filtering, duplicate caps, and per-thread
sampling address those problems at dataset construction time rather than trying
to compensate through decoding parameters.
