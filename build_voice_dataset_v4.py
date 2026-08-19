"""Rebuild the iMessage voice-clone SFT dataset with session-aware windowing.

The v1 dataset (build_voice_dataset.py, now lost) sliced each thread's full
message history into fixed-size chunks of ~6 messages, without regard to
real elapsed time between them. That meant "example 5" and "example 6" in
the same thread could be from conversations months apart, stitched together
as if they were one continuous exchange -- which produced training pairs
where the "reply" wasn't really a reply to the "question" shown, since the
model was learning from context that never actually preceded it in time.

This version:
  1. Only uses 1:1 DM threads (chat.style == 45 / room_name IS NULL) --
     group chats have multi-party turn-taking that doesn't map cleanly to
     a single user/assistant persona pair.
  2. Merges consecutive same-sender messages within MERGE_GAP_SECONDS into
     one "turn", capped at MAX_MERGED_CHARS -- v2 let a burst of 131 rapid
     texts (someone reading out an entire essay draft) merge into one
     4,606-character "turn", which is a merge artifact, not a natural
     single reply, and would have skewed training toward long garbled
     targets. Once a merge would exceed the cap, a new turn starts instead.
  3. Splits each thread's turns into SESSIONS using a real time-gap
     threshold (SESSION_GAP_SECONDS) -- a session boundary is drawn
     wherever there's a long silence, the same way a human would say
     "that was a different conversation."
  4. Builds one training example per your reply, using ONLY the turns
     that actually, chronologically preceded it within the SAME session
     as context -- never content from a different session or from days/
     weeks away. The example is only created if the turn immediately
     before your reply is genuinely from the other person (a real
     question/message you're responding to), not a delayed continuation
     of your own earlier turn.
  5. Filters out examples where a URL, junk placeholder char, or CS
     coursework marker appears ANYWHERE in the window -- your reply OR any
     of the other person's messages used as context. v3 only screened your
     own reply, so a shared link or pasted code in their message still
     slipped into ~20% of examples as noisy context. Caps how many times
     an exact-duplicate filler reply ("ok", "lol", "?") can appear, so
     those don't dominate the training signal.
  6. Caps examples per thread (MAX_PER_THREAD) -- v2's flat dataset was
     65.6% dominated by its top 3 threads (one single thread alone was
     37.9%), meaning the model would mostly learn that one relationship's
     register instead of a general texting voice. High-quality instruction
     datasets (UltraChat, FineWeb, Dolma) all cap per-source contribution
     for exactly this reason -- diversity over raw volume.
  7. Splits train/heldout by SESSION, not by flattened example -- v2
     shuffled all examples together before splitting, so two examples
     drawn from the same real conversation session could land on
     opposite sides of the split (measured: 70 of 211 heldout sessions
     also appeared in train). That's train/eval leakage: the heldout
     "coherence" check wasn't actually testing generalization.
  8. Computes a short, real style profile (punctuation/capitalization
     habits measured from the actual data) and appends it to the system
     prompt, since persona-conditioning measurably improves style
     consistency in fine-tunes without needing more data.
  9. Anonymizes each thread to Contact_N (deterministic via random.seed).
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"
OUT_DIR = Path(__file__).parent / "data"

MERGE_GAP_SECONDS = 120
MAX_MERGED_CHARS = 400           # cap on a single merged "turn" -- longer bursts split instead
MAX_RAW_MSG_CHARS = 500          # drop individual messages longer than this (document pastes,
                                  # not real texting -- e.g. a 4,558-char pasted essay in one bubble)
SESSION_GAP_SECONDS = 3 * 3600   # 3 hours of silence = new session
CONTEXT_TURNS = 8                # how many preceding turns (within the session) to include
DUPLICATE_CAP = 6                # max times an identical target reply may appear
MAX_PER_THREAD = 200             # cap per-thread contribution so one relationship can't dominate
MIN_TEXT_LEN = 1

APPLE_EPOCH_OFFSET = 978307200  # seconds between 1970-01-01 and 2001-01-01

BASE_SYSTEM_PROMPT = (
    "You're texting as a real person. Reply the way they naturally "
    "would in a casual iMessage conversation -- their own tone, length, "
    "and phrasing, not a generic assistant voice."
)

URL_PAT = re.compile(r"https?://")
JUNK_PAT = re.compile(r"^[￼�\s|]*$")  # object-replacement / replacement char only
JUNK_CHAR_PAT = re.compile(r"[￼�]")   # object-replacement / replacement char anywhere
CS_MARKERS = [
    "%%sql", "class Solution", "edstem.org", "zoom.us", "CREATE TRIGGER",
    "CREATE INDEX", "CREATE TABLE", "DROP TABLE", "playerID",
]


def is_junk_or_url(text: str) -> bool:
    if URL_PAT.search(text) or JUNK_PAT.match(text.strip()) or JUNK_CHAR_PAT.search(text):
        return True
    if any(marker in text for marker in CS_MARKERS):
        return True
    return False


def fetch_dm_messages() -> dict[str, list[tuple[float, bool, str]]]:
    """Returns {chat_identifier: [(unix_ts, is_from_me, text), ...]} for 1:1 chats."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.chat_identifier, m.date, m.is_from_me, m.text
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        JOIN chat c ON c.ROWID = cmj.chat_id
        WHERE c.style = 45
          AND m.text IS NOT NULL AND m.text != ''
          AND m.is_system_message = 0
        ORDER BY c.chat_identifier, m.date ASC
    """)
    by_chat: dict[str, list[tuple[float, bool, str]]] = defaultdict(list)
    for chat_id, date_ns, is_from_me, text in cur.fetchall():
        if len(text) > MAX_RAW_MSG_CHARS:
            continue  # document paste, not a real text -- treat as if never sent
        unix_ts = date_ns / 1e9 + APPLE_EPOCH_OFFSET
        by_chat[chat_id].append((unix_ts, bool(is_from_me), text))
    conn.close()
    return by_chat


def merge_into_turns(msgs: list[tuple[float, bool, str]]) -> list[dict]:
    """Merge consecutive same-sender messages within MERGE_GAP_SECONDS,
    capped at MAX_MERGED_CHARS so a long rapid-fire burst doesn't collapse
    into one unnaturally huge "turn" -- it splits into multiple turns
    instead once the cap is hit."""
    turns: list[dict] = []
    for ts, is_from_me, text in msgs:
        if turns and turns[-1]["is_from_me"] == is_from_me and \
                ts - turns[-1]["end_ts"] <= MERGE_GAP_SECONDS and \
                len(turns[-1]["text"]) < MAX_MERGED_CHARS:
            turns[-1]["text"] += "\n" + text
            turns[-1]["end_ts"] = ts
        else:
            turns.append({"start_ts": ts, "end_ts": ts, "is_from_me": is_from_me, "text": text})
    return turns


def split_into_sessions(turns: list[dict]) -> list[list[dict]]:
    sessions: list[list[dict]] = []
    for t in turns:
        if sessions and t["start_ts"] - sessions[-1][-1]["end_ts"] <= SESSION_GAP_SECONDS:
            sessions[-1].append(t)
        else:
            sessions.append([t])
    return sessions


def compute_style_profile(by_chat: dict[str, list[tuple[float, bool, str]]]) -> str:
    """Measure real punctuation/capitalization habits from your own messages
    and phrase them as an explicit style note. Persona-conditioning like this
    measurably improves style consistency in fine-tunes without needing more
    training data -- and some habits (e.g. almost never ending a text with
    terminal punctuation) are easy to state directly but hard for a small
    LoRA to reliably infer from examples alone."""
    my_texts = [text for msgs in by_chat.values() for _, is_from_me, text in msgs if is_from_me]
    n = len(my_texts)
    ends_punct = sum(1 for t in my_texts if t.strip() and t.rstrip()[-1:] in ".!?")
    has_emoji = sum(1 for t in my_texts if re.search(r"[\U0001F300-\U0001FAFF☀-➿]", t))

    notes = []
    punct_pct = 100 * ends_punct / n
    if punct_pct < 15:
        notes.append("You almost never end a text with a period, exclamation point, or question mark.")
    elif punct_pct > 70:
        notes.append("You usually end your texts with proper punctuation.")

    emoji_pct = 100 * has_emoji / n
    if emoji_pct < 10:
        notes.append("You rarely use emoji.")
    elif emoji_pct > 40:
        notes.append("You use emoji fairly often.")

    if not notes:
        return BASE_SYSTEM_PROMPT
    return BASE_SYSTEM_PROMPT + " " + " ".join(notes)


def build_examples_for_chat(turns: list[dict], system_prompt: str) -> list[dict]:
    examples = []
    for session in split_into_sessions(turns):
        for i, turn in enumerate(session):
            if not turn["is_from_me"]:
                continue
            if i == 0:
                continue  # no preceding context in this session
            prev = session[i - 1]
            if prev["is_from_me"]:
                continue  # not a genuine reply to the other person
            if is_junk_or_url(turn["text"]):
                continue

            context = session[max(0, i - CONTEXT_TURNS):i]
            # v3 only screened the target reply for URLs/junk/CS-boilerplate --
            # the other person's messages used as CONTEXT went unchecked, so a
            # shared link or pasted code in their turn still made it into the
            # example even though your reply was clean. Screen the whole
            # window now: skip the example if noise shows up anywhere in it.
            if any(is_junk_or_url(c["text"]) for c in context):
                continue

            messages = [{"role": "system", "content": system_prompt}]
            for c in context:
                messages.append({
                    "role": "assistant" if c["is_from_me"] else "user",
                    "content": c["text"],
                })
            messages.append({"role": "assistant", "content": turn["text"]})
            examples.append({
                "messages": messages,
                "session_start": context[0]["start_ts"],
            })
    return examples


def main() -> None:
    print(f"Reading {DB_PATH} ...")
    by_chat = fetch_dm_messages()
    print(f"Found {len(by_chat)} 1:1 DM threads")

    system_prompt = compute_style_profile(by_chat)
    print(f"Computed system prompt: {system_prompt!r}")

    all_examples: dict[str, list[dict]] = {}
    for chat_id, msgs in by_chat.items():
        turns = merge_into_turns(msgs)
        examples = build_examples_for_chat(turns, system_prompt)
        if examples:
            all_examples[chat_id] = examples

    total_raw = sum(len(v) for v in all_examples.values())
    print(f"{len(all_examples)} threads produced examples, {total_raw} raw examples before filtering")

    # Cap exact-duplicate target replies globally (case/whitespace-insensitive)
    dup_counts: Counter[str] = Counter()
    deduped: dict[str, list[dict]] = defaultdict(list)
    for chat_id, examples in all_examples.items():
        for ex in examples:
            target = ex["messages"][-1]["content"].strip().lower()
            if dup_counts[target] >= DUPLICATE_CAP:
                continue
            dup_counts[target] += 1
            deduped[chat_id].append(ex)

    total_deduped = sum(len(v) for v in deduped.values())
    print(f"{total_deduped} examples after duplicate-reply cap (max {DUPLICATE_CAP}x each)")

    # Cap per-thread contribution so one relationship's register can't
    # dominate the dataset (v2: top 3 threads alone were 65.6% of everything)
    rng_cap = random.Random(7)
    kept: dict[str, list[dict]] = {}
    for chat_id, examples in deduped.items():
        if len(examples) > MAX_PER_THREAD:
            examples = rng_cap.sample(examples, MAX_PER_THREAD)
        kept[chat_id] = examples

    total_kept = sum(len(v) for v in kept.values())
    print(f"{total_kept} examples after per-thread cap (max {MAX_PER_THREAD} each)")

    # Anonymize thread identifiers deterministically
    chat_ids = sorted(kept.keys())
    rng = random.Random(7)
    shuffled = chat_ids[:]
    rng.shuffle(shuffled)
    id_map = {cid: f"Contact_{i+1}" for i, cid in enumerate(shuffled)}

    labeled: list[dict] = []
    for chat_id, examples in kept.items():
        label = id_map[chat_id]
        for ex in examples:
            labeled.append({
                "thread": label,
                "messages": ex["messages"],
                "session_start": ex["session_start"],
                # (thread, session_start) is the atomic unit for the train/heldout
                # split below -- splitting by flattened example (v2's approach)
                # let two examples from the SAME real conversation land on
                # opposite sides, which is train/eval leakage.
                "session_id": (chat_id, ex["session_start"]),
            })

    # Split by SESSION, not by example, so no session straddles both files
    session_ids = sorted(set(ex["session_id"] for ex in labeled))
    rng2 = random.Random(7)
    rng2.shuffle(session_ids)
    n_heldout_sessions = max(20, len(session_ids) // 12)
    heldout_sessions = set(session_ids[:n_heldout_sessions])

    train, heldout = [], []
    for ex in labeled:
        row = {"thread": ex["thread"], "messages": ex["messages"], "session_start": ex["session_start"]}
        (heldout if ex["session_id"] in heldout_sessions else train).append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "voice_clone_train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(OUT_DIR / "voice_clone_heldout.jsonl", "w") as f:
        for ex in heldout:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train)} train / {len(heldout)} heldout examples to {OUT_DIR}")
    print(f"({len(session_ids) - n_heldout_sessions} train sessions / {n_heldout_sessions} heldout sessions, zero overlap)")


if __name__ == "__main__":
    main()
