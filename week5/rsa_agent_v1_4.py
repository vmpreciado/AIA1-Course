"""RSA v1.4 — the memory-equipped agent.

This is RSA-v1.3 (the guarded, traced loop from Module 4) with the three memory
additions from Module 5. The model and the tools' PURPOSE are unchanged; what is
new is how the loop manages the context window:

  1. COMPACTION ON A THRESHOLD — a check at the top of each step measures the
     context, and when it crosses a token budget it summarizes the old trajectory
     against the goal, so the context can never grow without bound.
  2. A NOTES TOOL BACKED BY SQLITE — save_note / read_note let the agent write and
     read durable facts as a deliberate action, stored in a local SQLite file.
  3. PERSISTENCE ACROSS SESSIONS — the durable state (the goal and the notes) lives
     in the same SQLite file, so a run can be killed mid-task and resumed later.

Nothing here makes the model smarter. The loop around it simply manages memory.

Prerequisites
  1. Install Python 3.
  2. Install Ollama, then run:  ollama pull qwen2.5:7b
  3. Install the Python packages:  pip install ollama pydantic
     (sqlite3 ships with Python; no install needed.)

Run it
  python rsa_agent_v1_4.py             # asks for your goal, then the trajectory
  python rsa_agent_v1_4.py --verbose   # also watch each pass as it happens
  python rsa_agent_v1_4.py --resume    # reload the saved goal + notes from agent.db
"""
import argparse
import functools
import json
import sqlite3
import sys

import ollama                                    # talks to the local model
from pydantic import BaseModel, ConfigDict, ValidationError  # tool schemas + validation

MODEL             = "qwen2.5:7b"   # the local model that does the thinking
MAX_STEPS         = 20             # THE STEP BUDGET: hard cap on passes (from v1.3)
COMPACT_THRESHOLD = 30000          # compact when the context crosses this many tokens
KEEP_RECENT       = 6              # messages kept verbatim (working memory) on compaction
DB_PATH           = "agent.db"     # the single SQLite file that holds durable state


# ===========================================================================
# MAP OF THIS FILE  —  find each block; you do not need to read every line.
#   1) THE CONSTITUTION      — the system prompt: who the agent is and its rules.
#   2) THE SQLITE STORE      — the one file that survives the process (notes + state).
#   3) THE TOOLBOX           — save_note / read_note, backed by SQLite, and schemas.
#   4) THE CONFIRMATION GATE — asks a human for yes / no before any write.
#   5) RUN ONE TOOL CALL     — checks arguments, runs the tool, errors-as-results.
#   6) TOKEN COUNT + COMPACTION — measure the context, summarize the old part.
#   7) THE TRACER            — records the run as a trajectory (stand-in for Langfuse).
#   8) PERSISTENCE           — save / load the durable state across sessions.
#   9) THE MEMORY-MANAGED LOOP — compaction check on top, guarded loop below.
# ===========================================================================


# ---------------------------------------------------------------------------
# 1) THE CONSTITUTION  (system prompt — role, goal, rules; no scripted plan)
# ---------------------------------------------------------------------------
SYSTEM = """\
# ROLE
You are RSA, a careful assistant with a durable memory.

# GOAL
Help the user by working step by step, and by saving facts you will need later.

# CONSTRAINTS
- Take one action at a time.
- When you discover a durable fact you will need again, use save_note to store it
  under a short, descriptive key. Do not rely on memory for saved facts.
- When you need a fact you saved earlier, use read_note with its key.
- Never invent a saved note's contents; read it to confirm.
- If a tool returns an error, read it and try a corrected call.
- If a message tells you a call was already tried, do not repeat it; try something else.

# TOOLS
- save_note(key, text): store a durable fact under a key. This is a write action.
- read_note(key): return the text stored under a key, or a not-found message.

# HOW TO FINISH
Call a tool whenever you need one. When the request is complete, reply in plain
text with a short confirmation for the user and do NOT call a tool.
"""


# ---------------------------------------------------------------------------
# 2) THE SQLITE STORE  (one ordinary file, no server; survives the process)
# ---------------------------------------------------------------------------
def db_connect(path):
    """Open (or create) the SQLite file and make sure its two tables exist."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS notes (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS state "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), goal TEXT, summary TEXT)")
    con.commit()
    return con


CON = None   # the open connection; set in __main__ so the tools can reach it


# ---------------------------------------------------------------------------
# 3) THE TOOLBOX  (the notes tool: schema class + worker function per tool)
# ---------------------------------------------------------------------------
class SaveNote(BaseModel):                      # the INPUTS for the write tool
    """Store a durable fact you will need later, under a short descriptive key."""
    model_config = ConfigDict(extra="forbid")
    key: str
    text: str


class ReadNote(BaseModel):                      # the INPUTS for the read tool
    """Read back a durable fact you saved earlier, by its key."""
    model_config = ConfigDict(extra="forbid")
    key: str


def save_note(key, text):                       # the WORK of the write tool
    """Write text under a key in the SQLite notes table (creating or replacing it)."""
    CON.execute("INSERT OR REPLACE INTO notes(key, value) VALUES (?, ?)", (key, text))
    CON.commit()
    return f"Saved note '{key}' ({len(text)} characters) to {DB_PATH}."


def read_note(key):                             # the WORK of the read tool
    """Return the text stored under a key, or a not-found message."""
    row = CON.execute("SELECT value FROM notes WHERE key = ?", (key,)).fetchone()
    return row[0] if row else f"No note found for key '{key}'."


TOOLBOX = {
    "save_note": (SaveNote, save_note),
    "read_note": (ReadNote, read_note),
}
DANGEROUS = {"save_note"}   # write actions that must pass the confirmation gate


def tool_specs():
    """Build the tool schemas we hand the model, straight from Pydantic."""
    specs = []
    for name, (schema, _) in TOOLBOX.items():
        specs.append({"type": "function", "function": {
            "name": name,
            "description": (schema.__doc__ or "").strip(),
            "parameters": schema.model_json_schema(),
        }})
    return specs


# ---------------------------------------------------------------------------
# 4) THE CONFIRMATION GATE  (a human "yes" before anything is written to disk)
# ---------------------------------------------------------------------------
def approved(name, args):
    print(f"\n[gate] The agent wants to run a write action: {name}({args})")
    return input("[gate] Approve this write? [y/n]: ").strip().lower().startswith("y")


# ---------------------------------------------------------------------------
# 5) RUN ONE TOOL CALL  (validate args, gate writes, run, errors-as-results)
# ---------------------------------------------------------------------------
def run_tool(name, args):
    if name not in TOOLBOX:
        return f"Error: no such tool '{name}'."
    schema, func = TOOLBOX[name]
    try:
        valid = schema(**args)                     # validate arguments (same class)
    except ValidationError as e:
        return f"Error: invalid arguments for {name}: {e}"
    if name in DANGEROUS and not approved(name, args):
        return "Cancelled by the user."
    try:
        return func(**valid.model_dump())          # run the tool
    except Exception as e:
        return f"Error: {e}"                        # error-as-result


# ---------------------------------------------------------------------------
# 6) TOKEN COUNT + COMPACTION  (measure the context; summarize the old part)
# ---------------------------------------------------------------------------
def token_count(messages):
    """A rough token estimate: about four characters per token. Good enough to
    decide WHEN to compact (a real system would use the model's tokenizer)."""
    chars = sum(len(str(m.get("content", "") or "")) for m in messages)
    return chars // 4


def compact(messages, goal, verbose=False):
    """Fold the OLD middle of the history into one goal-conditioned summary.
    Protected: the system prompt + the original goal (kept word for word) and the
    KEEP_RECENT most recent messages (working memory the model is about to use)."""
    head = messages[:2]                 # [system, the original goal] — never summarized
    recent = messages[-KEEP_RECENT:]    # the freshest steps — never summarized
    middle = messages[2:-KEEP_RECENT]   # everything else — this is what we compress
    if not middle:
        return messages                 # nothing old enough to compact yet
    transcript = "\n".join(f"{m['role']}: {str(m.get('content', '') or '')[:600]}"
                           for m in middle)
    prompt = ("You are compressing an AI agent's history so it fits in a smaller context.\n"
              f"The agent's GOAL is: {goal}\n\n"
              "Write a short summary of the steps below. KEEP the facts discovered, the "
              "sources, the steps already completed, and the questions still open that serve "
              "the goal. DROP raw text, dead ends, and anything the goal does not need.\n\n"
              f"STEPS:\n{transcript}")
    reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    summary = (reply["message"]["content"] or "").strip()
    if verbose:
        print(f"\n[compaction] context over {COMPACT_THRESHOLD} tokens; "
              f"folded {len(middle)} old messages into a summary.")
    summary_msg = {"role": "user", "content": "[Summary of earlier steps]\n" + summary}
    if CON is not None:                 # persist the summary as part of durable state
        CON.execute("UPDATE state SET summary = ? WHERE id = 1", (summary,))
        CON.commit()
    return head + [summary_msg] + recent


# ---------------------------------------------------------------------------
# 7) THE TRACER  (records the run as a trajectory; a tiny stand-in for Langfuse)
# ---------------------------------------------------------------------------
TRAJECTORY = []   # list of (kind, text) for the current run


def observe(fn):
    """Decorator that resets the trajectory at the start of each run."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        TRAJECTORY.clear()
        return fn(*args, **kwargs)
    return wrapper


def record(kind, text):
    TRAJECTORY.append((kind, str(text)))


def print_trajectory():
    """Print the recorded run as an indented tree."""
    print("\n=== trajectory (the trace) ===")
    print("run")
    step = 0
    for kind, text in TRAJECTORY:
        if kind == "thought":
            step += 1
            print(f"├─ step {step}")
            print(f"│    ├─ thought      {text[:70]}")
        elif kind == "action":
            print(f"│    ├─ action       {text[:70]}")
        elif kind == "compact":
            print(f"│    ├─ compaction   {text[:70]}")
        elif kind == "guard":
            print(f"│    ├─ guard        {text[:70]}")
        elif kind == "observation":
            print(f"│    └─ observation  {text[:70]}")
        elif kind == "answer":
            print(f"└─ answer  {text[:70]}")
        elif kind == "stop":
            print(f"└─ stop  {text[:70]}")


# ---------------------------------------------------------------------------
# 8) PERSISTENCE  (save / load the durable state across sessions)
# ---------------------------------------------------------------------------
def save_goal(goal):
    """Record the goal in the state table so a later run can resume it."""
    CON.execute("INSERT OR REPLACE INTO state(id, goal, summary) VALUES (1, ?, "
                "COALESCE((SELECT summary FROM state WHERE id = 1), NULL))", (goal,))
    CON.commit()


def load_state():
    """Return (goal, summary) from the state table, or None if nothing was saved."""
    return CON.execute("SELECT goal, summary FROM state WHERE id = 1").fetchone()


# ---------------------------------------------------------------------------
# 9) THE MEMORY-MANAGED LOOP  (compaction on top; guarded loop below; traced)
# ---------------------------------------------------------------------------
@observe
def agent(goal, verbose=False, resume_summary=None):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": goal}]
    if resume_summary:                             # reload prior progress on --resume
        messages.append({"role": "user",
                         "content": "[Summary of earlier steps]\n" + resume_summary})
    save_goal(goal)                                # PERSISTENCE: durable goal on disk
    seen = set()                                   # repetition guard's memory of past calls
    for step in range(1, MAX_STEPS + 1):           # STEP BUDGET: never more than MAX_STEPS
        # COMPACTION CHECK — the one new line at the top of each step
        if token_count(messages) > COMPACT_THRESHOLD:
            messages = compact(messages, goal, verbose=verbose)
            record("compact", "context compacted, goal-conditioned summary")
        if verbose:
            print(f"\n--- pass {step} of at most {MAX_STEPS}  "
                  f"(context ~{token_count(messages)} tokens) ---")
        reply = ollama.chat(model=MODEL, messages=messages, tools=tool_specs())
        msg = reply["message"]
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        calls = msg.get("tool_calls") or []
        if not calls:                              # no tool call -> the model is answering
            messages.append(msg)
            answer = (msg.get("content") or "").strip() or "(no answer)"
            record("answer", answer)
            return answer
        # ONE ACTION AT A TIME: keep only the first proposed call
        call = calls[0]
        msg["tool_calls"] = [call]
        messages.append(msg)
        name = call["function"]["name"]
        args = call["function"]["arguments"] or {}
        if isinstance(args, str):
            args = json.loads(args)
        thought = (msg.get("content") or msg.get("thinking") or "").strip()
        record("thought", thought or "(no explicit thought)")
        record("action", f"{name}({args})")
        if verbose:
            if thought:
                print(f"thought    : {thought}")
            print(f"action     : {name}({args})")
        # REPETITION GUARD: if this exact call was already made, nudge instead of running
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            result = (f"Repetition guard: you already called {name} with these exact arguments "
                      f"and it did not solve the task. Do not repeat it; try a different approach.")
            record("guard", f"repeated {name}, sent a nudge")
        else:
            seen.add(key)
            result = run_tool(name, args)          # validate + gate + run + error-as-result
        record("observation", result)
        if verbose:
            print(f"observation: {result}")
        messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    record("stop", "step budget reached")
    return "Stopped: step budget reached."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="watch each pass as it happens")
    ap.add_argument("--resume", action="store_true", help="reload the saved goal + notes")
    args = ap.parse_args()

    CON = db_connect(DB_PATH)
    resume_summary = None
    if args.resume:
        state = load_state()
        if state:
            goal, resume_summary = state[0], state[1]
            print(f"Resuming saved goal from {DB_PATH}: {goal}")
            notes = CON.execute("SELECT key FROM notes").fetchall()
            if notes:
                print("Notes available on disk:", ", ".join(k for (k,) in notes))
        else:
            goal = input("No saved state found. What would you like me to do? ")
    else:
        goal = input("What would you like me to do? ")

    print("\n" + agent(goal, verbose=args.verbose, resume_summary=resume_summary))
    print_trajectory()
