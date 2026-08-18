"""RSA v1.3 — the guarded, traced loop.

This is RSA with a reliable LOOP around the tool-using core from Module 3. The model
and the tools are unchanged from v1.2. What is new is the harness (the orchestration
loop) around the model, with four additions from Module 4:

  1. EXPLICIT ReAct — every pass is recorded as a labelled thought, action, and
     observation in the ledger and in the trajectory.
  2. A STEP BUDGET — a hard cap (MAX_STEPS) on the outside of the loop, so a runaway
     is impossible.
  3. A REPETITION GUARD — if the model asks for a tool call it already made, the guard
     answers with a short nudge instead of running it again.
  4. TRACE LOGGING — every run is recorded as a navigable trajectory. In production you
     would use a tracing tool such as Langfuse's @observe decorator; here we use a tiny
     built-in tracer so you can run this offline with no account. Swapping in the real
     @observe is a one-line change (see the note on `observe` below).

Nothing here makes the model smarter. The loop around it is simply safe.

Prerequisites
  1. Install Python 3.
  2. Install Ollama, then run:  ollama pull qwen2.5:7b
  3. Install the Python packages:  pip install ollama pydantic

Run it (it will ask you for your goal)
  python rsa_agent_v1_3.py             # the final reply, then the trajectory
  python rsa_agent_v1_3.py --verbose   # also watch each pass as it happens
"""
import functools
import json
import os
import sys

import ollama                                    # talks to the local model
from pydantic import BaseModel, ConfigDict, ValidationError  # tool schemas + validation

MODEL     = "qwen2.5:7b"   # the local model that does the thinking
MAX_STEPS = 20             # THE STEP BUDGET: hard cap on passes, so it can never run forever


# ===========================================================================
# MAP OF THIS FILE  —  find each block; you do not need to read every line.
#   1) THE CONSTITUTION      — the system prompt: who the agent is and its rules.
#   2) THE TOOLBOX           — the two tools (read_note, save_note) and their schemas.
#   3) THE CONFIRMATION GATE — asks a human for yes / no before any write to disk.
#   4) RUN ONE TOOL CALL     — checks arguments, runs the tool, turns errors into results.
#   5) THE TRACER            — records the run as a trajectory (stand-in for Langfuse).
#   6) THE GUARDED LOOP       — step budget on the outside, repetition guard on the inside.
# ===========================================================================


# ---------------------------------------------------------------------------
# 1) THE CONSTITUTION  (the system prompt — role, goal, rules; no scripted plan)
# ---------------------------------------------------------------------------
SYSTEM = """\
# ROLE
You are RSA, a careful note-taking assistant.

# GOAL
Help the user by reading facts from their notes and saving new notes when asked.

# CONSTRAINTS
- Use the tools to read and write notes; do not rely on memory for the user's saved facts.
- Take one action at a time.
- Never invent a note's contents; read the note to confirm.
- If a tool returns an error, read it and try a corrected call.
- If a message tells you a call was already tried, do not repeat it; try something different.

# TOOLS
- read_note(path): read a local note and return its text.
- save_note(path, text): write text to a local note. This is a write action.

# HOW TO FINISH
Call a tool whenever you need one. When the request is complete, reply in plain
text with a short confirmation for the user and do NOT call a tool.
"""


# ---------------------------------------------------------------------------
# 2) THE TOOLBOX  (unchanged from v1.2: schema class + worker function per tool)
# ---------------------------------------------------------------------------
class ReadNote(BaseModel):                      # the INPUTS for the read tool
    """Read a local note file and return its text. Use it to recall a fact the user saved earlier."""
    model_config = ConfigDict(extra="forbid")
    path: str


class SaveNote(BaseModel):                      # the INPUTS for the write tool
    """Write text to a local note file, creating folders if needed. Use it to save a note or summary for later."""
    model_config = ConfigDict(extra="forbid")
    path: str
    text: str


def read_note(path):                            # the WORK of the read tool
    """Safe read tool: return the file's text (raises if the path is wrong)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_note(path, text):                      # the WORK of the write tool
    """Risky write tool: write text to the file, creating parent folders."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved {len(text)} characters to {path}."


TOOLBOX = {
    "read_note": (ReadNote, read_note),
    "save_note": (SaveNote, save_note),
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
# 3) THE CONFIRMATION GATE  (a human "yes" before anything irreversible)
# ---------------------------------------------------------------------------
def approved(name, args):
    print(f"\n[gate] The agent wants to run a write action: {name}({args})")
    return input("[gate] Approve this write? [y/n]: ").strip().lower().startswith("y")


# ---------------------------------------------------------------------------
# 4) RUN ONE TOOL CALL  (validate args, gate writes, run, errors-as-results)
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
        return f"Error: {e}"                        # error-as-result (e.g. file not found)


# ---------------------------------------------------------------------------
# 5) THE TRACER  (records the run as a trajectory; a tiny stand-in for Langfuse)
#    To use the real thing instead, install langfuse and replace `observe` with
#    `from langfuse.decorators import observe`. Everything else stays the same.
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
        elif kind == "guard":
            print(f"│    ├─ guard        {text[:70]}")
        elif kind == "observation":
            print(f"│    └─ observation  {text[:70]}")
        elif kind == "answer":
            print(f"└─ answer  {text[:70]}")
        elif kind == "stop":
            print(f"└─ stop  {text[:70]}")


# ---------------------------------------------------------------------------
# 6) THE GUARDED LOOP  (step budget outside, repetition guard inside, traced)
# ---------------------------------------------------------------------------
@observe
def agent(goal, verbose=False):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": goal}]
    seen = set()                                   # repetition guard's memory of past calls
    for step in range(1, MAX_STEPS + 1):           # STEP BUDGET: never more than MAX_STEPS
        if verbose:
            print(f"\n--- pass {step} of at most {MAX_STEPS} ---")
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
        if thought:
            record("thought", thought)
        else:
            record("thought", "(no explicit thought)")
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
    verbose = "--verbose" in sys.argv
    goal = input("What would you like me to do? ")
    print("\n" + agent(goal, verbose=verbose))
    print_trajectory()
