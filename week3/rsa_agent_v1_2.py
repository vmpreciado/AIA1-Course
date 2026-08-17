"""RSA v1.2 — the tool-using agent, rebuilt on native function calling.

This is RSA with its ACTION side rebuilt for Chapter 3. The reasoning core is the
same idea as v1.1 — a constitution-style system prompt that gives the model only
its role, its tools, and how to answer, never a worked example or a scripted plan.
What changes is HOW the agent acts:

  1. NATIVE FUNCTION CALLING — we retire the hand-written JSON parser. We hand the
     model the tool schemas through Ollama's `tools=` argument, and the model
     replies with a structured tool call (which tool, with what arguments).
  2. A REAL TOOLBOX, generated from Pydantic — two tools: a safe read tool,
     `read_note`, and a risky write tool, `save_note`. Each tool's arguments are a
     Pydantic class; the SAME class both describes the tool to the model and
     validates the arguments when a call arrives, so schema and code cannot drift.
  3. A CONFIRMATION GATE — `save_note` writes to disk, which is not cheaply
     reversible, so it never runs without a human "yes".
  4. ERRORS AS RESULTS — a failed tool call is not a crash. We catch the error and
     hand it back to the model as the tool's result, so the model can read it and
     correct its next call.

The whole thing runs the function-calling loop over MESSAGES — the running ledger
of the conversation, which is the agent's memory.

Prerequisites
  1. Install Python 3.
  2. Install Ollama, then run:  ollama pull qwen2.5:7b
  3. Install the Python packages:  pip install ollama pydantic

Run it (it will ask you for your goal)
  python rsa_agent_v1_2.py             # just the final reply
  python rsa_agent_v1_2.py --verbose   # watch the loop, the tool calls, and the gate
"""
import json
import os
import sys

import ollama                                    # talks to the local model
from pydantic import BaseModel, ConfigDict, ValidationError  # tool schemas + argument validation

MODEL     = "qwen2.5:7b"   # the local model that does the thinking (see the homework for why not a smaller one)
MAX_STEPS = 8              # guard: cap on passes through the loop, so it can never run forever


# ===========================================================================
# MAP OF THIS FILE  —  five blocks. You do NOT need to read every line; the goal
# is to find each block and get a sense of what it does.
#   1) THE CONSTITUTION      — the system prompt: who the agent is and the rules it follows.
#   2) THE TOOLBOX           — the two tools (read_note, save_note): their input schemas
#                              and the plain functions that run them.
#   3) THE CONFIRMATION GATE — asks a human for yes / no before any write to disk.
#   4) RUN ONE TOOL CALL     — checks the arguments, runs the tool, and turns any error
#                              into a result (so a failure is never a crash).
#   5) THE LOOP              — the agent's memory and its reasoning -> action -> result cycle.
# ===========================================================================


# ---------------------------------------------------------------------------
# 1) THE CONSTITUTION  (the system prompt — the reasoning core, unchanged in spirit)
#    Role + Goal + Constraints + the tools it may use + how to finish. Nothing
#    here scripts the task or hands the model the answer.
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

# TOOLS
- read_note(path): read a local note and return its text.
- save_note(path, text): write text to a local note. This is a write action.

# HOW TO FINISH
Call a tool whenever you need one. When the request is complete, reply in plain
text with a short confirmation for the user and do NOT call a tool.
"""


# ---------------------------------------------------------------------------
# 2) THE TOOLBOX  (two tools; their argument schemas generated from Pydantic)
#    Each tool's arguments are a Pydantic class. The docstring becomes the tool's
#    description the model reads; the fields become its parameters. The very same
#    class validates the arguments when a call arrives (see `run_tool`).
#    In plain terms: each tool has TWO parts — a small "class" just below that lists
#    the tool's inputs and checks them, paired with a plain function further down
#    that actually does the work (open a file, write a file).
# ---------------------------------------------------------------------------
class ReadNote(BaseModel):                      # the INPUTS for the read tool: just a file path
    """Read a local note file and return its text. Use it to recall a fact the user saved earlier."""
    model_config = ConfigDict(extra="forbid")  # reject unknown arguments so a malformed call is caught
    path: str


class SaveNote(BaseModel):                      # the INPUTS for the write tool: where to write, and what
    """Write text to a local note file, creating folders if needed. Use it to save a note or summary for later."""
    model_config = ConfigDict(extra="forbid")  # reject unknown arguments so a malformed call is caught, not silently written
    path: str
    text: str


def read_note(path):                            # the WORK of the read tool: open the file and hand back its text
    """Safe read tool: return the file's text (raises if the path is wrong)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_note(path, text):                      # the WORK of the write tool: create folders if needed, then write
    """Risky write tool: write text to the file, creating parent folders."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved {len(text)} characters to {path}."


# Map each tool name to (argument schema class, implementation function).
TOOLBOX = {
    "read_note": (ReadNote, read_note),
    "save_note": (SaveNote, save_note),
}
DANGEROUS = {"save_note"}   # write actions that must pass the confirmation gate


def tool_specs():
    """Build the list of tool schemas we hand the model — straight from Pydantic."""
    specs = []
    for name, (schema, _) in TOOLBOX.items():
        specs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (schema.__doc__ or "").strip(),
                "parameters": schema.model_json_schema(),
            },
        })
    return specs


# ---------------------------------------------------------------------------
# 3) THE CONFIRMATION GATE  (a human "yes" before anything irreversible)
# ---------------------------------------------------------------------------
def approved(name, args):
    print(f"\n[gate] The agent wants to run a write action: {name}({args})")
    return input("[gate] Approve this write? [y/n]: ").strip().lower().startswith("y")


# ---------------------------------------------------------------------------
# 4) RUN ONE TOOL CALL  (validate args, gate writes, run — and errors-as-results)
#    Whatever happens, we return a STRING that becomes the tool's result message.
#    A failure is not a crash: the error text flows back so the model can recover.
# ---------------------------------------------------------------------------
def run_tool(name, args):
    if name not in TOOLBOX:
        return f"Error: no such tool '{name}'."
    schema, func = TOOLBOX[name]
    try:
        valid = schema(**args)                     # validate the arguments (same class)
    except ValidationError as e:
        return f"Error: invalid arguments for {name}: {e}"
    if name in DANGEROUS and not approved(name, args):
        return "Cancelled by the user."            # gate said no -> flows back as a result
    try:
        return func(**valid.model_dump())          # run the tool
    except Exception as e:
        return f"Error: {e}"                        # error-as-result (e.g. file not found)


# ---------------------------------------------------------------------------
# 5) THE FUNCTION-CALLING LOOP  (over MESSAGES — the ledger / the agent's memory)
#    Seed the ledger, ask the model with the tools, run the ONE action it asks for,
#    append the result so the model can SEE it, and repeat until it answers with no
#    tool call. One action per pass matters: a write must not be guessed before the
#    read it depends on has come back.
# ---------------------------------------------------------------------------
def agent(goal, verbose=False):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": goal}]
    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n--- pass {step} ---")
        reply = ollama.chat(model=MODEL, messages=messages, tools=tool_specs())
        msg = reply["message"]
        if hasattr(msg, "model_dump"):             # normalise to a plain dict
            msg = msg.model_dump()
        calls = msg.get("tool_calls") or []
        if not calls:                              # no tool call -> the model is done
            messages.append(msg)
            return (msg.get("content") or "").strip() or "(no answer)"
        # ONE ACTION AT A TIME: even if the model proposes several calls in one turn,
        # run only the first, so it must see the result before choosing the next.
        call = calls[0]
        msg["tool_calls"] = [call]                 # keep the ledger consistent with what we run
        messages.append(msg)                       # MEMORY (1 of 2): remember the action the agent chose
        name = call["function"]["name"]
        args = call["function"]["arguments"] or {}
        if isinstance(args, str):                  # some backends return args as JSON text
            args = json.loads(args)
        if verbose:                                # show the pass as reasoning -> action -> result
            reasoning = (msg.get("content") or msg.get("thinking") or "").strip()
            if reasoning:                          # chain-of-thought, shown only when the model gives it
                print(f"reasoning: {reasoning}")
            print(f"action   : {name}({args})")
        result = run_tool(name, args)              # validate + gate + run + error-as-result
        if verbose:
            print(f"result   : {result}")
        # MEMORY (2 of 2): remember the result too, tagged with the tool it came from.
        # These two appends per pass ARE the agent's memory — nothing else is stored.
        messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    return "Stopped: step budget reached."


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    goal = input("What would you like me to do? ")   # ask the user for the goal
    print("\n" + agent(goal, verbose=verbose))
