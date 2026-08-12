"""RSA v1.1 — the research agent, hardened with the Week 2 upgrades.

This is RSA v1.0 (the ~30-line Week 1 agent) with the four reliability upgrades
from Chapter 2 — every one of them layered onto the model's *reason* step:

  1. A CONSTITUTION for the system prompt — five labelled parts: Role, Goal,
     Constraints, Action space, and Output format (see SYSTEM below).
  2. CHAIN-OF-THOUGHT — the decision schema asks the model to write its
     `reasoning` FIRST, before it commits to an action.
  3. A DECISION SCHEMA with VALIDATE-AND-RETRY — every decision is a Pydantic
     `Decision`. We force the shape with Ollama's `format=` argument and check the
     reply with `model_validate_json`; a malformed reply is handed back for a retry.
  4. SELF-CONSISTENCY — instead of trusting one draw, we sample each decision a
     few times and keep the majority vote. The odd wrong sample gets outvoted.

As in v1.0, the system prompt gives the model only its role, its tools, and the
output format — never a worked example or a scripted plan. The agent still has to
reach the answer on its own; we have only made its decisions more reliable.

Prerequisites
  1. Install Python 3.
  2. Install Ollama, then run:  ollama pull llama3.2
  3. Install the Python packages:  pip install ollama ddgs pydantic

Run it (it will ask you for your question)
  python rsa_agent.py             # just the answer
  python rsa_agent.py --verbose   # watch the reasoning, the votes, and the loop
"""
import sys
from collections import Counter
from typing import Literal

import ollama                                    # talks to the local model
from ddgs import DDGS                            # the real web-search engine (DuckDuckGo)
from pydantic import BaseModel, ValidationError  # the schema + its validator

MODEL       = "llama3.2"
N_SAMPLES   = 3     # self-consistency: draws per decision, then majority vote
MAX_RETRIES = 2     # validate-and-retry: extra tries if a reply is malformed
TEMPERATURE = 0.8   # > 0 so the samples actually differ (voting needs variety)
MAX_STEPS   = 8     # guard: cap on passes through the loop


# ---------------------------------------------------------------------------
# 1) THE DECISION SCHEMA  (structured output + chain-of-thought)
#    `reasoning` is declared FIRST, so under schema-constrained decoding the
#    model must generate its reasoning before it picks an action.
# ---------------------------------------------------------------------------
class Decision(BaseModel):
    reasoning: str
    action: Literal["search", "finish"]
    input: str


# ---------------------------------------------------------------------------
# 2) THE CONSTITUTION  (the system prompt, in five parts)
#    Role + Goal + Constraints + Action space + Output format — and nothing that
#    scripts the search or hands the model the answer.
# ---------------------------------------------------------------------------
SYSTEM = """\
# ROLE
You are RSA, a careful research assistant.

# GOAL
Answer the user's question with a correct, well-supported answer.
Finish only once the answer is backed by something the search tool returned.

# CONSTRAINTS
- Do not rely on your own memory for facts; confirm them with the search tool.
- Take exactly one action per turn.
- Never invent facts, sources, or tool results.
- Think briefly before you act.

# ACTION SPACE
- search : run a web search for `input` (a query).
- finish : return `input` as the final answer to the user.

# OUTPUT FORMAT
Reply with ONE JSON object with exactly these fields:
- reasoning : one short sentence explaining your next step
- action    : "search" or "finish"
- input     : the search query, or the final answer
"""


# ---------------------------------------------------------------------------
# THE TOOL: a REAL web search (unchanged from v1.0)
#   When a decision is `action == "search"`, the loop calls this with the query.
#   It asks DuckDuckGo through the `ddgs` package and returns the top snippets.
# ---------------------------------------------------------------------------
def web_search(query):
    try:
        with DDGS() as ddg:
            hits = ddg.text(query, max_results=3)
        text = "  ".join(h.get("body", "") for h in hits if h.get("body"))
        return " ".join(text.split())[:400] or "No results found."
    except Exception as e:
        return f"Search error: {e}"


# ---------------------------------------------------------------------------
# 3) ONE VALIDATED DECISION  (force the shape + validate + retry)
#    `format=Decision.model_json_schema()` constrains Ollama's decoding to the
#    schema, so the raw reply is already schema-shaped JSON. `model_validate_json`
#    is the second safety layer; if it ever fails, we hand the error back and ask
#    the model to try again — up to MAX_RETRIES times.
# ---------------------------------------------------------------------------
def decide_once(memory):
    prompt = memory
    error = None
    for _ in range(MAX_RETRIES + 1):
        reply = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            format=Decision.model_json_schema(),   # <-- forces schema-shaped JSON
            options={"temperature": TEMPERATURE},
        )
        raw = reply["message"]["content"]
        try:
            return Decision.model_validate_json(raw)          # validate + parse
        except ValidationError as e:                          # tell the model what broke
            error = e
            prompt = (memory + "\n\nYour last reply was not valid:\n" + str(e)
                      + "\nReply again with a single valid JSON object.")
    raise error   # gave up after the retries


# ---------------------------------------------------------------------------
# 4) SELF-CONSISTENCY  (sample the decision N times, keep the majority vote)
#    We vote over the (action, input) pair and keep a winning sample so we also
#    keep its reasoning. Outliers get outvoted; the price is a few extra calls.
# ---------------------------------------------------------------------------
def decide(memory, verbose=False):
    samples = [decide_once(memory) for _ in range(N_SAMPLES)]
    key = lambda d: (d.action, d.input.strip().lower())
    tally = Counter(key(d) for d in samples)
    winner_key, votes = tally.most_common(1)[0]
    winner = next(d for d in samples if key(d) == winner_key)   # keep its reasoning
    if verbose:
        extra = "" if len(tally) == 1 else f"   ({len(tally)} distinct draws)"
        print(f'votes  : {votes}/{N_SAMPLES} for {winner.action} "{winner.input}"{extra}')
    return winner


# ---------------------------------------------------------------------------
# THE LOOP  (perceive -> reason+vote -> act -> observe -> update)
# ---------------------------------------------------------------------------
def agent(goal, verbose=False):
    memory = "User's question: " + goal
    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n--- pass {step} ---")
        d = decide(memory, verbose)                     # perceive + reason (+ vote)
        if verbose:
            print(f"reason : {d.reasoning}\naction : {d.action} {d.input}")
        if d.action == "finish":                        # the model decides it is done
            return d.input
        result = web_search(d.input)                    # act
        memory += (f"\nreasoning: {d.reasoning}"         # observe + update memory
                   f"\naction: search {d.input}"
                   f"\nresult: {result}")
        if verbose:
            print(f"result : {result}")
    return "Stopped: step budget reached."


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    goal = input("What would you like me to research? ")   # ask the user for the goal
    print("\nAnswer:", agent(goal, verbose=verbose))
