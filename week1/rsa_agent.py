"""RSA — the minimal research agent from Week 1 (about 30 lines).

It runs a local Llama 3.2 model through Ollama and has one tool: a web search.
For this course demo, `web_search` is a small BUILT-IN STUB that returns canned
results, so the agent runs offline and every student sees the same clean trace.
(In a later week we swap this stub for a real web search.)

Prerequisites
  1. Install Python 3.
  2. Install Ollama, then run:  ollama pull llama3.2
  3. Install the Python package:  pip install ollama

Run it
  python rsa_agent.py             # just the answer
  python rsa_agent.py --verbose   # watch the memory grow, pass by pass
"""
import sys
import ollama  # talks to the local model

MODEL = "llama3.2"
SYSTEM = (  # fixed prompt, sent on every call
    "You are a research assistant that answers the user's question by "
    "searching the web.\n"
    "Reply with a single line only:\n"
    "  action: <name> <input>\n"
    "where <name> is 'search' or 'finish'."
)


# --- The one tool: a stubbed web search (canned answers for the demo) -------
def web_search(query):
    q = query.lower()
    if "year" in q:
        return "We are in 2026."
    if "france" in q and "gdp" in q:
        return "France's GDP in 2026 is $3.6 trillion."
    return "No relevant result found."


TOOLS = {"search": web_search}


# --- The model call: memory in -> one line out -----------------------------
def call(memory):
    reply = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": memory},
    ])
    return reply["message"]["content"].strip()


# --- The parser: one line -> (action name, input) --------------------------
def parse(line):
    body = line.split("action:")[-1].strip()
    name, _, arg = body.partition(" ")
    return name, arg.strip()


# --- The loop: ask -> parse -> act -> remember -----------------------------
def agent(goal, max_steps=10, verbose=False):
    memory = "Goal: " + goal
    for step in range(1, max_steps + 1):
        line = call(memory)               # perceive + reason
        name, arg = parse(line)
        if verbose:
            print(f"\n--- pass {step} ---")
            print("model :", line)
        if name == "finish":              # the model says it is done
            return arg
        result = TOOLS[name](arg)         # act
        memory += "\n" + line             # observe + update memory
        memory += "\nresult: " + result
        if verbose:
            print("result:", result)
    return "Stopped: step budget reached."


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    goal = "What is France's GDP?"
    print("Goal:", goal)
    answer = agent(goal, verbose=verbose)
    print("\nAnswer:", answer)
