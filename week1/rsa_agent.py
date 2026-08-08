"""RSA — the minimal research agent from Week 1.

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
# The system prompt fixes the format AND shows one worked example, so the small
# local model reliably searches first and only then finishes with the answer.
SYSTEM = (
    "You are a research agent. You answer the user's question with a search tool.\n"
    "On EACH turn, reply with EXACTLY ONE line, in one of these forms:\n"
    "  action: search <what to look up>\n"
    "  action: finish <the final answer>\n"
    "Use 'search' to gather facts; use 'finish' only once you know the answer.\n"
    "Never write anything except a single 'action:' line.\n\n"
    "Example run:\n"
    "Goal: What is Japan's GDP?\n"
    "action: search current year\n"
    "result: We are in 2026.\n"
    "action: search Japan GDP in 2026\n"
    "result: Japan's GDP in 2026 is $4.2 trillion.\n"
    "action: finish Japan's GDP in 2026 is $4.2 trillion."
)


def web_search(query):  # the TOOL (stubbed with canned answers for the demo)
    q = query.lower()
    if "year" in q:
        return "We are in 2026."
    if "france" in q and "gdp" in q:
        return "France's GDP in 2026 is $3.6 trillion."
    return "No relevant result found."


def call(memory):  # MODEL: memory -> one 'action:' line
    reply = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": memory}])
    text = reply["message"]["content"].strip()
    for ln in text.splitlines():          # pick the line that holds the action
        if "action:" in ln.lower():
            return ln.strip()
    return text.splitlines()[0] if text else ""


def parse(line):  # PARSER: text -> (name, input)
    i = line.lower().find("action:")
    body = (line[i + 7:] if i >= 0 else line).strip()
    name, _, arg = body.partition(" ")
    return name.lower().strip(), arg.strip()


def agent(goal, max_steps=10, verbose=False):  # the LOOP
    memory = "Goal: " + goal
    last = ""                                   # remember the latest search result
    for step in range(1, max_steps + 1):
        line = call(memory)                     # perceive + reason
        name, arg = parse(line)
        if verbose:
            print(f"\n--- pass {step} ---\nmodel : {line}")
        if name == "finish" and arg:            # done, with a real answer
            return arg
        if name == "search":                    # act
            result = last = web_search(arg)
        else:                                   # empty/unknown -> nudge, don't crash
            result = "Search for the facts first, then finish with the answer."
        memory += f"\n{line}\nresult: {result}"  # observe + update memory
        if verbose:
            print(f"result: {result}")
    return last or "Stopped: step budget reached."  # never return an empty answer


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    goal = "What is France's GDP?"   # the question — edit this line to ask something else
    print("Goal:", goal)
    print("\nAnswer:", agent(goal, verbose=verbose))
