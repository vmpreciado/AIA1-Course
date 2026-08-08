"""RSA — the minimal research agent from Week 1 (about 30 lines).

It runs a local Llama 3.2 model through Ollama and has one tool: a REAL web search.
The search uses `ddgs`, a small Python package that queries the DuckDuckGo search
engine and hands back the top results as plain text — no browser, no API key.

The system prompt gives the model only its role, its tool, and the reply format.
It does NOT tell the model what to search for or when to stop — the agent decides
that on its own, one pass at a time. That autonomy is the whole point.

Prerequisites
  1. Install Python 3.
  2. Install Ollama, then run:  ollama pull llama3.2
  3. Install the Python packages:  pip install ollama ddgs

Run it
  python rsa_agent.py             # just the answer
  python rsa_agent.py --verbose   # watch the memory grow, pass by pass
"""
import sys
import ollama          # talks to the local model
from ddgs import DDGS  # the real web-search engine (DuckDuckGo)

MODEL = "llama3.2"
SYSTEM = (  # role + tool + format only — no plan, no examples
    "You are a research agent. Your only tool is a web search.\n"
    "You cannot rely on your own knowledge; use the search tool to find facts.\n"
    "On each turn, reply with a single line, in one of these two forms:\n"
    "  action: search <query>\n"
    "  action: finish <answer>\n"
    "Finish only when you can state the answer."
)


# The one TOOL: a REAL web search. When the model emits "action: search <query>",
# the loop below calls this function, which asks DuckDuckGo (via the ddgs package)
# and returns the top result snippets as text for the model to read next pass.
def web_search(query):
    try:
        with DDGS() as ddg:
            hits = ddg.text(query, max_results=3)
        text = "  ".join(h.get("body", "") for h in hits if h.get("body"))
        return " ".join(text.split())[:400] or "No results found."
    except Exception as e:
        return f"Search error: {e}"


def call(memory):  # MODEL: memory -> one 'action:' line
    reply = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": memory}])
    text = reply["message"]["content"].strip()
    for ln in text.splitlines():          # take the line that holds the action
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
    for step in range(1, max_steps + 1):
        line = call(memory)                       # perceive + reason
        name, arg = parse(line)
        if verbose:
            print(f"\n--- pass {step} ---\nmodel : {line}")
        if name == "finish":                      # the model decides it is done
            return arg
        result = web_search(arg) if name == "search" else \
            "Unknown action. Use 'action: search <query>' or 'action: finish <answer>'."
        memory += f"\n{line}\nresult: {result}"    # observe + update memory
        if verbose:
            print(f"result: {result}")
    return "Stopped: step budget reached."


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    goal = "What is France's GDP?"   # the question — edit this line to ask something else
    print("Goal:", goal)
    print("\nAnswer:", agent(goal, verbose=verbose))
