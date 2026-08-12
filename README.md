# AIA1 — Agentic AI (Course Code)

Companion code for the **AIA1** course (Agentic AI). Each `weekN/` folder holds the
runnable code for that week's videos. Files are version-tagged (e.g. `_v1_0`, `_v1_1`)
so each week's agent stays distinct.

## Week 1 — Build your first agent (RSA v1.0)

[`week1/rsa_agent_v1_0.py`](week1/rsa_agent_v1_0.py) is a complete AI agent in about 30 lines:
a **local Llama 3.2** model, one **web-search** tool, a **text memory**, and a **loop**
that ties them together.

The search is a **real web search**: it uses [`ddgs`](https://pypi.org/project/ddgs/),
a small Python package that queries the DuckDuckGo search engine and returns the top
results as text — no browser and no API key. When the model outputs
`action: search <query>`, the loop calls this tool and feeds the results back on the
next pass. The system prompt only tells the model its role, its tool, and the reply
format — **it does not script the steps**, so the agent decides for itself what to
search and when to finish.

### Prerequisites

1. **Python 3** — https://www.python.org/downloads (on Windows, tick *"Add python.exe to PATH"*).
2. **Ollama** — install from https://ollama.com/download, then download the model (~2 GB, one time):
   ```
   ollama pull llama3.2
   ```
3. **The Python packages** — talk to Ollama, plus the real web search:
   ```
   pip install ollama ddgs        # macOS: pip3 install ollama ddgs
   ```

### Run it

```
python  rsa_agent_v1_0.py         # Windows
python3 rsa_agent_v1_0.py         # macOS
```

Watch the loop turn, one pass at a time:

```
python  rsa_agent_v1_0.py --verbose
```

Because the search is live, the exact results (and the final answer) will vary from
run to run — that's a real agent working against the real web.

## Week 2 — Make the decisions reliable (RSA v1.1)

[`week2/rsa_agent_v1_1.py`](week2/rsa_agent_v1_1.py) is the same agent with the four
Chapter 2 upgrades, all in the model's **reason** step:

1. **A constitution** — the system prompt is written in five labelled parts: role,
   goal, constraints, action space, and output format.
2. **Chain-of-thought** — the decision schema asks the model to write its `reasoning`
   first, before it commits to an action.
3. **A Decision schema + validate-and-retry** — every decision is a Pydantic
   `Decision`. We force the shape with Ollama's `format=` argument and check the reply
   with `model_validate_json`; a malformed reply is handed back for a retry.
4. **Self-consistency** — each decision is sampled a few times and the majority vote
   is kept, so the odd wrong sample gets outvoted.

As in Week 1, the system prompt gives the model only its role, tools, and output
format — it never scripts the steps.

### Prerequisites

Same as Week 1, plus **Pydantic** (the schema library):

```
pip install ollama ddgs pydantic     # macOS: pip3 install ...
```

### Run it

Save `week2/rsa_agent_v1_1.py` into a folder (e.g. `~/Desktop/agent`), then:

```
cd ~/Desktop/agent
python  rsa_agent_v1_1.py            # Windows
python3 rsa_agent_v1_1.py            # macOS
python3 rsa_agent_v1_1.py --verbose  # watch the reasoning, the votes, and the loop
```

A single decision now costs several model calls (a few samples, each possibly
retried), but what comes out is a validated `Decision` the loop can trust.
