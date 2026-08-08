# AIA1 — Agentic AI (Course Code)

Companion code for the **AIA1** course (Agentic AI). Each `weekN/` folder holds the
runnable code for that week's videos.

## Week 1 — Build your first agent

[`week1/rsa_agent.py`](week1/rsa_agent.py) is a complete AI agent in about 30 lines:
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
python  rsa_agent.py         # Windows
python3 rsa_agent.py         # macOS
```

Watch the loop turn, one pass at a time:

```
python  rsa_agent.py --verbose
```

Because the search is live, the exact results (and the final answer) will vary from
run to run — that's a real agent working against the real web.
