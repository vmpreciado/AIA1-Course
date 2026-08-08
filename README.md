# AIA1 — Agentic AI (Course Code)

Companion code for the **AIA1** course (Agentic AI). Each `weekN/` folder holds the
runnable code for that week's videos.

## Week 1 — Build your first agent

[`week1/rsa_agent.py`](week1/rsa_agent.py) is a complete AI agent in about 30 lines:
a **local Llama 3.2** model, one **web-search** tool, a **text memory**, and a **loop**
that ties them together. For this course, the web search is a small built-in **stub**
(canned answers) so the agent runs offline and everyone sees the same result. In a later
week we swap the stub for a real web search.

### Prerequisites

1. **Python 3** — https://www.python.org/downloads (on Windows, tick *"Add python.exe to PATH"*).
2. **Ollama** — install from https://ollama.com/download, then download the model (~2 GB, one time):
   ```
   ollama pull llama3.2
   ```
3. **The Python package** that lets your code talk to Ollama:
   ```
   pip install ollama        # macOS: pip3 install ollama
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

Expected answer: `France's GDP in 2026 is $3.6 trillion.`
