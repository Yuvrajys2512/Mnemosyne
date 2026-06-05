# recall-ai

Multi-layer persistent memory for LangGraph and LangChain agents.

Most AI agents forget everything the moment a session ends. **recall-ai** gives your agent three memory layers that persist across sessions — the same way human memory works.

```python
from mnemosyne import Mnemosyne, EventType

memory = Mnemosyne(session_id="user_123", groq_api_key="gsk_...")

# Store what happens
memory.remember("I always prefer Python over JavaScript", event_type=EventType.USER_MESSAGE)

# Retrieve relevant context before each LLM call
context = memory.recall("what language does the user prefer?")
# → "User Preferences\n• User always prefers Python over JavaScript\n..."

# Compress episodes into durable facts (or let it happen automatically)
memory.consolidate()
```

Memory persists to disk. Restart your agent — it still knows who the user is.

---

## How it works

```
Episode store  ──→  Consolidation (LLM)  ──→  Semantic store
(raw events)        extracts durable facts      (compressed facts)
     │                                               │
     └───────────────────────────────────────────────┘
                          ↓
                   Working memory
              (top-k, token-budget-aware)
                          ↓
                   recall() → str
              (injected into system prompt)
```

**Three layers:**

| Layer | What it stores | Lifetime |
|---|---|---|
| Episodic | Raw timestamped events (messages, tool calls) | Hours to days |
| Semantic | LLM-compressed durable facts (preferences, context) | Weeks |
| Working | Top-k assembled context for the current prompt | One turn |

**Scoring** — every candidate memory is ranked by a composite score:

```
score = 0.40 × semantic_similarity
      + 0.30 × recency_decay
      + 0.20 × importance
      + 0.10 × access_frequency
```

---

## Installation

```bash
pip install recall-ai
```

For the LangGraph integration example:

```bash
pip install recall-ai langgraph
```

---

## Quick start

### Standalone

```python
from mnemosyne import Mnemosyne, EventType

memory = Mnemosyne(
    session_id="user_123",
    groq_api_key="gsk_...",          # free at console.groq.com
)

memory.remember("I prefer dark mode in all my tools")
memory.remember("The project uses FastAPI and PostgreSQL")

print(memory.recall("what tools is the user working with?"))
```

### LangGraph agent

```python
from langgraph.graph import StateGraph, END
from mnemosyne import Mnemosyne, EventType

memory = Mnemosyne(session_id="user_123", groq_api_key="gsk_...")

def agent_node(state):
    context = memory.recall(state["messages"][-1]["content"])
    # inject context into your system prompt, call LLM ...
    memory.remember(state["messages"][-1]["content"], event_type=EventType.USER_MESSAGE)
    memory.remember(reply, event_type=EventType.AGENT_RESPONSE)
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}]}
```

See [`examples/langgraph_agent.py`](examples/langgraph_agent.py) for a full working demo.

---

## Configuration

```python
memory = Mnemosyne(
    session_id="user_123",

    # Storage (default: ~/.mnemosyne/sessions/<session_id>)
    storage_path="/data/my_agent",

    # LLM for consolidation — Groq (primary) with Ollama fallback
    groq_api_key="gsk_...",
    groq_model="llama-3.3-70b-versatile",
    ollama_model="llama3.2",                   # used if Groq fails

    # Working memory token budget injected into prompts
    working_memory_tokens=1_500,

    # Auto-consolidate after this many unconsolidated events (async contexts only)
    auto_consolidate_threshold=20,
)
```

---

## LLM providers

Consolidation requires an LLM to extract semantic facts from raw episodes.

| Provider | Setup |
|---|---|
| **Groq** (recommended) | Free tier at [console.groq.com](https://console.groq.com). Pass `groq_api_key=`. |
| **Ollama** (local) | Run `ollama pull llama3.2`. Used automatically as fallback if Groq fails. |

---

## License

MIT
