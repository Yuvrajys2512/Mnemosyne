# Mnemosyne

Multi-layer persistent memory for LangGraph and LangChain agents.

Three memory layers — episodic, semantic, and working — that persist across sessions and retrieve intelligently using a hybrid recency-plus-relevance scoring function.

```python
from mnemosyne import Mnemosyne, EventType

memory = Mnemosyne(session_id="user_123")
memory.remember("I prefer async Python patterns")
context = memory.recall("help me write an API endpoint")
memory.consolidate()
```
