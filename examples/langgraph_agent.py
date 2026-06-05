"""
Mnemosyne + LangGraph: a conversational agent with persistent long-term memory.

What this demonstrates
──────────────────────
  • recall()     — injects relevant long-term context before every agent turn
  • remember()   — logs each user/agent exchange to episodic memory
  • consolidate()— extracts durable semantic facts from the session at the end
  • Persistence  — memory is written to disk; restart the script and the agent
                   remembers who you are and what you discussed before.

Graph
──────────────────────
  [recall] → [agent] → [remember] → END
     ↑                                ↓
     └──────── next user turn ────────┘

Dependencies
──────────────────────
    pip install mnemosyne langgraph

Run
──────────────────────
    Windows cmd:   set GROQ_API_KEY=gsk_...  &&  python examples/langgraph_agent.py
    PowerShell:    $env:GROQ_API_KEY="gsk_..."  ;  python examples/langgraph_agent.py
"""
from __future__ import annotations

import os
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from mnemosyne import EventType, Mnemosyne

# ── Configuration ─────────────────────────────────────────────────────────────

SESSION_ID   = "langgraph_demo"
STORAGE_PATH = os.path.join(os.path.expanduser("~"), ".mnemosyne", "langgraph_demo")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# How many tokens of long-term memory to inject into each prompt
MEMORY_TOKEN_BUDGET = 800

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list[dict]   # full conversation history (role / content dicts)
    memory_context: str    # formatted long-term memory for the current turn


# ── Node factories ────────────────────────────────────────────────────────────

def make_recall_node(memory: Mnemosyne):
    def recall_node(state: AgentState) -> dict:
        """Query long-term memory using the latest user message as the search key."""
        last_user = next(
            (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
            "",
        )
        context = memory.recall(last_user, token_budget=MEMORY_TOKEN_BUDGET)
        return {"memory_context": context}
    return recall_node


def make_agent_node(llm: OpenAI):
    def agent_node(state: AgentState) -> dict:
        """Build a prompt with memory context injected, call the LLM, return reply."""
        system_parts = ["You are a helpful assistant with long-term memory."]
        if state.get("memory_context", "").strip():
            system_parts.append(
                "\nWhat you remember about this user and their projects:\n"
                + state["memory_context"]
                + "\n\nUse this context naturally in your replies. "
                  "Do not say 'according to my memory' — just act like you know it."
            )

        response = llm.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "\n".join(system_parts)},
                *state["messages"],
            ],
        )
        reply = response.choices[0].message.content
        return {"messages": state["messages"] + [{"role": "assistant", "content": reply}]}
    return agent_node


def make_remember_node(memory: Mnemosyne):
    def remember_node(state: AgentState) -> dict:
        """Store the last user/assistant exchange in episodic memory."""
        msgs = state["messages"]
        # Walk backward to find the last user and assistant messages
        assistant_content = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None
        )
        user_content = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "user"), None
        )
        if user_content:
            memory.remember(user_content, event_type=EventType.USER_MESSAGE)
        if assistant_content:
            memory.remember(assistant_content, event_type=EventType.AGENT_RESPONSE)
        return {}
    return remember_node


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph(memory: Mnemosyne, llm: OpenAI):
    g = StateGraph(AgentState)

    g.add_node("recall",   make_recall_node(memory))
    g.add_node("agent",    make_agent_node(llm))
    g.add_node("remember", make_remember_node(memory))

    g.set_entry_point("recall")
    g.add_edge("recall",   "agent")
    g.add_edge("agent",    "remember")
    g.add_edge("remember", END)

    return g.compile()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    memory = Mnemosyne(
        session_id=SESSION_ID,
        storage_path=STORAGE_PATH,
        groq_api_key=api_key,
        groq_model=GROQ_MODEL,
    )
    llm = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    graph = build_graph(memory, llm)

    print("=" * 60)
    print("  Mnemosyne + LangGraph demo")
    print(f"  Session : {SESSION_ID}")
    print(f"  Memory  : {STORAGE_PATH}")
    print("  Type 'quit' to exit and consolidate memory.")
    print("=" * 60 + "\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = "quit"

        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        state = graph.invoke({"messages": history, "memory_context": ""})
        history = state["messages"]

        assistant_reply = next(
            m["content"] for m in reversed(history) if m["role"] == "assistant"
        )
        print(f"\nAgent: {assistant_reply}\n")

    # Consolidate episodic memory into durable semantic facts on exit
    print("\nConsolidating memory...")
    result = memory.consolidate()
    print(
        f"Done. {result.episodes_processed} events → "
        f"{result.facts_created} new facts, "
        f"{result.facts_updated} updated."
    )
    print("Memory saved. Run the script again to continue where you left off.\n")


if __name__ == "__main__":
    main()
