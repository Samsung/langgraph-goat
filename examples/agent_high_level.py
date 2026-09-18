"""agent.py — High-Level (create_agent) demonstration with GoatMiddleware.

This file demonstrates the high-level integration pattern using GoatMiddleware
with LangChain's create_agent API.
"""

import json
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage
# from langgraph.store.memory import InMemoryStore  # Disabled — using MarkdownStore instead
from langchain.agents import create_agent

# Import GoatMiddleware from langgraph_goat package (installed via pip)
from langgraph_goat import GoatMiddleware, GoatMiddlewareConfig, GoatLogger

# Import tools and world state
from examples.res.toolsNworld import WORLD, ALL_TOOLS

import logging

# ── 1. Define Context Schema (for user identification) ────────────────────────

@dataclass
class Context:
    user_id: str

# ── 3. Initialize Store and Model ─────────────────────────────────────────────

from customllm import CustomLLM
from examples.res.markdown_store import MarkdownStore

# store = InMemoryStore()  # Disabled — using MarkdownStore for persistent memory
store = MarkdownStore("examples/res/memory.md")
model = CustomLLM()


# ── 4. Configure GoatMiddleware ───────────────────────────────────────────────

# Output directory for all GoAT artifacts (graphs, telemetry, explanations)
# Change this to your desired output location
OUTPUT_DIR = "examples/output"

goat_config = GoatMiddlewareConfig(
    default_K=5,
    max_K=5,
    store=store,  # Pass store so GoAT can read memory for better plan generation
    output_dir=OUTPUT_DIR,  # All GoAT outputs will be saved here
    render_goat_graph=True,
    render_goat_tele=True,
    render_goat_trace=True,
    # Optional: customize components
    # classifier=CustomClassifier(),
    # critic=CustomCritic(),
    # explanation_generator=CustomExplanationGenerator(),
    # planning_prompt_template="...",
)
goat_middleware = GoatMiddleware(goat_config)


# ── 5. Create the Agent with GoAT middleware ──────────────────────────────────

# ALL_TOOLS is imported from toolsNworld.py (contains 67 tools for 20 test cases)

agent = create_agent(
    model=model,
    tools=ALL_TOOLS,
    middleware=[goat_middleware],
    context_schema=Context,
    store=store,
)


# ── 5b. Helper function for testing ───────────────────────────────────────────

# Configure logging to output DEBUG level messages to console

log = GoatLogger("middleware", level=logging.DEBUG)


# ── 6. Usage Simulation ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("GoAT Middleware")
    print("=" * 72)
    
    # Save a preference to Long-Term Memory
    store.put(("preferences",), "user_123", {"style": "Artistic and poetic."})
    print("[LOG] Saved user preference to memory store")
    
    # Invoke the Agent
    context = Context(user_id="user_123")
    user_input = "I feel ambitious."
    
    print(f"[LOG] Invoking agent with input: '{user_input}'")
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        context=context,
    )
    print("[LOG] Agent invocation complete")
    print(f"[LOG] Response keys: {list(response.keys())}")
    print(f"[LOG] Response type: {type(response)}")
    print(f"[LOG] message:\n {response['messages']}")

    # Extract GoatOutput from response or middleware instance
    goat_output = response.get("goat_output")
    if goat_output is None:
        # Fallback: get from middleware instance
        goat_output = goat_middleware.last_goat_output
        print("[LOG] GoatOutput retrieved from middleware instance")

    goat_graph = goat_output.graph if goat_output else None
    goat_graph_id = goat_graph.graph_id if goat_graph else "unknown"
    goat_winning_plan_id = goat_output.winning_plan_id if goat_output else None
    goat_complexity = goat_output.complexity if goat_output else None
    goat_director_result = goat_output.director_result if goat_output else None
    goat_explanation = goat_output.explanation if goat_output else None
    
    print(f"[LOG] GoAT Graph ID: {goat_graph_id}")
    print(f"[LOG] Winning Plan ID: {goat_winning_plan_id}")
    print(f"[LOG] Complexity K: {goat_complexity.suggested_K if goat_complexity else 'N/A'}")
    
    # ── Create output folder with graph_id ─────────────────────────────────────
    
    output_dir = Path(OUTPUT_DIR) / goat_graph_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LOG] Created output directory: {output_dir}")
    
    # ── Build complete response data ──────────────────────────────────────────

    goat_response = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "goat_graph_id": goat_graph_id,
        **(goat_output.to_dict() if goat_output else {}),
    }
    
    # ── Save response.json ─────────────────────────────────────────────────────
    
    response_file = output_dir / "response.json"
    with open(response_file, "w", encoding="utf-8") as f:
        json.dump(goat_response, f, indent=2, default=str, ensure_ascii=False)
    print(f"[LOG] Saved response to: {response_file}")
    
    # ── Save graph.json (from GoAT graph) ──────────────────────────────────────

    if goat_graph:
        goat_dict = goat_output.to_dict()
        graph_data = {
            **goat_dict.get("graph", {}),
            "plan_complexity": goat_dict.get("complexity"),
        }

        graph_file = output_dir / "graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, default=str, ensure_ascii=False)
        print(f"[LOG] Saved graph to: {graph_file}")
