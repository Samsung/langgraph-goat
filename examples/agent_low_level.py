"""agent_low_level.py — Low-Level (StateGraph) demonstration with GoAT node wrapper.

This file demonstrates the low-level integration pattern where:
1. The user defines their own planner node (with their own prompt/logic)
2. GoAT wraps that planner to ask it to generate K diverse plans, critique, and pick a winner
3. Uses tools and world state from toolsNworld.py

For high-level create_agent integration, see agent_high_level.py.
"""

import json
from datetime import datetime
from pathlib import Path

from typing import Annotated, TypedDict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, START, END

# Import tools and world state from toolsNworld
from examples.res.toolsNworld import WORLD, ALL_TOOLS

from examples.res.markdown_store import MarkdownStore

# Import GoAT node wrapper from installed langgraph_goat package
from langgraph_goat import GoatNodeConfig, GoatNode


# ── 1. State Schema ───────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    messages: Annotated[list, lambda x, y: x + y]
    user_input: str
    available_tools: list
    plan: list
    goat_output: Any
    step_outputs: list
    final_answer: str


# ── 2. Tool Implementations ───────────────────────────────────────────────────

# Build tool implementation map from ALL_TOOLS imported from toolsNworld
TOOL_IMPL = {t.name: lambda args, tool=t: tool.invoke(args) for t in ALL_TOOLS}

AVAILABLE_TOOLS_DESC = [
    {"name": t.name, "description": t.description} for t in ALL_TOOLS
]

OUTPUT_PATH = "examples/output"

# ── 3. User's Custom Planner Node ─────────────────────────────────────────────

def build_prompt(state: dict) -> str:
    """Build the planner's prompt from state.

    This is a standalone function that the planner uses internally. When
    wrapped with GoAT, the wrapper can call this to get the planner's prompt
    and append a K-plans instruction to it (for K>=2).

    The planner itself is completely unaware of GoAT.
    """
    user_input = state.get("user_input", "")
    available_tools = state.get("available_tools", AVAILABLE_TOOLS_DESC)

    tool_lines = []
    for t in available_tools:
        name = t.get("name", "unknown") if isinstance(t, dict) else getattr(t, "name", "unknown")
        desc = t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
        tool_lines.append(f"- {name}: {desc}")
    tools_desc = "\n".join(tool_lines) if tool_lines else "No tools available"

    memory_snippets = state.get("memory_snippets", [])
    memory_text = "\n".join(str(m) for m in memory_snippets) if memory_snippets else "No memory context"

    system_prompt = (
        "You are a smart assistant that creates action plans for user requests.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        f"Memory context:\n{memory_text}\n\n"
        "Given the user's request, create a plan of tool calls to fulfill it.\n"
        "Output ONLY a JSON array of steps. Each step must have:\n"
        '  - "tool_name": name of the tool to call (must be one of the available tools)\n'
        '  - "args": dict of arguments for the tool\n'
        '  - "reason": brief reason for this step\n\n'
        "If no tools are needed, output an empty array [].\n\n"
        "IMPORTANT: Output ONLY the JSON array, no other text before or after."
    )
    return system_prompt

def _parse_plan_json(content: str) -> list:
    """Parse JSON plan from LLM response content."""
    import re as _re
    # Try to find JSON array in the content
    match = _re.search(r'\[.*\]', content, _re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            plan = json.loads(json_str)
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass
    
    # If parsing fails, return empty plan
    return []

def my_planner(state: dict) -> dict:
    """Planner node — calls the LLM to produce a plan of tool calls.

    This is a COMPLETELY GENERIC master planner. It has its own prompt,
    its own output format, and ZERO knowledge of GoAT, K, or any goat-specific state keys.

    When wrapped with GoAT:
        - If the classifier decides K==1, this planner runs as-is (GoAT is
          not involved at all).
        - If K>=2, the GoAT wrapper calls build_prompt(state) to get this
          planner's prompt, appends a K-plans instruction, and calls the
          model itself. This planner is never aware of that.

    Outputs:
        - plan: list of step dicts with tool_name, args, reason
    """
    user_input = state.get("user_input", "")
    system_prompt = build_prompt(state)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

    print(f"[PLANNER] Generating plan for: '{user_input}'")
    try:
        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        plan = _parse_plan_json(content)
    except Exception as e:
        print(f"[PLANNER] LLM call failed: {e}")
        plan = []

    print(f"[PLANNER] Plan generated with {len(plan)} step(s)")
    for i, s in enumerate(plan, 1):
        print(f"  {i}. {s.get('tool_name', 'unknown')}(args={s.get('args', {})})")

    return {"plan": plan}


# ── 4. Initialize Store and Model ─────────────────────────────────────────────
from customllm import CustomLLM
store = MarkdownStore("examples/res/memory.md")
model = CustomLLM()


# ── 5. Wrap Planner with GoAT ─────────────────────────────────────────────────

goat_node_config = GoatNodeConfig(
    default_K=3,
    max_K=3,
    store=store,
    output_dir=OUTPUT_PATH,
    render_goat_graph=True,
    render_goat_tele=True,
    render_goat_trace=True,
)

wrapped_planner = GoatNode(
    planner_callable=my_planner,
    model=model,
    config=goat_node_config,
    build_prompt=build_prompt,
)


# ── 6. Executor Node ──────────────────────────────────────────────────────────

def executor_node(state: dict) -> dict:
    """Execute the winning plan steps and capture results."""
    plan = state.get("plan", [])
    step_outputs = []

    for step in plan:
        tool_name = step.get("tool_name") or step.get("tool")
        args = step.get("args", {})
        impl = TOOL_IMPL.get(tool_name)

        if impl:
            try:
                result = impl(args)
                step_outputs.append({"tool": tool_name, "args": args, "ok": True, "result": result})
            except Exception as e:
                step_outputs.append({"tool": tool_name, "args": args, "ok": False, "error": str(e)})
        else:
            step_outputs.append({"tool": tool_name, "ok": False, "error": f"Tool not found: {tool_name}"})

    return {"step_outputs": step_outputs}


# ── 7. Responder Node ─────────────────────────────────────────────────────────

def responder_node(state: dict) -> dict:
    """Generate final response from step outputs."""
    step_outputs = state.get("step_outputs", [])

    if not step_outputs:
        return {"final_answer": "No plan was executed.", "messages": [AIMessage(content="No plan was executed.")]}

    summaries = []
    for step_out in step_outputs:
        if step_out.get("ok"):
            result = step_out.get("result", {})
            summary = result.get("summary", str(result)) if isinstance(result, dict) else str(result)
            summaries.append(f"  - {step_out['tool']}: {summary}")
        else:
            summaries.append(f"  - {step_out['tool']}: Failed ({step_out.get('error', 'Unknown')})")

    answer = "Here's what I did:\n" + "\n".join(summaries)
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


# ── 8. Build the StateGraph ───────────────────────────────────────────────────

def build_custom_graph() -> Any:
    """Build and compile the custom StateGraph with GoAT integration."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", wrapped_planner)
    workflow.add_node("executor", executor_node)
    workflow.add_node("responder", responder_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()


# ── 9. Usage ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("GoAT Low-Level (StateGraph)")
    print("=" * 72)

    app = build_custom_graph()

    user_input = "I am tired."

    initial_state = {
        "user_input": user_input,
        "available_tools": AVAILABLE_TOOLS_DESC,
        "messages": [HumanMessage(content=user_input)],
    }

    print(f"[LOG] Invoking graph with input: '{user_input}'")
    final_state = app.invoke(initial_state)
    print("[LOG] Graph invocation complete")

    # Extract GoatOutput
    goat_output = final_state.get("goat_output")
    goat_graph = goat_output.graph if goat_output else None
    goat_graph_id = goat_graph.graph_id if goat_graph else "unknown"
    goat_winning_plan_id = goat_output.winning_plan_id if goat_output else None
    goat_complexity = goat_output.complexity if goat_output else None
    goat_director_result = goat_output.director_result if goat_output else None
    goat_explanation = goat_output.explanation if goat_output else None

    print(f"[LOG] GoAT Graph ID: {goat_graph_id}")
    print(f"[LOG] Winning Plan ID: {goat_winning_plan_id}")
    print(f"[LOG] Complexity K: {goat_complexity.suggested_K if goat_complexity else 'N/A'}")

    # ── Create output folder ───────────────────────────────────────────────────

    output_dir = Path(OUTPUT_PATH) / goat_graph_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LOG] Created output directory: {output_dir}")

    # ── Build response data ───────────────────────────────────────────────────

    goat_response = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "final_answer": final_state.get("final_answer", ""),
        "goat_graph_id": goat_graph_id,
        **(goat_output.to_dict() if goat_output else {}),
    }

    # ── Save response.json ─────────────────────────────────────────────────────

    response_file = output_dir / "response.json"
    with open(response_file, "w", encoding="utf-8") as f:
        json.dump(goat_response, f, indent=2, default=str, ensure_ascii=False)
    print(f"[LOG] Saved response to: {response_file}")

    # ── Save graph.json ────────────────────────────────────────────────────────

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
