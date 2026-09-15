"""node_wrapper.py — Robust wrapper for integrating GoAT with LangGraph planner nodes.

This module provides a flexible wrapper function `GoatNode` that:
1. Extracts the last human message from LangGraph state
2. Calls the GoAT complexity classifier to determine K
3. Routes execution based on K:
   - K <= 1: Run planner_callable directly (bypass GoAT)
   - K >= 2: Run full GoAT pipeline with modified prompt
4. Injects GoAT outputs into state (goat_director_result, goat_graph, etc.)
5. Provides robust error handling with fallback to normal planner

Usage:
    >>> from langgraph_goat import GoatNodeConfig, GoatNode
    >>>
    >>> goat_node_config = GoatNodeConfig(
    ...     default_K=3,
    ...     max_K=3,
    ...     store=store,
    ... )
    >>>
    >>> wrapped_planner = GoatNode(
    ...     planner_callable=my_planner,
    ...     model=model,
    ...     config=goat_node_config,
    ...     build_prompt=build_prompt,
    ... )
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ._core import (
    AvailableTool,
    Director,
    DirectorResult,
    Plan,
    PlanComplexity,
    PlanStep,
    PlanningContext,
)
from ._core.complexity import LLMClassifier
from ._core.critic import LLMJudgeCritic
from ._core.explanation import LLMExplanationGenerator as CoreLLMExplanationGenerator
from ._core.graph.stores import NetworkXStore
from ._core.interfaces import PlanComplexityClassifier, PlanGenerator
from ._core.utils.logger import GoatLogger, TraceContext, TraceSpan, get_sequence_collector, reset_sequence_collector
from ._core.utils.trace_logger import get_telemetry_collector, reset_telemetry_collector

from ._core.utils.json_extractor import extractJson
from ._core.utils.graph_renderer import renderGraph, renderTele, renderTraceFromCollector

# Import GoatOutput
from .goat_output import GoatOutput

# Import shared utilities from middleware
from .middleware import (
    BaseModelAsLLMClient,
    _retrieve_memory_from_store,
)

log = GoatLogger("langgraph_goat.node_wrapper")


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class GoatNodeConfig:
    """Configuration for the GoAT node wrapper.

    Attributes:
        default_K: Default K value when classifier is not configured or uncertain.
        max_K: Maximum allowed K value (hard cap).
        store: Optional LangGraph store for memory retrieval (read-only).
        classifier: Optional custom complexity classifier. If None, uses LLMClassifier.
        critic: Optional custom critic. If None, uses LLMJudgeCritic.
        explanation_generator: Optional custom explanation generator.
        graph_store: Optional custom graph store. If None, uses NetworkXStore.
        output_dir: Root directory for all GoAT outputs (graphs, telemetry). Defaults to "./output".
        force_K: If set, bypasses classifier and forces this K value.
        callbacks: Optional list of callback handlers to track LLM calls (tokens, latency).
        render_goat_graph: If True, render GoAT graph visualization to output_dir.
        render_goat_tele: If True, render telemetry reports to output_dir.
        render_goat_trace: If True, render execution trace diagram to output_dir.
    """
    default_K: int = 3
    max_K: int = 3
    store: Optional[Any] = None
    classifier: Optional[PlanComplexityClassifier] = None
    critic: Optional[Any] = None
    explanation_generator: Optional[Any] = None
    graph_store: Optional[Any] = None
    output_dir: str = "./output"
    force_K: Optional[int] = None
    callbacks: Optional[list] = None
    render_goat_graph: bool = False
    render_goat_tele: bool = False
    render_goat_trace: bool = False


# ── Planning Context Extraction ───────────────────────────────────────────────

def extract_context_from_state(
    state: dict,
    store: Optional[Any] = None,
) -> PlanningContext:
    """Build a PlanningContext from LangGraph state.
    
    Extracts:
    - user_input: From state["user_input"], state["input"], or last HumanMessage
    - available_tools: From state["available_tools"] or state["tools"]
    - memory_snippets: Retrieved from store if provided
    
    Args:
        state: LangGraph state dict.
        store: Optional LangGraph store for memory retrieval.
    
    Returns:
        PlanningContext with extracted data.
    """
    # Extract user input
    user_input = ""
    if state.get("user_input"):
        user_input = state["user_input"]
    elif state.get("input"):
        user_input = state["input"]
    elif state.get("messages"):
        # Find last human message
        for m in reversed(state["messages"]):
            if isinstance(m, dict) and m.get("role") == "user":
                user_input = m.get("content", "")
                break
            if hasattr(m, "type") and m.type == "human":
                user_input = getattr(m, "content", "")
                break
            if isinstance(m, HumanMessage):
                user_input = m.content
                break
    
    # Extract available tools
    available_tools: list[AvailableTool] = []
    raw_tools = state.get("tools") or state.get("available_tools") or []
    for t in raw_tools:
        if isinstance(t, dict):
            available_tools.append(AvailableTool(
                name=t.get("name", "unknown"),
                description=t.get("description", ""),
                parameters_schema=t.get("parameters", {}) or t.get("args_schema", {}),
            ))
        elif hasattr(t, "name"):
            available_tools.append(AvailableTool(
                name=t.name,
                description=getattr(t, "description", ""),
                parameters_schema=getattr(t, "args_schema", {}) or {},
            ))
    
    # Retrieve memory from store
    memory_snippets = _retrieve_memory_from_store(store, user_input) if store else []
    
    return PlanningContext(
        user_input=user_input,
        available_tools=available_tools,
        memory_snippets=memory_snippets,
        session_history=state.get("session_history") or [],
        metadata={"source": "node_wrapper", "raw_state_keys": list(state.keys())},
    )


# ── Plan Generator for Node Wrapper ───────────────────────────────────────────

class NodeWrapperPlanGenerator(PlanGenerator):
    """PlanGenerator that generates K plans in a SINGLE model call.
    
    This generator:
    1. Builds a prompt from PlanningContext using build_prompt
    2. Appends K-plans instruction with persona diversity hints
    3. Calls the model ONCE to get all K plans in a single JSON response
    4. Parses and returns K Plan objects
    
    Key: The model is asked to return all K plans in one JSON response.
    
    Attributes:
        build_prompt: Function that builds the planner's prompt from state.
        model: BaseChatModel to call for generating K plans (single call).
        callbacks: Optional list of callback handlers for tracking LLM calls.
    """
    
    name = "node_wrapper"
    
    def __init__(
        self,
        build_prompt: Callable[[dict], str],
        model: BaseChatModel,
        callbacks: Optional[list] = None,
    ):
        self.build_prompt = build_prompt
        self.model = model
        self.callbacks = callbacks or []
    
    def generate_K(
        self,
        context: PlanningContext,
        K: int,
        trace_context: Any = None,
    ) -> list[Plan]:
        """Generate K plans in a SINGLE model call.
        
        The model is prompted to return all K plans in one JSON response
        with the structure: {"goat_plans": [plan1, plan2, ..., planK]}
        
        Args:
            context: Planning context with user input, tools, memory.
            K: Number of plans to generate.
            trace_context: Optional trace context for logging.
        
        Returns:
            List of K Plan objects from single model response.
        """
        import uuid
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Build state for prompt generation
        state = {
            "user_input": context.user_input,
            "available_tools": [
                {"name": t.name, "description": t.description}
                for t in context.available_tools
            ],
            "memory_snippets": context.memory_snippets,
        }
        
        # Build base prompt from user's build_prompt function
        base_prompt = self.build_prompt(state)
        
        # Build K-plans instruction - asks for ALL K plans in ONE JSON response
        k_instruction = (
            f"\n\nIMPORTANT: You must create exactly {K} DIVERSE plans in a SINGLE response. "
            f"For implicit requests (e.g., 'I'm hungry'), create plans with DIFFERENT APPROACHES:\n"
            f"  - Plan 1: Quick/immediate approach\n"
            f"  - Plan 2: Thorough/comprehensive approach\n"
            f"  - Plan 3: Alternative/creative approach\n\n"
            f"Return ALL {K} plans in the following JSON format:\n"
            f'{{\n  "goat_plans": [\n'
        )
        for i in range(K):
            k_instruction += (
                f'    {{\n      "plan_name": "Plan {i+1}",\n      '
                f'"plan": [{{"tool": "tool_name", "args": {{}}, "justification": "reason"}}],\n'
                f'      "plan_reasoning": "overall reasoning for this plan"\n    }}'
            )
            if i < K - 1:
                k_instruction += ",\n"
        k_instruction += f'\n  ]\n}}\n\n'

        k_instruction += (
            f"CRITICAL: Return exactly {K} plans in a single JSON object with goat_plans array.\n"
            f"Each plan must be distinct and use different tool combinations, strategies, or approaches.\n"
            f"IMPORTANT: All tool arguments must be STRINGS (e.g., 'duration': '15-minutes', NOT 'duration': 15). "
            f"Strictly return ONLY the JSON object - no other text, no markdown, no code blocks."
        )
        
        modified_prompt = base_prompt + k_instruction

        # Build full context for the human message (tools + memory + user input)
        human_context_parts = []

        # Include available tools
        if context.available_tools:
            human_context_parts.append("AVAILABLE TOOLS:")
            for tool in context.available_tools:
                human_context_parts.append(f"  - {tool.name}: {tool.description}")

        # Include memory context
        if context.memory_snippets:
            human_context_parts.append("\nRELEVANT CONTEXT FROM MEMORY:")
            for snippet in context.memory_snippets[:3]:
                human_context_parts.append(f"  - {snippet}")

        # Add the actual user request
        human_context_parts.append(f"\nUSER REQUEST: {context.user_input}")

        full_human_context = "\n".join(human_context_parts)

        # SINGLE model call - returns all K plans in one response
        messages = [
            SystemMessage(content=modified_prompt),
            HumanMessage(content=full_human_context),
        ]
        
        try:
            # Pass callbacks to track tokens and latency
            config = {"callbacks": self.callbacks} if self.callbacks else None
            response = self.model.invoke(messages, config=config)
            content = response.content if hasattr(response, "content") else str(response)

            log.debug("NodeWrapperPlanGenerator.generate_K: raw response (first 1000 chars):\n%s",
                     content[:1000] if content else "EMPTY")

            # Parse JSON from response using extractJson (robust parser)
            plan_data = extractJson(content)
            log.debug("NodeWrapperPlanGenerator.generate_K: extracted JSON: %s", plan_data)

            # Extract plans from parsed data
            plans: list[Plan] = []
            goat_plans = []

            if isinstance(plan_data, dict):
                goat_plans = plan_data.get("goat_plans", [])
                log.debug("NodeWrapperPlanGenerator.generate_K: found 'goat_plans' key with %d items", len(goat_plans))
            elif isinstance(plan_data, list):
                # Handle case where response is directly a list of plans
                goat_plans = plan_data
                log.debug("NodeWrapperPlanGenerator.generate_K: response is list with %d items", len(goat_plans))
            else:
                log.debug("NodeWrapperPlanGenerator.generate_K: plan_data is %s (type: %s)", plan_data, type(plan_data).__name__)

            if isinstance(goat_plans, list) and goat_plans:
                for i, plan_dict in enumerate(goat_plans[:K]):
                    plan_id = f"plan_{i}_{uuid.uuid4().hex[:6]}"

                    if isinstance(plan_dict, Plan):
                        plan_dict.plan_id = plan_id
                        plans.append(plan_dict)
                    elif isinstance(plan_dict, dict):
                        raw_steps = plan_dict.get("plan", [])
                        steps = []
                        for step_idx, step_data in enumerate(raw_steps):
                            steps.append(PlanStep(
                                step_id=f"{plan_id}_step{step_idx}",
                                tool_name=step_data.get("tool", step_data.get("tool_name", "unknown")),
                                args=step_data.get("args", {}),
                                justification=step_data.get("justification", step_data.get("reason", "")),
                            ))
                        plans.append(Plan(
                            plan_id=plan_id,
                            plan_name=plan_dict.get("plan_name", plan_id),
                            steps=steps,
                            overall_reasoning=plan_dict.get("plan_reasoning", ""),
                        ))
            else:
                log.debug("NodeWrapperPlanGenerator.generate_K: goat_plans is empty or not a list (got %s)", type(goat_plans).__name__)

            log.info("NodeWrapperPlanGenerator: generated %d plans in single model call", len(plans))
            return plans

        except Exception as e:
            import traceback
            log.error("NodeWrapperPlanGenerator: single model call failed: %s", e, exc_info=True)
            log.error("NodeWrapperPlanGenerator: traceback: %s", traceback.format_exc())
            # Return empty plans on error - Director will handle fallback
            return []


# ── Serialization Helpers ─────────────────────────────────────────────────────

def serialize_plan_for_state(plan: Plan) -> list[dict]:
    """Convert a Plan into the raw [{tool, args, justification}] format.
    
    Args:
        plan: Plan object to serialize.
    
    Returns:
        List of step dicts for LangGraph state.
    """
    return [
        {
            "tool": step.tool_name,
            "args": step.args,
            "justification": step.justification,
            "step_id": step.step_id,
            "expected_output": step.expected_output,
        }
        for step in plan.steps
    ]


def create_state_update(
    state: dict,
    director_result: DirectorResult,
    config: Optional[GoatNodeConfig] = None,
) -> dict:
    """Create state update dict from DirectorResult with optional rendering.

    Args:
        state: Original state dict (for base values).
        director_result: Result from Director.run().
        config: Optional GoatNodeConfig to control rendering behavior.

    Returns:
        New state dict with GoAT outputs merged in.
    """
    new_state = dict(state)

    # Create GoatOutput object
    critic_verdict = None
    if hasattr(director_result, "graph") and director_result.graph.critic_verdict:
        critic_verdict = director_result.graph.critic_verdict

    goat_output = GoatOutput(
        graph=director_result.graph,
        winning_plan_id=director_result.winning_plan.plan_id,
        complexity=director_result.complexity,
        explanation=director_result.explanation,
        director_result=director_result,
        critic_verdict=critic_verdict,
    )

    new_state["goat_output"] = goat_output

    # Serialize winning plan for backward compatibility
    new_state["plan"] = serialize_plan_for_state(director_result.winning_plan)

    # Render graph visualization if configured (telemetry is already rendered in GoatNode)
    if config and director_result.graph and config.render_goat_graph:
        output_dir = Path(config.output_dir) / director_result.graph.graph_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            renderGraph(director_result.graph, str(output_dir))
            log.debug("GoatNode: rendered graph to %s", output_dir)
        except Exception as e:
            log.warning("GoatNode: failed to render graph: %s", e)

    return new_state


# ── Main Wrapper Function ─────────────────────────────────────────────────────

def GoatNode(
    planner_callable: Callable[[dict], dict],
    model: BaseChatModel,
    config: GoatNodeConfig,
    build_prompt: Callable[[dict], str],
) -> Callable[[dict], dict]:
    """Wrap a LangGraph planner node with GoAT integration.

    This wrapper:
    1. Extracts the last human message from state
    2. Calls the GoAT complexity classifier to determine K
    3. Routes execution:
       - K <= 1: Run planner_callable directly (bypass GoAT)
       - K >= 2: Run full GoAT pipeline with modified prompt
    4. Injects GoAT outputs into state

    Args:
        planner_callable: The user's planner function (state -> state).
        model: The BaseChatModel to use for classification/critique.
        config: GoatNodeConfig with wrapper settings.
        build_prompt: Function that builds the planner's prompt from state.

    Returns:
        Wrapped planner function that integrates GoAT.
    """
    # Create LLM client adapter from user's model with callbacks for token tracking
    llm_client = BaseModelAsLLMClient(model, callbacks=config.callbacks)
    
    # Create classifier for initial complexity check
    # Pass both max_K and default_K_when_uncertain from config to respect user's K configuration
    classifier = config.classifier or LLMClassifier(
        llm=llm_client,
        max_K=config.max_K,
        default_K_when_uncertain=config.default_K,
    )

    
    @functools.wraps(planner_callable)
    def wrapped(state: dict) -> dict:
        """Synchronous wrapped planner node."""
        user_input = state.get("user_input", "")
        log.debug("GoatNode: user_input='%s'", user_input)
        log.debug("GoatNode: config default_K=%d, max_K=%d, has_store=%s",
                  config.default_K, config.max_K, config.store is not None)

        try:
            # Step 1: Extract planning context
            context = extract_context_from_state(state, store=config.store)

            log.debug("GoatNode: extracted context user_input='%s', "
                      "tools=%d, memory=%d",
                      context.user_input, len(context.available_tools),
                      len(context.memory_snippets))
            
            if not context.user_input.strip():
                log.warning("GoatNode: empty user_input, running planner directly")
                return planner_callable(state)
            
            # Step 2: Classify complexity
            complexity: PlanComplexity
            if config.force_K is not None:
                K = max(1, min(config.force_K, config.max_K))
                complexity = PlanComplexity(
                    needs_multiple_plans=K > 1,
                    suggested_K=K,
                    reason=f"Forced by config (force_K={config.force_K})",
                    confidence=1.0,
                )
                log.debug("GoatNode: using forced K=%d", K)
            else:
                log.debug("GoatNode: calling classifier.classify()")
                complexity = classifier.classify(context)
                K = complexity.suggested_K
                log.debug("GoatNode: classifier returned K=%d, reason=%s", K, complexity.reason)
            
            log.info("GoatNode: K=%d (reason: %s)", K, complexity.reason)
            
            # Step 3: Route based on K
            if K <= 1:
                log.info("GoatNode: K=1, running planner directly (GoAT bypass)")
                result = planner_callable(state)
                # Still add complexity and empty GoatOutput for transparency
                result = dict(result)
                result["goat_complexity"] = complexity
                # Create empty GoatOutput to indicate GoAT didn't run
                result["goat_output"] = GoatOutput(
                    graph=None,
                    winning_plan_id=None,
                    complexity=complexity,
                    explanation=None,
                    director_result=None,
                    critic_verdict=None,
                )
                log.debug("GoatNode: planner returned result with keys=%s", list(result.keys()))
                return result

            # K >= 2: Run full GoAT pipeline
            log.info("GoatNode: K=%d, running GoAT pipeline", K)
            
            # Create plan generator - passes model for single-call K plan generation
            # Pass callbacks to track tokens and latency for GOAT internal calls
            plan_generator = NodeWrapperPlanGenerator(
                build_prompt=build_prompt,
                model=model,
                callbacks=config.callbacks,
            )
            
            # Build Director with components adapted from user's model
            # By default, graph_store is None — the caller is responsible for
            # saving graphs. This avoids creating a redundant "goat_graphs" subfolder.
            # Users can provide a custom graph_store via GoatNodeConfig if needed.
            
            # Create critic with callbacks for token tracking
            critic = config.critic or LLMJudgeCritic(llm=llm_client)
            if config.callbacks:
                critic._callbacks = config.callbacks
            
            # Create explanation generator with callbacks for token tracking
            explanation_generator = config.explanation_generator or CoreLLMExplanationGenerator(llm=llm_client)
            if config.callbacks:
                explanation_generator._callbacks = config.callbacks
            
            graph_store = config.graph_store  # None by default — no redundant goat_graphs folder
            
            director = Director(
                plan_generator=plan_generator,
                critic=critic,
                classifier=None,  # Already classified, skip internal classifier
                explanation_generator=explanation_generator,
                graph_store=graph_store,
                default_K=K,
                max_K=config.max_K,
                force_K=K,  # Force K to bypass internal classification
                root_dir=config.output_dir,  # Used only if graph_store is provided
            )

            
            # Run Director
            # Reset and enable the sequence collector BEFORE running the pipeline
            collector = reset_sequence_collector()
            collector.enable()
            
            trace_ctx = TraceContext()
            director_result: DirectorResult = director.run(context, trace_context=trace_ctx)
            
            # === TELEMETRY RENDERING (independently gated by config flags) ===
            graph_id = director_result.graph.graph_id if director_result.graph else "unknown"
            tele_output_dir = str(Path(config.output_dir) / graph_id)

            # Render telemetry report
            if config.render_goat_tele:
                try:
                    tc = get_telemetry_collector()
                    tc.set_graph(director_result.graph)
                    tc.extract_from_graph(director_result.graph)
                    tc.extract_from_sequence_collector()
                    # Write JSON trace to graph-specific subfolder
                    trace_data = tc.write_tele(output_dir=tele_output_dir)
                    log.info("GoatNode: telemetry JSON written to %s", tele_output_dir)

                    # Render HTML telemetry report
                    if director_result.graph:
                        renderTele(trace_data, director_result.graph.graph_id, tele_output_dir)
                        log.debug("GoatNode: rendered telemetry to %s", tele_output_dir)
                except Exception as e:
                    log.warning("GoatNode: Failed to render telemetry: %s", e)

            # Render trace diagram
            if config.render_goat_trace:
                try:
                    tc = get_telemetry_collector()
                    tc.set_graph(director_result.graph)
                    tc.extract_from_graph(director_result.graph)
                    tc.extract_from_sequence_collector()
                    # Render PlantUML sequence diagram from the SequenceTraceCollector
                    if director_result.graph:
                        renderTraceFromCollector(
                            collector,
                            graph_id=graph_id,
                            path=tele_output_dir,
                            title=f"GOAT Trace - {graph_id}",
                        )
                        log.debug("GoatNode: rendered trace to %s", tele_output_dir)
                except Exception as render_err:
                    log.warning("GoatNode: Failed to render trace: %s", render_err)
            # === END TELEMETRY ===#
            
            # Merge GoAT outputs into state (with optional rendering)
            new_state = create_state_update(state, director_result, config)
            
            log.info(
                "GoatNode: GoAT pipeline complete, winner=%s",
                director_result.winning_plan.plan_id,
            )
            log.debug(
                "GoatNode: returning state with keys=%s, has_graph=%s, has_director_result=%s",
                list(new_state.keys()),
                "goat_graph" in new_state,
                "goat_director_result" in new_state,
            )

            return new_state

            
        except Exception as e:
            # Error fallback: run planner directly and log warning
            log.warning("GoatNode: GoAT pipeline failed (%s), falling back to planner", e)
            result = planner_callable(state)
            result = dict(result)
            result["_goat_error"] = str(e)
            return result
    
    return wrapped


__all__ = [
    "GoatNodeConfig",
    "GoatNode",
    "NodeWrapperPlanGenerator",
    "extract_context_from_state",
    "serialize_plan_for_state",
    "create_state_update",
]
