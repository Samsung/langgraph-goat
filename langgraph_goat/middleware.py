"""langgraph_goat middleware — LangChain `create_agent` integration for GoAT.

This module provides middleware-based integration for LangChain's `create_agent` API.
It wraps the model call to inject GoAT's explainability pipeline without modifying
the agent's core behavior.

Two usage patterns:
    1. High-Level (create_agent): Use GoatMiddleware with middleware=[...]
    2. Low-Level (StateGraph): Use GoatNode() from node_wrapper.py
"""

from __future__ import annotations

import time
import json
import logging
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.runtime import Runtime

from ._core import (
    AvailableTool,
    CriticVerdict,
    Director,
    DirectorResult,
    Plan,
    PlanComplexity,
    PlanStep,
    PlanningContext,
    Explanation,
    GoATGraph,
)
from ._core.interfaces import (
    PlanComplexityClassifier,
    PlanCritic,
    PlanGenerator,
    ExplanationGenerator,
    GraphStore,
    LLMClient,
)
from ._core.complexity import LLMClassifier
from ._core.critic import LLMJudgeCritic
from ._core.explanation import LLMExplanationGenerator as CoreLLMExplanationGenerator
from ._core.graph import GraphBuilder
from ._core.graph.stores import NetworkXStore
from ._core.utils.logger import GoatLogger, TraceContext, TraceSpan, get_sequence_collector, reset_sequence_collector
from ._core.utils.trace_logger import get_telemetry_collector, reset_telemetry_collector
from ._core.utils.json_extractor import extractJson
from ._core.utils.graph_renderer import renderTele
from .goat_output import GoatOutput

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from typing_extensions import NotRequired, Annotated, TypedDict

from langchain.agents.middleware.types import OmitFromSchema

log = GoatLogger("langgraph_goat.middleware")

# ── Type annotations for state schema ─────────────────────────────────────────

# Omit from input (user doesn't provide) but include in output (user can access)
GoatStateAttr = OmitFromSchema(input=True, output=False)


# ── LLM Client Adapter ────────────────────────────────────────────────────────

class BaseModelAsLLMClient:
    """Adapts langchain_core BaseChatModel → the GoAT engine LLMClient protocol.
    
    This allows engine components (classifier, critic) to use the agent's
    own model without creating a new LLM instance.
    
    Includes telemetry recording for LLM calls.
    
    Attributes:
        _model: The wrapped langchain BaseChatModel.
        _callbacks: Optional list of callback handlers for token/latency tracking.
    """
    
    def __init__(self, model: BaseChatModel, callbacks: Optional[list] = None):
        self._model = model
        self._callbacks = callbacks or []
    
    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        trace_context: Any = None,
        **kwargs: Any,
    ) -> str:
        """Call the wrapped model with the engine message format.
        
        Records telemetry including token usage, latency, model config,
        prompts, and completions.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Ignored (uses wrapped model).
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            trace_context: Optional trace context for correlation.
            **kwargs: Additional model kwargs.
        
        Returns:
            The model's response content as a string.
        """
        start_time = time.perf_counter()
        
        # Convert goat message format to langchain format
        lc_messages = self._convert_messages(messages)
        
        # Bind parameters and invoke with callbacks if available
        bind_kwargs = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}
        bound_model = self._model.bind(**bind_kwargs)
        
        # Invoke with callbacks via config (not through bind to avoid conflicts)
        invoke_config = {"callbacks": self._callbacks} if self._callbacks else None
        response = bound_model.invoke(lc_messages, config=invoke_config)
        
        elapsed = time.perf_counter() - start_time
        
        # Record telemetry - wrap AIMessage in a ModelResponse-like dict structure
        try:
            tc = get_telemetry_collector()
            
            # Extract token usage from Ollama/LangChain response
            # Ollama returns tokens in response_metadata with various possible field names
            usage_data = getattr(response, "response_metadata", {}) or {}
            
            # Try multiple possible field names for Ollama token counts
            prompt_tokens = (
                usage_data.get("prompt_tokens", 0) or 
                usage_data.get("prompt_eval_count", 0) or
                usage_data.get("eval_count", 0) or  # Some Ollama versions use this for prompt
                0
            )
            completion_tokens = (
                usage_data.get("completion_tokens", 0) or 
                usage_data.get("eval_count", 0) or  # Some Ollama versions use this for completion
                0
            )
            # If we still don't have tokens, estimate from content
            if prompt_tokens == 0 and completion_tokens == 0:
                # Rough estimation: ~4 chars per token for English text
                prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4 if messages else 0
                completion_tokens = len(response.content) // 4 if response.content else 0
            
            total_tokens = prompt_tokens + completion_tokens
            
            # Create a ModelResponse-like dict structure from AIMessage
            model_response_dict = {
                "id": getattr(response, "id", str(uuid.uuid4())),
                "choices": [{
                    "message": {
                        "content": response.content,
                        "tool_calls": getattr(response, "tool_calls", None),
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            tc.record_llm_call(
                model_response=model_response_dict,
                messages=[{"role": m.__class__.__name__.replace("Message", "").lower(), "content": m.content} 
                          for m in lc_messages if hasattr(m, "content")],
                model=getattr(self._model, "model_name", "") or getattr(self._model, "model", ""),
                temperature=temperature,
                max_tokens=max_tokens,
                latency_total=elapsed,
                latency_inference=elapsed,
            )
        except Exception as e:
            log.debug("Failed to record LLM telemetry: %s", e)
        
        return response.content
    
    def _convert_messages(self, messages: list[dict]) -> list[BaseMessage]:
        """Convert goat message format to langchain BaseMessage list."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))
            else:
                # Default to human message for unknown roles
                result.append(HumanMessage(content=content))
        
        return result


# ── Middleware Configuration ──────────────────────────────────────────────────

@dataclass
class GoatMiddlewareConfig:
    """User-configurable settings for GoatMiddleware.

    All fields are optional. Defaults are created from the agent's model
    when the middleware is first invoked.
    """

    # Director components (all optional — defaults created from agent's model)
    classifier: Optional[PlanComplexityClassifier] = None
    critic: Optional[PlanCritic] = None
    explanation_generator: Optional[ExplanationGenerator] = None
    graph_store: Optional[GraphStore] = None

    # LangGraph store reference for memory retrieval (GoAT reads only — never writes)
    store: Optional[Any] = None

    # Director configuration
    default_K: int = 3
    max_K: int = 3
    force_K: Optional[int] = None

    # Planning prompt customization
    planning_prompt_template: Optional[str] = None

    # Output directory for all GoAT artifacts (graphs, telemetry, responses)
    # If not specified, defaults to "./output" in the current working directory
    output_dir: str = "./output"

    # Rendering options
    render_goat_graph: bool = False
    render_goat_tele: bool = False
    render_goat_trace: bool = False

    # Callbacks for token/latency tracking
    callbacks: Optional[list] = None

    def __post_init__(self):
        """Validate output directory at config creation time."""
        out_path = Path(self.output_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise ValueError(
                f"Cannot write to output_dir '{self.output_dir}': {e}. "
                "Ensure directory path is valid and you have write permissions."
            ) from e


# ── Plan Generator for Middleware Context ─────────────────────────────────────

class MiddlewarePlanGenerator(PlanGenerator):
    """PlanGenerator that calls the agent's model with a planning prompt.
    
    This generator is designed for the middleware context where there's no
    pre-existing planner node. It generates K plans by calling the model
    once with a prompt that asks for K diverse plans.
    
    Attributes:
        model: The BaseChatModel to use for planning.
        messages: The conversation history (excluding system message).
        system_message: Optional system message.
        tools: Available tools for planning.
    """
    
    name = "middleware"
    
    def __init__(self):
        self._model: Optional[BaseChatModel] = None
        self._messages: list[BaseMessage] = []
        self._system_message: Optional[SystemMessage] = None
        self._tools: list[BaseTool | dict[str, Any]] = []
        self._prompt_template: Optional[str] = None
    
    def set_request_context(
        self,
        model: BaseChatModel,
        messages: list[BaseMessage],
        system_message: Optional[SystemMessage],
        tools: list[BaseTool | dict[str, Any]],
        prompt_template: Optional[str] = None,
    ) -> None:
        """Set the model context before Director.run().
        
        Args:
            model: The agent's BaseChatModel.
            messages: Conversation history (excluding system message).
            system_message: Optional system message.
            tools: Available tools.
            prompt_template: Optional custom planning prompt template.
        """
        self._model = model
        self._messages = list(messages)
        self._system_message = system_message
        self._tools = list(tools)
        self._prompt_template = prompt_template
    
    def generate_K(
        self,
        context: PlanningContext,
        K: int,
        trace_context: Optional[TraceContext] = None,
    ) -> list[Plan]:
        """Generate K plans in a single model call.
        
        Args:
            context: Planning context with user input, tools, memory.
            K: Number of plans to generate.
            trace_context: Optional trace context for logging.
        
        Returns:
            List of K Plan objects.
        """
        if self._model is None:
            raise RuntimeError("set_request_context() must be called before generate_K()")
        
        # Build planning prompt
        prompt = self._build_planning_prompt(context, K)
        
        # Build messages list
        messages = list(self._messages)
        if self._system_message:
            messages = [self._system_message] + messages
        messages.append(HumanMessage(content=prompt))
        
        # Call model
        log.info("MiddlewarePlanGenerator: calling model for K=%d plans", K)
        bound_model = self._model.bind(temperature=0.7, max_tokens=2024)
        response = bound_model.invoke(messages)
        
        # Debug: Log raw response for troubleshooting
        log.info("MiddlewarePlanGenerator: raw response (first 500 chars): %s", response.content[:500] if response.content else "EMPTY")
        
        # Parse K plans from response
        plan_data = extractJson(response.content)
        log.info("MiddlewarePlanGenerator: extracted JSON: %s", plan_data)
        plans = self._parse_plans(plan_data, K)
        
        log.info("MiddlewarePlanGenerator: parsed %d plans", len(plans))
        return plans
    
    def _build_planning_prompt(
        self,
        context: PlanningContext,
        K: int,
    ) -> str:
        """Build the K-plans prompt for the model.

        This is ONLY called when K >= 2 (the middleware short-circuits K==1
        before the Director is ever invoked). It builds a prompt that asks
        the model to generate K diverse plans.

        Args:
            context: Planning context.
            K: Number of plans to generate (always >= 2).

        Returns:
            The planning prompt as a string.
        """
        # Get tool descriptions
        tool_names = []
        for t in context.available_tools:
            tool_names.append(f"- {t.name}: {t.description}")
        tools_desc = "\n".join(tool_names) if tool_names else "No tools available"

        # Get memory
        memory_text = "\n".join(context.memory_snippets) if context.memory_snippets else "No memory context"

        # Check if custom template is provided
        if self._prompt_template:
            return self._prompt_template.format(
                user_input=context.user_input,
                tools=tools_desc,
                memory=memory_text,
                K=K,
            )

        prompt = f"""
You are an AI assistant that creates action plans based on user requests.

Available tools:
{tools_desc}

Memory context:
{memory_text}

User request: {context.user_input}

You must create exactly {K} diverse plans.

Return a JSON object with the following structure:
{{
    "goat_plans": [
        {{
            "plan_name": "Unique plan name",
            "plan": [
                {{"tool": "tool_name", "args": {{"arg1": "value1"}}, "justification": "reason for this step"}},
                ...
            ],
            "plan_reasoning": "overall reasoning for this plan"
        }},
        ... (exactly {K} plans)
    ]
}}

Strictly return only the parsable JSON object with exactly {K} plans.
Each plan must be distinct and reflect its assigned persona/context.
""".strip()

        return prompt
    
    def _parse_plans(
        self,
        plan_data: dict,
        K: int,
    ) -> list[Plan]:
        """Parse K plans from the model's JSON response.
        
        Args:
            plan_data: Parsed JSON from model response.
            K: Expected number of plans.
        
        Returns:
            List of Plan objects.
        """
        plans = []
        
        # Handle None or empty response from model
        if not plan_data:
            log.warning("_parse_plans: received empty/None plan_data")
            return plans
        
        # Handle goat_plans list format (K > 1)
        if "goat_plans" in plan_data:
            plans_list = plan_data["goat_plans"]
            for i, plan_dict in enumerate(plans_list[:K]):
                plan = self._parse_single_plan(plan_dict, i)
                plans.append(plan)
        # Handle single plan format (K = 1)
        elif "plan" in plan_data:
            plan = self._parse_single_plan(plan_data, 0)
            plans.append(plan)
        
        # If we got fewer plans than K, that's okay - Director will handle it
        # For K>1, if we only got 1 plan, Director will use K=1 path (no critic)
        # This is better than creating stub plans that have no real content
        
        return plans
    
    def _parse_single_plan(
        self,
        plan_dict: dict,
        index: int,
    ) -> Plan:
        """Parse a single plan from a dict."""
        import uuid
        
        plan_id = f"plan_{index}_{uuid.uuid4().hex[:6]}"
        plan_name = plan_dict.get("plan_name", plan_id)
        plan_reasoning = plan_dict.get("plan_reasoning", "")
        
        # Parse steps
        steps = []
        raw_steps = plan_dict.get("plan", [])
        for step_idx, step_dict in enumerate(raw_steps):
            step = PlanStep(
                step_id=f"{plan_id}_step{step_idx}",
                tool_name=step_dict.get("tool", "unknown"),
                args=step_dict.get("args", {}),
                expected_output=step_dict.get("expected_output"),
                justification=step_dict.get("justification", ""),
            )
            steps.append(step)
        
        return Plan(
            plan_id=plan_id,
            plan_name=plan_name,
            steps=steps,
            overall_reasoning=plan_reasoning,
        )


# ── Context Extraction ────────────────────────────────────────────────────────

def extract_context_from_request(
    request: ModelRequest,
    store: Optional[Any] = None,
) -> PlanningContext:
    """Extract PlanningContext from a ModelRequest.
    
    Reads memory from the provided store (if any) to populate
    memory_snippets for GoAT's planning pipeline. GoAT never writes
    to the store — it only reads user-provided context.
    
    Args:
        request: The ModelRequest from LangChain middleware.
        store: Optional LangGraph store reference for memory retrieval.
            Must support .search(namespace_tuple, query=str) or
            .get(namespace_tuple, key=str).
    
    Returns:
        PlanningContext with user input, tools, and memory snippets.
    """
    # Extract user input from last message
    user_input = ""
    if request.messages:
        last_msg = request.messages[-1]
        if hasattr(last_msg, "content"):
            user_input = str(last_msg.content)
    
    # Extract available tools
    available_tools = []
    for t in request.tools:
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
    
    # Extract memory from store (GoAT reads only — never writes)
    memory_snippets = _retrieve_memory_from_store(store, user_input)
    
    return PlanningContext(
        user_input=user_input,
        available_tools=available_tools,
        memory_snippets=memory_snippets,
        session_history=[],
        metadata={"source": "middleware"},
    )


def _retrieve_memory_from_store(
    store: Optional[Any],
    query: str,
    namespaces: Optional[list[tuple]] = None,
) -> list[str]:
    """Retrieve memory snippets from a LangGraph store (synchronous).

    GoAT only reads from the store — it never writes. The store is
    provided by the user via GoatMiddlewareConfig.store.

    Supports:
        - LangGraph InMemoryStore / PostgresStore with .search() and .get()
        - Any store with .search(namespace, query=...) or .get(namespace, key=...)

    Args:
        store: LangGraph store instance (or any compatible store).
        query: The user input to search for relevant memories.
        namespaces: Optional list of namespace tuples to search in.
            Defaults to [("preferences",), ("memory",), ("context",)].

    Returns:
        List of memory snippet strings for GoAT's planning context.
    """
    if store is None:
        return []

    if namespaces is None:
        namespaces = [("preferences",), ("memory",), ("context",)]

    snippets: list[str] = []

    for namespace in namespaces:
        try:
            # Try .search() first (semantic/keyword search)
            if hasattr(store, "search"):
                results = store.search(namespace, query=query)
                for item in results:
                    # Item may be an Item object with .value, or a dict
                    if hasattr(item, "value"):
                        val = item.value
                    elif isinstance(item, dict):
                        val = item.get("value", item.get("data", str(item)))
                    else:
                        val = str(item)
                    if isinstance(val, dict):
                        snippets.append(f"[{namespace}] {val}")
                    else:
                        snippets.append(f"[{namespace}] {val}")
            # Fallback: try .get() with the query as key
            elif hasattr(store, "get"):
                item = store.get(namespace, query)
                if item is not None:
                    if hasattr(item, "value"):
                        val = item.value
                    elif isinstance(item, dict):
                        val = item.get("value", item.get("data", str(item)))
                    else:
                        val = str(item)
                    if isinstance(val, dict):
                        snippets.append(f"[{namespace}] {val}")
                    else:
                        snippets.append(f"[{namespace}] {val}")
        except Exception as e:
            log.debug("Memory retrieval from namespace %s failed: %s", namespace, e)

    if snippets:
        log.info("Retrieved %d memory snippet(s) from store for query: '%s'", len(snippets), query[:80])

    return snippets

# ── Plan to ModelResponse Conversion ──────────────────────────────────────────

def plan_to_tool_calls(plan: Plan) -> list[dict]:
    """Convert a Plan to a list of tool_call dicts for AIMessage.
    
    Args:
        plan: The Plan to convert.
    
    Returns:
        List of tool_call dicts compatible with AIMessage.
    """
    tool_calls = []
    for step in plan.steps:
        tool_call = {
            "id": step.step_id,
            "name": step.tool_name,
            "args": step.args or {},
        }
        tool_calls.append(tool_call)
    return tool_calls


def plan_to_model_response(plan: Plan) -> ModelResponse:
    """Convert a winning Plan to a ModelResponse.
    
    Args:
        plan: The winning plan.
    
    Returns:
        ModelResponse with AIMessage containing tool_calls.
    """
    tool_calls = plan_to_tool_calls(plan)
    
    # Create AIMessage with tool_calls
    ai_message = AIMessage(
        content="",  # Empty content since we have tool_calls
        tool_calls=tool_calls,
    )
    
    return ModelResponse(
        result=[ai_message],
        structured_response=None,
    )


# ── Goat Middleware State Schema ──────────────────────────────────────────────

class GoatAgentState(TypedDict, total=False):
    """Extended AgentState with GoAT explainability fields.
    
    All goat_* fields are marked with OmitFromInput so they don't
    need to be provided by the user, but ARE included in output.
    """
    # GoAT outputs - omitted from input but included in output
    goat_graph: NotRequired[Annotated[Any, GoatStateAttr]]
    goat_winning_plan_id: NotRequired[Annotated[str, GoatStateAttr]]
    goat_complexity: NotRequired[Annotated[Any, GoatStateAttr]]
    goat_explanation: NotRequired[Annotated[Any, GoatStateAttr]]
    goat_director_result: NotRequired[Annotated[Any, GoatStateAttr]]


# ── Goat Middleware Implementation ────────────────────────────────────────────

class GoatMiddleware(AgentMiddleware[GoatAgentState, Any]):
    """LangChain AgentMiddleware that integrates GoAT explainability.
    
    This middleware intercepts the model call and runs the GoAT pipeline:
    1. Extract PlanningContext from the request
    2. Classify complexity → determine K
    3. Generate K plans (single model call)
    4. Critic selects winner (if K>1)
    5. Build GoAT graph
    6. Generate explanation
    7. Convert winner plan to ModelResponse
    8. Return ModelResponse (agent continues normally)
    
    The goat_output are injected into the agent state via after_model hook.
    """
    
    state_schema = GoatAgentState
    
    def __init__(self, config: Optional[GoatMiddlewareConfig] = None):
        """Initialize GoatMiddleware.

        Args:
            config: Optional configuration. Defaults are used if not provided.
        """
        self.config = config or GoatMiddlewareConfig()
        self._plan_generator = MiddlewarePlanGenerator()
        self._director: Optional[Director] = None
        self._pending_result: Optional[DirectorResult] = None
        self._pending_complexity: Optional[PlanComplexity] = None
        self.last_goat_output: Optional[GoatOutput] = None
    
    def _get_or_create_director(self, model: BaseChatModel) -> Director:
        """Lazy init Director with model-adapted LLMClient.
        
        Args:
            model: The agent's BaseChatModel.
        
        Returns:
            Configured Director instance.
        """
        if self._director is None:
            # Create LLM client adapter from agent's model with callbacks for token tracking
            llm_client = BaseModelAsLLMClient(model, callbacks=self.config.callbacks)
            
            # Create default components if not provided
            classifier = self.config.classifier
            if classifier is None:
                # Pass both max_K and default_K_when_uncertain from config
                # This ensures the classifier respects the user's K configuration
                classifier = LLMClassifier(
                    llm=llm_client,
                    max_K=self.config.max_K,
                    default_K_when_uncertain=self.config.default_K,
                )

            
            critic = self.config.critic
            if critic is None:
                critic = LLMJudgeCritic(llm=llm_client)
            
            explanation_generator = self.config.explanation_generator
            if explanation_generator is None:
                # Use actual LLM explanation generator for real explanations
                explanation_generator = CoreLLMExplanationGenerator(llm=llm_client)
            
            # Graph store for persisting GoAT graphs.
            # By default, graph_store is None — the agent script's save_goat_logs()
            # function already saves the graph as graph.json in the output folder.
            # This avoids creating a redundant "goat_graphs" subfolder.
            # Users can still provide a custom graph_store via GoatMiddlewareConfig
            # if they want the Director to persist graphs separately.
            graph_store = self.config.graph_store  # None by default — no redundant goat_graphs folder
            
            self._director = Director(
                plan_generator=self._plan_generator,
                critic=critic,
                classifier=classifier,
                explanation_generator=explanation_generator,
                graph_store=graph_store,
                default_K=self.config.default_K,
                max_K=self.config.max_K,
                force_K=self.config.force_K,
                root_dir=self.config.output_dir,  # Used only if graph_store is provided
            )

            
            log.info("GoatMiddleware: Director initialized with max_K=%d", self.config.max_K)
        
        return self._director
    
    def _reset_telemetry_for_new_run(self) -> None:
        """Reset telemetry collector and sequence collector for a new run."""
        from ._core.utils.trace_logger import reset_telemetry_collector
        from ._core.utils.logger import reset_sequence_collector
        reset_telemetry_collector()
        collector = reset_sequence_collector()
        collector.enable()  # Enable the sequence collector to record events
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage | Any:
        """Intercept model call and conditionally run GoAT pipeline.

        The middleware classifies complexity ONCE (using the classifier directly,
        NOT through the Director):
            - K == 1: the model works INDEPENDENTLY of GoAT. The handler is
              called directly — no Director, no graph, no explanation, no goat_*
              fields. GoAT is not involved at all.
            - K >= 2: the full GoAT pipeline runs (generate K plans → critic →
              graph → explanation). The Director's own classifier is bypassed
              via force_K to avoid double-classification.

        GoAT pipeline ONLY runs on initial user requests (HumanMessage).
        When the last message is a ToolMessage, the agent is generating
        the final response after tool execution — let it proceed normally.

        Args:
            request: ModelRequest with model, messages, tools, etc.
            handler: Callback to execute normal model call (used as fallback
                and for K==1 path).

        Returns:
            ModelResponse with winner plan as tool_calls, or handler result.
        """
        # Reset telemetry and sequence collectors for a fresh run
        self._reset_telemetry_for_new_run()
        
        # Create trace context at the entry point
        trace_ctx = TraceContext()
        
        with TraceSpan(
            "middleware.intercept",
            caller="LangChain",
            callee="GoatMiddleware",
            trace_context=trace_ctx,
            package="adapter",
        ) as span:
            try:
                # Check if this is a tool response turn (last message is ToolMessage)
                # If so, let the model generate the final response normally
                if request.messages:
                    last_msg = request.messages[-1]
                    if isinstance(last_msg, ToolMessage):
                        log.debug("ToolMessage detected — letting model generate final response", trace_context=trace_ctx)
                        return handler(request)

                # Step 1: Extract PlanningContext (pass store for memory retrieval)
                context = extract_context_from_request(request, store=self.config.store)

                if not context.user_input.strip():
                    log.debug("Empty user input — skipping GoAT pipeline", trace_context=trace_ctx)
                    return handler(request)

                # Step 2: Classify complexity ONCE (in the middleware, not through Director)
                director = self._get_or_create_director(request.model)
                classifier = director.classifier
                if classifier is None:
                    # No classifier configured — use default_K
                    K = max(1, min(director.default_K, director.max_K))
                    complexity = PlanComplexity(
                        needs_multiple_plans=K > 1,
                        suggested_K=K,
                        reason=f"No classifier configured. Using default_K={director.default_K}.",
                        confidence=1.0,
                    )
                else:
                    with TraceSpan(
                        "classify_complexity",
                        caller="GoatMiddleware",
                        callee="LLMClassifier",
                        trace_context=trace_ctx,
                        package="core",
                    ):
                        complexity = classifier.classify(context)
                    K = complexity.suggested_K

                log.info(
                    "GoatMiddleware: classified K=%d (reason: %s, confidence=%.2f)",
                    K, complexity.reason, complexity.confidence,
                    trace_context=trace_ctx,
                )

                # Step 3: If K == 1, let the model work INDEPENDENTLY of GoAT
                # But still store complexity for goat_output in after_model hook
                if K <= 1:
                    log.info("GoatMiddleware: K=1, letting model work independently (no goat pipeline)", trace_context=trace_ctx)
                    self._pending_complexity = complexity
                    return handler(request)

                # Step 4: K >= 2 — run the full GoAT pipeline
                # Set up plan generator with current context
                self._plan_generator.set_request_context(
                    model=request.model,
                    messages=request.messages,
                    system_message=request.system_message,
                    tools=request.tools,
                    prompt_template=self.config.planning_prompt_template,
                )

                # Set force_K so Director skips its own classification (no double-classify)
                director.force_K = K

                # Run Director pipeline with trace span
                with TraceSpan(
                    "director.run",
                    caller="GoatMiddleware",
                    callee="Director",
                    trace_context=trace_ctx,
                    package="core",
                ) as director_span:
                    result = director.run(context)
                    # Update trace context with graph_id if available
                    if result and result.graph:
                        trace_ctx.graph_id = result.graph.graph_id

                # If GoAT produced 0 plans, fall back to normal model call
                # The model will create and execute its own plan anyway
                if not result.all_plans:
                    log.warning("GoatMiddleware: GoAT produced 0 plans — falling back to normal model call", trace_context=trace_ctx)
                    return handler(request)

                # Store result for after_model hook
                self._pending_result = result

                log.info(
                    "GoatMiddleware: pipeline complete, winner=%s, K=%d",
                    result.winning_plan.plan_id,
                    result.complexity.suggested_K,
                    trace_context=trace_ctx,
                )

                # Step 5: Convert winner plan to ModelResponse (tool calls for agent to execute)
                model_response = plan_to_model_response(result.winning_plan)

                return model_response

            except Exception as e:
                log.warning("GoAT pipeline failed: %s. Falling back to normal model call.", e, trace_context=trace_ctx)
                # Record error for telemetry
                try:
                    tc = get_telemetry_collector()
                    tc.record_error(
                        api_failure=str(e),
                        retries=0,
                    )
                except Exception:
                    pass  # Don't fail if telemetry recording fails
                # Step 6 (fallback): Use normal model call
                return handler(request)

    
    def after_model(
        self,
        state: GoatAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inject GoAT outputs into agent state and write telemetry.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State updates with goat_* fields, or None if no GoAT result.
        """
        result = self._pending_result
        complexity = self._pending_complexity
        self._pending_result = None
        self._pending_complexity = None

        # Handle K >= 2 case (full GoAT pipeline ran)
        if result is not None:
            # Create GoatOutput object
            critic_verdict = None
            if hasattr(result, "graph") and result.graph.critic_verdict:
                critic_verdict = result.graph.critic_verdict

            goat_output = GoatOutput(
                graph=result.graph,
                winning_plan_id=result.winning_plan.plan_id,
                complexity=result.complexity,
                explanation=result.explanation,
                director_result=result,
                critic_verdict=critic_verdict,
            )
        # Handle K == 1 case (no full pipeline, but still have complexity)
        elif complexity is not None:
            goat_output = GoatOutput(
                graph=None,
                winning_plan_id=None,
                complexity=complexity,
                explanation=None,
                director_result=None,
                critic_verdict=None,
            )
        else:
            # No GoAT result at all
            return None

        # Store goat_output on middleware instance for user access after invocation
        self.last_goat_output = goat_output

        updates: dict[str, Any] = {
            "goat_output": goat_output,
        }

        # Render outputs based on config flags (only for K >= 2, where result.graph exists)
        if result is not None and result.graph:
            graph_id = result.graph.graph_id
            output_subdir = str(Path(self.config.output_dir) / graph_id)

            # Render graph visualization
            if self.config.render_goat_graph:
                try:
                    from ._core.utils.graph_renderer import renderGraph as render_goat_graph
                    render_goat_graph(result.graph, output_subdir)
                    log.debug("GoatMiddleware: rendered graph to %s", output_subdir)
                except Exception as e:
                    log.warning("GoatMiddleware: failed to render graph: %s", e)

            # Render telemetry
            if self.config.render_goat_tele:
                try:
                    tc = get_telemetry_collector()
                    tc.set_graph(result.graph)
                    tc.extract_from_graph(result.graph)
                    tc.extract_from_sequence_collector()
                    trace_data = tc.write_tele(output_dir=output_subdir)
                    renderTele(trace_data, result.graph.graph_id, output_subdir)
                    log.debug("GoatMiddleware: rendered telemetry to %s", output_subdir)
                except Exception as e:
                    log.warning("GoatMiddleware: failed to render telemetry: %s", e)

            # Render trace diagram
            if self.config.render_goat_trace:
                try:
                    from ._core.utils.graph_renderer import renderTraceFromCollector
                    collector = get_sequence_collector()
                    if renderTraceFromCollector(
                        collector,
                        graph_id=graph_id,
                        path=output_subdir,
                        title=f"GOAT Trace - {graph_id}",
                    ):
                        log.debug("GoatMiddleware: rendered trace to %s", output_subdir)
                except Exception as e:
                    log.warning("GoatMiddleware: failed to render trace: %s", e)


        if result is not None:
            log.debug("GoatMiddleware: injected goat_* fields into state (K >= 2)")
        else:
            log.debug("GoatMiddleware: injected goat_* fields into state (K == 1, complexity only)")
        return updates

__all__ = [
    "GoatMiddleware",
    "GoatMiddlewareConfig",
    "GoatAgentState",
    "BaseModelAsLLMClient",
    "MiddlewarePlanGenerator",
    "extract_context_from_request",
    "plan_to_model_response",
]
