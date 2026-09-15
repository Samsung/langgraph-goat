"""Pluggable component interfaces.

Every interface is a Protocol — implementations don't have to inherit,
they just have to match the shape. This keeps the plugin honestly
modular: any class with the right methods plugs in.
"""

from __future__ import annotations
from typing import Optional, Protocol, runtime_checkable
from pathlib import Path

from .types import (
    AvailableTool,
    CriticVerdict,
    ExecutionRecord,
    GoATGraph,
    Plan,
    PlanComplexity,
    PlanningContext,
    StepExecutionRecord,
)

# TraceContext is imported as Any for type-hint flexibility — concrete
# implementations may or may not use it, but the slot is always there.
from typing import Any


@runtime_checkable
class PlanComplexityClassifier(Protocol):
    """Decides whether an utterance needs K>1 plans.

    Optional in the Director — if None, K is decided by config alone.
    """

    def classify(
        self,
        context: PlanningContext,
        trace_context: Any = None,
    ) -> PlanComplexity:
        ...


@runtime_checkable
class PlanGenerator(Protocol):
    """Generates one or more candidate plans.

    The adapter's PlanGenerator is what wraps the framework's actual planner
    (LangGraph node, OpenClaw skill, CrewAI planner agent).
    """

    def generate_K(
        self,
        context: PlanningContext,
        K: int,
        trace_context: Any = None,
    ) -> list[Plan]:
        """Generate K plans.
        """
        ...


@runtime_checkable
class PlanCritic(Protocol):
    """Scores K candidate plans and produces rejection justifications.

    Default: LLM-as-judge. Pluggable so users can swap in rule-based,
    embedding-similarity, or human-in-the-loop critics.
    """

    def score(
        self,
        plans: list[Plan],
        context: PlanningContext,
        trace_context: Any = None,
    ) -> CriticVerdict:
        ...


@runtime_checkable
class ExecutionRecorder(Protocol):
    """Adapter reports execution events back to the Director.

    The recorder is the data path from "plan executed" back to "graph
    enriched with what actually happened."
    """

    def record_step(self, plan_id: str, record: StepExecutionRecord) -> None:
        ...

    def finalize_plan(self, plan_id: str, final_answer: str) -> ExecutionRecord:
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Persists / retrieves GoAT graphs.

    Default: NetworkXStore (in-memory). Production: Neo4j, Kùzu, etc.
    """

    def save(self, path: Path, graph: GoATGraph, trace_context: Any = None) -> str:
        """Returns a storage key/id."""
        ...

    def load(self, path: Path, graph_id: str, trace_context: Any = None) -> GoATGraph:
        ...

    def list_recent(self, limit: int = 20, trace_context: Any = None) -> list[str]:
        ...


@runtime_checkable
class ExplanationGenerator(Protocol):
    """Renders a GoAT graph as natural-language explanation."""

    def generate(self, graph: GoATGraph, trace_context: Any = None):  # returns Explanation
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Thin abstraction over LiteLLM so the core never imports providers directly.

    Adapters and components receive an LLMClient — they don't construct one.
    This is what lets users swap Anthropic for Ollama for OpenAI without
    touching component code.
    """

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        trace_context: Any = None,
        **kwargs,
    ) -> str:
        ...
