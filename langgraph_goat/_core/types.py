"""Core types for the GoAT engine.

These dataclasses are the data contract between the framework adapter,
the GoAT engine, and downstream consumers (graph viewers, evaluators, etc.).
Adapters convert their native plan/execution objects to/from these types.

Deliberately framework-agnostic — no LangGraph, OpenClaw, CrewAI imports allowed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Literal


# ── Planning context ─────────────────────────────────────────────────────────


@dataclass
class AvailableTool:
    """A tool/sub-agent the planner can invoke.

    Adapters populate this from their framework's tool registry.
    """
    name: str
    description: str
    parameters_schema: dict = field(default_factory=dict)
    cost_hint: Optional[str] = None  # "fast"/"slow"/"expensive" — informs critic


@dataclass
class PlanningContext:
    """Everything the planner+critic need beyond the user input.

    This is what makes plan generation reproducible — same input + same
    context = same diversity space.
    """
    user_input: str
    available_tools: list[AvailableTool] = field(default_factory=list)
    memory_snippets: list[str] = field(default_factory=list)  # retrieved from LTM
    session_history: list[dict] = field(default_factory=list)  # prior turns
    metadata: dict = field(default_factory=dict)  # adapter-specific extras


# ── Plan & PlanStep ──────────────────────────────────────────────────────────


@dataclass
class PlanStep:
    """One step of a candidate plan.

    Steps are intent-level: 'check weather' not 'call weather.json's get() function'.
    The adapter resolves intent to actual framework calls.
    """
    step_id: str
    tool_name: str  # must match an AvailableTool.name
    args: dict = field(default_factory=dict)
    expected_output: Optional[str] = None
    justification: str = ""  # why this step, why now


@dataclass
class Plan:
    """A candidate plan: ordered sequence of steps + reasoning."""
    plan_id: str  # unique within a single Director.run() invocation
    plan_name: str
    steps: list[PlanStep] = field(default_factory=list)
    overall_reasoning: str = ""  # planner's CoT explaining the plan


# ── Plan complexity classification ───────────────────────────────────────────


@dataclass
class PlanComplexity:
    """Output of the PlanComplexityClassifier — does this need K>1 plans?"""
    needs_multiple_plans: bool
    suggested_K: int  # 1 if needs_multiple_plans is False, else 2-K_max
    reason: str  # natural-language reason — surfaces in the graph
    confidence: float = 1.0  # 0-1


# ── Critic output ────────────────────────────────────────────────────────────


@dataclass
class CriticVerdict:
    """The Plan Critic's output for K plans (only used when K>1)."""
    winner_plan_id: str
    scores: dict[str, float]  # plan_id → score (0-10)
    rejection_justifications: dict[str, str]  # plan_id → why it lost
    overall_reasoning: str = ""
    judge_model: Optional[str] = None  # for bias tracking later


# ── Execution recording ──────────────────────────────────────────────────────


@dataclass
class StepExecutionRecord:
    """Adapter reports back what actually happened when a step ran.

    This is what makes GoAT a reasoning record, not a planning record —
    we capture both predicted and actual outcomes.
    """
    step_id: str
    started_at: datetime
    completed_at: datetime
    success: bool
    output: Any = None
    output_summary: str = ""
    error: Optional[str] = None


@dataclass
class ExecutionRecord:
    """Aggregate record for a full plan execution."""
    plan_id: str
    steps: list[StepExecutionRecord] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: int = 0
    tokens_used: int = 0


# ── GoAT graph types ─────────────────────────────────────────────────────────


class NodeType(str, Enum):
    DIRECTOR = "director"
    PLAN_STEP = "plan_step"
    MEMORY_LOAD = "memory_load"
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    GHOST = "ghost"


class EdgeKind(str, Enum):
    CAUSAL_FLOW = "causal_flow"
    MEMORY_RETRIEVAL = "memory_retrieval"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class GoATNode:
    """A node in the GoAT graph."""
    node_id: str
    node_type: NodeType
    label: str
    reasoning: str = ""  # the "why" attached to this node
    payload: dict = field(default_factory=dict)  # arbitrary extras


@dataclass
class GoATEdge:
    """A directed edge connecting two GoAT nodes."""
    edge_id: str
    src: str
    dst: str
    kind: EdgeKind
    reasoning: str = ""


@dataclass
class GoATGraph:
    """The output of one Director.run() invocation.

    Always called a GoATGraph regardless of K — single-intent runs (K=1)
    have no ghost nodes and no counterfactual edges, but they still carry
    the director's reasoning and the causal flow of execution.
    """
    graph_id: str
    user_input: str
    nodes: list[GoATNode] = field(default_factory=list)
    edges: list[GoATEdge] = field(default_factory=list)
    plan_complexity: Optional[PlanComplexity] = None
    critic_verdict: Optional[CriticVerdict] = None  # None when K=1
    execution_record: Optional[ExecutionRecord] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_counterfactuals(self) -> bool:
        return any(n.node_type == NodeType.GHOST for n in self.nodes)

    @property
    def factual_path_nodes(self) -> list[GoATNode]:
        return [n for n in self.nodes if n.node_type != NodeType.GHOST]

    @property
    def ghost_nodes(self) -> list[GoATNode]:
        return [n for n in self.nodes if n.node_type == NodeType.GHOST]


# ── Explanation ──────────────────────────────────────────────────────────────


@dataclass
class Explanation:
    """Audience-adapted explanation of the GoAT graph."""
    developer_explanation: str = ""
    user_explanation: str = ""
    raw_output: str = ""  # the full LLM response for debugging
