"""langgraph_goat._core — framework-agnostic GoAT engine (PRIVATE).

Internal implementation detail. Do **not** import from this subpackage
directly — its layout and signatures may change without notice. Everything
supported is re-exported from the top-level ``langgraph_goat`` package::

    from langgraph_goat import Director, Plan, PlanningContext

Contents:
    Director — the orchestrator
    Plan, PlanStep, PlanningContext, AvailableTool — input types
    GoATGraph, GoATNode, GoATEdge — output types
    ExecutionRecord, StepExecutionRecord — execution feedback
    Various default implementations (LLMClassifier, LLMJudgeCritic, etc.)
"""

from .director import Director, DirectorResult
from .complexity import LLMClassifier
from .critic import LLMJudgeCritic
from .explanation import LLMExplanationGenerator
from .graph import GraphBuilder
from .graph.stores import NetworkXStore
from .types import (
    AvailableTool,
    CriticVerdict,
    EdgeKind,
    ExecutionRecord,
    Explanation,
    GoATEdge,
    GoATGraph,
    GoATNode,
    NodeType,
    Plan,
    PlanComplexity,
    PlanningContext,
    PlanStep,
    StepExecutionRecord,
)
from .utils import (
    extractJson,
    renderGraph,
    renderTrace,
    renderTele
)

from .utils import (
    CallEvent,
    SequenceTraceCollector,
    TraceContext,
    TraceFormatter,
    TraceLogger,
    TraceSpan,
    GoatLogger,
    get_sequence_collector,
    traced,
)

from .utils import (
    TelemetryCollector,
    get_telemetry_collector,
    reset_telemetry_collector,
)

__all__ = [
    # Orchestrator
    "Director",
    "DirectorResult",
    # Defaults
    "LLMClassifier",
    "LLMJudgeCritic",
    "LLMExplanationGenerator",
    "GraphBuilder",
    "NetworkXStore",
    # Types
    "AvailableTool",
    "CriticVerdict",
    "EdgeKind",
    "ExecutionRecord",
    "Explanation",
    "GoATEdge",
    "GoATGraph",
    "GoATNode",
    "NodeType",
    "Plan",
    "PlanComplexity",
    "PlanningContext",
    "PlanStep",
    "StepExecutionRecord",
    # Tools
    "extractJson",
    "renderGraph",
    "renderTrace",
    "renderTele",
    # Traceability logger
    "CallEvent",
    "SequenceTraceCollector",
    "TraceContext",
    "TraceFormatter",
    "TraceLogger",
    "TraceSpan",
    "GoatLogger",
    "get_sequence_collector",
    "traced",
    # Telemetry trace logger
    "TelemetryCollector",
    "get_telemetry_collector",
    "reset_telemetry_collector",
]

__version__ = "1.0.0"
