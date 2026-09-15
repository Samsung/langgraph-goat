"""GoatOutput — unified output container for GoAT results.

Provides a consistent interface for both high-level (create_agent + GoatMiddleware)
and low-level (StateGraph + GoatNode) integration patterns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from ._core import GoATGraph, PlanComplexity, DirectorResult, Explanation, CriticVerdict


@dataclass
class GoatOutput:
    """Unified container for GoAT pipeline outputs.

    Returned as part of the agent/app response to provide structured access
    to explainability artifacts without tying users to a specific integration pattern.

    Attributes:
        graph: The GoAT reasoning graph (nodes, edges, plan complexity, counterfactuals).
        winning_plan_id: The ID of the selected plan.
        complexity: Complexity classification result (K, reasoning, confidence).
        explanation: Human-readable explanations (developer + user facing).
        director_result: Full DirectorResult with all plans, verdict, and metadata.
        critic_verdict: LLM critic's scoring and rejection justifications.
    """

    graph: Optional[GoATGraph] = None
    winning_plan_id: Optional[str] = None
    complexity: Optional[PlanComplexity] = None
    explanation: Optional[Explanation] = None
    director_result: Optional[DirectorResult] = None
    critic_verdict: Optional[CriticVerdict] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "graph": self._serialize_graph() if self.graph else None,
            "winning_plan_id": self.winning_plan_id,
            "complexity": self._serialize_complexity() if self.complexity else None,
            "explanation": {
                "developer": self.explanation.developer_explanation,
                "user": self.explanation.user_explanation,
            } if self.explanation else None,
            "critic_verdict": self._serialize_verdict() if self.critic_verdict else None,
            "all_plans": self._serialize_plans() if self.director_result else None,
        }

    def _serialize_graph(self) -> dict[str, Any]:
        """Serialize GoATGraph."""
        return {
            "graph_id": self.graph.graph_id,
            "user_input": self.graph.user_input,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type.value,
                    "label": n.label,
                    "reasoning": n.reasoning,
                    "payload": n.payload,
                }
                for n in self.graph.nodes
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "src": e.src,
                    "dst": e.dst,
                    "kind": e.kind.value,
                    "reasoning": e.reasoning,
                }
                for e in self.graph.edges
            ],
            "has_counterfactuals": self.graph.has_counterfactuals,
            "created_at": self.graph.created_at.isoformat() if self.graph.created_at else None,
        }

    def _serialize_complexity(self) -> dict[str, Any]:
        """Serialize PlanComplexity."""
        return {
            "needs_multiple_plans": self.complexity.needs_multiple_plans,
            "suggested_K": self.complexity.suggested_K,
            "reason": self.complexity.reason,
            "confidence": self.complexity.confidence,
        }

    def _serialize_verdict(self) -> dict[str, Any]:
        """Serialize CriticVerdict."""
        return {
            "scores": self.critic_verdict.scores,
            "rejection_justifications": self.critic_verdict.rejection_justifications,
            "overall_reasoning": self.critic_verdict.overall_reasoning,
        }

    def _serialize_plans(self) -> list[dict[str, Any]]:
        """Serialize all plans from DirectorResult."""
        if not hasattr(self.director_result, "all_plans") or not self.director_result.all_plans:
            return []

        return [
            {
                "plan_id": plan.plan_id,
                "plan_name": plan.plan_name,
                "is_winner": plan.plan_id == self.winning_plan_id,
                "steps": [
                    {"tool": s.tool_name, "args": s.args, "justification": s.justification}
                    for s in plan.steps
                ],
                "reasoning": plan.overall_reasoning,
            }
            for plan in self.director_result.all_plans
        ]
