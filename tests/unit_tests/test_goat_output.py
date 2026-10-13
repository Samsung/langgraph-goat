"""Unit tests for GoatOutput."""

import json

from langgraph_goat import GoatOutput
from langgraph_goat._core import (
    CriticVerdict,
    DirectorResult,
    EdgeKind,
    Explanation,
    GoATEdge,
    GoATGraph,
    GoATNode,
    NodeType,
    Plan,
    PlanComplexity,
    PlanStep,
)


def _plan(plan_id: str) -> Plan:
    return Plan(
        plan_id=plan_id,
        plan_name=f"name-{plan_id}",
        steps=[PlanStep(step_id="s1", tool_name="get_weather", args={"city": "NYC"}, justification="why")],
        overall_reasoning="because",
    )


def _full_output() -> GoatOutput:
    winner, loser = _plan("p1"), _plan("p2")
    complexity = PlanComplexity(needs_multiple_plans=True, suggested_K=2, reason="hard", confidence=0.8)
    verdict = CriticVerdict(
        winner_plan_id="p1",
        scores={"p1": 0.9, "p2": 0.4},
        rejection_justifications={"p2": "worse"},
        overall_reasoning="p1 wins",
    )
    graph = GoATGraph(
        graph_id="g1",
        user_input="weather?",
        nodes=[GoATNode(node_id="n1", node_type=NodeType.DIRECTOR, label="Director")],
        edges=[GoATEdge(edge_id="e1", src="n1", dst="n1", kind=EdgeKind.CAUSAL_FLOW)],
        plan_complexity=complexity,
        critic_verdict=verdict,
    )
    explanation = Explanation(developer_explanation="dev", user_explanation="user")
    result = DirectorResult(
        graph=graph,
        winning_plan=winner,
        all_plans=[winner, loser],
        complexity=complexity,
        explanation=explanation,
    )
    return GoatOutput(
        graph=graph,
        winning_plan_id="p1",
        complexity=complexity,
        explanation=explanation,
        director_result=result,
        critic_verdict=verdict,
    )


class TestGoatOutput:
    def test_defaults_are_all_none(self) -> None:
        out = GoatOutput()
        assert out.graph is None
        assert out.winning_plan_id is None
        assert out.complexity is None
        assert out.explanation is None
        assert out.director_result is None
        assert out.critic_verdict is None

    def test_empty_to_dict(self) -> None:
        assert GoatOutput().to_dict() == {
            "graph": None,
            "winning_plan_id": None,
            "complexity": None,
            "explanation": None,
            "critic_verdict": None,
            "all_plans": None,
        }

    def test_complexity_only_to_dict(self) -> None:
        """K == 1 bypass produces an output holding only the complexity."""
        complexity = PlanComplexity(needs_multiple_plans=False, suggested_K=1, reason="easy")
        data = GoatOutput(complexity=complexity).to_dict()
        assert data["complexity"] == {
            "needs_multiple_plans": False,
            "suggested_K": 1,
            "reason": "easy",
            "confidence": 1.0,
        }
        assert data["graph"] is None
        assert data["all_plans"] is None

    def test_full_to_dict(self) -> None:
        data = _full_output().to_dict()

        assert data["winning_plan_id"] == "p1"
        assert data["explanation"] == {"developer": "dev", "user": "user"}
        assert data["critic_verdict"] == {
            "scores": {"p1": 0.9, "p2": 0.4},
            "rejection_justifications": {"p2": "worse"},
            "overall_reasoning": "p1 wins",
        }

        graph = data["graph"]
        assert graph["graph_id"] == "g1"
        assert graph["user_input"] == "weather?"
        assert graph["nodes"][0]["node_type"] == "director"
        assert graph["edges"][0]["kind"] == "causal_flow"
        assert graph["created_at"] is not None

    def test_all_plans_flags_winner(self) -> None:
        plans = _full_output().to_dict()["all_plans"]
        assert [(p["plan_id"], p["is_winner"]) for p in plans] == [("p1", True), ("p2", False)]
        assert plans[0]["steps"] == [{"tool": "get_weather", "args": {"city": "NYC"}, "justification": "why"}]
        assert plans[0]["reasoning"] == "because"

    def test_all_plans_empty_when_director_result_has_none(self) -> None:
        out = _full_output()
        out.director_result.all_plans = []
        assert out.to_dict()["all_plans"] == []

    def test_to_dict_is_json_serializable(self) -> None:
        json.dumps(_full_output().to_dict())
