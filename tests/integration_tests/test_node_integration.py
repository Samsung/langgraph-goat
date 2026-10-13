"""Integration tests: GoatNode inside a real LangGraph ``StateGraph``.

No external service is required. A deterministic fake chat model stands in for
the LLM, while the GoAT engine (Director, critic, explanation, graph building)
and LangGraph itself run for real.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langgraph_goat import GoatNode, GoatNodeConfig, GoatOutput

PLANS_JSON = json.dumps(
    {
        "goat_plans": [
            {
                "plan_name": "Direct lookup",
                "plan": [{"tool": "get_weather", "args": {"city": "Paris"}, "justification": "fast"}],
                "plan_reasoning": "Ask the weather tool directly.",
            },
            {
                "plan_name": "Forecast first",
                "plan": [
                    {"tool": "get_forecast", "args": {"city": "Paris"}, "justification": "context"},
                    {"tool": "get_weather", "args": {"city": "Paris"}, "justification": "confirm"},
                ],
                "plan_reasoning": "Check the forecast, then confirm.",
            },
        ]
    }
)


class ScriptedChatModel(BaseChatModel):
    """Returns planner JSON for the planning call and plain text for everything else."""

    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        is_planning = "goat_plans" in prompt
        self.calls.append("plan" if is_planning else "other")
        text = PLANS_JSON if is_planning else "No structured verdict available."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class PlannerState(TypedDict, total=False):
    user_input: str
    plan: list[dict]
    goat_output: GoatOutput
    goat_complexity: Any
    _goat_error: str


def _planner(state: PlannerState) -> dict:
    return {"plan": [{"tool": "get_weather", "args": {}}]}


def _build_prompt(state: dict) -> str:
    return "You are a planner. Produce tool plans for the user's request."


def _build_app(model: BaseChatModel, config: GoatNodeConfig):
    graph = StateGraph(PlannerState)
    graph.add_node("planner", GoatNode(_planner, model, config, _build_prompt))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    return graph.compile()


@pytest.fixture
def model() -> ScriptedChatModel:
    return ScriptedChatModel(calls=[])


def test_k1_bypasses_goat_inside_state_graph(model: ScriptedChatModel, tmp_path) -> None:
    app = _build_app(model, GoatNodeConfig(force_K=1, output_dir=str(tmp_path)))

    result = app.invoke({"user_input": "What is the weather in Paris?"})

    assert result["plan"] == [{"tool": "get_weather", "args": {}}]
    output = result["goat_output"]
    assert isinstance(output, GoatOutput)
    assert output.graph is None
    assert output.complexity.suggested_K == 1
    assert model.calls == []  # planner ran directly, the model was never called


def test_k2_runs_full_goat_pipeline_inside_state_graph(model: ScriptedChatModel, tmp_path) -> None:
    app = _build_app(model, GoatNodeConfig(force_K=2, max_K=2, output_dir=str(tmp_path)))

    result = app.invoke({"user_input": "What is the weather in Paris?"})

    assert "_goat_error" not in result
    output = result["goat_output"]
    assert isinstance(output, GoatOutput)
    assert output.graph is not None
    assert output.complexity.suggested_K == 2

    plan_ids = {p["plan_id"] for p in output.to_dict()["all_plans"]}
    assert len(plan_ids) == 2
    assert output.winning_plan_id in plan_ids
    # backward-compatible serialized winning plan is merged into state
    assert result["plan"] and {"tool", "args", "justification"} <= set(result["plan"][0])


def test_goat_output_from_pipeline_is_json_serializable(model: ScriptedChatModel, tmp_path) -> None:
    app = _build_app(model, GoatNodeConfig(force_K=2, max_K=2, output_dir=str(tmp_path)))

    output = app.invoke({"user_input": "What is the weather in Paris?"})["goat_output"]

    payload = json.loads(json.dumps(output.to_dict()))
    assert payload["winning_plan_id"] == output.winning_plan_id
    assert payload["graph"]["nodes"]


def test_render_flag_writes_artifacts_to_output_dir(model: ScriptedChatModel, tmp_path) -> None:
    config = GoatNodeConfig(force_K=2, max_K=2, output_dir=str(tmp_path), render_goat_tele=True)
    app = _build_app(model, config)

    output = app.invoke({"user_input": "What is the weather in Paris?"})["goat_output"]

    assert (tmp_path / output.graph.graph_id).is_dir()


def test_model_failure_falls_back_to_planner(tmp_path) -> None:
    class FailingModel(ScriptedChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
            raise RuntimeError("model down")

    app = _build_app(FailingModel(calls=[]), GoatNodeConfig(force_K=2, max_K=2, output_dir=str(tmp_path)))

    result = app.invoke({"user_input": "What is the weather in Paris?"})

    # Planner output still reaches the caller; the pipeline never raises into the graph.
    assert result["plan"]
