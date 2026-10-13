"""Integration tests: GoatMiddleware inside a real ``langchain.agents.create_agent``.

No external service is required; a deterministic fake chat model stands in for the LLM.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from langgraph_goat import GoatMiddleware, GoatMiddlewareConfig, GoatOutput

PLANS_JSON = json.dumps(
    {
        "goat_plans": [
            {
                "plan_name": "Direct",
                "plan": [{"tool": "get_weather", "args": {"city": "Paris"}, "justification": "fast"}],
                "plan_reasoning": "Call the weather tool.",
            },
            {
                "plan_name": "Careful",
                "plan": [{"tool": "get_weather", "args": {"city": "Paris, FR"}, "justification": "precise"}],
                "plan_reasoning": "Be explicit about the country.",
            },
        ]
    }
)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


class ScriptedChatModel(BaseChatModel):
    """Planner JSON for GoAT planning calls; a final answer for plain agent turns."""

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        text = PLANS_JSON if "goat_plans" in prompt else "It is sunny in Paris."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _agent(config: GoatMiddlewareConfig):
    middleware = GoatMiddleware(config)
    agent = create_agent(ScriptedChatModel(), tools=[get_weather], middleware=[middleware])
    return agent, middleware


def test_force_k1_agent_run_bypasses_goat(tmp_path) -> None:
    agent, middleware = _agent(GoatMiddlewareConfig(force_K=1, output_dir=str(tmp_path)))

    result = agent.invoke({"messages": [HumanMessage(content="Weather in Paris?")]})

    assert result["messages"][-1].content == "It is sunny in Paris."
    assert middleware.last_goat_output is None or middleware.last_goat_output.graph is None


def test_k2_agent_run_produces_goat_output(tmp_path) -> None:
    agent, middleware = _agent(GoatMiddlewareConfig(force_K=2, max_K=2, output_dir=str(tmp_path)))

    result = agent.invoke({"messages": [HumanMessage(content="Weather in Paris?")]})

    output = middleware.last_goat_output
    assert isinstance(output, GoatOutput)
    assert output.graph is not None
    assert output.winning_plan_id is not None
    assert result["messages"], "agent produced no messages"


def test_goat_output_is_exposed_in_agent_result(tmp_path) -> None:
    """The documented contract: ``goat_output`` is returned as part of the agent response."""
    agent, _ = _agent(GoatMiddlewareConfig(force_K=2, max_K=2, output_dir=str(tmp_path)))

    result = agent.invoke({"messages": [HumanMessage(content="Weather in Paris?")]})

    assert isinstance(result.get("goat_output"), GoatOutput)


def test_middleware_is_reusable_across_runs(tmp_path) -> None:
    agent, middleware = _agent(GoatMiddlewareConfig(force_K=2, max_K=2, output_dir=str(tmp_path)))

    agent.invoke({"messages": [HumanMessage(content="Weather in Paris?")]})
    first = middleware.last_goat_output
    agent.invoke({"messages": [HumanMessage(content="Weather in Rome?")]})
    second = middleware.last_goat_output

    assert first is not None and second is not None
    assert first.graph.graph_id != second.graph.graph_id


def test_invalid_output_dir_is_rejected_at_config_time(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ValueError, match="Cannot write to output_dir"):
        GoatMiddlewareConfig(output_dir=str(blocker / "sub"))
