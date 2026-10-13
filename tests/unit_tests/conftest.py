"""Pytest fixtures for goat-langgraph tests."""

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run every test from a temp dir so the default ``./output`` never lands in the repo."""
    monkeypatch.chdir(tmp_path)


# ── Mock Model Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_chat_model():
    """Create a mock BaseChatModel for testing."""
    model = MagicMock()
    model.bind.return_value.invoke.return_value = AIMessage(
        content='{"goat_plans": [{"plan_name": "Test Plan", "plan": [{"tool": "test_tool", "args": {"arg": "value"}], "plan_reasoning": "test reasoning"}]}'
    )
    return model


@pytest.fixture
def mock_chat_model_k1():
    """Create a mock BaseChatModel that returns K=1 (simple task)."""
    model = MagicMock()
    model.bind.return_value.invoke.return_value = AIMessage(
        content='{"plan": {"plan_name": "Simple Plan", "plan": [{"tool": "test_tool", "args": {"arg": "value"}}], "plan_reasoning": "simple reasoning"}}'
    )
    return model


# ── Mock Director Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_director():
    """Create a mock Director for testing."""
    director = MagicMock()
    director.classifier = MagicMock()
    director.classifier.classify.return_value = MagicMock(
        needs_multiple_plans=True,
        suggested_K=3,
        reason="Complex task requiring multiple approaches",
        confidence=0.9
    )
    director.default_K = 3
    director.max_K = 3
    director.force_K = None
    return director


@pytest.fixture
def mock_director_result():
    """Create a mock DirectorResult for testing."""
    from langgraph_goat._core import Plan, PlanStep, PlanComplexity, CriticVerdict, Explanation, GoATGraph
    
    winning_plan = Plan(
        plan_id="plan_1",
        plan_name="Winning Plan",
        steps=[
            PlanStep(
                step_id="step_1",
                tool_name="test_tool",
                args={"arg": "value"},
                justification="Test justification"
            )
        ],
        overall_reasoning="Winning plan reasoning"
    )
    
    all_plans = [
        winning_plan,
        Plan(
            plan_id="plan_2",
            plan_name="Alternative Plan",
            steps=[
                PlanStep(
                    step_id="step_2",
                    tool_name="other_tool",
                    args={"other_arg": "other_value"},
                    justification="Alternative justification"
                )
            ],
            overall_reasoning="Alternative plan reasoning"
        )
    ]
    
    complexity = PlanComplexity(
        needs_multiple_plans=True,
        suggested_K=2,
        reason="Complex task",
        confidence=0.85
    )
    
    verdict = CriticVerdict(
        winner_plan_id="plan_1",
        scores={"plan_1": 8.5, "plan_2": 7.0},
        rejection_justifications={"plan_2": "Less optimal approach"},
        overall_reasoning="Plan 1 is better because..."
    )
    
    explanation = Explanation(
        developer_explanation="Developer explanation text",
        user_explanation="User explanation text",
        raw_output="Raw output"
    )
    
    graph = MagicMock(spec=GoATGraph)
    graph.graph_id = "test_graph_1"
    
    result = MagicMock()
    result.winning_plan = winning_plan
    result.all_plans = all_plans
    result.graph = graph
    result.complexity = complexity
    result.critic_verdict = verdict
    result.explanation = explanation
    
    return result


# ── Message Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def human_message():
    """Create a HumanMessage for testing."""
    return HumanMessage(content="What is the weather in New York?")


@pytest.fixture
def ai_message():
    """Create an AIMessage for testing."""
    return AIMessage(content="The weather in New York is sunny.")


@pytest.fixture
def system_message():
    """Create a SystemMessage for testing."""
    return SystemMessage(content="You are a helpful assistant.")


@pytest.fixture
def tool_message():
    """Create a ToolMessage for testing."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(content="Weather data retrieved", tool_call_id="call_123")


# ── Tool Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city.
        
        Args:
            city: The city name.
        """
        return f"Weather in {city}: Sunny"
    
    return get_weather


@pytest.fixture
def sample_tools(sample_tool):
    """Create a list of sample tools for testing."""
    @tool
    def search_web(query: str) -> str:
        """Search the web for a query.
        
        Args:
            query: The search query.
        """
        return f"Search results for: {query}"
    
    return [sample_tool, search_web]


# ── Request/Response Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def mock_model_request(mock_chat_model, human_message, sample_tools):
    """Create a mock ModelRequest for testing."""
    from langchain.agents.middleware import ModelRequest
    
    return ModelRequest(
        model=mock_chat_model,
        messages=[human_message],
        tools=sample_tools,
        system_message=None,
    )


@pytest.fixture
def mock_model_response():
    """Create a mock ModelResponse for testing."""
    from langchain.agents.middleware import ModelResponse
    
    return ModelResponse(
        result=[AIMessage(content="Test response")],
        structured_response=None,
    )


# ── Runtime Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_runtime():
    """Create a mock Runtime for testing."""
    from langgraph.runtime import Runtime
    
    runtime = MagicMock(spec=Runtime)
    runtime.store = None
    return runtime


# ── State Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def empty_goat_state():
    """Create an empty GoatAgentState for testing."""
    return {}


# ── Store Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_store():
    """Create a mock LangGraph store for testing."""
    store = MagicMock()
    store.search.return_value = [
        MagicMock(value="Memory snippet 1"),
        MagicMock(value="Memory snippet 2"),
    ]
    return store


# ── Planning Context Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def planning_context():
    """Create a PlanningContext for testing."""
    from langgraph_goat._core import PlanningContext, AvailableTool
    
    return PlanningContext(
        user_input="What is the weather in New York?",
        available_tools=[
            AvailableTool(
                name="get_weather",
                description="Get the weather for a city",
                parameters_schema={"type": "object", "properties": {"city": {"type": "string"}}}
            )
        ],
        memory_snippets=["User prefers metric units"],
        session_history=[],
        metadata={"source": "test"}
    )


# ── Plan Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_plan():
    """Create a sample Plan for testing."""
    from langgraph_goat._core import Plan, PlanStep
    
    return Plan(
        plan_id="test_plan_1",
        plan_name="Test Plan",
        steps=[
            PlanStep(
                step_id="step_1",
                tool_name="get_weather",
                args={"city": "New York"},
                expected_output="Weather data",
                justification="Need weather info to answer user's question"
            ),
            PlanStep(
                step_id="step_2",
                tool_name="format_response",
                args={"format": "json"},
                expected_output="Formatted response",
                justification="Format the response for the user"
            )
        ],
        overall_reasoning="This plan retrieves weather data and formats it for the user"
    )
