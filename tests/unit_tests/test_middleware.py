"""Unit tests for GoatMiddleware.

These tests follow the isolation strategy from the documentation:
- Mock AgentState and Runtime objects
- Call hook methods directly (wrap_model_call, after_model)
- Assert returned state dictionaries and handler behavior
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import ModelRequest, ModelResponse

from langgraph_goat import GoatOutput
from langgraph_goat.middleware import (
    GoatMiddleware,
    GoatMiddlewareConfig,
    GoatAgentState,
)


# ── Test GoatMiddlewareConfig ────────────────────────────────────────────────


class TestGoatMiddlewareConfig:
    """Tests for GoatMiddlewareConfig dataclass."""

    def test_default_config(self):
        """Test that default config has expected values."""
        config = GoatMiddlewareConfig()
        
        assert config.classifier is None
        assert config.critic is None
        assert config.explanation_generator is None
        assert config.graph_store is None
        assert config.store is None
        assert config.default_K == 3
        assert config.max_K == 3
        assert config.force_K is None
        assert config.planning_prompt_template is None

# ── Test wrap_model_call: ToolMessage Bypass ─────────────────────────────────


class TestWrapModelCallToolMessageBypass:
    """Tests for ToolMessage bypass behavior."""

    def test_tool_message_bypasses_goat_pipeline(self):
        """When last message is ToolMessage, handler is called directly."""
        middleware = GoatMiddleware()
        
        tool_msg = ToolMessage(content="Result from tool", tool_call_id="call_123")
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="What's the weather?"), tool_msg],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="Final response")]))
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        # Handler should be called directly without GoAT processing
        mock_handler.assert_called_once_with(request)
        assert isinstance(result, ModelResponse)
        assert len(result.result) == 1
        assert result.result[0].content == "Final response"

    def test_tool_message_with_multiple_messages_bypasses(self):
        """ToolMessage at end of conversation bypasses GoAT."""
        middleware = GoatMiddleware()
        
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(content="Let me check"),
            ToolMessage(content="Sunny, 25C", tool_call_id="call_1"),
        ]
        request = ModelRequest(
            model=MagicMock(),
            messages=messages,
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="It's sunny")]))
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        mock_handler.assert_called_once()


# ── Test wrap_model_call: Empty Input Bypass ─────────────────────────────────


class TestWrapModelCallEmptyInput:
    """Tests for empty input bypass behavior."""

    def test_empty_user_input_bypasses_goat_pipeline(self):
        """Empty user input skips GoAT pipeline."""
        middleware = GoatMiddleware()
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="")],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="Response")]))
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        mock_handler.assert_called_once()

    def test_whitespace_only_input_bypasses(self):
        """Whitespace-only input skips GoAT pipeline."""
        middleware = GoatMiddleware()
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="   \n\t  ")],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="Response")]))
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        mock_handler.assert_called_once()


# ── Test wrap_model_call: K=1 Bypass ─────────────────────────────────────────


class TestWrapModelCallK1Bypass:
    """Tests for K=1 (simple task) bypass behavior."""

    @patch('langgraph_goat.middleware.Director')
    def test_k_equals_1_bypasses_goat_pipeline(self, mock_director_cls):
        """When K=1, model works independently without GoAT."""
        # Setup mock classifier to return K=1
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            needs_multiple_plans=False,
            suggested_K=1,
            reason="Simple task",
            confidence=1.0
        )
        
        mock_director = MagicMock()
        mock_director.classifier = mock_classifier
        mock_director.default_K = 3
        mock_director.max_K = 3
        mock_director_cls.return_value = mock_director
        
        middleware = GoatMiddleware()
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Hello")],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="Hi there!")]))
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        # Handler should be called, not the GoAT pipeline
        mock_handler.assert_called_once()
        # Director.run should NOT be called for K=1
        mock_director.run.assert_not_called()


class TestWrapModelCallForceK:
    """config.force_K must bypass the classifier (parity with GoatNode)."""

    @staticmethod
    def _setup(mock_director_cls, **config_kwargs):
        mock_director = MagicMock()
        mock_director.default_K = 3
        mock_director.max_K = config_kwargs.get("max_K", 3)
        mock_director_cls.return_value = mock_director

        middleware = GoatMiddleware(GoatMiddlewareConfig(**config_kwargs))
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Hello")],
            tools=[],
            system_message=None,
        )
        handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="Hi")]))
        return middleware, mock_director, request, handler

    @patch('langgraph_goat.middleware.Director')
    def test_force_k1_skips_classifier_and_pipeline(self, mock_director_cls):
        middleware, director, request, handler = self._setup(mock_director_cls, force_K=1)

        middleware.wrap_model_call(request, handler)

        director.classifier.classify.assert_not_called()
        director.run.assert_not_called()
        handler.assert_called_once()
        assert middleware._pending_complexity.suggested_K == 1
        assert middleware._pending_complexity.confidence == 1.0

    @patch('langgraph_goat.middleware.Director')
    def test_force_k_is_clamped_to_max_k(self, mock_director_cls):
        middleware, director, request, handler = self._setup(
            mock_director_cls, force_K=5, max_K=2
        )
        director.run.side_effect = RuntimeError("stop after K is decided")

        middleware.wrap_model_call(request, handler)  # falls back to handler on error

        director.classifier.classify.assert_not_called()
        assert director.force_K == 2

    @patch('langgraph_goat.middleware.Director')
    def test_force_k_below_one_is_clamped_to_one(self, mock_director_cls):
        middleware, director, request, handler = self._setup(mock_director_cls, force_K=0)

        middleware.wrap_model_call(request, handler)

        director.run.assert_not_called()
        assert middleware._pending_complexity.suggested_K == 1


# ── Test wrap_model_call: K>1 GoAT Pipeline ──────────────────────────────────


class TestWrapModelCallK2RunsGoAT:
    """Tests for K>=2 (complex task) GoAT pipeline execution."""

    @patch('langgraph_goat.middleware.Director')
    @patch('langgraph_goat.middleware.extract_context_from_request')
    def test_k_greater_than_1_runs_goat_pipeline(
        self, 
        mock_extract_context, 
        mock_director_cls
    ):
        """When K>=2, full GoAT pipeline runs."""
        # Setup mocks
        mock_context = MagicMock()
        mock_context.user_input = "Complex task"
        mock_extract_context.return_value = mock_context
        
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            needs_multiple_plans=True,
            suggested_K=3,
            reason="Complex task",
            confidence=0.9
        )
        
        mock_winning_plan = MagicMock()
        mock_winning_plan.plan_id = "plan_1"
        mock_winning_plan.steps = []
        
        mock_result = MagicMock()
        mock_result.winning_plan = mock_winning_plan
        mock_result.all_plans = [mock_winning_plan]
        mock_result.complexity = MagicMock(suggested_K=3)
        
        mock_director = MagicMock()
        mock_director.classifier = mock_classifier
        mock_director.default_K = 3
        mock_director.max_K = 3
        mock_director.force_K = None
        mock_director.run.return_value = mock_result
        mock_director_cls.return_value = mock_director
        
        middleware = GoatMiddleware()
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Complex task requiring planning")],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock()
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        # GoAT pipeline should run
        mock_director.run.assert_called_once()
        # Handler should NOT be called (GoAT provides the response)
        mock_handler.assert_not_called()
        # Result should be a ModelResponse
        assert isinstance(result, ModelResponse)

    @patch('langgraph_goat.middleware.Director')
    @patch('langgraph_goat.middleware.extract_context_from_request')
    def test_zero_plans_fallback_to_handler(
        self,
        mock_extract_context,
        mock_director_cls
    ):
        """When GoAT produces 0 plans, falls back to normal handler."""
        mock_context = MagicMock()
        mock_context.user_input = "Task"
        mock_extract_context.return_value = mock_context
        
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            needs_multiple_plans=True,
            suggested_K=2,
            reason="Complex task",
            confidence=0.8
        )
        
        mock_result = MagicMock()
        mock_result.all_plans = []  # No plans generated
        
        mock_director = MagicMock()
        mock_director.classifier = mock_classifier
        mock_director.default_K = 3
        mock_director.max_K = 3
        mock_director.run.return_value = mock_result
        mock_director_cls.return_value = mock_director
        
        middleware = GoatMiddleware()
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Task")],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="Handler response")]))
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        # Should fallback to handler
        mock_handler.assert_called_once()


# ── Test wrap_model_call: Exception Handling ─────────────────────────────────


class TestWrapModelCallExceptionHandling:
    """Tests for exception handling and fallback behavior."""

    @patch('langgraph_goat.middleware.extract_context_from_request')
    def test_exception_fallback_to_handler(self, mock_extract_context):
        """Pipeline exceptions trigger fallback to handler."""
        mock_extract_context.side_effect = Exception("Pipeline error")
        
        middleware = GoatMiddleware()
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Test")],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="Fallback response")]))
        
        result = middleware.wrap_model_call(request, mock_handler)
        
        # Should fallback to handler on exception
        mock_handler.assert_called_once()


# ── Test after_model ─────────────────────────────────────────────────────────


class TestAfterModel:
    """Tests for after_model hook."""

    @patch('langgraph_goat.middleware.Director')
    def test_injects_goat_outputs(self, mock_director_cls):
        """When pending_result exists, goat_* fields are injected."""
        # Setup mock result
        mock_winning_plan = MagicMock()
        mock_winning_plan.plan_id = "plan_1"
        
        mock_result = MagicMock()
        mock_result.winning_plan = mock_winning_plan
        mock_result.graph = MagicMock()
        mock_result.complexity = MagicMock()
        mock_result.explanation = MagicMock()
        mock_result.critic_verdict = MagicMock()
        
        mock_director = MagicMock()
        mock_director.run.return_value = mock_result
        mock_director_cls.return_value = mock_director
        
        middleware = GoatMiddleware()
        
        # First call wrap_model_call to set pending_result
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Complex task")],
            tools=[],
            system_message=None,
        )
        mock_handler = MagicMock()
        
        # Mock classifier to return K>1
        mock_director.classifier.classify.return_value = MagicMock(
            needs_multiple_plans=True,
            suggested_K=2,
            reason="Complex",
            confidence=0.9
        )
        
        try:
            middleware.wrap_model_call(request, mock_handler)
        except Exception:
            pass  # Ignore any errors, we just want to set pending_result
        
        # Manually set pending_result for testing
        middleware._pending_result = mock_result
        
        # Now test after_model
        state = {}
        runtime = MagicMock()
        
        updates = middleware.after_model(state, runtime)
        
        assert updates is not None
        goat_output = updates["goat_output"]
        assert isinstance(goat_output, GoatOutput)
        assert goat_output.winning_plan_id == "plan_1"
        assert goat_output.graph is mock_result.graph
        assert goat_output.complexity is mock_result.complexity
        assert goat_output.explanation is mock_result.explanation
        assert goat_output.director_result is mock_result
        assert middleware.last_goat_output is goat_output

    def test_returns_none_when_no_pending_result(self):
        """No pending_result returns None."""
        middleware = GoatMiddleware()
        middleware._pending_result = None
        
        state = {}
        runtime = MagicMock()
        
        updates = middleware.after_model(state, runtime)
        
        assert updates is None

    def test_clears_pending_result(self):
        """Pending result is cleared after injection."""
        middleware = GoatMiddleware()
        middleware._pending_result = MagicMock()
        
        state = {}
        runtime = MagicMock()
        
        middleware.after_model(state, runtime)
        
        assert middleware._pending_result is None


# ── Test State Schema ────────────────────────────────────────────────────────


class TestGoatAgentState:
    """Tests for GoatAgentState schema."""

    def test_state_schema_declares_goat_output(self):
        """GoatAgentState declares goat_output, the only key after_model writes."""
        from langgraph_goat.middleware import GoatAgentState

        assert "goat_output" in GoatAgentState.__annotations__

    def test_after_model_keys_are_declared_in_state_schema(self):
        """Every key returned by after_model must exist in the state schema,
        otherwise LangGraph silently drops it from the agent result."""
        middleware = GoatMiddleware()
        middleware._pending_complexity = MagicMock()

        updates = middleware.after_model({}, MagicMock())

        assert updates is not None
        assert set(updates) <= set(GoatAgentState.__annotations__)
