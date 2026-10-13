"""Unit tests for helper functions in langgraph_goat.middleware.

Tests for:
- extract_context_from_request
- _retrieve_memory_from_store
- _aretrieve_memory_from_store
- plan_to_tool_calls
- plan_to_model_response
"""

from unittest.mock import MagicMock, PropertyMock, patch

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from langgraph_goat.middleware import (
    extract_context_from_request,
    plan_to_tool_calls,
    plan_to_model_response,
    _retrieve_memory_from_store,
)
from langgraph_goat.middleware import ModelRequest, ModelResponse


# ── Test extract_context_from_request ────────────────────────────────────────


class TestExtractContextFromRequest:
    """Tests for extract_context_from_request function."""

    def test_extracts_user_input_from_last_message(self):
        """User input is extracted from the last message."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="Response"),
            HumanMessage(content="What is the weather?"),
        ]
        request = ModelRequest(
            model=MagicMock(),
            messages=messages,
            tools=[],
            system_message=None,
        )
        
        context = extract_context_from_request(request)
        
        assert context.user_input == "What is the weather?"

    def test_extracts_tools_from_request(self):
        """Tools are extracted from the request."""
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}"
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Weather?")],
            tools=[get_weather],
            system_message=None,
        )
        
        context = extract_context_from_request(request)
        
        assert len(context.available_tools) == 1
        assert context.available_tools[0].name == "get_weather"

    def test_extracts_tools_from_dict_format(self):
        """Tools in dict format are extracted correctly."""
        tool_dict = {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Search")],
            tools=[tool_dict],
            system_message=None,
        )
        
        context = extract_context_from_request(request)
        
        assert len(context.available_tools) == 1
        assert context.available_tools[0].name == "search"
        assert context.available_tools[0].description == "Search the web"

    def test_retrieves_memory_from_store(self):
        """Memory is retrieved from provided store."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            MagicMock(value="User likes coffee"),
            MagicMock(value="User prefers morning meetings"),
        ]
        
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Schedule a meeting")],
            tools=[],
            system_message=None,
        )
        
        context = extract_context_from_request(request, store=mock_store)
        
        # Store searches 3 default namespaces, each returning 2 snippets = 6 total
        assert len(context.memory_snippets) == 6
        assert mock_store.search.call_count == 3

    def test_handles_empty_store(self):
        """Empty or None store returns empty memory snippets."""
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Test")],
            tools=[],
            system_message=None,
        )
        
        context = extract_context_from_request(request, store=None)
        
        assert context.memory_snippets == []

    def test_handles_empty_messages(self):
        """Empty messages list results in empty user input."""
        request = ModelRequest(
            model=MagicMock(),
            messages=[],
            tools=[],
            system_message=None,
        )
        
        context = extract_context_from_request(request)
        
        assert context.user_input == ""

    def test_handles_message_without_content_attr(self):
        """Messages without content attribute are handled."""
        mock_msg = MagicMock(spec=[])  # No content attribute
        request = ModelRequest(
            model=MagicMock(),
            messages=[mock_msg],
            tools=[],
            system_message=None,
        )
        
        # This may raise an error depending on implementation
        # The function should handle this gracefully
        context = extract_context_from_request(request)
        
        # Should not crash, may have empty user_input
        assert context is not None


# ── Test _retrieve_memory_from_store ────────────────────────────────────────


class TestRetrieveMemoryFromStore:
    """Tests for _retrieve_memory_from_store function."""

    def test_retrieves_from_multiple_namespaces(self):
        """Memory is retrieved from multiple default namespaces."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            MagicMock(value="Memory from namespace"),
        ]
        
        snippets = _retrieve_memory_from_store(mock_store, "test query")
        
        # Should search multiple namespaces
        assert mock_store.search.call_count >= 1

    def test_handles_item_with_value_attr(self):
        """Items with value attribute are handled."""
        mock_store = MagicMock()
        mock_item = MagicMock()
        mock_item.value = "Test value"
        # Mock search to return results only for first namespace, empty for others
        mock_store.search.side_effect = [
            [mock_item],  # First namespace returns result
            [],  # Second namespace returns empty
            [],  # Third namespace returns empty
        ]
        
        snippets = _retrieve_memory_from_store(mock_store, "query")
        
        assert len(snippets) == 1
        assert "Test value" in snippets[0]

    def test_handles_item_as_dict(self):
        """Items that are dicts are handled."""
        mock_store = MagicMock()
        mock_store.search.side_effect = [
            [{"value": "Dict value"}],
            [],
            [],
        ]
        
        snippets = _retrieve_memory_from_store(mock_store, "query")
        
        assert len(snippets) == 1
        assert "Dict value" in snippets[0]

    def test_handles_item_as_string(self):
        """Items that are strings are handled."""
        mock_store = MagicMock()
        mock_store.search.side_effect = [
            ["Plain string value"],
            [],
            [],
        ]
        
        snippets = _retrieve_memory_from_store(mock_store, "query")
        
        assert len(snippets) == 1

    def test_handles_search_exception(self):
        """Search exceptions are caught and logged."""
        mock_store = MagicMock()
        mock_store.search.side_effect = Exception("Search failed")
        
        # Should not raise, just return empty list
        snippets = _retrieve_memory_from_store(mock_store, "query")
        
        assert snippets == []

    def test_falls_back_to_get_method(self):
        """Falls back to .get() if .search() not available."""
        # Use hasattr to simulate a store without search method
        mock_store = MagicMock()
        # Remove search attribute by making hasattr return False
        type(mock_store).search = PropertyMock(side_effect=AttributeError)
        mock_store.get.return_value = MagicMock(value="Got value")
        
        # Patch hasattr to return False for search
        import builtins
        original_hasattr = builtins.hasattr
        
        def mock_hasattr(obj, name):
            if obj is mock_store and name == 'search':
                return False
            return original_hasattr(obj, name)
        
        with patch('builtins.hasattr', mock_hasattr):
            snippets = _retrieve_memory_from_store(mock_store, "key")
        
        mock_store.get.assert_called()

    def test_custom_namespaces(self):
        """Custom namespaces can be provided."""
        mock_store = MagicMock()
        mock_store.search.return_value = [MagicMock(value="Custom namespace value")]
        
        custom_namespaces = [("custom", "namespace")]
        snippets = _retrieve_memory_from_store(
            mock_store, "query", namespaces=custom_namespaces
        )
        
        # Should search custom namespace
        assert mock_store.search.called


# ── Test plan_to_tool_calls ──────────────────────────────────────────────────


class TestPlanToToolCalls:
    """Tests for plan_to_tool_calls function."""

    def test_converts_single_step_plan(self):
        """Single step plan converts to single tool call."""
        from langgraph_goat._core import Plan, PlanStep
        
        plan = Plan(
            plan_id="plan_1",
            plan_name="Test Plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="get_weather",
                    args={"city": "New York"},
                )
            ],
            overall_reasoning="Test"
        )
        
        tool_calls = plan_to_tool_calls(plan)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "step_1"
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"city": "New York"}

    def test_converts_multi_step_plan(self):
        """Multi-step plan converts to multiple tool calls."""
        from langgraph_goat._core import Plan, PlanStep
        
        plan = Plan(
            plan_id="plan_1",
            plan_name="Test Plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="get_weather",
                    args={"city": "New York"},
                ),
                PlanStep(
                    step_id="step_2",
                    tool_name="format_response",
                    args={"format": "json"},
                ),
            ],
            overall_reasoning="Test"
        )
        
        tool_calls = plan_to_tool_calls(plan)
        
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[1]["name"] == "format_response"

    def test_handles_empty_args(self):
        """Plan steps with empty args are handled."""
        from langgraph_goat._core import Plan, PlanStep
        
        plan = Plan(
            plan_id="plan_1",
            plan_name="Test Plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="list_items",
                    args={},
                )
            ],
            overall_reasoning="Test"
        )
        
        tool_calls = plan_to_tool_calls(plan)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["args"] == {}

# ── Test plan_to_model_response ──────────────────────────────────────────────


class TestPlanToModelResponse:
    """Tests for plan_to_model_response function."""

    def test_converts_plan_to_model_response(self):
        """Plan converts to ModelResponse with AIMessage."""
        from langgraph_goat._core import Plan, PlanStep
        
        plan = Plan(
            plan_id="plan_1",
            plan_name="Test Plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="get_weather",
                    args={"city": "New York"},
                )
            ],
            overall_reasoning="Test"
        )
        
        response = plan_to_model_response(plan)
        
        assert isinstance(response, ModelResponse)
        assert len(response.result) == 1
        assert isinstance(response.result[0], AIMessage)

    def test_ai_message_has_empty_content(self):
        """AIMessage has empty content since we have tool_calls."""
        from langgraph_goat._core import Plan, PlanStep
        
        plan = Plan(
            plan_id="plan_1",
            plan_name="Test Plan",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="get_weather",
                    args={"city": "New York"},
                )
            ],
            overall_reasoning="Test"
        )
        
        response = plan_to_model_response(plan)
        ai_message = response.result[0]
        
        assert ai_message.content == ""

