"""Unit tests for node_wrapper.py.

These tests follow the isolation strategy from the documentation:
- Mock AgentState and Runtime objects
- Call functions and classes directly with mock objects
- Assert returned state dictionaries and behavior
"""

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from langgraph_goat.node_wrapper import (
    GoatNodeConfig,
    GoatNode,
)


# ── Test GoatNodeConfig ────────────────────────────────────────────────────────


class TestGoatNodeConfig:
    """Tests for GoatNodeConfig dataclass."""

    def test_default_config(self):
        """Test that default config has expected values."""
        config = GoatNodeConfig()
        
        assert config.default_K == 3
        assert config.max_K == 3
        assert config.store is None
        assert config.classifier is None
        assert config.critic is None
        assert config.explanation_generator is None
        assert config.graph_store is None
        assert config.output_dir == "./output"
        assert config.force_K is None

class TestWrapNodeWithGoatDiverse:
    """Additional diverse tests for GoatNode."""

    def test_with_custom_classifier(self):
        """Test GoatNode with custom classifier."""
        mock_planner = MagicMock(return_value={"result": "test"})
        mock_model = MagicMock()
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MagicMock(
            needs_multiple_plans=False,
            suggested_K=1,
            reason="Custom classifier",
            confidence=0.9,
        )
        
        config = GoatNodeConfig(classifier=mock_classifier)
        mock_build_prompt = MagicMock(return_value="Prompt")
        
        wrapped = GoatNode(
            planner_callable=mock_planner,
            model=mock_model,
            config=config,
            build_prompt=mock_build_prompt,
        )
        
        state = {"user_input": "Test", "messages": [HumanMessage(content="Test")]}
        result = wrapped(state)
        
        # Custom classifier should be used
        mock_classifier.classify.assert_called_once()
        mock_planner.assert_called_once()

    def test_with_input_fallback(self):
        """Test GoatNode with 'input' key instead of 'user_input'."""
        mock_planner = MagicMock(return_value={"result": "test"})
        mock_model = MagicMock()
        
        config = GoatNodeConfig(force_K=1)
        mock_build_prompt = MagicMock(return_value="Prompt")
        
        wrapped = GoatNode(
            planner_callable=mock_planner,
            model=mock_model,
            config=config,
            build_prompt=mock_build_prompt,
        )
        
        state = {"input": "Test via input key", "messages": [HumanMessage(content="Test")]}
        result = wrapped(state)
        
        mock_planner.assert_called_once()
        assert "goat_complexity" in result

    def test_preserves_state_keys_in_result(self):
        """Test that original state keys are preserved in result."""
        mock_planner = MagicMock(return_value={
            "result": "test",
            "original_key": "original_value",
        })
        mock_model = MagicMock()
        
        config = GoatNodeConfig(force_K=1)
        mock_build_prompt = MagicMock(return_value="Prompt")
        
        wrapped = GoatNode(
            planner_callable=mock_planner,
            model=mock_model,
            config=config,
            build_prompt=mock_build_prompt,
        )
        
        state = {"user_input": "Test", "messages": [HumanMessage(content="Test")]}
        result = wrapped(state)
        
        assert "original_key" in result
        assert result["original_key"] == "original_value"


class TestGoatNodeFallback:
    """GoatNode falls back to the planner when the GoAT pipeline fails."""

    def test_pipeline_error_falls_back_to_planner(self):
        """Test full flow with K=2 (triggers GoAT pipeline)."""
        mock_planner = MagicMock(return_value={"messages": ["Planned"]})
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("Simulated GoAT error")
        
        config = GoatNodeConfig(force_K=2)
        mock_build_prompt = MagicMock(return_value="Prompt")
        
        wrapped = GoatNode(
            planner_callable=mock_planner,
            model=mock_model,
            config=config,
            build_prompt=mock_build_prompt,
        )
        
        state = {
            "user_input": "Complex task",
            "messages": [HumanMessage(content="Complex task")],
            "tools": [],
        }
        result = wrapped(state)
        
        # Should fallback to planner on error
        mock_planner.assert_called()
        assert "_goat_error" in result
