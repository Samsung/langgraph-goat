"""customllm.py - Custom LLM

This module provides a custom LLM implementation that connects to an LLM server.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Any, List, Sequence
import requests

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langsmith import traceable

logger = logging.getLogger(__name__)

ENDPOINT_URL = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "")
REQUEST_TIMEOUT = 180  # seconds;

# Matches a single {"name": "...", "args": {...}} tool-call block.
TOOL_CALL_PATTERN = r"\{\s*\"name\"\s*:\s*\"(?P<name>[^\"]+)\"\s*,\s*\"args\"\s*:\s*(?P<args>\{[^}]*\})\s*\}"


class CustomLLM(BaseChatModel):
    """Custom chat model for LLM API with tool calling support.

    This implementation connects to an LLM server via REST API and supports:
    1. Configurable endpoint and model name
    2. Tool calling via JSON parsing
    3. Caller-supplied system messages, passed through as the API's system field
    """

    _endpoint_url: str = ""
    _model_name: str = ""
    _bound_tools: List[BaseTool] = []  # List of tools bound via ``bind_tools``

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._endpoint_url = ENDPOINT_URL
        self._model_name = MODEL_NAME
        logger.info("Initialized LLM CustomLLM: endpoint=%s, model=%s", self._endpoint_url, self._model_name)

    @property
    def _llm_type(self) -> str:
        """Identify the LLM type for LangChain."""
        return "llm_custom"

    @property
    def _identifying_params(self) -> dict:
        return {
            "endpoint_url": self._endpoint_url,
            "model_name": self._model_name,
        }

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "CustomLLM":
        """Return a new model instance with the given tools bound.

        The returned instance carries a copy of the configuration
        and the list of tools.
        """
        cloned = CustomLLM()
        cloned._endpoint_url = self._endpoint_url
        cloned._model_name = self._model_name
        cloned._bound_tools = list(tools)
        # Copy _additional_kwargs if it exists (from previous bind() calls)
        if hasattr(self, "_additional_kwargs"):
            cloned._additional_kwargs = dict(self._additional_kwargs)
        return cloned

    def bind(self, *args: Any, **kwargs: Any) -> "CustomLLM":
        """Return a new model instance with the given parameters bound.

        This overrides BaseChatModel.bind() to properly handle our custom attributes.
        """
        bound = super().bind(*args, **kwargs)
        bound._endpoint_url = self._endpoint_url
        bound._model_name = self._model_name
        bound._bound_tools = self._bound_tools
        return bound

    @traceable(name="CustomLLM._generate", run_type="llm", tags=["llm"])
    def _generate(
        self,
        messages: List[Any],
        stop: List[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the LLM API.

        This method supports tool calling. The caller's system messages become the
        LLM ``system`` field, the remaining messages become the prompt, and any
        ``tool_call`` JSON blocks in the response are parsed out.
        """
        # A summary turn is one where the last message is a tool result
        is_summary_turn = bool(messages) and getattr(messages[-1], "type", "") == "tool"

        # Caller-supplied system messages go in the API's dedicated system field
        system_prompt = "\n".join(
            str(msg.content) for msg in messages
            if getattr(msg, "type", "") == "system" and hasattr(msg, "content")
        )
        prompt = "\n".join(
            f"{getattr(msg, 'type', 'human')}: {msg.content}" for msg in messages
            if hasattr(msg, "content") and getattr(msg, "type", "") != "system"
        )

        # Bound parameters come from _additional_kwargs (set via bind()), else defaults
        additional_kwargs = getattr(self, "_additional_kwargs", {})
        temperature = additional_kwargs.get("temperature", 0.8)
        max_tokens = additional_kwargs.get("max_tokens", 2000)

        body = {
            "model": self._model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.94,
                "top_k": 14,
            },
        }

        logger.info(
            "LLM API call: model=%s, temperature=%.1f, max_tokens=%d, prompt_length=%d, system_prompt_length=%d",
            self._model_name, temperature, max_tokens, len(prompt), len(system_prompt)
        )
        if len(prompt) + len(system_prompt) > 10000:
            logger.warning(
                "LLM API call: very long prompt (%d chars). This may cause timeouts or empty responses.",
                len(prompt) + len(system_prompt)
            )

        api_start = time.monotonic()
        response = requests.post(
            f"{self._endpoint_url}/api/generate",
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
        logger.info(
            "LLM API call completed: status=%d, duration_ms=%.1f",
            response.status_code, round((time.monotonic() - api_start) * 1000, 2)
        )

        if response.status_code != 200:
            raise ValueError(f"API request failed with status {response.status_code}: {response.text}")

        response_json = response.json()
        raw_content = str(response_json.get("response", "")).strip()
        if not raw_content:
            raise ValueError("LLM API returned empty response content")

        # Token counts, estimated when LLM doesn't report them
        prompt_eval_count = response_json.get("prompt_eval_count", 0) or response_json.get("prompt_tokens", 0)
        eval_count = response_json.get("eval_count", 0) or response_json.get("completion_tokens", 0)
        if not prompt_eval_count:
            prompt_eval_count = max(1, int(len(prompt.split()) * 1.3))  # ~1.3 tokens per English word
        if not eval_count:
            eval_count = len(raw_content) // 4  # ~4 chars per token

        response_metadata = {
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,
            "model": self._model_name,
            "prompt_char_count": len(prompt),
        }

        # Parse tool calls if any (only when not a summary turn)
        tool_calls = []
        cleaned_content = raw_content
        if not is_summary_turn and self._bound_tools:
            for match in re.finditer(TOOL_CALL_PATTERN, raw_content, re.DOTALL):
                try:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": match.group("name"),
                        "args": json.loads(match.group("args")),
                    })
                except Exception as e:
                    logger.debug("Failed to parse tool call block: %s", e)
            cleaned_content = re.sub(TOOL_CALL_PATTERN, "", raw_content, flags=re.DOTALL).strip()

        ai_message = AIMessage(
            content=cleaned_content,
            tool_calls=tool_calls,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
