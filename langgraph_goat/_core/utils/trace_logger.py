"""Trace Logger — assembles and writes the full telemetry trace JSON.

Collects telemetry data from across the GoAT pipeline (LLM ModelResponse,
TraceSpan timings, tool execution records, GoATGraph, etc.) and produces
a structured ``trace_{graph_id}.json`` file in the output directory.

Trace JSON structure::

    {
      "telemetry": [
        { "type": "token", ... },
        { "type": "latency", ... },
        { "type": "model", ... },
        { "type": "prompt", ... },
        { "type": "completion", ... },
        { "type": "tool", ... },
        { "type": "explainability", ... }
      ]
    }
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..types import (
    CriticVerdict,
    GoATGraph,
    GoATNode,
    NodeType,
    Plan,
    StepExecutionRecord,
)
from .logger import GoatLogger, TraceContext, TraceSpan, get_sequence_collector

log = GoatLogger("langgraph_goat._core.trace_logger")


# ── Data containers for each telemetry type ────────────────────────────────────


@dataclass
class TokenMetrics:
    """Token usage from a single LLM call."""

    id: str = ""
    input: int = 0
    output: int = 0
    total: int = 0
    reasoning: int = 0
    tool_call: Optional[Any] = None
    tokens_per_sec: float = 0.0


@dataclass
class LatencyMetrics:
    """Latency breakdown for a single LLM call / pipeline stage."""

    total: float = 0.0
    inference: float = 0.0
    tool: float = 0.0
    ttft: float = 0.0  # time-to-first-token


@dataclass
class ModelMetrics:
    """Model configuration used for an LLM call."""

    name: str = ""
    temperature: float = 0.7
    top_p: float = 1.0
    seed: Optional[int] = None
    max_tokens: int = 2000


@dataclass
class PromptMetrics:
    """Prompt details for a single LLM call."""

    id: str = ""
    system: str = ""
    user: str = ""
    size: int = 0
    version: str = "1.0"


@dataclass
class CompletionMetrics:
    """Completion details from a single LLM call."""

    output: str = ""
    finish_reason: str = ""

@dataclass
class PipelineErrorMetrics:
    """Error details from a specific pipeline stage."""
    stage: str = ""  # e.g., "classifier", "plan_generator", "critic", "graph_builder", "explanation_generator"
    error_type: str = ""
    message: str = ""
    traceback: str = ""


@dataclass
class ErrorMetrics:
    """Error details from complete execution."""
    api_failure: str = ""
    rate_limits: str = ""
    quota_exceeded: str = ""
    retries: int = 0
    timeouts: str = ""
    pipeline_errors: list[PipelineErrorMetrics] = field(default_factory=list)


@dataclass
class ToolMetrics:
    """Tool execution record."""

    name: str = ""
    args: dict = field(default_factory=dict)
    output: Any = None
    result: str = ""  # "success" or "fail"


@dataclass
class SpanMetrics:
    """A named timing span (e.g. director_agent, sub_agent)."""

    name: str = ""
    time: float = 0.0


@dataclass
class CounterfactualMetrics:
    """Counterfactual plan info from the critic."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    plan_name: str = ""
    score: float = 0.0
    reasoning: str = ""


@dataclass
class LLMCallRecord:
    """Everything captured from a single llm call."""

    token: TokenMetrics = field(default_factory=TokenMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    model: ModelMetrics = field(default_factory=ModelMetrics)
    prompt: PromptMetrics = field(default_factory=PromptMetrics)
    completion: CompletionMetrics = field(default_factory=CompletionMetrics)
    error: ErrorMetrics = field(default_factory=ErrorMetrics)
    # When the call happened (monotonic offset from trace start)
    timestamp_offset: float = 0.0


# ── TelemetryCollector ────────────────────────────────────────────────────────


class TelemetryCollector:
    """Collects telemetry data across a single Director.run() pipeline execution.

    Usage::

        tc = TelemetryCollector()
        tc.record_llm_call(...)
        tc.record_tool_execution(...)
        tc.set_graph(...)
        tc.write_tele(output_dir="project/output")
    """

    def __init__(self) -> None:
        self._llm_calls: list[LLMCallRecord] = []
        self._tool_records: list[ToolMetrics] = []
        self._spans: list[SpanMetrics] = []
        self._counterfactuals: list[CounterfactualMetrics] = []
        self._global_error: Optional[ErrorMetrics] = None
        self._graph: Optional[GoATGraph] = None
        self._director_agent_time: float = 0.0
        self._start_time: float = time.perf_counter()

    # ── LLM call recording ─────────────────────────────────────────────────

    def record_llm_call(
        self,
        model_response: Any = None,
        messages: list[dict] | None = None,
        model: str = "",
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        max_tokens: int = 2000,
        latency_total: float = 0.0,
        latency_inference: float = 0.0,
        latency_ttft: float = 0.0,
    ) -> LLMCallRecord:
        """Record a single LLM call with its full ModelResponse and metadata.

        Args:
            model_response: The raw litellm ModelResponse object (or dict-like).
            messages: The messages list sent to the LLM.
            model: Model name used.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus sampling) value.
            seed: Seed value for reproducibility.
            max_tokens: Maximum tokens for the completion.
            latency_total: Total wall-clock time for this call (seconds).
            latency_inference: Inference-only time (seconds).
            latency_ttft: Time-to-first-token (seconds).
        """
        record = LLMCallRecord(
            timestamp_offset=time.perf_counter() - self._start_time,
            model=ModelMetrics(
                name=model,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_tokens=max_tokens,
            ),
            latency=LatencyMetrics(
                total=latency_total,
                inference=latency_inference,
                tool=0.0,
                ttft=latency_ttft,
            ),
        )

        # ── Extract the response id from ModelResponse (shared across token & prompt) ──
        call_id: str = uuid.uuid4().hex[:8]
        if model_response is not None:
            resp_id = getattr(model_response, "id", None) or (model_response.get("id", "") if hasattr(model_response, "get") else "")
            if resp_id:
                call_id = resp_id

        # ── Extract token metrics from ModelResponse ───────────────────────
        if model_response is not None:
            try:
                usage = getattr(model_response, "usage", None) or model_response.get("usage", {})
                if usage:
                    record.token = TokenMetrics(
                        id=call_id,
                        input=getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens", 0),
                        output=getattr(usage, "completion_tokens", None) or usage.get("completion_tokens", 0),
                        total=getattr(usage, "total_tokens", None) or usage.get("total_tokens", 0),
                        reasoning=getattr(usage, "completion_tokens", None) or usage.get("completion_tokens", 0),
                    )
                    # Compute tokens/sec if we have output tokens and inference latency
                    if record.token.output > 0 and latency_inference > 0:
                        record.token.tokens_per_sec = round(record.token.output / latency_inference, 2)

                # ── Extract completion metrics ──────────────────────────────
                choices = getattr(model_response, "choices", None) or model_response.get("choices", [])
                if choices:
                    first_choice = choices[0]
                    message = getattr(first_choice, "message", None) or first_choice.get("message", {})
                    record.completion = CompletionMetrics(
                        output=getattr(message, "content", None) or message.get("content", ""),
                        finish_reason=getattr(first_choice, "finish_reason", None) or first_choice.get("finish_reason", ""),
                    )
                    # Extract tool_calls if present
                    tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls", None)
                    if tool_calls:
                        record.token.tool_call = [
                            {"name": getattr(tc.function, "name", ""), "arguments": getattr(tc.function, "arguments", "")}
                            if hasattr(tc, "function") else tc
                            for tc in tool_calls
                        ]
            except Exception as e:
                log.warning("TelemetryCollector: error extracting ModelResponse data: %s", e)

        # ── Extract prompt metrics from messages ───────────────────────────
        if messages:
            system_parts: list[str] = []
            user_parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_parts.append(content)
                elif role == "user":
                    user_parts.append(content)
            record.prompt = PromptMetrics(
                id=call_id,
                system="\n".join(system_parts),
                user="\n".join(user_parts),
                size=sum(len(m.get("content", "")) for m in messages),
            )

        self._llm_calls.append(record)
        log.debug(
            "TelemetryCollector: recorded LLM call #%d (model=%s, tokens=%d)",
            len(self._llm_calls), model, record.token.total,
        )
        return record

    # ── Tool execution recording ───────────────────────────────────────────

    def record_tool_execution(
        self,
        name: str,
        args: dict | None = None,
        output: Any = None,
        result: str = "success",
    ) -> None:
        """Record a tool execution outcome."""
        self._tool_records.append(ToolMetrics(
            name=name,
            args=args or {},
            output=output,
            result=result,
        ))

    # ── Span / explainability recording ────────────────────────────────────

    def record_span(self, name: str, time_taken: float) -> None:
        """Record a named timing span for explainability."""
        self._spans.append(SpanMetrics(name=name, time=round(time_taken, 6)))

    def set_director_agent_time(self, time_taken: float) -> None:
        """Set the total time taken by the director_agent."""
        self._director_agent_time = round(time_taken, 6)

    # ── Graph & counterfactual recording ───────────────────────────────────

    def set_graph(self, graph: GoATGraph) -> None:
        """Set the GoATGraph reference for explainability data extraction."""
        self._graph = graph

    def record_counterfactual(
        self,
        plan_id: str = "",
        plan_name: str = "",
        score: float = 0.0,
        reasoning: str = "",
    ) -> None:
        """Record a counterfactual plan from the critic."""
        self._counterfactuals.append(CounterfactualMetrics(
            id=plan_id or uuid.uuid4().hex[:8],
            plan_name=plan_name,
            score=score,
            reasoning=reasoning,
        ))

    # ── Error recording ────────────────────────────────────────────────────

    def record_error(
        self,
        api_failure: str = "",
        rate_limits: str = "",
        quota_exceeded: str = "",
        retries: int = 0,
        timeouts: str = "",
    ) -> None:
        """Record error details from the execution.

        If there is an active LLM call record (the last one), the error
        is attached to it. Otherwise the error is stored at the collector
        level for inclusion in the aggregated error telemetry.

        Args:
            api_failure: Description of any API failure.
            rate_limits: Rate limit error details.
            quota_exceeded: Quota exceeded error details.
            retries: Number of retries attempted.
            timeouts: Timeout error details.
        """
        error = ErrorMetrics(
            api_failure=api_failure,
            rate_limits=rate_limits,
            quota_exceeded=quota_exceeded,
            retries=retries,
            timeouts=timeouts,
        )
        # Attach to the most recent LLM call if one exists
        if self._llm_calls:
            last_call = self._llm_calls[-1]
            # Merge: only overwrite non-empty values
            if api_failure:
                last_call.error.api_failure = api_failure
            if rate_limits:
                last_call.error.rate_limits = rate_limits
            if quota_exceeded:
                last_call.error.quota_exceeded = quota_exceeded
            if retries:
                last_call.error.retries = retries
            if timeouts:
                last_call.error.timeouts = timeouts
        else:
            # Store at collector level for standalone error cases (e.g. tool errors)
            self._global_error = error

        log.debug(
            "TelemetryCollector: recorded error (api_failure=%s, rate_limits=%s, quota=%s, retries=%d, timeouts=%s)",
            bool(api_failure), bool(rate_limits), bool(quota_exceeded), retries, bool(timeouts),
        )

    def record_pipeline_error(
        self,
        stage: str,
        error_type: str = "",
        message: str = "",
        traceback: str = "",
    ) -> None:
        """Record an error from a specific pipeline stage.

        This method captures errors from any stage of the GoAT pipeline
        (classifier, plan_generator, critic, graph_builder, explanation_generator)
        for inclusion in the telemetry error section.

        Args:
            stage: The pipeline stage where the error occurred
                (e.g., "classifier", "plan_generator", "critic", "graph_builder", "explanation_generator").
            error_type: The type of error (e.g., "ValueError", "TimeoutError").
            message: The error message.
            traceback: The full traceback string.
        """
        pipeline_error = PipelineErrorMetrics(
            stage=stage,
            error_type=error_type,
            message=message,
            traceback=traceback,
        )

        # Attach to the most recent LLM call if one exists
        if self._llm_calls:
            last_call = self._llm_calls[-1]
            last_call.error.pipeline_errors.append(pipeline_error)
        else:
            # Store at collector level for standalone errors
            if self._global_error is None:
                self._global_error = ErrorMetrics()
            self._global_error.pipeline_errors.append(pipeline_error)

        log.debug(
            "TelemetryCollector: recorded pipeline error at stage=%s (error_type=%s)",
            stage, error_type,
        )

    # ── Build the trace JSON ───────────────────────────────────────────────

    def build_trace(self) -> dict[str, Any]:
        """Assemble all collected telemetry into the final trace JSON structure.

        Returns:
            The complete trace dict ready for JSON serialization.
        """
        telemetry: list[dict[str, Any]] = []

        # ── 1. Token telemetry (one entry per LLM call) ────────────────────
        token_metrics = []
        for call in self._llm_calls:
            token_metrics.append({
                "id": call.token.id,
                "input": call.token.input,
                "output": call.token.output,
                "total": call.token.total,
                "reasoning": call.token.reasoning,
                "tool-call": call.token.tool_call,
                "tokens-per-sec": call.token.tokens_per_sec,
            })
        if token_metrics:
            telemetry.append({"type": "token", "metrics": token_metrics})

        # ── 2. Latency telemetry (aggregated) ──────────────────────────────
        total_latency = sum(c.latency.total for c in self._llm_calls)
        inference_latency = sum(c.latency.inference for c in self._llm_calls)
        tool_latency = sum(t_record.output.get("duration_ms", 0) / 1000.0
                          if isinstance(t_record.output, dict) else 0.0
                          for t_record in self._tool_records)
        ttft = next((c.latency.ttft for c in self._llm_calls if c.latency.ttft > 0), 0.0)

        # Add tool execution time from tool records
        tool_latency_total = 0.0
        for tr in self._tool_records:
            if isinstance(tr.output, dict) and "duration_ms" in tr.output:
                tool_latency_total += tr.output["duration_ms"] / 1000.0

        latency_entry = {
            "type": "latency",
            "total": round(total_latency + self._director_agent_time, 6),
            "inference": round(inference_latency, 6),
            "tool": round(tool_latency_total, 6),
            "TTFT": round(ttft, 6),
        }
        telemetry.append(latency_entry)

        # ── 3. Model telemetry (from the first LLM call) ──────────────────
        if self._llm_calls:
            first_call = self._llm_calls[0]
            telemetry.append({
                "type": "model",
                "name": first_call.model.name,
                "temperature": first_call.model.temperature,
                "top_p": first_call.model.top_p,
                "seed": first_call.model.seed,
                "max_tokens": first_call.model.max_tokens,
            })

        # ── 4. Prompt telemetry (one entry per LLM call) ───────────────────
        prompt_metrics = []
        for call in self._llm_calls:
            prompt_metrics.append({
                "id": call.prompt.id,
                "system": call.prompt.system,
                "user": call.prompt.user,
                "size": call.prompt.size,
                "version": call.prompt.version,
            })
        if prompt_metrics:
            telemetry.append({"type": "prompt", "metrics": prompt_metrics})

        # ── 5. Completion telemetry (from the last LLM call, typically) ────
        if self._llm_calls:
            last_call = self._llm_calls[-1]
            telemetry.append({
                "type": "completion",
                "output": last_call.completion.output,
                "finish_reason": last_call.completion.finish_reason,
            })

        # ── 6. Tool telemetry ──────────────────────────────────────────────
        if self._tool_records:
            tool_metrics_list = []
            for tr in self._tool_records:
                tool_metrics_list.append({
                    "name": tr.name,
                    "args": tr.args,
                    "output": tr.output,
                    "result": tr.result,
                })
            telemetry.append({"type": "tool", "metrics": tool_metrics_list})

        # ── 7. Explainability telemetry ────────────────────────────────────
        explainability: dict[str, Any] = {
            "type": "explainability",
            "director_agent": self._director_agent_time,
            "span": [{"name": s.name, "time": s.time} for s in self._spans],
            "counterfactuals": [
                {
                    "id": cf.id,
                    "plan_name": cf.plan_name,
                    "score": cf.score,
                    "reasoning": cf.reasoning,
                }
                for cf in self._counterfactuals
            ],
        }

        # Attach the GoAT graph (serialized to dict)
        if self._graph is not None:
            explainability["graph"] = self._serialize_graph(self._graph)
            # Add user_input at top level for easy access (not buried in graph)
            explainability["user_input"] = self._graph.user_input
            # Add winning_plan section with steps and justifications
            explainability["winning_plan"] = self._extract_winning_plan(self._graph)
        else:
            explainability["graph"] = None
            explainability["user_input"] = ""
            explainability["winning_plan"] = None

        telemetry.append(explainability)


        # ── 8. Error telemetry (aggregated from all LLM calls + global) ────
        # Aggregate all errors across LLM calls and any global errors
        agg_api_failure: list[str] = []
        agg_rate_limits: list[str] = []
        agg_quota_exceeded: list[str] = []
        agg_retries = 0
        agg_timeouts: list[str] = []

        for call in self._llm_calls:
            if call.error.api_failure:
                agg_api_failure.append(call.error.api_failure)
            if call.error.rate_limits:
                agg_rate_limits.append(call.error.rate_limits)
            if call.error.quota_exceeded:
                agg_quota_exceeded.append(call.error.quota_exceeded)
            agg_retries += call.error.retries
            if call.error.timeouts:
                agg_timeouts.append(call.error.timeouts)

        # Include global error (e.g. tool errors without an LLM call)
        if self._global_error is not None:
            if self._global_error.api_failure:
                agg_api_failure.append(self._global_error.api_failure)
            if self._global_error.rate_limits:
                agg_rate_limits.append(self._global_error.rate_limits)
            if self._global_error.quota_exceeded:
                agg_quota_exceeded.append(self._global_error.quota_exceeded)
            agg_retries += self._global_error.retries
            if self._global_error.timeouts:
                agg_timeouts.append(self._global_error.timeouts)

        # Collect pipeline errors from all LLM calls and global errors
        all_pipeline_errors = []
        for call in self._llm_calls:
            all_pipeline_errors.extend(call.error.pipeline_errors)
        if self._global_error is not None:
            all_pipeline_errors.extend(self._global_error.pipeline_errors)

        error_entry = {
            "type": "error",
            "api_failure": "; ".join(agg_api_failure) if agg_api_failure else "",
            "rate_limits": "; ".join(agg_rate_limits) if agg_rate_limits else "",
            "quota_exceeded": "; ".join(agg_quota_exceeded) if agg_quota_exceeded else "",
            "retries": agg_retries,
            "timeouts": "; ".join(agg_timeouts) if agg_timeouts else "",
            "pipeline_errors": [
                {
                    "stage": pe.stage,
                    "error_type": pe.error_type,
                    "message": pe.message,
                    "traceback": pe.traceback,
                }
                for pe in all_pipeline_errors
            ] if all_pipeline_errors else [],
        }
        telemetry.append(error_entry)

        return {"telemetry": telemetry}


    # ── Write trace to file ────────────────────────────────────────────────

    def write_tele(self, output_dir: str) -> dict[str, Any]:
        """Assemble and write the trace JSON file.

        Args:
            output_dir: Directory to write the trace file into.

        Returns:
            The path of the written trace file.
        """
        graph_id = self._graph.graph_id if self._graph else uuid.uuid4().hex[:8]
        trace_data = self.build_trace()

        out_path = Path(output_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            log.error("TelemetryCollector: cannot create output directory '%s': %s", output_dir, e)
            raise

        filename = f"tele_{graph_id}.json"
        filepath = out_path / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
            log.info("TelemetryCollector: trace written to %s", filepath)
        except (PermissionError, OSError) as e:
            log.error("TelemetryCollector: cannot write trace to '%s': %s", filepath, e)
            raise
        except (TypeError, ValueError) as e:
            log.error("TelemetryCollector: JSON serialization failed: %s", e)
            raise

        return trace_data

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _serialize_graph(graph: GoATGraph) -> dict[str, Any]:
        """Serialize a GoATGraph to a JSON-safe dict."""
        from dataclasses import asdict

        result: dict[str, Any] = {
            "graph_id": graph.graph_id,
            "user_input": graph.user_input,
            "nodes": [],
            "edges": [],
        }

        for node in graph.nodes:
            result["nodes"].append({
                "node_id": node.node_id,
                "node_type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
                "label": node.label,
                "reasoning": node.reasoning,
                "payload": node.payload,
            })

        for edge in graph.edges:
            result["edges"].append({
                "edge_id": edge.edge_id,
                "src": edge.src,
                "dst": edge.dst,
                "kind": edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind),
                "reasoning": edge.reasoning,
            })

        if graph.plan_complexity:
            result["plan_complexity"] = {
                "needs_multiple_plans": graph.plan_complexity.needs_multiple_plans,
                "suggested_K": graph.plan_complexity.suggested_K,
                "reason": graph.plan_complexity.reason,
                "confidence": graph.plan_complexity.confidence,
            }

        if graph.critic_verdict:
            result["critic_verdict"] = {
                "winner_plan_id": graph.critic_verdict.winner_plan_id,
                "scores": graph.critic_verdict.scores,
                "rejection_justifications": graph.critic_verdict.rejection_justifications,
                "overall_reasoning": graph.critic_verdict.overall_reasoning,
            }

        if graph.execution_record:
            result["execution_record"] = {
                "plan_id": graph.execution_record.plan_id,
                "final_answer": graph.execution_record.final_answer,
                "total_duration_ms": graph.execution_record.total_duration_ms,
            }

        return result

    @staticmethod
    def _extract_winning_plan(graph: GoATGraph) -> dict[str, Any] | None:
        """Extract the winning plan with steps and justifications from the graph.
        
        Args:
            graph: The GoATGraph containing the winning plan information.
        
        Returns:
            A dict with plan_id, plan_name, and steps (with tool_name, args, justification),
            or None if no winning plan is found.
        """
        if graph.critic_verdict is None:
            return None
        
        winner_plan_id = graph.critic_verdict.winner_plan_id
        
        # Find the plan node (the first plan_step node after director that belongs to winning plan)
        plan_name = ""
        plan_score = None
        steps = []
        
        for node in graph.nodes:
            # Find the plan node (has plan_id in payload matching winner)
            if node.node_type == NodeType.PLAN_STEP:
                payload = node.payload or {}
                if payload.get("plan_id") == winner_plan_id:
                    plan_name = node.label.split("\n")[0] if "\n" in node.label else node.label
                    # Extract score if available
                    if "(score =" in node.label:
                        try:
                            score_part = node.label.split("(score =")[1].split(")")[0].strip()
                            plan_score = float(score_part)
                        except (ValueError, IndexError):
                            pass
            
            # Find tool call nodes that belong to the winning plan
            elif node.node_type == NodeType.TOOL_CALL:
                payload = node.payload or {}
                # Check if this tool call is part of the winning plan by checking edges
                # Tool calls are children of the plan node in the causal chain
                steps.append({
                    "tool_name": payload.get("tool_name", node.label),
                    "args": payload.get("args", {}),
                    "justification": node.reasoning or "",
                })
        
        if not plan_name and winner_plan_id:
            # Fallback: use plan_id as name if we couldn't extract a better name
            plan_name = winner_plan_id
        
        return {
            "plan_id": winner_plan_id,
            "plan_name": plan_name,
            "score": plan_score,
            "steps": steps,
        }

    # ── Extract from GoATGraph after pipeline completes ────────────────────

    def extract_from_graph(self, graph: GoATGraph) -> None:

        """Extract counterfactual and tool data from a GoATGraph.

        Call this after the Director pipeline completes to automatically
        populate counterfactual and tool records from the graph structure.
        """
        self.set_graph(graph)

        for node in graph.nodes:
            # Ghost nodes → counterfactuals
            if node.node_type == NodeType.GHOST:
                payload = node.payload or {}
                # Only add plan-level ghost nodes (not step-level)
                if payload.get("plan_id") and payload.get("score") is not None:
                    self.record_counterfactual(
                        plan_id=payload["plan_id"],
                        plan_name=node.label.split("\n")[0] if "\n" in node.label else node.label,
                        score=payload.get("score", 0.0),
                        reasoning=node.reasoning or "",
                    )
            # Tool call nodes → tool records (if execution data present)
            elif node.node_type == NodeType.TOOL_CALL:
                payload = node.payload or {}
                if "actual_success" in payload:
                    self.record_tool_execution(
                        name=payload.get("tool_name", ""),
                        args=payload.get("args", {}),
                        output=payload.get("actual_output", ""),
                        result="success" if payload.get("actual_success", True) else "fail",
                    )

    # ── Extract timing from SequenceTraceCollector spans ───────────────────

    def extract_from_sequence_collector(self) -> None:
        """Extract span timings from the global SequenceTraceCollector.

        Pairs call/return events to compute durations per participant
        and populates the explainability spans.
        """
        collector = get_sequence_collector()
        if not collector.enabled and not collector.events:
            return

        # Group return events by callee to get per-participant timings
        participant_times: dict[str, list[float]] = {}
        for ev in collector.events:
            if ev.event_type == "return" and ev.duration_s is not None:
                participant_times.setdefault(ev.callee, []).append(ev.duration_s)

        for participant, durations in participant_times.items():
            total_time = sum(durations)
            self.record_span(name=participant, time_taken=total_time)

        # Director agent time = sum of all Director-related spans
        director_time = 0.0
        for participant, durations in participant_times.items():
            if participant.lower() in ("director", "directorate", "goatnode"):
                director_time += sum(durations)
        if director_time > 0:
            self.set_director_agent_time(director_time)

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all collected telemetry data."""
        self._llm_calls.clear()
        self._tool_records.clear()
        self._spans.clear()
        self._counterfactuals.clear()
        self._global_error = None
        self._graph = None
        self._director_agent_time = 0.0
        self._start_time = time.perf_counter()


# ── Module-level singleton ────────────────────────────────────────────────────

_telemetry_collector: Optional[TelemetryCollector] = None


def get_telemetry_collector() -> TelemetryCollector:
    """Return the global :class:`TelemetryCollector` singleton.

    Lazily created on first call. Use :meth:`reset` to clear between runs.
    """
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector


def reset_telemetry_collector() -> None:
    """Reset the global TelemetryCollector (call between independent runs)."""
    global _telemetry_collector
    if _telemetry_collector is not None:
        _telemetry_collector.reset()
