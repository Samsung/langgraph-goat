"""LangGraphExecutionRecorder — captures execution outcomes during graph execution.

Use this as a callback hook in your LangGraph after the planner has chosen
a plan. As nodes execute, call record_step() with each step's outcome.
At the end, call finalize_plan() to get an ExecutionRecord that you can
feed back into Director.record_execution().
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional

from ._core import ExecutionRecord, StepExecutionRecord
from ._core.utils.logger import GoatLogger, TraceContext, traced

log = GoatLogger("langgraph_goat.recorder")


class LangGraphExecutionRecorder:
    """Collects per-step execution records during plan execution."""

    name = "langgraph"

    def __init__(self):
        self._step_records: dict[str, list[StepExecutionRecord]] = {}
        self._step_starts: dict[str, datetime] = {}

    @traced("langgraph_goat.recorder")
    def begin_step(self, plan_id: str, step_id: str, trace_context: Optional[TraceContext] = None) -> None:
        """Mark a step's start time. Call this just before the node executes."""
        key = f"{plan_id}::{step_id}"
        self._step_starts[key] = datetime.now(timezone.utc)
        log.debug(
            "LangGraphExecutionRecorder.begin_step: plan_id=%s step_id=%s",
            plan_id, step_id,
            trace_context=trace_context,
        )

    @traced("langgraph_goat.recorder")
    def end_step(
        self,
        plan_id: str,
        step_id: str,
        success: bool,
        output: Any = None,
        output_summary: str = "",
        error: str | None = None,
        trace_context: Optional[TraceContext] = None,
    ) -> None:
        """Mark a step's completion."""
        key = f"{plan_id}::{step_id}"
        started = self._step_starts.pop(key, datetime.now(timezone.utc))
        record = StepExecutionRecord(
            step_id=step_id,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            success=success,
            output=output,
            output_summary=output_summary or self._truncate_output(output),
            error=error,
        )
        self._step_records.setdefault(plan_id, []).append(record)
        log.debug(
            "LangGraphExecutionRecorder.end_step: plan_id=%s step_id=%s success=%s",
            plan_id, step_id, success,
            trace_context=trace_context,
        )

    @traced("langgraph_goat.recorder")
    def record_step(self, plan_id: str, record: StepExecutionRecord, trace_context: Optional[TraceContext] = None) -> None:
        """Direct record submission — use when you already have a StepExecutionRecord."""
        self._step_records.setdefault(plan_id, []).append(record)
        log.debug(
            "LangGraphExecutionRecorder.record_step: plan_id=%s step_id=%s",
            plan_id, record.step_id,
            trace_context=trace_context,
        )

    @traced("langgraph_goat.recorder")
    def finalize_plan(self, plan_id: str, final_answer: str, trace_context: Optional[TraceContext] = None) -> ExecutionRecord:
        """Produce the final ExecutionRecord for the plan. Adapter feeds this
        back into Director.record_execution() to update the graph."""
        steps = self._step_records.get(plan_id, [])
        total_duration = 0
        for s in steps:
            total_duration += int(
                (s.completed_at - s.started_at).total_seconds() * 1000
            )
        log.info(
            "LangGraphExecutionRecorder.finalize_plan: plan_id=%s steps=%d total_duration_ms=%d",
            plan_id, len(steps), total_duration,
            trace_context=trace_context,
        )
        return ExecutionRecord(
            plan_id=plan_id,
            steps=steps,
            final_answer=final_answer,
            total_duration_ms=total_duration,
        )

    @traced("langgraph_goat.recorder")
    def reset(self, trace_context: Optional[TraceContext] = None) -> None:
        """Clear recorded state — call between independent runs."""
        self._step_records.clear()
        self._step_starts.clear()
        log.debug("LangGraphExecutionRecorder.reset: cleared all records", trace_context=trace_context)

    @staticmethod
    def _truncate_output(output: Any) -> str:
        if output is None:
            return ""
        s = str(output)
        return s[:300] + "..." if len(s) > 300 else s
