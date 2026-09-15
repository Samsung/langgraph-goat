"""LLMJudgeCritic — default Plan Critic implementation.

Uses an LLM to score K candidate plans and produce rejection justifications
for losing plans. Built-in mitigations against known LLM-as-judge biases:

  1. Optional cross-model critic: pass a different model than the planner
     to reduce self-preference bias.
  2. Score-distribution logging: if all scores are within 0.5 of each other,
     log a warning — that's a sign the critic isn't discriminating.
"""

from __future__ import annotations
from typing import Optional

from ..interfaces import LLMClient
from ..types import CriticVerdict, Plan, PlanningContext
from ..utils import extractJson
from ..utils.logger import GoatLogger, TraceContext, traced, TraceSpan

log = GoatLogger("langgraph_goat._core.critic")


_CRITIC_SYSTEM = """\
You are the Plan Critic for a multi-agent system. You will receive a user \
request and several candidate plans. Score each plan from 0.0 to 10.0 on \
overall quality (relevance, correctness, efficiency, robustness). \
Pick the best plan and explain why each non-winning plan was rejected \
relative to the winner. Be specific — cite what the winner does better \
than each loser.

Return ONLY a valid, complete, parsable JSON object with this exact shape:
{
  "winner_plan": "<plan_id of the best plan>",
  "scores": {
    "plan_id_1": {
      "score": 10.0,
      "justification": "winning justification"
    },
    "plan_id_2": {
      "score": 7.5,
      "justification": "rejection reason relative to winner"
    },
    "plan_id_3": {
      "score": 5.0,
      "justification": "rejection reason relative to winner"
    }
  },
  "overall_reasoning": "your overall analysis"
}

CRITICAL RULES:
1. Return ONLY the JSON object - no other text, no markdown code blocks, no explanation.
2. Include ALL plans in the scores object with complete, closed braces for each plan entry.
3. EVERY opening brace "{" must have a matching closing brace "}".
4. EVERY array "[" must have a matching closing bracket "]".
5. Ensure the entire JSON is valid and complete - do not truncate or cut off entries.
6. For the winning plan, the justification should explain why it won.
7. For losing plans, the justification should explain why they lost relative to the winner.
"""


def _format_plan_for_judge(plan: Plan, idx: int) -> str:
    """Render one plan as a human-readable block for the critic prompt."""
    lines = [f"Plan: {plan.plan_id}", f"Reasoning: {plan.overall_reasoning or '(none)'}"]
    if plan.steps:
        lines.append("Steps:")
        for i, step in enumerate(plan.steps, 1):
            arg_str = ", ".join(f"{k}={v!r}" for k, v in (step.args or {}).items())
            lines.append(
                f"  {i}. {step.tool_name}({arg_str}) — {step.justification or '(no justification)'}"
            )
    else:
        lines.append("Steps: (none)")
    return "\n".join(lines)


class LLMJudgeCritic:
    """Default LLM-as-judge Plan Critic."""

    name = "llm_judge"

    def __init__(
        self,
        llm: LLMClient,
        model: str | None = None,
    ):
        """
        Args:
            llm: any LLMClient.
            model: override the LLM's default model
        """
        self.llm = llm
        self.model = model

    @traced("langgraph_goat._core.critic")
    def score(
        self,
        plans: list[Plan],
        context: PlanningContext,
        trace_context: Optional[TraceContext] = None,
    ) -> CriticVerdict:
        """Score K candidate plans and return a CriticVerdict."""
        ctx = trace_context or TraceContext()
        log.info(
            "LLMJudgeCritic.score: scoring %d plans", len(plans),
            trace_context=ctx,
        )

        if len(plans) < 2:
            raise ValueError("Critic needs at least 2 plans to score.")

        # Format plans for critique (no shuffling with new format)
        plan_blocks = [_format_plan_for_judge(p, i) for i, p in enumerate(plans)]

        user_msg = (
            f"USER REQUEST:\n{context.user_input}\n\n"
            f"CANDIDATE PLANS:\n\n" + "\n\n".join(plan_blocks)
        )
        log.debug(user_msg)

        with TraceSpan("critic_llm_complete", caller="LLMJudgeCritic", callee="LLMClient", trace_context=ctx):
            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": _CRITIC_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                model=self.model,
                temperature=0.2,  # judges should be steady, not creative
                max_tokens=4096,  # Increased to ensure response completes without truncation
            )
            log.debug("LLMJudgeCritic.score: LLM response received", trace_context=ctx)
            log.debug("LLMJudgeCritic.score: raw response (first 2000 chars):\n%s", response[:2000] if response else "EMPTY")

        with TraceSpan("critic_parse_response", caller="LLMJudgeCritic", callee="LLMJudgeCritic", trace_context=ctx):
            parsed = extractJson(response)
            if not parsed:
                log.warning(
                    "LLMJudgeCritic.score: JSON parsing failed, using fallback verdict",
                    trace_context=ctx,
                )
                # Record error to telemetry
                try:
                    from ..utils.trace_logger import get_telemetry_collector
                    tc = get_telemetry_collector()
                    tc.record_pipeline_error(
                        stage="critic",
                        error_type="JSONParseError",
                        message="LLMJudgeCritic.score: JSON parsing failed, using fallback verdict",
                        traceback=f"Raw LLM response:\n{response}",
                    )
                except Exception:
                    pass  # Don't fail if telemetry recording fails
                return self._fallback_verdict(plans, response, trace_context=ctx)


            # Parse new format: winner_plan (string) and scores as objects
            winner_plan_id: str = ""
            scores_data: list = []
            try:
                winner_plan_id = parsed["winner_plan"]
                scores_data = parsed.get("scores", [])
            except (KeyError, TypeError):
                log.warning(
                    "LLMJudgeCritic.score: verdict parsing failed, using fallback verdict",
                    trace_context=ctx,
                )
                return self._fallback_verdict(plans, response, trace_context=ctx)

        # Build scores_by_plan_id and rejections_by_plan_id from the new format
        scores_by_plan_id: dict[str, float] = {}
        rejections_by_plan_id: dict[str, str] = {}

        # Parse scores object which maps plan_id to {score, justification}
        if isinstance(scores_data, dict):
            for plan_id, score_obj in scores_data.items():
                if isinstance(score_obj, dict):
                    try:
                        score = float(score_obj.get("score", 0.0))
                        justification = str(score_obj.get("justification", ""))
                        scores_by_plan_id[plan_id] = score

                        # Only store rejection justification if plan didn't win
                        if plan_id != winner_plan_id and justification:
                            rejections_by_plan_id[plan_id] = justification
                    except (TypeError, ValueError):
                        scores_by_plan_id[plan_id] = 0.0

        # Ensure all plans have a score (fallback to 0.0 if missing)
        for plan in plans:
            if plan.plan_id not in scores_by_plan_id:
                scores_by_plan_id[plan.plan_id] = 0.0


        # Validate: the winner must have the highest score.
        # LLMs can be inconsistent (winner_plan ≠ highest score); correct it.
        actual_winner_plan_id = winner_plan_id
        actual_winner_score = scores_by_plan_id.get(actual_winner_plan_id, 0.0)
        best_plan_id = max(scores_by_plan_id, key=lambda pid: scores_by_plan_id[pid]) if scores_by_plan_id else ""
        best_score = scores_by_plan_id.get(best_plan_id, 0.0) if best_plan_id else 0.0

        if best_plan_id != actual_winner_plan_id:
            log.info(
                "LLMJudgeCritic.score: correcting inconsistent winner from %s (score=%.1f) to %s (score=%.1f)",
                actual_winner_plan_id, actual_winner_score,
                best_plan_id, best_score,
                trace_context=ctx,
            )
            # LLM was inconsistent — override with the true highest-scoring plan
            # Move rejection justification from old winner to new winner if needed
            old_winner_rejection = rejections_by_plan_id.pop(actual_winner_plan_id, "")
            if old_winner_rejection:
                rejections_by_plan_id[actual_winner_plan_id] = old_winner_rejection
            # Remove rejection for the new winner (it won)
            rejections_by_plan_id.pop(best_plan_id, None)
            actual_winner_plan_id = best_plan_id

        # Score-distribution check: warn if all scores are within 0.5
        score_values = list(scores_by_plan_id.values())
        if score_values and (max(score_values) - min(score_values)) < 0.5:
            log.warning(
                "LLMJudgeCritic.score: scores are tightly clustered (range=%.2f), critic may not be discriminating",
                max(score_values) - min(score_values),
                trace_context=ctx,
            )

        log.info(
            "LLMJudgeCritic.score: winner=%s scores=%s",
            actual_winner_plan_id, scores_by_plan_id,
            trace_context=ctx,
        )

        return CriticVerdict(
            winner_plan_id=actual_winner_plan_id,
            scores=scores_by_plan_id,
            rejection_justifications=rejections_by_plan_id,
            overall_reasoning=parsed.get("overall_reasoning", ""),
            judge_model=self.model,
        )

    def _fallback_verdict(
        self,
        plans: list[Plan],
        raw_response: str,
        trace_context: Optional[TraceContext] = None,
    ) -> CriticVerdict:
        """When JSON extraction fails — degrade gracefully.

        First plan wins by default (no info to discriminate). All non-winners
        get a placeholder rejection so the graph still renders.
        """
        log.warning(
            "LLMJudgeCritic._fallback_verdict: using fallback scoring",
            trace_context=trace_context,
        )
        return CriticVerdict(
            winner_plan_id=plans[0].plan_id,
            scores={p.plan_id: 5.0 for p in plans},
            rejection_justifications={
                p.plan_id: "(critic returned unparseable response — defaulting)"
                for p in plans[1:]
            },
            overall_reasoning=f"FALLBACK: critic JSON parsing failed. Raw response:\n{raw_response[:500]}",
            judge_model=self.model,
        )
