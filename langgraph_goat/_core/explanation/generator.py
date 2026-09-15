"""LLM-based audience-adapted explanation generator.

Given a GoATGraph, produces:
  - developer_explanation: technical, references node IDs and rejection
    justifications, includes counterfactual reasoning
  - user_explanation: plain-language, brief, hints at alternatives only
    where they meaningfully changed the outcome
"""

from __future__ import annotations
from typing import Any, Optional

from ..interfaces import LLMClient
from ..types import Explanation, GoATGraph, NodeType
from ..utils import extractJson
from ..utils.logger import GoatLogger, TraceContext, traced, TraceSpan

log = GoatLogger("langgraph_goat._core.explanation")


_EXPLAIN_SYSTEM = """\
You are an Explanation Generator for GoAT (Graph of Agentic Thoughts).
You will receive a structured summary of a reasoning graph below.
Your job is to produce TWO DIFFERENT explanations as a single JSON object.

You MUST read ALL the data provided below carefully, including:
- The factual plan (the winning plan) and its steps
- The counterfactual plans (rejected alternatives) with their rejection reasons
- The critic scores and verdict reasoning
- Any memory/context used

KEY 1 - "developer_explanation":
- Written for a software developer.
- MUST be detailed and thorough. At least 5-8 sentences.
- Start with the flow: Director -> [Config] -> [LTM] memory(if any) -> [Plan] winning plan name
- List EVERY step of the winning plan with its tool name, args, and justification.
- For EACH rejected (counterfactual) plan: state the plan name, its steps, the critic score, and the rejection reason.
- Include the critic's overall reasoning if available.
- Use arrow notation (->) and bracket tags like [Plan], [Counterfactual], [LTM], [Tool], [Critic].
- Do NOT skip counterfactual plans. If counterfactual plans are listed in the data, you MUST include ALL of them.
- Only say "no counterfactual plans" if the data truly lists zero rejected plans.

KEY 2 - "user_explanation":
- Written for a non-technical end user.
- Exactly 3 to 5 sentences. Plain language. No jargon.
- Explain what the agent decided to do and why, in simple words.
- Briefly mention that other approaches were considered but not chosen (without technical details).
- Do NOT use words like "counterfactual", "node", "graph", "plan_id", "critic", or "score".
- Do NOT copy or paraphrase the developer_explanation. Use completely different wording.

OUTPUT RULES:
1. Output ONLY a JSON object. No text before or after it.
2. The JSON must have exactly two keys: "developer_explanation" and "user_explanation".
3. Use double quotes for all strings and keys.
4. Do not add trailing commas.
5. The developer_explanation must be LONGER and more detailed than the user_explanation.

OUTPUT FORMAT (follow this exactly):
{
  "developer_explanation": "Your detailed technical explanation here.",
  "user_explanation": "Your 3-5 sentence plain language explanation here."
}

EXAMPLE:
{
  "developer_explanation": "Director -> [Config] K=2 -> [LTM] memory(context: user hungry, preference: vegetarian) -> [Plan] order_food (Factual, score=8.5, reason: directly addresses hunger with minimal effort). Steps: 1) order_food(item=biryani, justification: user preference) 2) set_packing_reminder(justification: proactive). -> [Counterfactual] cook_food (Rejected, score=6.0, reason: higher effort, slower). -> [Counterfactual] play_music (Rejected, score=4.0, reason: does not address hunger). Critic verdict: order_food scored highest due to relevance and efficiency.",
  "user_explanation": "The assistant ordered food for you because you said you were hungry. It chose food delivery over cooking because it is faster. Other options like listening to music were considered but did not directly solve your hunger."
}
"""




def _summarize_graph_for_explainer(graph: GoATGraph) -> str:
    """Build a compact text representation of the graph for the LLM.

    Includes: user request, plan complexity, factual path nodes,
    ghost/counterfactual nodes, critic verdict (scores + rejections),
    and execution record (final answer).
    """
    parts: list[str] = []
    parts.append(f"User request: {graph.user_input}")

    if graph.plan_complexity:
        parts.append(
            f"Plan complexity: K={graph.plan_complexity.suggested_K} "
            f"(reason: {graph.plan_complexity.reason})"
        )

    # ── Factual path (winning plan steps) ─────────────────────────────────
    parts.append("\nFactual path (winning plan):")
    for node in graph.factual_path_nodes:
        reasoning_text = node.reasoning if node.reasoning else "No reasoning"
        if isinstance(reasoning_text, str):
            reasoning_text = reasoning_text[:300]
        parts.append(
            f"  - [{node.node_type.value}] {node.label}: {reasoning_text}"
        )

    # ── Ghost/counterfactual nodes (from graph structure) ─────────────────
    ghosts = graph.ghost_nodes
    if ghosts:
        parts.append("\nCounterfactual (rejected) plan nodes:")
        for ghost in ghosts:
            parts.append(f"  - {ghost.label}")
            rejection_text = ghost.reasoning if ghost.reasoning else "No rejection reason"
            if isinstance(rejection_text, str):
                rejection_text = rejection_text[:400]
            parts.append(f"    REJECTION: {rejection_text}")

    # ── Critic verdict (scores + rejection justifications) ────────────────
    # This is the PRIMARY source of counterfactual data — without it the
    # LLM has no visibility into why plans were rejected.
    cv = graph.critic_verdict
    if cv:
        parts.append("\nCritic verdict:")
        parts.append(f"  Winning plan ID: {cv.winner_plan_id}")
        if cv.scores:
            parts.append("  Scores:")
            for plan_id, score in cv.scores.items():
                marker = " (WINNER)" if plan_id == cv.winner_plan_id else ""
                parts.append(f"    - {plan_id}: {score}{marker}")
        if cv.rejection_justifications:
            parts.append("  Rejection justifications (counterfactual reasons):")
            for plan_id, reason in cv.rejection_justifications.items():
                # reason may be a dict or string
                if isinstance(reason, dict):
                    reason = "; ".join(f"{k}: {v}" for k, v in reason.items())
                elif not isinstance(reason, str):
                    reason = str(reason)
                parts.append(f"    - {plan_id}: {reason[:400]}")
        if cv.overall_reasoning:
            parts.append(f"  Overall reasoning: {cv.overall_reasoning[:500]}")

    # ── Execution record ──────────────────────────────────────────────────
    if graph.execution_record:
        final_answer = graph.execution_record.final_answer if graph.execution_record.final_answer else "No final answer"
        if isinstance(final_answer, str):
            final_answer = final_answer[:400]
        parts.append(f"\nFinal answer: {final_answer}")

    # ── Metadata (memory snippets, etc.) ──────────────────────────────────
    if graph.metadata:
        mem = graph.metadata.get("memory_snippets") or graph.metadata.get("memory")
        if mem:
            parts.append(f"\nMemory context used: {mem}")

    return "\n".join(parts)



class LLMExplanationGenerator:
    """Default explanation generator using an LLM."""

    name = "llm"

    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        self.model = model

    @traced("langgraph_goat._core.explanation")
    def generate(
        self,
        graph: GoATGraph,
        trace_context: Optional[TraceContext] = None,
    ) -> Explanation:
        """Generate audience-adapted explanations from a GoAT graph."""
        ctx = trace_context or TraceContext()
        log.info(
            "LLMExplanationGenerator.generate: graph_id=%s", graph.graph_id,
            trace_context=ctx,
        )

        with TraceSpan("summarize_graph", caller="LLMExplanationGenerator", callee="LLMExplanationGenerator", trace_context=ctx):
            summary = _summarize_graph_for_explainer(graph)

        with TraceSpan("explanation_llm_complete", caller="LLMExplanationGenerator", callee="LLMClient", trace_context=ctx):
            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": _EXPLAIN_SYSTEM},
                    {"role": "user", "content": summary},
                ],
                model=self.model,
                temperature=0.4,
                max_tokens=1500,
            )
            log.debug(
                "LLMExplanationGenerator.generate: LLM response received",
                trace_context=ctx,
            )

        with TraceSpan("explanation_parse_response", caller="LLMExplanationGenerator", callee="LLMExplanationGenerator", trace_context=ctx):
            parsed = extractJson(response)
            if parsed:
                explanation = Explanation(
                    developer_explanation=parsed.get("developer_explanation", ""),
                    user_explanation=parsed.get("user_explanation", ""),
                    raw_output=response,
                )
                log.info(
                    "LLMExplanationGenerator.generate: explanation generated successfully",
                    trace_context=ctx,
                )
                return explanation

        # Fallback: use raw text for both
        log.warning(
            "LLMExplanationGenerator.generate: JSON parsing failed, using raw response as fallback",
            trace_context=ctx,
        )
        return Explanation(
            developer_explanation=response,
            user_explanation=response[:400],
            raw_output=response,
        )
