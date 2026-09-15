"""LLMClassifier — LLM-based plan-complexity classifier.

Uses an LLM to determine whether a user utterance needs multiple plans (K>1).
This replaces the previous pattern-matching approach with more sophisticated
semantic understanding.

Returns confidence based on the LLM's certainty in its classification.
"""

from __future__ import annotations
from typing import Any, Optional

from ..interfaces import LLMClient
from ..types import PlanComplexity, PlanningContext
from ..utils import extractJson
from ..utils.logger import GoatLogger, TraceContext, traced, TraceSpan

log = GoatLogger("langgraph_goat._core.complexity")


_COMPLEXITY_SYSTEM_PROMPT = """\
You are a Plan Complexity Classifier for a multi-agent system. Your task is to analyze user requests and determine whether they need a single plan (K=1) or multiple alternative plans (K>1) to be properly addressed.

CRITICAL DISTINCTION - Classify based on ACTION clarity, not goal clarity:
- K=1: User explicitly states a SPECIFIC ACTION (verb + clear target)
- K>=2: User states a STATE, GOAL, or IMPLICIT NEED without specifying HOW to achieve it

Analyze the user's request and decide:
- K=1: Direct action commands (e.g., "call mom", "play music", "set an alarm")
- K>=2: Implicit statements, goals, or state descriptions (e.g., "I'm hungry", "I'm bored", "I'm cold", "plan my morning", "I need to travel")

Return ONLY a valid JSON object with this exact shape:
{
  "needs_multiple_plans": <boolean>,
  "suggested_K": <integer, 1 to max_K (configured by system, typically 3-5)>,
  "reason": "<brief explanation of your decision>",
  "confidence": <float, 0.0-1.0>
}

Guidelines:
- Direct action commands with explicit verb and target → K=1, high confidence (0.8-0.95)
  - Examples: "call mom", "play a song", "set an alarm", "send an email"
  - MUST have explicit ACTION (verb) and clear target
  - NOT just stating a goal or state

- Implicit statements, goals, or needs without explicit action → K>=2, high confidence (0.75-0.95)
  - Examples: "I'm hungry", "I'm bored", "I'm cold", "I'm tired"
  - "plan my morning", "help me prep for travel", "I need to relax"
  - These state a state/goal but NOT HOW to achieve it
  - Multiple valid approaches exist (order food vs cook vs go out for "I'm hungry")

- Complex, multi-faceted requests with many dependencies → K=4-5
- Consider the available tools when making your decision
- ALWAYS treat statements starting with "I'm [state]" or "I feel [state]" as K>=2
"""



def _format_context_for_classifier(context: PlanningContext) -> str:
    """Format the planning context as a human-readable prompt for the classifier."""
    lines = [f"USER REQUEST: {context.user_input}"]
    
    if context.available_tools:
        lines.append("\nAVAILABLE TOOLS:")
        for tool in context.available_tools:
            lines.append(f"  - {tool.name}: {tool.description}")
    
    if context.memory_snippets:
        lines.append("\nRELEVANT CONTEXT:")
        for snippet in context.memory_snippets[:3]:  # Limit to first 3 snippets
            lines.append(f"  - {snippet}")
    
    return "\n".join(lines)


class LLMClassifier:
    """LLM-based PlanComplexityClassifier implementation."""

    name = "llm"

    def __init__(
        self, 
        llm: LLMClient | None = None,
        model: str | None = None,
        default_K_when_uncertain: int = 3, 
        max_K: int = 5
    ):
        """
        Args:
            llm: LLMClient instance.
            model: Override the LLM's default model.
            default_K_when_uncertain: Default K value when uncertain.
            max_K: Maximum allowed K value.
        """
        self.llm = llm
        self.model = model
        self.default_K_when_uncertain = default_K_when_uncertain
        self.max_K = max_K

    @traced("langgraph_goat._core.complexity")
    def classify(
        self,
        context: PlanningContext,
        trace_context: Optional[TraceContext] = None,
    ) -> PlanComplexity:
        """Classify plan complexity using LLM reasoning."""
        ctx = trace_context or TraceContext()
        log.info("LLMClassifier.classify: starting classification", trace_context=ctx)

        formatted_context = _format_context_for_classifier(context)
        
        user_message = (
            f"Analyze this request and determine if it needs multiple plans:\n\n"
            f"{formatted_context}"
        )
        
        response: str = ""
        with TraceSpan("llm_classifier_complete", caller="LLMClassifier", callee="LLMClient", trace_context=ctx):
            try:
                response = self.llm.complete(
                    messages=[
                        {"role": "system", "content": _COMPLEXITY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    model=self.model,
                    temperature=0.3,  # Low temperature for consistent classification
                    max_tokens=1024,
                )
                log.debug("LLMClassifier.classify: LLM response received", trace_context=ctx)
            except Exception as e:
                log.warning("LLMClassifier.classify: LLM call failed: %s", e, trace_context=ctx)
                # Fallback to default K>1 with moderate confidence
                K = min(self.default_K_when_uncertain, self.max_K)
                return PlanComplexity(
                    needs_multiple_plans=True,
                    suggested_K=K,
                    reason="LLM classification failed or returned invalid response. Defaulting to K>1 for safety.",
                    confidence=0.5,
                )

        with TraceSpan("llm_classifier_parse", caller="LLMClassifier", callee="LLMClassifier", trace_context=ctx):
            parsed = extractJson(response)
            if parsed:
                # Extract and validate the classification results
                needs_multiple_plans = bool(parsed.get("needs_multiple_plans", False))
                suggested_K = int(parsed.get("suggested_K", self.default_K_when_uncertain))
                reason = str(parsed.get("reason", "LLM classification completed"))
                confidence = float(parsed.get("confidence", 0.7))
                
                # Apply bounds
                suggested_K = max(1, min(suggested_K, self.max_K))
                confidence = max(0.0, min(confidence, 1.0))

                log.info(
                    "LLMClassifier.classify: result K=%d confidence=%.2f reason=%s",
                    suggested_K, confidence, reason,
                    trace_context=ctx,
                )
                
                return PlanComplexity(
                    needs_multiple_plans=needs_multiple_plans,
                    suggested_K=suggested_K,
                    reason=reason,
                    confidence=confidence,
                )
            
            # Fallback: JSON parsing failed - default to K>1 with moderate confidence
            K = min(self.default_K_when_uncertain, self.max_K)
            log.warning(
                "LLMClassifier.classify: JSON parsing failed, defaulting to K=%d",
                K,
                trace_context=ctx,
            )
            # Record error to telemetry
            try:
                from ..utils.trace_logger import get_telemetry_collector
                tc = get_telemetry_collector()
                tc.record_pipeline_error(
                    stage="classifier",
                    error_type="JSONParseError",
                    message=f"LLMClassifier.classify: JSON parsing failed, defaulting to K={K}",
                    traceback=f"Raw LLM response:\n{response}",
                )
            except Exception:
                pass  # Don't fail if telemetry recording fails

        return PlanComplexity(
            needs_multiple_plans=True,
            suggested_K=K,
            reason="LLM classification failed or returned invalid response. Defaulting to K>1 for safety.",
            confidence=0.5,
        )

