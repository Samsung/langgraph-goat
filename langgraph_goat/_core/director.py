"""Director — the GoAT engine orchestrator.

NOT a "DirectorAgent" in the LLM sense. This is the controller that
sequences: classify → generate → critique → build graph → explain.

The framework adapter constructs a Director with the components it
wants (classifier, generator, critic, etc.) and calls run() per query.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .graph.builder import GraphBuilder
from .interfaces import (
    ExplanationGenerator,
    GraphStore,
    PlanComplexityClassifier,
    PlanCritic,
    PlanGenerator,
)
from .types import (
    ExecutionRecord,
    Explanation,
    GoATGraph,
    Plan,
    PlanComplexity,
    PlanningContext,
)
from .utils.logger import GoatLogger, TraceContext, traced, TraceSpan

log = GoatLogger("langgraph_goat._core.director")


@dataclass
class DirectorResult:
    """What Director.run returns — graph + winning plan + explanation.

    The adapter consumes this: it executes `winning_plan` (or its
    framework-native equivalent) and feeds the result back via
    record_execution() to get a final enriched graph.
    """
    graph: GoATGraph
    winning_plan: Plan
    all_plans: list[Plan]
    complexity: PlanComplexity
    explanation: Optional[Explanation] = None


class Director:
    """Orchestrates the GoAT pipeline.

    Required components:
        plan_generator (PlanGenerator) — adapter-provided
        critic (PlanCritic) — only used when K>1; can be None if K is forced to 1
        graph_builder (GraphBuilder)

    Optional components:
        classifier (PlanComplexityClassifier) — if None, uses default_K config
        explanation_generator (ExplanationGenerator) — if None, no explanation
        graph_store (GraphStore) — if None, graph is returned only, not persisted

    Config:
        default_K: int — used when classifier is None or returns uncertain
        max_K: int — hard cap
        force_K: Optional[int] — bypass classifier entirely if set
    """

    def __init__(
        self,
        plan_generator: PlanGenerator,
        root_dir: str,
        critic: Optional[PlanCritic] = None,
        graph_builder: Optional[GraphBuilder] = None,
        classifier: Optional[PlanComplexityClassifier] = None,
        explanation_generator: Optional[ExplanationGenerator] = None,
        graph_store: Optional[GraphStore] = None,
        default_K: int = 3,
        max_K: int = 5,
        force_K: Optional[int] = None,
    ):
        self.plan_generator = plan_generator
        self.critic = critic
        self.graph_builder = graph_builder or GraphBuilder()
        self.classifier = classifier
        self.explanation_generator = explanation_generator
        self.graph_store = graph_store
        self.default_K = default_K
        self.max_K = max_K
        self.force_K = force_K

        # Validate root_dir early
        root_path = Path(root_dir)
        try:
            root_path.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = root_path / ".goat_write_test"
            test_file.touch()
            test_file.unlink()
            self.root_dir = str(root_path.resolve())
        except (PermissionError, OSError) as e:
            raise ValueError(
                f"Cannot write to root_dir '{root_dir}': {e}. "
                "Ensure directory exists and you have write permissions."
            ) from e

    @traced("langgraph_goat._core.director")
    def run(self, context: PlanningContext, trace_context: Optional[TraceContext] = None) -> DirectorResult:
        """Generate plans, score (if K>1), build graph. Does NOT execute the plan.

        Adapter's job is to execute `result.winning_plan` and call
        record_execution() to enrich the graph with actual outcomes.
        """
        # Create a root trace context for this run if not provided
        ctx = trace_context or TraceContext()
        log.info("Director.run: starting pipeline", trace_context=ctx)

        # Step 1: classify complexity (or force K)
        with TraceSpan("classify_complexity", caller="Director", callee="LLMClassifier", trace_context=ctx):
            complexity = self._determine_complexity(context, trace_context=ctx)
        K = complexity.suggested_K
        log.info("Director.run: K=%d (reason: %s)", K, complexity.reason, trace_context=ctx)

        
        # Only require critic when K > 1
        if K > 1 and not self.critic:
            raise ValueError(
                f"K={K} > 1 but no critic configured. "
                "Either provide a PlanCritic or use K=1."
            )
        with TraceSpan("generate_K_plans", caller="Director", callee="PlanGenerator", trace_context=ctx):
            plans = self.plan_generator.generate_K(context, K, trace_context=ctx)
        if len(plans) != K:
            log.warning(
                "PlanGenerator returned %d plans, expected %d", len(plans), K,
                trace_context=ctx,
            )
        with TraceSpan("critic_score", caller="Director", callee="Critic", trace_context=ctx):
            verdict = self.critic.score(plans, context, trace_context=ctx)
        log.info(
            "Director.run: critic winner=%s scores=%s",
            verdict.winner_plan_id, verdict.scores,
            trace_context=ctx,
        )
        with TraceSpan("build_graph_with_ghosts", caller="Director", callee="GraphBuilder", trace_context=ctx):
            graph = self.graph_builder.build_with_ghosts(
                context=context,
                plans=plans,
                verdict=verdict,
                complexity=complexity,
                trace_context=ctx,
            )
        ctx = ctx.with_ids(graph_id=graph.graph_id)
        winner = next(
            (p for p in plans if p.plan_id == verdict.winner_plan_id),
            plans[0],
        )
        result = DirectorResult(
            graph=graph,
            winning_plan=winner,
            all_plans=plans,
            complexity=complexity,
        )

        # Step 3: explanation (deferred if no generator configured)
        if self.explanation_generator:
            with TraceSpan("generate_explanation", caller="Director", callee="ExplanationGenerator", trace_context=ctx):
                try:
                    result.explanation = self.explanation_generator.generate(result.graph, trace_context=ctx)
                    log.info("Director.run: explanation generated", trace_context=ctx)
                except Exception as e:
                    log.warning("Explanation generation failed: %s", e, trace_context=ctx)

        # Step 4: persist
        if self.graph_store:
            with TraceSpan("persist_graph", caller="Director", callee="GraphStore", trace_context=ctx):
                try:
                    # Use proper Path operations for cross-platform compatibility
                    output_path = Path(self.root_dir) / graph.graph_id
                    output_path.mkdir(parents=True, exist_ok=True)
                    log.debug("Director.run: graph store path = %s", output_path, trace_context=ctx)
                    file_path = output_path / f"{graph.graph_id}.json"
                    self.graph_store.save(file_path, result.graph, trace_context=ctx)
                    log.info("Director.run: graph persisted", trace_context=ctx)
                except Exception as e:
                    log.warning("Graph store save failed: %s", e, trace_context=ctx)

        log.info("Director.run: pipeline complete", trace_context=ctx)
        return result

    @traced("langgraph_goat._core.director")
    def record_execution(
        self,
        result: DirectorResult,
        execution: ExecutionRecord,
        trace_context: Optional[TraceContext] = None,
    ) -> DirectorResult:
        """Re-build the graph with execution data merged in.

        Call this after the adapter has executed result.winning_plan and
        captured StepExecutionRecords. Returns an updated result with
        actual-outcome data on each step node.
        """
        ctx = trace_context or TraceContext()
        log.info("Director.record_execution: rebuilding graph with execution data", trace_context=ctx)

        # Rebuild the graph with execution data
        if result.graph.has_counterfactuals and result.graph.critic_verdict:
            with TraceSpan("rebuild_graph_with_ghosts", caller="Director", callee="GraphBuilder", trace_context=ctx):
                new_graph = self.graph_builder.build_with_ghosts(
                    context=PlanningContext(user_input=result.graph.user_input),
                    plans=result.all_plans,
                    verdict=result.graph.critic_verdict,
                    complexity=result.complexity,
                    execution=execution,
                    trace_context=ctx,
                )
        else:
            with TraceSpan("rebuild_simple_graph", caller="Director", callee="GraphBuilder", trace_context=ctx):
                new_graph = self.graph_builder.build_simple(
                    context=PlanningContext(user_input=result.graph.user_input),
                    plan=result.winning_plan,
                    complexity=result.complexity,
                    execution=execution,
                    trace_context=ctx,
                )

        # Preserve original graph_id so consumers can correlate
        new_graph.graph_id = result.graph.graph_id

        result.graph = new_graph

        # Regenerate explanation now that we have actual outcomes
        if self.explanation_generator:
            with TraceSpan("regenerate_explanation", caller="Director", callee="ExplanationGenerator", trace_context=ctx):
                try:
                    result.explanation = self.explanation_generator.generate(new_graph, trace_context=ctx)
                    log.info("Director.record_execution: explanation regenerated", trace_context=ctx)
                except Exception as e:
                    log.warning("Post-execution explanation regeneration failed: %s", e, trace_context=ctx)

        if self.graph_store:
            with TraceSpan("persist_graph_after_execution", caller="Director", callee="GraphStore", trace_context=ctx):
                try:
                    # Reconstruct the file path using the preserved graph_id
                    output_path = Path(self.root_dir) / result.graph.graph_id
                    output_path.mkdir(parents=True, exist_ok=True)
                    file_path = output_path / f"{result.graph.graph_id}.json"
                    self.graph_store.save(file_path, new_graph, trace_context=ctx)
                    log.info("Director.record_execution: graph persisted after execution", trace_context=ctx)
                except Exception as e:
                    log.warning("Graph store save failed: %s", e, trace_context=ctx)

        log.info("Director.record_execution: complete", trace_context=ctx)
        return result

    # ── Internals ────────────────────────────────────────────────────────────

    def _determine_complexity(
        self,
        context: PlanningContext,
        trace_context: Optional[TraceContext] = None,
    ) -> PlanComplexity:
        if self.force_K is not None:
            K = max(1, min(self.force_K, self.max_K))
            log.debug(
                "Director._determine_complexity: force_K=%d → K=%d",
                self.force_K, K,
                trace_context=trace_context,
            )
            return PlanComplexity(
                needs_multiple_plans=K > 1,
                suggested_K=K,
                reason=f"Forced by Director config (force_K={self.force_K}).",
                confidence=1.0,
            )

        if self.classifier:
            log.debug("Director._determine_complexity: delegating to classifier", trace_context=trace_context)
            complexity = self.classifier.classify(context, trace_context=trace_context)
            # Honor max_K cap
            if complexity.suggested_K > self.max_K:
                complexity.suggested_K = self.max_K
                complexity.reason += f" (capped at max_K={self.max_K})"
            log.debug(
                "Director._determine_complexity: classifier returned K=%d",
                complexity.suggested_K,
                trace_context=trace_context,
            )
            return complexity

        # No classifier configured — use default_K
        K = max(1, min(self.default_K, self.max_K))
        log.debug(
            "Director._determine_complexity: no classifier, default_K=%d → K=%d",
            self.default_K, K,
            trace_context=trace_context,
        )
        return PlanComplexity(
            needs_multiple_plans=K > 1,
            suggested_K=K,
            reason=f"No classifier configured. Using default_K={self.default_K}.",
            confidence=1.0,
        )
