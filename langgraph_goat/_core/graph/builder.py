"""GoAT graph builder.

Produces a unified GoATGraph regardless of K. Single-intent runs (K=1)
get the same node types but no ghost nodes. The graph schema is
intentionally identical so downstream tools (viewers, evaluators) handle
one shape.
"""

from __future__ import annotations
import uuid
from typing import Optional

from ..types import (
    CriticVerdict,
    EdgeKind,
    ExecutionRecord,
    GoATEdge,
    GoATGraph,
    GoATNode,
    NodeType,
    Plan,
    PlanComplexity,
    PlanningContext,
)
from ..utils.logger import GoatLogger, TraceContext, traced, TraceSpan

log = GoatLogger("langgraph_goat._core.graph")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class GraphBuilder:
    """Stateless graph constructor.

    Two entry points:
        build_simple(...) — K=1 path, no ghost nodes
        build_with_ghosts(...) — K>1 path, ghost nodes with rejections
    """

    @traced("langgraph_goat._core.graph")
    def build_simple(
        self,
        context: PlanningContext,
        plan: Plan,
        complexity: PlanComplexity,
        execution: ExecutionRecord | None = None,
        trace_context: Optional[TraceContext] = None,
    ) -> GoATGraph:
        """Build a simple K=1 graph with no ghost nodes."""
        ctx = trace_context or TraceContext()
        log.info(
            "GraphBuilder.build_simple: plan_id=%s", plan.plan_id,
            trace_context=ctx,
        )

        with TraceSpan("create_simple_graph", caller="GraphBuilder", callee="GraphBuilder", trace_context=ctx):
            graph = GoATGraph(
                graph_id=_new_id("graph"),
                user_input=context.user_input,
                plan_complexity=complexity,
                execution_record=execution,
            )

            # Director node — captures the classifier's reasoning
            director = GoATNode(
                node_id="director",
                node_type=NodeType.DIRECTOR,
                label="DirectorAgent",
                reasoning=(
                    f"Plan complexity: K=1 (single intent). {complexity.reason}"
                ),
                payload={"selected_plan_id": plan.plan_id},
            )
            graph.nodes.append(director)

            last_id = self._add_factual_path(graph, plan, execution, parent_id="director", trace_context=ctx)

            # Final answer
            final_id = "answer"
            final_text = execution.final_answer if execution else "(not yet executed)"
            graph.nodes.append(GoATNode(
                node_id=final_id,
                node_type=NodeType.FINAL_ANSWER,
                label=final_text,
                reasoning="Terminal output delivered to user",
                payload={"content": final_text},
            ))
            graph.edges.append(GoATEdge(
                edge_id=_new_id("e"),
                src=last_id,
                dst=final_id,
                kind=EdgeKind.CAUSAL_FLOW,
                reasoning="Answer synthesized from executed plan",
            ))

        log.info(
            "GraphBuilder.build_simple: graph_id=%s nodes=%d edges=%d",
            graph.graph_id, len(graph.nodes), len(graph.edges),
            trace_context=ctx,
        )
        return graph

    @traced("langgraph_goat._core.graph")
    def build_with_ghosts(
        self,
        context: PlanningContext,
        plans: list[Plan],
        verdict: CriticVerdict,
        complexity: PlanComplexity,
        execution: ExecutionRecord | None = None,
        trace_context: Optional[TraceContext] = None,
    ) -> GoATGraph:
        """Build a K>1 graph with ghost nodes for rejected plans."""
        ctx = trace_context or TraceContext()
        log.info(
            "GraphBuilder.build_with_ghosts: K=%d winner=%s",
            len(plans), verdict.winner_plan_id,
            trace_context=ctx,
        )

        with TraceSpan("create_graph_with_ghosts", caller="GraphBuilder", callee="GraphBuilder", trace_context=ctx):
            graph = GoATGraph(
                graph_id=_new_id("graph"),
                user_input=context.user_input,
                plan_complexity=complexity,
                critic_verdict=verdict,
                execution_record=execution,
            )

            winner = next((p for p in plans if p.plan_id == verdict.winner_plan_id), plans[0])

            # Director with full reasoning chain
            winner_score = verdict.scores.get(winner.plan_id, 0.0)
            director = GoATNode(
                node_id="director",
                node_type=NodeType.DIRECTOR,
                label="DirectorAgent",
                reasoning=(
                    f"Plan complexity: K={len(plans)}. {complexity.reason}\n\n"
                    f"Critic selected plan {winner.plan_id}\n(score: {winner_score:.1f}). "
                    f"{verdict.overall_reasoning}"
                ),
                payload={
                    "selected_plan_id": winner.plan_id,
                    "all_scores": verdict.scores,
                    "K": len(plans),
                },
            )
            graph.nodes.append(director)

            # Factual path (winner)
            last_id = self._add_factual_path(graph, winner, execution, parent_id="director", score=winner_score, trace_context=ctx)

            # Final answer
            final_id = "answer"
            final_text = execution.final_answer if execution else "(not yet executed)"
            graph.nodes.append(GoATNode(
                node_id=final_id,
                node_type=NodeType.FINAL_ANSWER,
                label="Final Answer",
                reasoning="Terminal output delivered to user",
                payload={"content": final_text},
            ))
            graph.edges.append(GoATEdge(
                edge_id=_new_id("e"),
                src=last_id,
                dst=final_id,
                kind=EdgeKind.CAUSAL_FLOW,
                reasoning="Answer synthesized from winning plan",
            ))

            # Ghost nodes (rejected plans) — plan header + per-step tool nodes
            with TraceSpan("add_ghost_nodes", caller="GraphBuilder", callee="GraphBuilder", trace_context=ctx):
                for plan in plans:
                    if plan.plan_id == winner.plan_id:
                        continue
                    score = verdict.scores.get(plan.plan_id, 0.0)
                    rejection = verdict.rejection_justifications.get(
                        plan.plan_id, "(no rejection justification provided)"
                    )
                    ghost_plan_id = _new_id("ghost")
                    graph.nodes.append(GoATNode(
                        node_id=ghost_plan_id,
                        node_type=NodeType.GHOST,
                        label=f"{plan.plan_name}\n(score = {score:.1f})",
                        reasoning=rejection,
                        payload={
                            "plan_id": plan.plan_id,
                            "score": score,
                        },
                    ))
                    graph.edges.append(GoATEdge(
                        edge_id=_new_id("e"),
                        src="director",
                        dst=ghost_plan_id,
                        kind=EdgeKind.COUNTERFACTUAL,
                        reasoning=rejection,
                    ))

                    # Ghost tool-call nodes for each step in the rejected plan
                    last_ghost_id = ghost_plan_id
                    for step in plan.steps:
                        ghost_step_id = _new_id("ghost")
                        graph.nodes.append(GoATNode(
                            node_id=ghost_step_id,
                            node_type=NodeType.GHOST,
                            label=f"🔧 {step.tool_name}",
                            reasoning=step.justification,
                            payload={
                                "tool_name": step.tool_name,
                                "args": step.args,
                                "expected_output": step.expected_output,
                            },
                        ))
                        graph.edges.append(GoATEdge(
                            edge_id=_new_id("e"),
                            src=last_ghost_id,
                            dst=ghost_step_id,
                            kind=EdgeKind.COUNTERFACTUAL,
                            reasoning=step.justification or "Would have executed",
                        ))
                        last_ghost_id = ghost_step_id

        log.info(
            "GraphBuilder.build_with_ghosts: graph_id=%s nodes=%d edges=%d ghosts=%d",
            graph.graph_id, len(graph.nodes), len(graph.edges), len(graph.ghost_nodes),
            trace_context=ctx,
        )
        return graph

    def _add_factual_path(
        self,
        graph: GoATGraph,
        plan: Plan,
        execution: ExecutionRecord | None,
        parent_id: str,
        score: float | None = None,
        trace_context: Optional[TraceContext] = None,
    ) -> str:
        """Add plan-name + tool-call nodes for the winning plan. Returns last node id.

        A PLAN_STEP node carrying the plan name is inserted between the director
        and the first tool call, making the causal chain:
            DirectorAgent → plan_name → tool_call(s) → Final answer
        Tool-call nodes added per plan step, enriched with execution outcomes
        if execution data is available.
        """
        ctx = trace_context
        log.debug(
            "GraphBuilder._add_factual_path: plan_id=%s steps=%d",
            plan.plan_id, len(plan.steps),
            trace_context=ctx,
        )

        # ── Plan-name node ───────────────────────────────────────────────────
        plan_node_id = _new_id("plan")
        plan_label = f"{plan.plan_name}\n(score = {score:.1f})" if score is not None else plan.plan_name
        plan_reasoning = plan.overall_reasoning or f"Plan {plan.plan_id} selected for execution"
        graph.nodes.append(GoATNode(
            node_id=plan_node_id,
            node_type=NodeType.PLAN_STEP,
            label=f"{plan_label}",
            reasoning=plan_reasoning,
            payload={
                "plan_id": plan.plan_id,
                "plan_name": plan.plan_name,
                "score": score,
                "num_steps": len(plan.steps),
            },
        ))
        graph.edges.append(GoATEdge(
            edge_id=_new_id("e"),
            src=parent_id,
            dst=plan_node_id,
            kind=EdgeKind.CAUSAL_FLOW,
            reasoning=plan_reasoning or "Plan selected",
        ))

        last_id = plan_node_id
        exec_by_step: dict = {}
        if execution:
            exec_by_step = {r.step_id: r for r in execution.steps}

        for step in plan.steps:
            step_node_id = _new_id("step")
            exec_record = exec_by_step.get(step.step_id)

            label = f"{step.tool_name}"
            payload: dict = {
                "tool_name": step.tool_name,
                "args": step.args,
                "expected_output": step.expected_output,
            }
            if exec_record:
                payload["actual_success"] = exec_record.success
                payload["actual_output"] = exec_record.output_summary
                payload["duration_ms"] = int(
                    (exec_record.completed_at - exec_record.started_at).total_seconds() * 1000
                )

            graph.nodes.append(GoATNode(
                node_id=step_node_id,
                node_type=NodeType.TOOL_CALL,
                label=label,
                reasoning=step.justification,
                payload=payload,
            ))
            graph.edges.append(GoATEdge(
                edge_id=_new_id("e"),
                src=last_id,
                dst=step_node_id,
                kind=EdgeKind.CAUSAL_FLOW,
                reasoning=step.justification or "Causal flow",
            ))
            last_id = step_node_id

        return last_id
