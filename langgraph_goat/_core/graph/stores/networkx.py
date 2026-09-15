"""NetworkXStore — default GraphStore backed by NetworkX + JSON files.

Stores GoATGraphs as JSON files in a directory. Optional NetworkX
representation for users who want graph-algorithm operations
(centrality, paths, etc.). The JSON format is canonical so users can
migrate to Neo4j / Kùzu later without rewriting their graphs.

If networkx isn't installed, falls back to JSON-only storage.
"""

from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ...types import (
    CriticVerdict,
    EdgeKind,
    ExecutionRecord,
    GoATEdge,
    GoATGraph,
    GoATNode,
    NodeType,
    Plan,
    PlanComplexity,
    PlanStep,
    StepExecutionRecord,
)
from ...utils.logger import TraceContext


def _serialize(obj: Any) -> Any:
    """Recursive JSON serializer for our dataclasses + enums + datetime."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (NodeType, EdgeKind)):
        return obj.value
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _serialize(asdict(obj))
    return str(obj)


class NetworkXStore:
    """File-based GoATGraph store. NetworkX is optional."""

    name = "networkx"

    def __init__(self, root_dir: str | Path = "./goat_graphs"):
        self.root = Path(root_dir)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise ValueError(
                f"Cannot write to graph store directory '{root_dir}': {e}. "
                "Ensure directory exists and you have write permissions."
            ) from e
        try:
            import networkx  # noqa: F401
            self._nx_available = True
        except ImportError:
            self._nx_available = False

    def save(self, path: Path, graph: GoATGraph, trace_context: Optional[TraceContext] = None) -> str:
        """Persist a GoATGraph as a JSON file."""
        try:
            data = _serialize(graph)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Graph serialization failed: {e}") from e
        except (PermissionError, OSError, UnicodeEncodeError) as e:
            raise IOError(f"Cannot write graph to '{path}': {e}") from e
        return graph.graph_id

    def load(self, path: Path, graph_id: str, trace_context: Optional[TraceContext] = None) -> GoATGraph:
        """Load a GoATGraph from a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"No graph found at {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = self._deserialize(data)
            return graph
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in graph file '{path}': {e}") from e
        except (PermissionError, OSError, UnicodeDecodeError) as e:
            raise IOError(f"Cannot read graph from '{path}': {e}") from e

    def list_recent(self, limit: int = 20, trace_context: Optional[TraceContext] = None) -> list[str]:
        """List recent graph IDs sorted by modification time."""
        files = sorted(
            self.root.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [f.stem for f in files[:limit]]

    def to_networkx(self, graph: GoATGraph, trace_context: Optional[TraceContext] = None):
        """Convert to a NetworkX DiGraph for algorithm use. Optional."""
        if not self._nx_available:
            raise RuntimeError(
                "NetworkX not installed. Install with: pip install networkx"
            )
        import networkx as nx
        g = nx.DiGraph(graph_id=graph.graph_id, user_input=graph.user_input)
        for node in graph.nodes:
            g.add_node(
                node.node_id,
                node_type=node.node_type.value,
                label=node.label,
                reasoning=node.reasoning,
                **node.payload,
            )
        for edge in graph.edges:
            g.add_edge(
                edge.src,
                edge.dst,
                kind=edge.kind.value,
                reasoning=edge.reasoning,
                edge_id=edge.edge_id,
            )
        return g

    # ── Deserialization ─────────────────────────────────────────────────────

    @staticmethod
    def _deserialize(data: dict) -> GoATGraph:
        nodes = [
            GoATNode(
                node_id=n["node_id"],
                node_type=NodeType(n["node_type"]),
                label=n["label"],
                reasoning=n.get("reasoning", ""),
                payload=n.get("payload", {}),
            )
            for n in data.get("nodes", [])
        ]
        edges = [
            GoATEdge(
                edge_id=e["edge_id"],
                src=e["src"],
                dst=e["dst"],
                kind=EdgeKind(e["kind"]),
                reasoning=e.get("reasoning", ""),
            )
            for e in data.get("edges", [])
        ]
        complexity = None
        if data.get("plan_complexity"):
            pc = data["plan_complexity"]
            complexity = PlanComplexity(
                needs_multiple_plans=pc["needs_multiple_plans"],
                suggested_K=pc["suggested_K"],
                reason=pc["reason"],
                confidence=pc.get("confidence", 1.0),
            )
        verdict = None
        if data.get("critic_verdict"):
            cv = data["critic_verdict"]
            verdict = CriticVerdict(
                winner_plan_id=cv["winner_plan_id"],
                scores=cv.get("scores", {}),
                rejection_justifications=cv.get("rejection_justifications", {}),
                overall_reasoning=cv.get("overall_reasoning", ""),
                judge_model=cv.get("judge_model"),
            )
        execution = None
        if data.get("execution_record"):
            er = data["execution_record"]
            steps = [
                StepExecutionRecord(
                    step_id=s["step_id"],
                    started_at=datetime.fromisoformat(s["started_at"]),
                    completed_at=datetime.fromisoformat(s["completed_at"]),
                    success=s["success"],
                    output=s.get("output"),
                    output_summary=s.get("output_summary", ""),
                    error=s.get("error"),
                )
                for s in er.get("steps", [])
            ]
            execution = ExecutionRecord(
                plan_id=er["plan_id"],
                steps=steps,
                final_answer=er.get("final_answer", ""),
                total_duration_ms=er.get("total_duration_ms", 0),
                tokens_used=er.get("tokens_used", 0),
            )
        return GoATGraph(
            graph_id=data["graph_id"],
            user_input=data["user_input"],
            nodes=nodes,
            edges=edges,
            plan_complexity=complexity,
            critic_verdict=verdict,
            execution_record=execution,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else datetime.now(timezone.utc),
        )
