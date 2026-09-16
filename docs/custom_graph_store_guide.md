# Custom Graph Store

Persists (and retrieves) `GoATGraph` objects.

## When it runs

- `graph_store` is `None` by default — nothing is persisted unless you configure one.
- Once configured, `save()` is called twice per run: after the winning plan is selected, and again at the end of `record_execution()` (post-execution graph with actual outcomes).
- Failures in `save()` are caught and logged — persistence errors never fail the run.
- The Director decides the file path: it creates `<output_dir>/<graph_id>/` and passes `<graph_id>.json` in as a `Path` — your `save`/`load` don't need to invent paths.

NOTE: `render_goat_graph` is unrelated: it writes a GraphViz SVG into that same `<output_dir>/<graph_id>/` folder, but never calls your store.

## Interface to implement

`GraphStore` is a `typing.Protocol`. Any object with these three methods is accepted:

```python
def save(self, path: Path, graph: GoATGraph, trace_context=None) -> str:
    ...  # returns a storage key/id

def load(self, path: Path, graph_id: str, trace_context=None) -> GoATGraph:
    ...

def list_recent(self, limit: int = 20, trace_context=None) -> list[str]:
    ...  # returns graph_ids, most recent first
```

`load` and `list_recent` are not called by the Director itself — they exist for your own retrieval code (dashboards, replay tooling, etc.). Only `save` is on the hot path.

## Required output shape

`save` must accept a full `GoATGraph` and be able to serialize every field on it, including nested dataclasses/enums (`PlanComplexity`, `CriticVerdict`, `ExecutionRecord`, `NodeType`, `EdgeKind`). `load` must reconstruct an equivalent `GoATGraph` — round-tripping through your storage medium.

## Example

```python
import json
from dataclasses import asdict
from pathlib import Path
from langgraph_goat._core.types import GoATGraph

class JsonFileStore:
    def save(self, path: Path, graph: GoATGraph, trace_context=None) -> str:
        path.write_text(json.dumps(asdict(graph), default=str), encoding="utf-8")
        return graph.graph_id

    def load(self, path: Path, graph_id: str, trace_context=None) -> GoATGraph:
        raise NotImplementedError("reconstruct GoATGraph from your storage format")

    def list_recent(self, limit: int = 20, trace_context=None) -> list[str]:
        return []
```

`asdict(graph)` alone loses enum/datetime fidelity on reload — see `langgraph_goat/_core/graph/stores/networkx.py` for a complete serialize/deserialize reference if you need a fully round-trippable store.

## Plugging it in

```python
config = GoatNodeConfig(graph_store=JsonFileStore())
```