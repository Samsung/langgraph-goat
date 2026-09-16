# Custom Explanation Generator

Turns a finished `GoATGraph` into audience-adapted explaination text for human understanding.

## When it runs

- Only for `K>1` runs (GoAT pipeline). Always runs in that case — `GoatNode`/`GoatMiddleware` fall back to the built-in `LLMExplanationGenerator` if `GoatNodeConfig.explanation_generator` is `None`; there is no "off" state once GoAT runs.
- Failures are caught and logged — a broken explanation generator degrades `result.explanation` to `None`/unset rather than failing the run.

## Interface to implement

`ExplanationGenerator` is a `typing.Protocol`. Any object with this one method is accepted:

```python
def generate(self, graph: GoATGraph, trace_context=None) -> Explanation:
    ...
```

- `graph: GoATGraph` — key fields: `user_input`, `nodes` (list of `GoATNode`), `edges`, `plan_complexity`, `critic_verdict` (`None` when K=1), `execution_record` (`None` before execution is recorded). Helper properties: `graph.factual_path_nodes`, `graph.ghost_nodes`, `graph.has_counterfactuals`.

## Required output shape

Must return an `Explanation`:

```python
@dataclass
class Explanation:
    developer_explanation: str = ""  # technical — can reference node ids, scores, rejection reasons
    user_explanation: str = ""        # plain language, no jargon
    raw_output: str = ""              # anything you want kept for debugging (e.g. raw LLM text)
```

There's no format validation beyond "these are strings" — the fields are surfaced verbatim wherever the caller reads `result.explanation`.

## Example

`graph` is always a real `GoATGraph` built by the GoAT Director — you never construct one yourself, so no import is needed to *read* it. For the return value, define your own plain `@dataclass` with matching field names instead of importing the framework's `Explanation`:

```python
from dataclasses import dataclass

@dataclass
class Explanation:
    developer_explanation: str = ""
    user_explanation: str = ""
    raw_output: str = ""

class SimpleExplanationGenerator:
    def generate(self, graph, trace_context=None) -> Explanation:
        winner_label = next(
            (n.label for n in graph.factual_path_nodes), "unknown plan"
        )
        return Explanation(
            developer_explanation=f"Selected plan: {winner_label}. Nodes: {len(graph.nodes)}.",
            user_explanation=f"I decided to go with: {winner_label}.",
            raw_output="",
        )
```

## Plugging it in

```python
config = GoatNodeConfig(explanation_generator=SimpleExplanationGenerator())
```