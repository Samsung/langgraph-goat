# Custom Plan Complexity Classifier

The classifier decides whether an incoming request needs one plan (`K=1`, GoAT bypassed) or multiple candidate plans (`K>1`, GoAT runs and the critic picks a winner).

## When it runs

- Called once per run, before plan generation.
- **Skipped entirely** if `GoatNodeConfig.force_K` is set — the classifier is never invoked in that case.
- If `GoatNodeConfig.classifier` is `None`, GoAT falls back to the built-in `LLMClassifier`.
- `suggested_K` is clamped to `GoatNodeConfig.max_K` automatically — you don't need to clamp it yourself.

## Interface to implement

`PlanComplexityClassifier` is a `typing.Protocol` — there is no base class to inherit. Any object with this one method is accepted:

```python
def classify(self, context: PlanningContext, trace_context=None) -> PlanComplexity:
    ...
```

- `context.user_input: str`, `context.available_tools: list[AvailableTool]`, `context.memory_snippets: list[str]` are the fields most classifiers key off of.
- `trace_context` is an opaque tracing object — accept it and ignore it if you don't need it.

## Required output shape

Must return a `PlanComplexity`:

```python
@dataclass
class PlanComplexity:
    needs_multiple_plans: bool
    suggested_K: int        # 1 if needs_multiple_plans is False, else 2..max_K
    reason: str              # surfaced in the graph/explanation
    confidence: float = 1.0  # 0.0-1.0
```

## Example

`context` is an already-built object handed to you — just read its attributes, no import needed. For the return value, define your own plain `@dataclass` with matching field names instead of importing the framework's `PlanComplexity`:

```python
from dataclasses import dataclass

@dataclass
class PlanComplexity:
    needs_multiple_plans: bool
    suggested_K: int
    reason: str
    confidence: float = 1.0

class KeywordClassifier:
    def classify(self, context, trace_context=None) -> PlanComplexity:
        if "plan" in context.user_input.lower():
            return PlanComplexity(True, 3, "Contains a planning keyword", 0.8)
        return PlanComplexity(False, 1, "No planning keyword found", 0.9)
```

## Plugging it in

```python
config = GoatNodeConfig(classifier=KeywordClassifier())
```