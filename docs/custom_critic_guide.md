# Custom Plan Critic

The critic scores K candidate plans (K>1 only) and picks a winner, with justifications for why each loser was rejected.

## When it runs

- Only when `K>1` (classifier or `force_K` decided more than one plan is needed).
- **Mandatory in that case**: if `K>1` and no critic is configured, the Director raises `ValueError` — there is no silent fallback like there is for the explanation generator or graph store.
- If `GoatNodeConfig.critic` is `None`, GoAT falls back to the built-in `LLMJudgeCritic`.
- Unlike the explanation generator and graph store, a critic exception is **not** caught — it propagates and fails the run.

## Interface to implement

`PlanCritic` is a `typing.Protocol`. Any object with this one method is accepted:

```python
def score(self, plans: list[Plan], context: PlanningContext, trace_context=None) -> CriticVerdict:
    ...
```

- `plans` is guaranteed to have at least 2 entries when the critic is called. Each `Plan` has `plan_id`, `plan_name`, `steps: list[PlanStep]`, `overall_reasoning`.
- The plan generator isn't guaranteed to return exactly `K` plans.

## Required output shape

Must return a `CriticVerdict`:

```python
@dataclass
class CriticVerdict:
    winner_plan_id: str                       # must match one of plans[i].plan_id
    scores: dict[str, float]                   # plan_id -> score (0-10 by convention)
    rejection_justifications: dict[str, str]   # plan_id -> why it lost (omit the winner)
    overall_reasoning: str = ""
    judge_model: Optional[str] = None
```

`winner_plan_id` must be a real `plan_id` from the input list — the graph builder looks the winner up by id and silently falls back to `plans[0]` if it isn't found, which loses accurate counterfactual reasoning in the resulting graph.

## Example

`plans`/`context` are already-built objects handed to you — just read their attributes, no import needed. For the return value, define your own plain `@dataclass` with matching field names instead of importing the framework's `CriticVerdict`:

```python
from dataclasses import dataclass

@dataclass
class CriticVerdict:
    winner_plan_id: str
    scores: dict
    rejection_justifications: dict
    overall_reasoning: str = ""
    judge_model: str | None = None

class ShortestPlanCritic:
    def score(self, plans, context, trace_context=None) -> CriticVerdict:
        winner = min(plans, key=lambda p: len(p.steps))
        return CriticVerdict(
            winner_plan_id=winner.plan_id,
            scores={p.plan_id: (10.0 if p is winner else 5.0) for p in plans},
            rejection_justifications={
                p.plan_id: "more steps than the winning plan" for p in plans if p is not winner
            },
            overall_reasoning="Picked the plan with the fewest steps.",
        )
```

## Plugging it in

```python
config = GoatNodeConfig(critic=ShortestPlanCritic())
```