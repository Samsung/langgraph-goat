<!--
Thanks for filing an issue. Please search existing issues first:
https://github.com/Samsung/langgraph-goat/issues

Delete the sections that don't apply. For usage questions, the guides in docs/
may answer it faster: middleware_usage_guide.md (high-level, create_agent) and
node_wrapper_usage_guide.md (low-level, StateGraph).

Do NOT paste API keys, tokens, or internal hostnames/endpoints.
-->

## Type

<!-- Delete the ones that don't apply. -->

- Bug
- Feature request
- Documentation
- Question

## Summary

<!-- One or two sentences. -->

## Integration pattern

<!-- Which one are you using? Delete the other. -->

- **High-level** — `GoatMiddleware` with `create_agent`
- **Low-level** — `GoatNode` wrapping a planner node in a `StateGraph`

## Environment

| | |
|---|---|
| langgraph-goat | <!-- python -c "import langgraph_goat; print(langgraph_goat.__version__)" --> |
| Python | <!-- python --version --> |
| OS | |
| langchain / langgraph | <!-- pip show langchain langgraph \| grep -i version --> |
| Model / provider | <!-- e.g. Ollama llama3.1:8b, OpenAI gpt-4o, Anthropic claude-sonnet-5 --> |

## Configuration

<!--
Paste your GoatMiddlewareConfig / GoatNodeConfig. K settings matter most.
Most "GoAT didn't run" reports turn out to be the classifier choosing K=1,
which bypasses the pipeline by design.
-->

```python
config = GoatMiddlewareConfig(
    default_K=3,
    max_K=3,
    force_K=None,
    # custom classifier / critic / explanation_generator / graph_store, if any
)
```

## Reproduction

<!--
Smallest snippet that shows the problem. A mock model that returns a fixed
string is usually enough and makes the issue reproducible without a live LLM.
-->

```python
# your code here
```

**Input that triggers it:**

## Expected behavior

## Actual behavior

<!--
GoAT is fail-soft: when the pipeline raises, it logs a warning and falls back
to a normal model call, so you may see no traceback at all. Check for:
  - a "GoAT pipeline failed" / "falling back" warning in the logs
  - the `_goat_error` key in the returned state (GoatNode)
  - `goat_output.complexity.suggested_K` to see which K was chosen
-->

## Logs

<!--
Run with debug logging and paste the relevant portion:

    from langgraph_goat import GoatLogger
    import logging
    GoatLogger("my_agent", level=logging.DEBUG)   # or "node_wrapper"
-->

```
paste/attach logs here
```

## Artifacts (optional)

<!--
If rendering is enabled, `<output_dir>/<graph_id>/` holds the graph JSON,
telemetry and trace diagram. Attaching graph.json helps a lot for issues
about plan selection, critic verdicts, or explanations.
-->

- [ ] Attached `graph.json`
- [ ] Attached telemetry / trace output
- [ ] `render_goat_graph` / `render_goat_tele` / `render_goat_trace` were enabled

---

<!-- For feature requests, replace the bug sections above with: -->

## Problem this would solve

## Proposed solution

<!-- If it fits an existing extension point (PlanComplexityClassifier, PlanCritic,
     ExplanationGenerator, GraphStore, PlanGenerator), say which — those are
     Protocols and can often be implemented without changing the library. -->

## Alternatives considered (if any)
