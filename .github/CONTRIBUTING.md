# Contributing to langgraph-goat

Issues and pull requests are welcome. This guide covers how to set up the
project, what the test and style expectations are, and the one architectural
rule that matters most here: the public API boundary.

## Before you start

- Search [existing issues](https://github.com/Samsung/langgraph-goat/issues) first.
- For anything beyond a small fix, open an issue before writing code so the
  approach can be agreed on.
- Many customizations do **not** require changing this library — see
  [Extension points](#extension-points) below.

## Running tests

The test suite is the gate for a pull request. It uses mock models — no
network, no API keys, no live LLM:

```bash
pytest tests -q
```

Coverage for the module you touched:

```bash
pytest tests --cov=langgraph_goat --cov-report=term-missing
```

## The public API boundary

`langgraph_goat/_core` is private. Users import from `langgraph_goat` only.
When contributing:

- **Adding a public name?** Import it in `langgraph_goat/__init__.py` and add
  it to `__all__`. A name that isn't in `__all__` isn't part of the API.
- **Don't** tell users to import from `langgraph_goat._core` in docs, examples
  or docstrings. If they need something from there, export it properly.
- Tests import from the public surface (`from langgraph_goat import Plan`) so
  that accidental removals from `__all__` fail the suite.

## Code style

Match the surrounding code. The conventions in use:

- `from __future__ import annotations`, with type hints on public functions.
- Dataclasses for config and data-contract types; Protocols for extension points.
- Google-style docstrings on public functions and classes, with an `Args:` and
  `Returns:` section where the signature isn't self-explanatory.
- Logging through `GoatLogger`, never `print()`, in library code. Use lazy
  `%s` formatting rather than f-strings in log calls.
- Keep GoAT fail-soft: the adapters catch pipeline errors, log a warning and
  fall back to the user's planner or a normal model call. A contribution
  should not turn an internal failure into a crash for the caller.

`ruff check langgraph_goat` is available via the `dev` extra. The project
currently ships no lint configuration, so treat its output as advisory and
don't reformat unrelated code in your PR.

## Tests for your change

- Add tests to the file matching the module you changed
  (`tests/test_middleware.py`, `tests/test_node_wrapper.py`, …).
- Use mocks for models and stores. Unit tests must not make network calls.
- Assert observable behavior — the returned state, the response, the recorded
  call — rather than that a mock was merely touched.
- `pytest.mark.parametrize` for input variations instead of near-duplicate tests.
- Tests run in a temp working directory, so nothing should write into the repo.

## Documentation

Update the docs alongside the code when you change public behavior:

- `docs/middleware_usage_guide.md` and `docs/node_wrapper_usage_guide.md` for
  the two integration patterns.
- The relevant `docs/custom_*_guide.md` when you touch a component contract.
- `README.md` if the change affects the quick-start or the output artifacts.

## Pull requests

1. Branch from `main`.
2. Keep the change focused; unrelated cleanups belong in their own PR.
3. Make sure `pytest tests -q` passes.
4. In the description, explain what changed and why, link the issue it closes,
   and call out any change to the public API or to default behavior.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
