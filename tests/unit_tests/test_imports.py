"""Unit tests for the public import surface of ``langgraph_goat``."""

import importlib
import re
from pathlib import Path

import pytest

import langgraph_goat

EXPECTED_PUBLIC_API = [
    "GoatMiddleware",
    "GoatMiddlewareConfig",
    "GoatNode",
    "GoatNodeConfig",
    "GoatOutput",
    "GoatLogger",
]


def test_all_matches_expected_public_api() -> None:
    assert sorted(langgraph_goat.__all__) == sorted(EXPECTED_PUBLIC_API)


@pytest.mark.parametrize("name", EXPECTED_PUBLIC_API)
def test_public_name_is_importable(name: str) -> None:
    assert getattr(langgraph_goat, name) is not None


def test_core_is_not_part_of_public_api() -> None:
    for name in langgraph_goat.__all__:
        obj = getattr(langgraph_goat, name)
        # GoatLogger is re-exported
        if name != "GoatLogger":
            assert not obj.__module__.startswith("langgraph_goat._core")


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    assert match is not None
    assert langgraph_goat.__version__ == match.group(1)


def test_package_reimport_is_idempotent() -> None:
    assert importlib.reload(langgraph_goat).__all__ == langgraph_goat.__all__
