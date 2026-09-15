"""langgraph-goat — Graph of Agentic Thoughts for LangGraph.

This is the only import surface of the package. Everything users need is
re-exported here::

    import langgraph_goat
    from langgraph_goat import GoatMiddleware, GoatNodeConfig, GoatNode

The engine lives in the private ``langgraph_goat._core`` subpackage — it is
an implementation detail and is not part of the public API. Import from
``langgraph_goat`` instead.
"""

# ── Adapter: high-level (create_agent) integration ────────────────────────────
from .middleware import (
    GoatMiddleware,
    GoatMiddlewareConfig,
)

# ── Adapter: low-level (StateGraph) integration ───────────────────────────────
from .node_wrapper import (
    GoatNode,
    GoatNodeConfig,
)
# ── Unified output container ──────────────────────────────────────────────────
from .goat_output import GoatOutput

from ._core.utils.logger import GoatLogger

__all__ = [
    # High-Level (create_agent) integration
    "GoatMiddleware",
    "GoatMiddlewareConfig",
    # Low-Level (StateGraph) integration
    "GoatNode",
    "GoatNodeConfig",
    # Output container
    "GoatOutput",
    # Logging
    "GoatLogger",
]

__version__ = "1.0.0"
