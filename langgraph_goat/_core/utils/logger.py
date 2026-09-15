"""Traceability-aware logger for the GoAT engine.

Provides structured, context-rich logging that enables end-to-end traceability
across the GoAT pipeline (classify → generate → critique → build graph → explain).

Key features:
    - **TraceContext**: A container for correlation IDs (graph_id, plan_id,
      step_id, etc.) that flow through every log entry, making it trivial to
      reconstruct the full lifecycle of a request.
    - **TraceLogger**: Wraps the standard ``logging.Logger`` and automatically
      injects the current ``TraceContext`` into every log record.
    - **@traced**: A decorator that logs entry/exit/timing of functions and
      records exceptions with full trace context.
    - **TraceSpan**: A context manager for timing arbitrary code blocks and
      nesting spans (parent/child relationships).
    - **SequenceTraceCollector**: Collects call/return events and exports them
      as Mermaid sequence diagrams, PlantUML, or JSON — enabling visual
      reconstruction of the pipeline execution flow.

Usage::

    from langgraph_goat import GoatLogger, TraceContext, traced, TraceSpan

    # 1. Obtain a trace-aware logger (names are rooted at "langgraph_goat")
    log = GoatLogger("director")

    # 2. Bind a trace context — all subsequent calls carry the IDs
    ctx = TraceContext(graph_id="g-001", plan_id="p-3")
    log.info("Director.run: K=%d", 3, trace_context=ctx)

    # 3. Decorator for automatic function tracing
    @traced("langgraph_goat._core.critic")
    def score(plans, context):
        ...

    # 4. Context manager for timing blocks
    with TraceSpan("build_graph", trace_context=ctx):
        graph = builder.build(...)

    # 5. Generate a sequence diagram from collected traces
    from langgraph_goat import get_sequence_collector
    collector = get_sequence_collector()
    print(collector.to_mermaid())   # Mermaid sequenceDiagram
    print(collector.to_plantuml())  # PlantUML sequence diagram
    print(collector.to_json())      # JSON array of events
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, overload

# ── Constants ──────────────────────────────────────────────────────────────────

# Root of the logging hierarchy — matches the importable package name, so
# ``logging.getLogger("langgraph_goat")`` controls every logger the library emits.
_LOG_NAMESPACE = "langgraph_goat"

# The engine's sub-hierarchy. Logger names mirror module paths, so engine logs
# read ``langgraph_goat._core.director`` — the leading underscore is the same
# "private, don't import this" signal the module layout carries.
_CORE_NAMESPACE = "langgraph_goat._core"

# Layer labels used to color-code sequence-diagram participants. These are
# display keys for PACKAGE_COLORS, not module paths.
_LAYER_CORE = "core"
_LAYER_ADAPTER = "adapter"
_LAYER_DEFAULT = "_default"

_TRACE_CONTEXT_ATTR = "trace_context"


# ── TraceContext ───────────────────────────────────────────────────────────────


@dataclass
class TraceContext:
    """Correlation IDs that flow through the GoAT pipeline.

    Every log record emitted while a TraceContext is active carries these IDs,
    enabling full traceability from user input through plan generation, critic
    scoring, graph building, and explanation.

    Attributes:
        trace_id: Root-level correlation ID (one per Director.run invocation).
        graph_id: ID of the GoATGraph being built.
        plan_id: ID of the current candidate plan.
        step_id: ID of the current plan step.
        node_id: ID of the current graph node.
        edge_id: ID of the current graph edge.
        session_id: External session identifier (set by the adapter).
        user_id: End-user identifier (for audit trails).
        extras: Arbitrary key-value pairs for adapter-specific trace data.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    graph_id: Optional[str] = None
    plan_id: Optional[str] = None
    step_id: Optional[str] = None
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ── Derived helpers ────────────────────────────────────────────────────

    def with_ids(
        self,
        *,
        graph_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        step_id: Optional[str] = None,
        node_id: Optional[str] = None,
        edge_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **extras: Any,
    ) -> TraceContext:
        """Return a new TraceContext with the given IDs merged in.

        Values provided here override existing ones; the ``trace_id`` is
        always preserved so the root correlation is never lost.
        """
        return TraceContext(
            trace_id=self.trace_id,
            graph_id=graph_id or self.graph_id,
            plan_id=plan_id or self.plan_id,
            step_id=step_id or self.step_id,
            node_id=node_id or self.node_id,
            edge_id=edge_id or self.edge_id,
            session_id=session_id or self.session_id,
            user_id=user_id or self.user_id,
            extras={**self.extras, **extras},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize non-None fields to a dict (for structured log handlers)."""
        result: dict[str, Any] = {"trace_id": self.trace_id}
        for key in (
            "graph_id",
            "plan_id",
            "step_id",
            "node_id",
            "edge_id",
            "session_id",
            "user_id",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.extras:
            result["extras"] = self.extras
        return result

    def format_compact(self) -> str:
        """Return a compact, human-readable representation for log lines.

        Example::

            [trace=a1b2c3d4e5f6 graph=g-001 plan=p-3 step=s-1]
        """
        parts = [f"trace={self.trace_id}"]
        for key in (
            "graph_id",
            "plan_id",
            "step_id",
            "node_id",
            "edge_id",
            "session_id",
            "user_id",
        ):
            value = getattr(self, key)
            if value is not None:
                short_key = key.replace("_id", "").replace("graph", "graph").replace("plan", "plan").replace("step", "step").replace("node", "node").replace("edge", "edge").replace("session", "sess").replace("user", "user")
                parts.append(f"{short_key}={value}")
        if self.extras:
            for k, v in self.extras.items():
                parts.append(f"{k}={v}")
        return "[" + " ".join(parts) + "]"


# ── Trace-aware LogRecord filter ──────────────────────────────────────────────


class _TraceContextFilter(logging.Filter):
    """Logging filter that attaches the current TraceContext to LogRecords.

    When a ``TraceContext`` is provided via ``TraceLogger``, this filter
    injects ``record.trace_context`` and ``record.trace_info`` (the compact
    string form) so that formatters can include trace data automatically.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # If a trace_context was attached by TraceLogger, populate convenience attrs
        ctx = getattr(record, _TRACE_CONTEXT_ATTR, None)
        if ctx is not None:
            record.trace_info = ctx.format_compact()  # type: ignore[attr-defined]
            record.trace_dict = ctx.to_dict()  # type: ignore[attr-defined]
        else:
            record.trace_info = ""  # type: ignore[attr-defined]
            record.trace_dict = {}  # type: ignore[attr-defined]
        return True


# ── TraceLogger ────────────────────────────────────────────────────────────────


class TraceLogger:
    """A logger that automatically injects ``TraceContext`` into log records.

    Drop-in replacement for ``logging.Logger`` — same call signature, but
    accepts an optional ``trace_context`` keyword argument. When provided,
    the context is attached to the ``LogRecord`` and can be used by
    formatters/handlers downstream.

    Example::

        log = GoatLogger("director")
        ctx = TraceContext(graph_id="g-42")
        log.info("Building graph", trace_context=ctx)
        # LogRecord will have record.trace_info == "[trace=... graph=g-42]"
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        # Ensure our filter is present so trace_info is always available
        if not any(isinstance(f, _TraceContextFilter) for f in self._logger.filters):
            self._logger.addFilter(_TraceContextFilter())

    # ── Standard logging passthrough ───────────────────────────────────────

    def debug(self, msg: str, *args: Any, trace_context: Optional[TraceContext] = None, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, args, trace_context=trace_context, **kwargs)

    def info(self, msg: str, *args: Any, trace_context: Optional[TraceContext] = None, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, args, trace_context=trace_context, **kwargs)

    def warning(self, msg: str, *args: Any, trace_context: Optional[TraceContext] = None, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, args, trace_context=trace_context, **kwargs)

    def error(self, msg: str, *args: Any, trace_context: Optional[TraceContext] = None, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, args, trace_context=trace_context, **kwargs)

    def critical(self, msg: str, *args: Any, trace_context: Optional[TraceContext] = None, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, args, trace_context=trace_context, **kwargs)

    def exception(self, msg: str, *args: Any, trace_context: Optional[TraceContext] = None, **kwargs: Any) -> None:
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, args, trace_context=trace_context, **kwargs)

    def log(self, level: int, msg: str, *args: Any, trace_context: Optional[TraceContext] = None, **kwargs: Any) -> None:
        """Log at the given level with optional trace context."""
        self._log(level, msg, args, trace_context=trace_context, **kwargs)

    # ── Internal ───────────────────────────────────────────────────────────

    def _log(
        self,
        level: int,
        msg: str,
        args: tuple,
        trace_context: Optional[TraceContext] = None,
        **kwargs: Any,
    ) -> None:
        extra = kwargs.pop("extra", None) or {}
        if trace_context is not None:
            extra[_TRACE_CONTEXT_ATTR] = trace_context
        if extra:
            kwargs["extra"] = extra
        self._logger._log(level, msg, args, **kwargs)

    # ── Proxy attributes ───────────────────────────────────────────────────

    @property
    def level(self) -> int:
        return self._logger.level

    @level.setter
    def level(self, value: int) -> None:
        self._logger.level = value

    @property
    def name(self) -> str:
        return self._logger.name

    def setLevel(self, level: int) -> None:
        self._logger.setLevel(level)

    def addHandler(self, handler: logging.Handler) -> None:
        self._logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        self._logger.removeHandler(handler)

    def isEnabledFor(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)

    @staticmethod
    def configure(
        level: int = logging.INFO,
        handler: Optional[logging.Handler] = None,
        fmt: Optional[str] = None,
    ) -> None:
        """Configure the ``langgraph_goat`` logging hierarchy.

        Creates a ``StreamHandler`` (if none provided), attaches a
        ``TraceFormatter``, and sets the root ``langgraph_goat`` logger to *level*.

        This is optional — if the host application already configures logging,
        the ``TraceLogger`` and ``TraceContext`` still work; they just rely on
        whatever handlers and formatters the host has set up.

        Example::

            from langgraph_goat import GoatLogger
            GoatLogger.configure(level=logging.DEBUG)
        """
        goat_logger = logging.getLogger(_LOG_NAMESPACE)
        goat_logger.setLevel(level)

        # Avoid adding duplicate handlers on repeated calls
        if not handler:
            if not goat_logger.handlers:
                handler = logging.StreamHandler()
            else:
                return  # already configured

        handler.setFormatter(TraceFormatter(fmt=fmt))
        handler.addFilter(_TraceContextFilter())
        goat_logger.addHandler(handler)


# ── Public factory ─────────────────────────────────────────────────────────────


def GoatLogger(package_name: str, level: Optional[int] = None) -> TraceLogger:
    """Return a ``TraceLogger`` under the ``langgraph_goat`` namespace.

    If *name* is already inside the namespace it is used as-is; otherwise
    ``langgraph_goat.`` is prepended automatically. Every logger the library
    emits therefore lives under one root, so a host application can silence or
    redirect all of it with ``logging.getLogger("langgraph_goat")``.

    Args:
        package_name: Logger name (module path). Auto-prefixed with ``langgraph_goat.``
              if not already namespaced.
        level: Optional logging level for this logger. If provided, automatically
               configures the root logger with a StreamHandler (if not already
               configured) and sets this logger to the given level. Allows you to
               skip calling TraceLogger.configure() explicitly.

    Example::

        log = GoatLogger("director")                       # → langgraph_goat.director
        log = GoatLogger("langgraph_goat.middleware")      # as-is
        # Auto-configure logging on first use with level
        log = GoatLogger("langgraph_goat._core.critic", level=logging.DEBUG)
    """
    if package_name != _LOG_NAMESPACE and not package_name.startswith(_LOG_NAMESPACE + "."):
        package_name = f"{_LOG_NAMESPACE}.{package_name}"

    # Auto-configure root logger if level is provided and not yet configured
    if level is not None:
        root_logger = logging.getLogger(_LOG_NAMESPACE)
        # Only configure if no handlers exist yet
        if not root_logger.handlers:
            TraceLogger.configure(level=level)

    logger = TraceLogger(package_name)
    if level is not None:
        logger.setLevel(level)
    return logger


# ── TraceSpan (context manager for timing) ─────────────────────────────────────


@dataclass
class _SpanRecord:
    """Internal record for a completed span."""

    name: str
    duration_s: float
    trace_context: Optional[TraceContext]
    success: bool
    error: Optional[str] = None


class TraceSpan:
    """Context manager that times a code block and logs entry/exit.

    Spans can be nested — each span tracks its parent, producing a
    call-tree that can be reconstructed from the log output.

    In sequence diagrams, **participants are Class names** (e.g.
    ``Director``, ``LLMClassifier``), not operation names.  The ``caller``
    and ``callee`` parameters control which participant boxes appear;
    ``name`` is the method/operation label shown on the arrow.

    Example::

        ctx = TraceContext(graph_id="g-1")
        with TraceSpan("build_graph", caller="Director", callee="GraphBuilder",
                        trace_context=ctx) as span:
            graph = builder.build(...)
        # Sequence diagram: Director ->> GraphBuilder: build_graph
    """

    # Class-level stack for tracking nesting
    _active_spans: list[TraceSpan] = []

    def __init__(
        self,
        name: str,
        trace_context: Optional[TraceContext] = None,
        logger: Optional[TraceLogger] = None,
        level: int = logging.DEBUG,
        caller: Optional[str] = None,
        callee: Optional[str] = None,
        package: Optional[str] = None,
    ) -> None:
        self.name = name
        self.trace_context = trace_context
        self.caller = caller
        self.callee = callee
        self.package = package
        self._logger = logger or GoatLogger("trace.span")
        self._level = level
        self._start_time: float = 0.0
        self._record: Optional[_SpanRecord] = None
        self._parent: Optional[TraceSpan] = None
        self._depth: int = 0

    @property
    def duration_s(self) -> Optional[float]:
        """Duration in seconds, or None if the span hasn't completed."""
        return self._record.duration_s if self._record else None

    @property
    def record(self) -> Optional[_SpanRecord]:
        """The completed span record, or None if still active."""
        return self._record

    def __enter__(self) -> TraceSpan:
        self._parent = TraceSpan._active_spans[-1] if TraceSpan._active_spans else None
        self._depth = len(TraceSpan._active_spans)
        TraceSpan._active_spans.append(self)
        self._start_time = time.perf_counter()
        indent = "  " * self._depth
        self._logger.debug(
            "%sENTER %s",
            indent,
            self.name,
            trace_context=self.trace_context,
        )
        # Emit sequence-diagram call event
        # Participants are Class names (caller/callee); name is the method label
        caller_participant = self.caller or (
            self._parent.callee if self._parent and self._parent.callee else "Caller"
        )
        callee_participant = self.callee or self.name
        _seq = get_sequence_collector()
        # Derive the layer for color-coding: spans default to the engine layer
        pkg = self.package or _LAYER_CORE
        _seq.record_call(
            caller=caller_participant,
            callee=callee_participant,
            method=self.name,
            trace_context=self.trace_context,
            caller_package=pkg,
            callee_package=pkg,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        duration = time.perf_counter() - self._start_time
        success = exc_type is None
        error_msg = str(exc_val) if exc_val else None
        self._record = _SpanRecord(
            name=self.name,
            duration_s=duration,
            trace_context=self.trace_context,
            success=success,
            error=error_msg,
        )
        # Pop self from the active stack
        if TraceSpan._active_spans and TraceSpan._active_spans[-1] is self:
            TraceSpan._active_spans.pop()

        indent = "  " * self._depth
        status = "OK" if success else f"FAIL ({error_msg})"
        self._logger.log(
            logging.ERROR if not success else self._level,
            "%sEXIT  %s (%.4fs) [%s]",
            indent,
            self.name,
            duration,
            status,
            trace_context=self.trace_context,
        )
        # Emit sequence-diagram return/error event
        caller_participant = self.caller or (
            self._parent.callee if self._parent and self._parent.callee else "Caller"
        )
        callee_participant = self.callee or self.name
        _seq = get_sequence_collector()
        if success:
            _seq.record_return(
                caller=caller_participant,
                callee=callee_participant,
                method=self.name,
                duration_s=duration,
                trace_context=self.trace_context,
            )
        else:
            _seq.record_error(
                caller=caller_participant,
                callee=callee_participant,
                method=self.name,
                duration_s=duration,
                error=error_msg or "unknown error",
                trace_context=self.trace_context,
            )
        # Do not suppress exceptions
        return False


# ── Participant-name helpers (for sequence diagrams) ──────────────────────────


def _derive_caller_participant(logger_name: str) -> str:
    """Derive the caller participant name from a logger name.

    ``"langgraph_goat._core.director"`` → ``"Director"``
    ``"langgraph_goat._core.critic.llm_judge"`` → ``"Critic"``
    ``"langgraph_goat._core.graph.builder"`` → ``"Graph"``
    ``"langgraph_goat.node"`` → ``"Node"``
    ``"langgraph_goat.generator"`` → ``"Generator"``
    """
    # Strip known namespace prefixes — longest first, so the engine's
    # sub-hierarchy is stripped whole rather than down to "_core".
    stripped = logger_name
    for prefix in (_CORE_NAMESPACE + ".", _LOG_NAMESPACE + "."):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    # Take the first dot-separated segment
    first_segment = stripped.split(".")[0]
    # PascalCase: "director" → "Director", "llm" → "LLM"
    return first_segment[0].upper() + first_segment[1:] if first_segment else "Caller"


def _derive_callee_participant(func_name: str) -> str:
    """Derive the callee participant name from a function qualname.

    ``"LLMJudgeCritic.score"`` → ``"LLMJudgeCritic"``
    ``"Director.run"`` → ``"Director"``
    ``"classify"`` → ``"Classify"``
    """
    # If it's a dotted qualname like "ClassName.method", take the class part
    if "." in func_name:
        class_part = func_name.split(".")[0]
        return class_part
    # Plain function name — PascalCase it
    return func_name[0].upper() + func_name[1:] if func_name else "Callee"


# ── @traced decorator ──────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])


def traced(
    logger_name: Optional[str] = None,
    level: int = logging.DEBUG,
    emit_args: bool = False,
) -> Callable[[F], F]:
    """Decorator that adds traceability logging to a function.

    Logs function entry, exit, timing, and any exceptions with full
    ``TraceContext`` if the function accepts a ``trace_context`` parameter.

    Args:
        logger_name: Dotted logger name under ``langgraph_goat.*``.
            Defaults to the module + function name.
        level: Logging level for entry/exit messages.
        emit_args: If True, log the function arguments on entry.

    Example::

        @traced("langgraph_goat._core.critic")
        def score(plans, context, trace_context=None):
            ...
    """

    def decorator(func: F) -> F:
        _log = GoatLogger(logger_name or func.__module__ or "traced")
        func_name = func.__qualname__ or func.__name__
        # Derive participant names for sequence diagrams:
        #   logger_name "langgraph_goat._core.critic" → caller="Director", callee="Critic"
        #   func_name "LLMJudgeCritic.score" → caller="LLMJudgeCritic", callee="score"
        _caller_participant = _derive_caller_participant(
            logger_name or func.__module__ or "caller"
        )
        _callee_participant = _derive_callee_participant(func_name)
        # Derive the layer from logger_name for color-coding. Check the engine's
        # sub-hierarchy first — it is a prefix-match of the root namespace.
        _resolved_logger = logger_name or func.__module__ or ""
        if _resolved_logger.startswith(_CORE_NAMESPACE):
            _package = _LAYER_CORE
        elif _resolved_logger.startswith(_LOG_NAMESPACE):
            _package = _LAYER_ADAPTER
        else:
            _package = _LAYER_DEFAULT

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract trace_context if passed
            ctx: Optional[TraceContext] = kwargs.get("trace_context")
            if ctx is None:
                # Check positional args — look for TraceContext type
                from inspect import signature as _sig

                try:
                    params = list(_sig(func).parameters.keys())
                    if "trace_context" in params:
                        idx = params.index("trace_context")
                        if idx < len(args):
                            ctx = args[idx]
                except (ValueError, TypeError):
                    pass

            args_desc = ""
            if emit_args:
                args_desc = f" args={args!r} kwargs={kwargs!r}"

            _log.log(
                level,
                "→ %s()%s",
                func_name,
                args_desc,
                trace_context=ctx,
            )

            # Derive caller from the active span stack (class-based participants)
            active_caller = _caller_participant
            if TraceSpan._active_spans:
                parent_span = TraceSpan._active_spans[-1]
                active_caller = parent_span.callee if parent_span.callee else active_caller

            # Emit sequence-diagram call event
            # method label: just the function name (without class prefix)
            method_label = func_name.split(".")[-1] if "." in func_name else func_name
            _seq = get_sequence_collector()
            _seq.record_call(
                caller=active_caller,
                callee=_callee_participant,
                method=method_label,
                trace_context=ctx,
                caller_package=_package,
                callee_package=_package,
            )

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                duration = time.perf_counter() - start
                _log.error(
                    "✗ %s() raised %s after %.4fs",
                    func_name,
                    type(exc).__name__,
                    duration,
                    trace_context=ctx,
                )
                _seq.record_error(
                    caller=active_caller,
                    callee=_callee_participant,
                    method=method_label,
                    duration_s=duration,
                    error=f"{type(exc).__name__}: {exc}",
                    trace_context=ctx,
                )
                raise
            else:
                duration = time.perf_counter() - start
                _log.log(
                    level,
                    "← %s() (%.4fs)",
                    func_name,
                    duration,
                    trace_context=ctx,
                )
                _seq.record_return(
                    caller=active_caller,
                    callee=_callee_participant,
                    method=method_label,
                    duration_s=duration,
                    trace_context=ctx,
                )
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Structured formatter for handlers ──────────────────────────────────────────


class TraceFormatter(logging.Formatter):
    """Log formatter that includes trace context in the output.

    If a ``TraceContext`` is attached to the record, the compact trace
    string is appended after the logger name.

    Default format::

        2026-05-20 14:00:00.123 [INFO] langgraph_goat._core.director [trace=a1b2 graph=g-1] Building graph

    Usage::

        import logging
        from langgraph_goat import TraceFormatter, GoatLogger

        handler = logging.StreamHandler()
        handler.setFormatter(TraceFormatter())
        logging.getLogger("langgraph_goat").addHandler(handler)
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        include_trace: bool = True,
    ) -> None:
        if fmt is None:
            fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s"
            if include_trace:
                fmt += " %(trace_info)s"
            fmt += " %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)


# ── Sequence-diagram trace generation ──────────────────────────────────────────


@dataclass
class CallEvent:
    """A single call or return event in the execution trace.

    These events are the raw material for generating sequence diagrams.
    Each event records who called whom (or who returned to whom), with
    timing and optional trace context for correlation.

    Attributes:
        event_id: Unique identifier for this event.
        event_type: One of ``"call"``, ``"return"``, or ``"error"``.
        caller: Name of the calling participant (e.g. ``"Director"``).
        callee: Name of the called participant (e.g. ``"LLMClassifier"``).
        method: The function/method name being invoked.
        timestamp: Monotonic timestamp (seconds from epoch).
        duration_s: Elapsed time (only meaningful for ``"return"``/``"error"``).
        trace_context: Optional TraceContext for correlation.
        label: Optional human-readable annotation on the arrow.
        error: Error message (only for ``"error"`` events).
    """

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    event_type: str = "call"  # "call" | "return" | "error"
    caller: str = ""
    callee: str = ""
    method: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_s: Optional[float] = None
    trace_context: Optional[TraceContext] = None
    label: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict (for JSON export)."""
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "caller": self.caller,
            "callee": self.callee,
            "method": self.method,
            "timestamp": self.timestamp,
        }
        if self.duration_s is not None:
            result["duration_s"] = round(self.duration_s, 6)
        if self.trace_context is not None:
            result["trace_context"] = self.trace_context.to_dict()
        if self.label is not None:
            result["label"] = self.label
        if self.error is not None:
            result["error"] = self.error
        return result


class SequenceTraceCollector:
    """Collects :class:`CallEvent` instances and exports sequence diagrams.

    The collector is designed to be used as a singleton (via
    :func:`get_sequence_collector`). The ``@traced`` decorator and
    ``TraceSpan`` context manager automatically emit events to the
    active collector when it is enabled.

    Participants are automatically color-coded by layer:
        - **core** — engine classes (Director, Critic, …) → blue tones
        - **adapter** — LangGraph-facing classes (GoatMiddleware, …) → green tones
        - Other/unknown → grey tones

    Supported export formats:
        - **Mermaid** (``sequenceDiagram``) — renderable in GitHub, Notion,
          VS Code preview, etc.
        - **PlantUML** — widely supported diagram tool.
        - **JSON** — structured array of :class:`CallEvent` dicts.

    Example::

        from langgraph_goat import (
            get_sequence_collector, traced, TraceContext, TraceLogger,
        )

        TraceLogger.configure(level=10)
        collector = get_sequence_collector()
        collector.enable()

        @traced("langgraph_goat._core.critic")
        def score(plans, trace_context=None):
            ...

        ctx = TraceContext(graph_id="g-1")
        score(plans, trace_context=ctx)

        print(collector.to_mermaid())
        collector.clear()
    """

    # Color scheme per layer
    PACKAGE_COLORS: dict[str, dict[str, str]] = {
        _LAYER_CORE: {
            "mermaid_fill": "#dbeafe",
            "mermaid_stroke": "#3b82f6",
            "plantuml_color": "LightBlue",
        },
        _LAYER_ADAPTER: {
            "mermaid_fill": "#dcfce7",
            "mermaid_stroke": "#22c55e",
            "plantuml_color": "LightGreen",
        },
        _LAYER_DEFAULT: {
            "mermaid_fill": "#f3f4f6",
            "mermaid_stroke": "#9ca3af",
            "plantuml_color": "LightGrey",
        },
    }

    def __init__(self) -> None:
        self._events: list[CallEvent] = []
        self._enabled: bool = False
        # Track pending calls so we can pair call→return and compute duration
        self._pending: dict[str, CallEvent] = {}  # key = caller.callee.method
        # Track which package each participant belongs to (for color-coding)
        self._participant_packages: dict[str, str] = {}

    # ── Enable / disable ───────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """Whether the collector is recording events."""
        return self._enabled

    def enable(self) -> None:
        """Start recording call/return events."""
        self._enabled = True

    def disable(self) -> None:
        """Stop recording events (already-collected events are retained)."""
        self._enabled = False

    # ── Event recording ────────────────────────────────────────────────────

    def record(self, event: CallEvent) -> None:
        """Record a call event. No-op if the collector is disabled."""
        if not self._enabled:
            return
        self._events.append(event)
        # Track pending calls so we can enrich return events with duration
        if event.event_type == "call":
            key = f"{event.caller}->{event.callee}.{event.method}"
            self._pending[key] = event

    def register_participant(self, name: str, package: str) -> None:
        """Register a participant as belonging to a specific layer (for color-coding).

        *package* is a :attr:`PACKAGE_COLORS` key — ``"core"`` or ``"adapter"``.
        """
        self._participant_packages[name] = package

    def _get_package_color(self, participant: str) -> dict[str, str]:
        """Get the color scheme for a participant based on its layer."""
        pkg = self._participant_packages.get(participant, _LAYER_DEFAULT)
        return self.PACKAGE_COLORS.get(pkg, self.PACKAGE_COLORS[_LAYER_DEFAULT])

    def record_call(
        self,
        caller: str,
        callee: str,
        method: str,
        trace_context: Optional[TraceContext] = None,
        label: Optional[str] = None,
        caller_package: Optional[str] = None,
        callee_package: Optional[str] = None,
    ) -> CallEvent:
        """Convenience: create and record a ``"call"`` event."""
        # Register package affiliations for color-coding
        if caller_package and caller:
            self.register_participant(caller, caller_package)
        if callee_package and callee:
            self.register_participant(callee, callee_package)
        event = CallEvent(
            event_type="call",
            caller=caller,
            callee=callee,
            method=method,
            trace_context=trace_context,
            label=label,
        )
        self.record(event)
        return event

    def record_return(
        self,
        caller: str,
        callee: str,
        method: str,
        duration_s: float,
        trace_context: Optional[TraceContext] = None,
        label: Optional[str] = None,
    ) -> CallEvent:
        """Convenience: create and record a ``"return"`` event."""
        event = CallEvent(
            event_type="return",
            caller=caller,
            callee=callee,
            method=method,
            duration_s=duration_s,
            trace_context=trace_context,
            label=label,
        )
        self.record(event)
        return event

    def record_error(
        self,
        caller: str,
        callee: str,
        method: str,
        duration_s: float,
        error: str,
        trace_context: Optional[TraceContext] = None,
    ) -> CallEvent:
        """Convenience: create and record an ``"error"`` event."""
        event = CallEvent(
            event_type="error",
            caller=caller,
            callee=callee,
            method=method,
            duration_s=duration_s,
            trace_context=trace_context,
            error=error,
        )
        self.record(event)
        return event

    # ── Access ─────────────────────────────────────────────────────────────

    @property
    def events(self) -> list[CallEvent]:
        """Return a shallow copy of the collected events."""
        return list(self._events)

    def clear(self) -> None:
        """Remove all collected events and reset pending calls."""
        self._events.clear()
        self._pending.clear()

    # ── Participants ───────────────────────────────────────────────────────

    def participants(self) -> list[str]:
        """Return the unique participant names in order of first appearance."""
        seen: set[str] = set()
        result: list[str] = []
        for ev in self._events:
            for name in (ev.caller, ev.callee):
                if name and name not in seen:
                    seen.add(name)
                    result.append(name)
        return result

    # ── Mermaid export ────────────────────────────────────────────────────

    def to_mermaid(self, title: Optional[str] = None) -> str:
        """Export the collected events as a Mermaid ``sequenceDiagram``.

        The output can be pasted directly into Markdown files that support
        Mermaid rendering (GitHub, GitLab, Notion, etc.).

        Args:
            title: Optional diagram title (rendered as a Mermaid note).

        Returns:
            A string containing the Mermaid sequence diagram source.

        Example output::

            sequenceDiagram
                participant Director
                participant LLMClassifier
                Director->>LLMClassifier: classify()
                LLMClassifier-->>Director: classify() (0.012s)
        """
        lines: list[str] = ["sequenceDiagram"]
        for p in self.participants():
            colors = self._get_package_color(p)
            fill = colors["mermaid_fill"]
            stroke = colors["mermaid_stroke"]
            lines.append(f"    participant {p}")
            # Mermaid styling for colored participant boxes
            lines.append(f"    style {p} fill:{fill},stroke:{stroke},stroke-width:2px,color:#1e293b")
        if title:
            lines.append(f"    Note over {self.participants()[0] if self.participants() else 'System'}: {title}")
        for ev in self._events:
            if ev.event_type == "call":
                arrow = "->>"
                lbl = ev.method
                if ev.label:
                    lbl = f"{ev.method}: {ev.label}"
                lines.append(f"    {ev.caller}{arrow}{ev.callee}: {lbl}")
            elif ev.event_type == "return":
                arrow = "-->>"
                dur = f" ({ev.duration_s:.4f}s)" if ev.duration_s is not None else ""
                lbl = f"{ev.method}{dur}"
                if ev.label:
                    lbl = f"{ev.method}: {ev.label}{dur}"
                lines.append(f"    {ev.callee}{arrow}{ev.caller}: {lbl}")
            elif ev.event_type == "error":
                arrow = "--x"
                dur = f" ({ev.duration_s:.4f}s)" if ev.duration_s is not None else ""
                lines.append(f"    {ev.callee}{arrow}{ev.caller}: {ev.method} ERROR: {ev.error}{dur}")
        return "\n".join(lines)

    # ── PlantUML export ───────────────────────────────────────────────────

    def to_plantuml(self, title: Optional[str] = None) -> str:
        """Export the collected events as a PlantUML sequence diagram.

        Args:
            title: Optional diagram title.

        Returns:
            A string containing the PlantUML source.

        Example output::

            @startuml
            title GoAT Pipeline Trace
            Director -> LLMClassifier : classify()
            LLMClassifier --> Director : classify() (0.012s)
            @enduml
        """
        def _puml_alias(name: str) -> str:
            """Create a PlantUML-safe alias (no underscores/spaces) for use in arrows."""
            return name.replace("_", "").replace(" ", "")

        lines: list[str] = ["@startuml"]
        if title:
            lines.append(f"title {title}")
        for p in self.participants():
            colors = self._get_package_color(p)
            puml_color = colors["plantuml_color"]
            alias = _puml_alias(p)
            # Quote the display name if it has underscores/spaces; use alias for arrows
            if "_" in p or " " in p:
                lines.append(f'participant "{p}" as {alias} #{puml_color}')
            else:
                lines.append(f"participant {p} #{puml_color}")
        for ev in self._events:
            caller_alias = _puml_alias(ev.caller)
            callee_alias = _puml_alias(ev.callee)
            if ev.event_type == "call":
                lbl = ev.method
                if ev.label:
                    lbl = f"{ev.method}: {ev.label}"
                lines.append(f"{caller_alias} -> {callee_alias} : {lbl}")
            elif ev.event_type == "return":
                dur = f" ({ev.duration_s:.4f}s)" if ev.duration_s is not None else ""
                lbl = f"{ev.method}{dur}"
                if ev.label:
                    lbl = f"{ev.method}: {ev.label}{dur}"
                lines.append(f"{callee_alias} --> {caller_alias} : {lbl}")
            elif ev.event_type == "error":
                dur = f" ({ev.duration_s:.4f}s)" if ev.duration_s is not None else ""
                lines.append(f"{callee_alias} -x {caller_alias} : {ev.method} ERROR: {ev.error}{dur}")
        lines.append("@enduml")
        return "\n".join(lines)

    # ── JSON export ───────────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        """Export the collected events as a JSON array.

        Each event is serialized via :meth:`CallEvent.to_dict`.

        Args:
            indent: JSON indentation level (default 2).

        Returns:
            A JSON string.
        """
        import json

        return json.dumps(
            [ev.to_dict() for ev in self._events],
            indent=indent,
        )

    # ── Filtering ─────────────────────────────────────────────────────────

    def filter_by_trace_id(self, trace_id: str) -> SequenceTraceCollector:
        """Return a new collector containing only events with the given trace_id."""
        filtered = SequenceTraceCollector()
        filtered._events = [
            ev for ev in self._events
            if ev.trace_context and ev.trace_context.trace_id == trace_id
        ]
        return filtered

    def filter_by_participant(self, name: str) -> SequenceTraceCollector:
        """Return a new collector containing only events involving the given participant."""
        filtered = SequenceTraceCollector()
        filtered._events = [
            ev for ev in self._events
            if ev.caller == name or ev.callee == name
        ]
        return filtered


# ── Global singleton ───────────────────────────────────────────────────────────

_sequence_collector: Optional[SequenceTraceCollector] = None


def get_sequence_collector() -> SequenceTraceCollector:
    """Return the global :class:`SequenceTraceCollector` singleton.

    The collector is lazily created on first call. Use :meth:`enable` to
    start recording and :meth:`clear` to reset between runs.

    Example::

        from langgraph_goat import get_sequence_collector

        collector = get_sequence_collector()
        collector.enable()

        # ... run pipeline ...

        print(collector.to_mermaid())
        collector.clear()
    """
    global _sequence_collector
    if _sequence_collector is None:
        _sequence_collector = SequenceTraceCollector()
    return _sequence_collector


def reset_sequence_collector() -> SequenceTraceCollector:
    """Reset and return the global :class:`SequenceTraceCollector` singleton.
    
    This clears all collected events and returns a fresh collector for a new run.
    Use this between independent Director.run() invocations to avoid mixing
    telemetry from different runs.
    
    Example::
    
        from langgraph_goat import reset_sequence_collector
        
        # At the start of each run
        collector = reset_sequence_collector()
        collector.enable()
    """
    global _sequence_collector
    _sequence_collector = SequenceTraceCollector()
    return _sequence_collector
