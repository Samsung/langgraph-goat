import html as _html_module
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

try:
    import graphviz
except ImportError:
    graphviz = None

from ..types import (
    NodeType,
    EdgeKind,
    GoATGraph
)

def _get_graphviz_path() -> str:
    """Get Graphviz bin path from environment or use platform-specific default.

    Returns empty string if Graphviz is assumed to be in system PATH.
    """
    # Check if user has explicitly set custom path
    if 'GRAPHVIZ_BIN_PATH' in os.environ:
        return os.environ['GRAPHVIZ_BIN_PATH']

    # User should either set GRAPHVIZ_BIN_PATH or install graphviz in system PATH
    if sys.platform == 'win32':
        default_path = 'C:/Program Files (x86)/Graphviz/bin/'
        if Path(default_path).exists():
            return default_path

    # Assume graphviz is in system PATH
    return ''


def _ensure_graphviz_in_path() -> None:
    """Add Graphviz to PATH if needed.

    This is deferred to render-time to avoid modifying PATH on every import.
    """
    if not graphviz_path:
        return

    # Only add to PATH if not already present
    current_paths = os.environ.get("PATH", "").split(os.pathsep)
    if graphviz_path not in current_paths:
        os.environ["PATH"] += os.pathsep + graphviz_path


graphviz_path = _get_graphviz_path()

# ── Visual style mapping per NodeType ──────────────────────────────────────────

_NODE_STYLES: dict[NodeType, dict] = {
    NodeType.DIRECTOR: {
        "fillcolor": "#e1f5ff",
        "color": "#0288d1",
        "penwidth": "2",
        "label_prefix": "🎯 ",
    },
    NodeType.PLAN_STEP: {
        "fillcolor": "#fff3e0",
        "color": "#f57c00",
        "penwidth": "1.5",
        "label_prefix": "👑 ",
    },
    NodeType.MEMORY_LOAD: {
        "fillcolor": "#f3e5f5",
        "color": "#7b1fa2",
        "penwidth": "1",
        "label_prefix": "📂 ",
    },
    NodeType.TOOL_CALL: {
        "fillcolor": "#e8f5e9",
        "color": "#388e3c",
        "penwidth": "1",
        "label_prefix": "🔧 ",
    },
    NodeType.FINAL_ANSWER: {
        "fillcolor": "#fce4ec",
        "color": "#c2185b",
        "penwidth": "2",
        "label_prefix": "✅ ",
    },
    NodeType.GHOST: {
        "fillcolor": "#f5f5f5",
        "color": "#bdbdbd",
        "penwidth": "1",
        "label_prefix": "👻 ",
        "style": "dashed",
        "fontcolor": "#9e9e9e",
    },
}

# ── Visual style mapping per EdgeKind ─────────────────────────────────────────

_EDGE_STYLES: dict[EdgeKind, dict] = {
    EdgeKind.CAUSAL_FLOW: {
        "color": "#333333",
        "penwidth": "2",
        "arrowsize": "1.0",
    },
    EdgeKind.MEMORY_RETRIEVAL: {
        "color": "#7b1fa2",
        "penwidth": "1.5",
        "arrowsize": "0.8",
    },
    EdgeKind.COUNTERFACTUAL: {
        "color": "#9e9e9e",
        "penwidth": "1",
        "arrowsize": "0.7",
        "style": "dashed",
        "fontcolor": "#bdbdbd",
    },
}


def renderGraph(graph: GoATGraph, path: Path|str):
    """Render the GoATGraph using graphviz and display it."""
    if graphviz is None:
        print("\n[ERROR] Graphviz not installed. Install with: pip install graphviz")
        return

    # Ensure Graphviz is in PATH (deferred from import time)
    _ensure_graphviz_in_path()

    output_dir = Path(path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"\n[ERROR] Cannot create output directory '{path}': {e}")
        return

    try:
        # 1. Create a graphviz Digraph
        dot = graphviz.Digraph(
            name=graph.graph_id,
            format="svg",
            graph_attr={
                "rankdir": "TB",
                "label": f"Graph ID: {graph.graph_id}\t|\tPrompt: {graph.user_input}\n",
                "labelloc": "t",
                "fontsize": "14",
                "fontname": "Courier New Bold",
                "bgcolor": "white",
                "pad": "0.5",
                "nodesep": "0.6",
                "ranksep": "0.8",
                "dpi": "96",
            },
            node_attr={
                "shape": "box",
                "style": "filled,rounded",
                "fontname": "Helvetica",
                "fontsize": "11",
            },
            edge_attr={
                "fontname": "Helvetica",
                "fontsize": "9",
                "fontcolor": "#b71c1c",
            },
        )

        # 2. Add nodes with type-specific styling
        for node in graph.nodes:
            style = _NODE_STYLES.get(node.node_type, {})
            attrs: dict = {
                "fillcolor": style.get("fillcolor", "#f5f5f5"),
                "color": style.get("color", "#666666"),
                "penwidth": style.get("penwidth", "1"),
                "label": f"{style.get('label_prefix', '')}{node.label}",
            }
            # Ghost nodes get dashed border and faded font
            if "style" in style:
                attrs["style"] = f"filled,rounded,{style['style']}"
            if "fontcolor" in style:
                attrs["fontcolor"] = style["fontcolor"]
            dot.node(node.node_id, **attrs)

        # 3. Add edges with kind-specific styling
        for edge in graph.edges:
            style = _EDGE_STYLES.get(edge.kind, {})
            attrs: dict = {
                "color": style.get("color", "#666666"),
                "penwidth": style.get("penwidth", "1.5"),
                "arrowsize": style.get("arrowsize", "0.8"),
            }
            if "style" in style:
                attrs["style"] = style["style"]
            if "fontcolor" in style:
                attrs["fontcolor"] = style["fontcolor"]
            if edge.reasoning:
                attrs["label"] = textwrap.fill(str(edge.reasoning), width=30)
            dot.edge(edge.src, edge.dst, **attrs)

        # 4. Render and display
        dot.render(directory=str(output_dir), view=True, cleanup=True, format='svg')
        print(f"\nGraph rendered successfully to: {output_dir}")
    except graphviz.backend.ExecutableNotFound as e:
        print(f"\n[ERROR] Graphviz not found: {e}")
        print(f"   Please install Graphviz:")
        if sys.platform == 'win32':
            print(f"   - Download from: https://graphviz.org/download/")
            print(f"   - Or set GRAPHVIZ_BIN_PATH env var to custom installation path")
        else:
            print(f"   - Linux: sudo apt-get install graphviz")
            print(f"   - macOS: brew install graphviz")
    except Exception as e:
        print(f"\n[WARN] Graph rendering failed: {e}")

def renderTrace(trace: str, id: str, path: str) -> bool:
    """Render PlantUML sequence diagram to SVG.

    Args:
        trace: PlantUML source code (sequence diagram).
        id: Identifier used in the output filename.
        path: Output directory where the SVG file will be written.
    """
    output_dir = Path(path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"\n[ERROR] Cannot create output directory '{path}': {e}")
        return False

    try:
        import plantuml as plantuml_lib
        server = plantuml_lib.PlantUML(url='http://www.plantuml.com/plantuml/svg/')
        svg_data = server.processes(trace)
        svg_path = output_dir / f"trace_{id}.svg"
        try:
            if isinstance(svg_data, bytes):
                svg_path.write_bytes(svg_data)
            else:
                svg_path.write_text(str(svg_data), encoding="utf-8")
            print(f"\nPlantUML sequence diagram (SVG) saved to: {svg_path}")
            return True
        except (PermissionError, OSError, UnicodeEncodeError) as e:
            print(f"\n[ERROR] Cannot write to '{svg_path}': {e}")
            return False
    except Exception as e:
        print(f"\n[WARN] PlantUML SVG rendering failed: {e}")
        return False


def renderTraceFromCollector(collector, graph_id: str, path: str, title: Optional[str] = None) -> bool:
    """Generate and render PlantUML sequence diagram from a SequenceTraceCollector.
    
    This method:
    1. Generates PlantUML source from the collector using to_plantuml()
    2. Renders it to SVG using renderTrace()
    
    Args:
        collector: SequenceTraceCollector instance with recorded events.
        graph_id: Graph identifier used in the output filename.
        path: Output directory where the SVG file will be written.
        title: Optional title for the sequence diagram.
    
    Example:
        >>> from langgraph_goat import get_sequence_collector
        >>> collector = get_sequence_collector()
        >>> collector.enable()
        >>> # ... run GOAT pipeline ...
        >>> renderTraceFromCollector(collector, graph_id="g-001", path="./output")
    """
    if collector is None:
        print("\n[WARN] renderTraceFromCollector: collector is None")
        return
    
    # Generate PlantUML source from collector
    plantuml_source = collector.to_plantuml(title=title or f"GOAT Trace - {graph_id}")
    
    # Render to SVG
    return renderTrace(plantuml_source, graph_id, path)


# ── Telemetry HTML rendering helpers ───────────────────────────────────────────

def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return _html_module.escape(str(text), quote=True)


def _render_token_section(token_data: dict[str, Any]) -> str:
    """Render the token telemetry section as HTML."""
    metrics = token_data.get("metrics", [])
    if not metrics:
        return ""

    rows = ""
    total_input = total_output = total_total = 0
    for i, m in enumerate(metrics, 1):
        inp = m.get("input", 0)
        out = m.get("output", 0)
        tot = m.get("total", 0)
        tps = m.get("tokens-per-sec", 0)
        tc = m.get("tool-call")
        total_input += inp
        total_output += out
        total_total += tot

        tc_badge = ""
        if tc:
            tc_badge = f'<span class="badge badge-tool">tool-call</span>'

        rows += f"""
            <tr>
                <td>{i}</td>
                <td><code>{_html_escape(m.get("id", ""))}</code></td>
                <td class="num">{inp:,}</td>
                <td class="num">{out:,}</td>
                <td class="num">{tot:,}</td>
                <td class="num">{m.get("reasoning", 0):,}</td>
                <td class="num">{tps:.2f}</td>
                <td>{tc_badge}</td>
            </tr>"""

    return f"""
    <div class="section" id="tokens">
        <h2>🔢 Token Usage</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>#</th><th>ID</th><th>Input</th><th>Output</th>
                        <th>Total</th><th>Reasoning</th><th>Tokens/sec</th><th>Tool Call</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                    <tr class="summary-row">
                        <td colspan="2"><strong>Total</strong></td>
                        <td class="num"><strong>{total_input:,}</strong></td>
                        <td class="num"><strong>{total_output:,}</strong></td>
                        <td class="num"><strong>{total_total:,}</strong></td>
                        <td colspan="3"></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>"""


def _render_latency_section(latency_data: dict[str, Any]) -> str:
    """Render the latency telemetry section as HTML."""
    total = latency_data.get("total", 0)
    inference = latency_data.get("inference", 0)
    tool = latency_data.get("tool", 0)
    ttft = latency_data.get("TTFT", 0)

    def pct(val: float) -> float:
        return round(val / total * 100, 1) if total > 0 else 0

    return f"""
    <div class="section" id="latency">
        <h2>⏱️ Latency</h2>
        <div class="cards">
            <div class="card card-total">
                <div class="card-label">Total</div>
                <div class="card-value">{total:.3f}s</div>
                <div class="card-bar"><div class="card-bar-fill" style="width:100%"></div></div>
            </div>
            <div class="card card-inference">
                <div class="card-label">Inference</div>
                <div class="card-value">{inference:.3f}s</div>
                <div class="card-bar"><div class="card-bar-fill" style="width:{pct(inference)}%"></div></div>
                <div class="card-pct">{pct(inference)}%</div>
            </div>
            <div class="card card-tool">
                <div class="card-label">Tool</div>
                <div class="card-value">{tool:.3f}s</div>
                <div class="card-bar"><div class="card-bar-fill" style="width:{pct(tool)}%"></div></div>
                <div class="card-pct">{pct(tool)}%</div>
            </div>
            <div class="card card-ttft">
                <div class="card-label">TTFT</div>
                <div class="card-value">{ttft:.3f}s</div>
                <div class="card-bar"><div class="card-bar-fill" style="width:{pct(ttft)}%"></div></div>
                <div class="card-pct">{pct(ttft)}%</div>
            </div>
        </div>
    </div>"""


def _render_model_section(model_data: dict[str, Any]) -> str:
    """Render the model telemetry section as HTML."""
    return f"""
    <div class="section" id="model">
        <h2>🤖 Model Configuration</h2>
        <div class="cards">
            <div class="card card-model">
                <div class="card-label">Model</div>
                <div class="card-value card-value-sm">{_html_escape(model_data.get('name', 'N/A'))}</div>
            </div>
            <div class="card card-model">
                <div class="card-label">Temperature</div>
                <div class="card-value">{model_data.get('temperature', 0.7)}</div>
            </div>
            <div class="card card-model">
                <div class="card-label">Top P</div>
                <div class="card-value">{model_data.get('top_p', 1.0)}</div>
            </div>
            <div class="card card-model">
                <div class="card-label">Seed</div>
                <div class="card-value">{model_data.get('seed', '—')}</div>
            </div>
            <div class="card card-model">
                <div class="card-label">Max Tokens</div>
                <div class="card-value">{model_data.get('max_tokens', 2000):,}</div>
            </div>
        </div>
    </div>"""


def _render_prompt_section(prompt_data: dict[str, Any]) -> str:
    """Render the prompt telemetry section as HTML."""
    metrics = prompt_data.get("metrics", [])
    if not metrics:
        return ""

    details = ""
    for i, m in enumerate(metrics, 1):
        sys_text = _html_escape(m.get("system", ""))
        usr_text = _html_escape(m.get("user", ""))
        size = m.get("size", 0)
        ver = m.get("version", "1.0")

        sys_block = ""
        if sys_text:
            sys_block = f"""
                <div class="prompt-block">
                    <div class="prompt-label">System</div>
                    <pre>{sys_text}</pre>
                </div>"""

        details += f"""
        <details class="prompt-details">
            <summary>
                <span class="prompt-summary-title">Prompt #{i}</span>
                <span class="badge">ID: {_html_escape(m.get('id', ''))}</span>
                <span class="badge">Size: {size:,} chars</span>
                <span class="badge">v{ver}</span>
            </summary>
            <div class="prompt-body">
                {sys_block}
                <div class="prompt-block">
                    <div class="prompt-label">User</div>
                    <pre>{usr_text}</pre>
                </div>
            </div>
        </details>"""

    return f"""
    <div class="section" id="prompts">
        <h2>📝 Prompts</h2>
        {details}
    </div>"""


def _render_completion_section(completion_data: dict[str, Any]) -> str:
    """Render the completion telemetry section as HTML."""
    output = _html_escape(completion_data.get("output", ""))
    finish = _html_escape(completion_data.get("finish_reason", ""))

    finish_class = "badge-success" if finish == "stop" else "badge-warn"

    return f"""
    <div class="section" id="completion">
        <h2>💬 Completion</h2>
        <div class="completion-wrap">
            <div class="completion-header">
                <span class="badge {finish_class}">finish_reason: {finish or 'N/A'}</span>
            </div>
            <pre class="completion-output">{output}</pre>
        </div>
    </div>"""


def _render_tool_section(tool_data: dict[str, Any]) -> str:
    """Render the tool telemetry section as HTML."""
    metrics = tool_data.get("metrics", [])
    if not metrics:
        return ""

    rows = ""
    for i, m in enumerate(metrics, 1):
        name = _html_escape(m.get("name", ""))
        args = _html_escape(json.dumps(m.get("args", {}), indent=2, default=str))
        output = _html_escape(json.dumps(m.get("output"), default=str) if m.get("output") is not None else "")
        result = m.get("result", "")
        result_class = "badge-success" if result == "success" else "badge-error"

        rows += f"""
            <tr>
                <td>{i}</td>
                <td><code>{name}</code></td>
                <td><pre class="inline-pre">{args}</pre></td>
                <td><pre class="inline-pre">{output}</pre></td>
                <td><span class="badge {result_class}">{_html_escape(result)}</span></td>
            </tr>"""

    return f"""
    <div class="section" id="tools">
        <h2>🔧 Tool Executions</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr><th>#</th><th>Tool</th><th>Args</th><th>Output</th><th>Result</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>"""


def _render_explainability_section(expl_data: dict[str, Any]) -> str:
    """Render the explainability telemetry section as HTML."""
    director_time = expl_data.get("director_agent", 0)
    spans = expl_data.get("span", [])
    counterfactuals = expl_data.get("counterfactuals", [])
    graph = expl_data.get("graph")
    
    # New fields from trace_logger.py (top-level, not buried in graph)
    user_input_top = expl_data.get("user_input", "")
    winning_plan_top = expl_data.get("winning_plan")


    # ── Spans timeline ──
    max_time = max((s.get("time", 0) for s in spans), default=0) if spans else 1
    if max_time == 0:
        max_time = 1

    span_rows = ""
    for s in spans:
        name = _html_escape(s.get("name", ""))
        t = s.get("time", 0)
        pct = round(t / max_time * 100, 1)
        bar_color = "#0288d1"
        if "LLM" in name or "LiteLLM" in name:
            bar_color = "#f57c00"
        elif "Critic" in name or "Judge" in name:
            bar_color = "#7b1fa2"
        elif "Planner" in name or "Plan" in name:
            bar_color = "#388e3c"
        elif "Graph" in name:
            bar_color = "#c2185b"

        span_rows += f"""
            <tr>
                <td class="span-name">{name}</td>
                <td class="num">{t:.4f}s</td>
                <td class="span-bar-cell">
                    <div class="span-bar"><div class="span-bar-fill" style="width:{pct}%;background:{bar_color}"></div></div>
                </td>
            </tr>"""

    spans_html = f"""
        <div class="subsection">
            <h3>📊 Span Timings</h3>
            <div class="director-badge">Director Agent: {director_time:.4f}s</div>
            <div class="table-wrap">
                <table>
                    <thead><tr><th>Participant</th><th>Time</th><th>Share</th></tr></thead>
                    <tbody>{span_rows}</tbody>
                </table>
            </div>
        </div>"""

    # ── Counterfactuals ──
    cf_html = ""
    if counterfactuals:
        cf_cards = ""
        for cf in counterfactuals:
            score = cf.get("score", 0)
            score_color = "#388e3c" if score >= 8 else "#f57c00" if score >= 5 else "#c2185b"
            cf_cards += f"""
            <div class="cf-card">
                <div class="cf-header">
                    <span class="cf-name">{_html_escape(cf.get('plan_name', ''))}</span>
                    <span class="cf-score" style="color:{score_color}">{score:.1f}</span>
                </div>
                <div class="cf-id"><code>{_html_escape(cf.get('id', ''))}</code></div>
                <div class="cf-reasoning">{_html_escape(cf.get('reasoning', ''))}</div>
            </div>"""
        cf_html = f"""
        <div class="subsection">
            <h3>👻 Counterfactual Plans</h3>
            <div class="cf-grid">{cf_cards}</div>
        </div>"""

    # ── User Input (from top-level explainability.user_input) ──
    user_input_html = ""
    if user_input_top:
        user_input_escaped = _html_escape(user_input_top)
        user_input_html = f"""
        <div class="subsection">
            <h3>🗣️ User Utterance</h3>
            <div class="user-input-box" style="background:#e3f2fd;padding:1rem;border-radius:8px;border-left:4px solid #0288d1;margin-bottom:1rem;">
                <p style="margin:0;font-size:0.95rem;line-height:1.6;">{user_input_escaped}</p>
            </div>
        </div>"""
    
    # ── Winning Plan (from top-level explainability.winning_plan) ──
    winning_plan_html = ""
    if winning_plan_top:
        plan_id = _html_escape(winning_plan_top.get("plan_id", ""))
        plan_name = _html_escape(winning_plan_top.get("plan_name", ""))
        plan_score = winning_plan_top.get("score", "N/A")
        steps = winning_plan_top.get("steps", [])
        
        if steps:
            steps_rows = ""
            for i, step in enumerate(steps, 1):
                tool_name = _html_escape(str(step.get("tool_name", "unknown")))
                args = json.dumps(step.get("args", {}), indent=2)
                justification = _html_escape(str(step.get("justification", "")))
                steps_rows += f"""
                <tr>
                    <td style="width:50px;text-align:center;"><strong>#{i}</strong></td>
                    <td><code style="background:#e8f5e9;color:#388e3c;padding:0.2rem 0.5rem;border-radius:4px;">{tool_name}</code></td>
                    <td><pre class="inline-pre">{_html_escape(args)}</pre></td>
                    <td>{justification}</td>
                </tr>"""
            
            winning_plan_html = f"""
            <div class="subsection">
                <h3>🏆 Winning Plan</h3>
                <div style="background:#f1f8e9;padding:0.75rem 1rem;border-radius:8px;border-left:4px solid #388e3c;margin-bottom:1rem;">
                    <strong>Plan:</strong> {plan_name} <span style="font-size:0.85rem;color:#666;">({plan_id})</span>
                    <span style="float:right;background:#388e3c;color:white;padding:0.2rem 0.6rem;border-radius:999px;font-size:0.8rem;font-weight:600;">Score: {plan_score}</span>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th style="width:50px;">#</th>
                                <th>Tool</th>
                                <th>Arguments</th>
                                <th>Justification</th>
                            </tr>
                        </thead>
                        <tbody>{steps_rows}</tbody>
                    </table>
                </div>
            </div>"""

    # ── Graph summary ──

    graph_html = ""
    if graph:
        graph_id = _html_escape(graph.get("graph_id", ""))
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        node_type_counts: dict[str, int] = {}
        for n in nodes:
            nt = n.get("node_type", "unknown")
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

        type_badges = ""
        type_colors = {
            "director": "#0288d1",
            "plan_step": "#f57c00",
            "tool_call": "#388e3c",
            "final_answer": "#c2185b",
            "ghost": "#9e9e9e",
            "memory_load": "#7b1fa2",
        }
        for nt, count in node_type_counts.items():
            c = type_colors.get(nt, "#666")
            type_badges += f'<span class="badge" style="background:{c}20;color:{c};border:1px solid {c}40">{_html_escape(nt)}: {count}</span> '

        graph_html = f"""
        <div class="subsection">
            <h3>🔗 Graph Summary</h3>
            <div class="graph-summary">
                <div class="graph-meta">
                    <span><strong>Graph ID:</strong> <code>{graph_id}</code></span>
                    <span><strong>Nodes:</strong> {len(nodes)} &nbsp;|&nbsp; <strong>Edges:</strong> {len(edges)}</span>
                </div>
                <div class="graph-badges">{type_badges}</div>
            </div>
        </div>"""


    return f"""
    <div class="section" id="explainability">
        <h2>🧠 Explainability</h2>
        {user_input_html}
        {winning_plan_html}
        {spans_html}
        {cf_html}
        {graph_html}
    </div>"""



def _render_error_section(error_data: dict[str, Any]) -> str:
    """Render the error telemetry section as HTML."""
    has_errors = any(
        error_data.get(k) for k in ("api_failure", "rate_limits", "quota_exceeded", "timeouts")
    )
    retries = error_data.get("retries", 0)
    pipeline_errors = error_data.get("pipeline_errors", [])
    
    if not has_errors and retries == 0 and not pipeline_errors:
        return """
        <div class="section" id="errors">
            <h2>⚠️ Errors</h2>
            <div class="no-errors">✅ No errors recorded</div>
        </div>"""

    rows = ""
    for key, label in [
        ("api_failure", "API Failure"),
        ("rate_limits", "Rate Limits"),
        ("quota_exceeded", "Quota Exceeded"),
        ("timeouts", "Timeouts"),
    ]:
        val = error_data.get(key, "")
        if val:
            rows += f"""
            <tr>
                <td>{label}</td>
                <td class="error-val">{_html_escape(str(val))}</td>
            </tr>"""

    if retries:
        rows += f"""
        <tr>
            <td>Retries</td>
            <td class="num">{retries}</td>
        </tr>"""

    # Pipeline errors section
    pipeline_errors_html = ""
    if pipeline_errors:
        pipeline_rows = ""
        for pe in pipeline_errors:
            stage = _html_escape(pe.get("stage", "unknown"))
            error_type = _html_escape(pe.get("error_type", ""))
            message = _html_escape(pe.get("message", ""))
            traceback_text = _html_escape(pe.get("traceback", ""))
            pipeline_rows += f"""
            <tr>
                <td><code>{stage}</code></td>
                <td><code>{error_type}</code></td>
                <td>{message}</td>
                <td><pre class="inline-pre">{traceback_text}</pre></td>
            </tr>"""
        
        pipeline_errors_html = f"""
        <div class="subsection">
            <h3>🔧 Pipeline Errors</h3>
            <div class="table-wrap">
                <table>
                    <thead><tr><th>Stage</th><th>Error Type</th><th>Message</th><th>Traceback</th></tr></thead>
                    <tbody>{pipeline_rows}</tbody>
                </table>
            </div>
        </div>"""

    return f"""
    <div class="section" id="errors">
        <h2>⚠️ Errors</h2>
        <div class="table-wrap">
            <table>
                <thead><tr><th>Type</th><th>Details</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        {pipeline_errors_html}
    </div>"""



# ── Section dispatcher ─────────────────────────────────────────────────────────

_SECTION_RENDERERS = {
    "token": _render_token_section,
    "latency": _render_latency_section,
    "model": _render_model_section,
    "prompt": _render_prompt_section,
    "completion": _render_completion_section,
    "tool": _render_tool_section,
    "explainability": _render_explainability_section,
    "error": _render_error_section,
}


# ── Main renderTele function ──────────────────────────────────────────────────

def renderTele(tele: str | dict[str, Any], id: str, path: str):
    """Render telemetry data into a self-contained HTML report.

    Similar to :func:`renderTrace`, this takes the telemetry payload and
    produces a rich, interactive HTML file in the output directory.

    Args:
        tele: Telemetry data. Can be either:
            - A dict with a ``"telemetry"`` key (as produced by
              :meth:`TelemetryCollector.build_trace`), or
            - A path string to a ``tele_*.json`` file.
        id: Identifier used in the output filename (typically the graph_id).
        path: Output directory where the HTML file will be written.
    """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load telemetry data ─────────────────────────────────────────────────
    if isinstance(tele, dict):
        tele_data = tele
    elif isinstance(tele, str) and Path(tele).is_file():
        with open(tele, "r", encoding="utf-8") as f:
            tele_data = json.load(f)
    elif isinstance(tele, str):
        try:
            tele_data = json.loads(tele)
        except json.JSONDecodeError:
            print(f"\n[WARN] renderTele: could not parse telemetry input as JSON or file path")
            return
    else:
        print(f"\n[WARN] renderTele: unsupported telemetry input type: {type(tele)}")
        return

    telemetry_list: list[dict[str, Any]] = tele_data.get("telemetry", [])
    if not telemetry_list:
        print(f"\n[WARN] renderTele: telemetry data is empty")
        return

    # ── Build HTML sections ─────────────────────────────────────────────────
    sections_html = ""
    nav_items = ""
    section_icons = {
        "token": "🔢", "latency": "⏱️", "model": "🤖", "prompt": "📝",
        "completion": "💬", "tool": "🔧", "explainability": "🧠", "error": "⚠️",
    }

    for entry in telemetry_list:
        entry_type = entry.get("type", "")
        renderer = _SECTION_RENDERERS.get(entry_type)
        if renderer:
            sections_html += renderer(entry)
            icon = section_icons.get(entry_type, "📋")
            nav_items += f'<a href="#{entry_type}">{icon} {entry_type.title()}</a>\n'

    # ── Assemble full HTML ──────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Telemetry Report — {_html_escape(id)}</title>
<style>
  :root {{
    --bg: #f8f9fa;
    --surface: #ffffff;
    --border: #e0e0e0;
    --text: #212121;
    --text-secondary: #616161;
    --accent: #0288d1;
    --accent-light: #e1f5ff;
    --success: #388e3c;
    --warn: #f57c00;
    --error: #c2185b;
    --radius: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}

  /* ── Header ─────────────────────────────────────── */
  .header {{
    background: linear-gradient(135deg, #0288d1 0%, #7b1fa2 100%);
    color: white;
    padding: 2rem 2rem 1.5rem;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }}
  .header .subtitle {{ opacity: 0.85; font-size: 0.9rem; }}

  /* ── Nav ────────────────────────────────────────── */
  .nav {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0.5rem 2rem;
    display: flex;
    gap: 0.25rem;
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: var(--shadow);
  }}
  .nav a {{
    text-decoration: none;
    color: var(--text-secondary);
    padding: 0.4rem 0.75rem;
    border-radius: var(--radius);
    font-size: 0.82rem;
    font-weight: 500;
    transition: all 0.15s;
  }}
  .nav a:hover {{ background: var(--accent-light); color: var(--accent); }}

  /* ── Main ───────────────────────────────────────── */
  .main {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}

  /* ── Section ────────────────────────────────────── */
  .section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow);
  }}
  .section h2 {{
    font-size: 1.15rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
  }}
  .subsection {{
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px dashed var(--border);
  }}
  .subsection h3 {{
    font-size: 0.95rem;
    margin-bottom: 0.75rem;
    color: var(--text-secondary);
  }}

  /* ── Tables ─────────────────────────────────────── */
  .table-wrap {{ overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    background: #f5f5f5;
    text-align: left;
    padding: 0.6rem 0.75rem;
    font-weight: 600;
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
  }}
  td {{
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: top;
  }}
  tr:hover {{ background: #fafafa; }}
  .summary-row {{ background: #f5f5f5; font-weight: 600; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  /* ── Cards ──────────────────────────────────────── */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
  }}
  .card-label {{ font-size: 0.78rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.03em; }}
  .card-value {{ font-size: 1.5rem; font-weight: 700; margin: 0.25rem 0; font-variant-numeric: tabular-nums; }}
  .card-value-sm {{ font-size: 0.85rem; font-weight: 600; word-break: break-all; }}
  .card-pct {{ font-size: 0.75rem; color: var(--text-secondary); }}
  .card-bar {{ height: 4px; background: #eee; border-radius: 2px; margin-top: 0.5rem; }}
  .card-bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}
  .card-total .card-bar-fill {{ background: #0288d1; }}
  .card-inference .card-bar-fill {{ background: #f57c00; }}
  .card-tool .card-bar-fill {{ background: #388e3c; }}
  .card-ttft .card-bar-fill {{ background: #7b1fa2; }}
  .card-model {{ border-left: 3px solid var(--accent); }}

  /* ── Badges ─────────────────────────────────────── */
  .badge {{
    display: inline-block;
    font-size: 0.72rem;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    background: #f0f0f0;
    color: var(--text-secondary);
    font-weight: 500;
    margin-left: 0.35rem;
  }}
  .badge-tool {{ background: #e8f5e9; color: #388e3c; }}
  .badge-success {{ background: #e8f5e9; color: #388e3c; }}
  .badge-warn {{ background: #fff3e0; color: #f57c00; }}
  .badge-error {{ background: #fce4ec; color: #c2185b; }}

  /* ── Prompts ────────────────────────────────────── */
  .prompt-details {{
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 0.5rem;
    overflow: hidden;
  }}
  .prompt-details summary {{
    padding: 0.6rem 1rem;
    cursor: pointer;
    font-weight: 500;
    font-size: 0.85rem;
    background: #fafafa;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .prompt-details summary::-webkit-details-marker {{ display: none; }}
  .prompt-details summary::before {{
    content: "▶";
    font-size: 0.7rem;
    transition: transform 0.15s;
  }}
  .prompt-details[open] summary::before {{ transform: rotate(90deg); }}
  .prompt-details summary:hover {{ background: var(--accent-light); }}
  .prompt-summary-title {{ font-weight: 600; }}
  .prompt-body {{ padding: 0.75rem 1rem; border-top: 1px solid var(--border); }}
  .prompt-block {{ margin-bottom: 0.75rem; }}
  .prompt-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
    font-weight: 600;
  }}
  .prompt-block pre {{
    background: #263238;
    color: #eeffff;
    padding: 0.75rem;
    border-radius: 6px;
    font-size: 0.78rem;
    line-height: 1.5;
    overflow-x: auto;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }}

  /* ── Completion ─────────────────────────────────── */
  .completion-wrap {{ border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }}
  .completion-header {{ padding: 0.5rem 1rem; background: #fafafa; border-bottom: 1px solid var(--border); }}
  .completion-output {{
    padding: 1rem;
    background: #263238;
    color: #eeffff;
    font-size: 0.82rem;
    line-height: 1.5;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }}

  /* ── Inline pre ─────────────────────────────────── */
  .inline-pre {{
    margin: 0;
    font-size: 0.78rem;
    background: #f5f5f5;
    padding: 0.35rem 0.5rem;
    border-radius: 4px;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }}

  /* ── Span bars ──────────────────────────────────── */
  .span-name {{ font-weight: 500; white-space: nowrap; }}
  .span-bar-cell {{ width: 50%; }}
  .span-bar {{ height: 18px; background: #eee; border-radius: 4px; overflow: hidden; }}
  .span-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; min-width: 2px; }}
  .director-badge {{
    display: inline-block;
    background: var(--accent-light);
    color: var(--accent);
    padding: 0.3rem 0.75rem;
    border-radius: var(--radius);
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
  }}

  /* ── Counterfactuals ────────────────────────────── */
  .cf-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.75rem; }}
  .cf-card {{
    border: 1px dashed #bdbdbd;
    border-radius: var(--radius);
    padding: 1rem;
    background: #fafafa;
  }}
  .cf-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; }}
  .cf-name {{ font-weight: 600; font-size: 0.9rem; }}
  .cf-score {{ font-size: 1.2rem; font-weight: 700; }}
  .cf-id {{ font-size: 0.72rem; color: var(--text-secondary); margin-bottom: 0.5rem; }}
  .cf-reasoning {{ font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; }}

  /* ── Graph summary ──────────────────────────────── */
  .graph-summary {{ padding: 0.5rem 0; }}
  .graph-meta {{ display: flex; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 0.75rem; font-size: 0.85rem; }}
  .graph-badges {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}

  /* ── Errors ─────────────────────────────────────── */
  .no-errors {{
    text-align: center;
    padding: 1.5rem;
    color: var(--success);
    font-weight: 600;
    font-size: 0.95rem;
  }}
  .error-val {{ color: var(--error); font-weight: 500; }}

  /* ── Footer ──────────────────────────────────────── */
  .footer {{
    text-align: center;
    padding: 1.5rem;
    color: var(--text-secondary);
    font-size: 0.78rem;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 Telemetry Report</h1>
  <div class="subtitle">ID: {_html_escape(id)}</div>
</div>

<nav class="nav">
  {nav_items}
</nav>

<div class="main">
  {sections_html}
</div>

<div class="footer">
  Built with ❤️ by <strong>GoAT</strong>.
</div>

</body>
</html>"""

    # ── Write HTML file ─────────────────────────────────────────────────────
    html_path = output_dir / f"tele_{id}.html"
    try:
        html_path.write_text(html, encoding="utf-8")
        print(f"\nTelemetry HTML report saved to: {html_path}")
    except (PermissionError, OSError, UnicodeEncodeError) as e:
        print(f"\n[ERROR] Cannot write telemetry report to '{html_path}': {e}")
