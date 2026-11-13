"""
agent_viz.py
------------
Live visualisations for Step 5 (multi-agent patterns).

    run_sequential(graph, date, wish)   -> animates a LangGraph StateGraph
    run_supervisor(app, date, wish)     -> animates a supervisor whose specialists are tools (ask_<agent>)
    run_network(swarm, date, wish)      -> animates a langgraph-swarm app
    compare_runs()                      -> table comparing all runs so far
    show_patterns()                     -> static overview of the three patterns

The animation is driven by LangChain callbacks, so it shows in real time which
agent is thinking, which tool it calls and whom it hands the work to. The
diagrams are derived from the compiled graphs themselves: if you rewire a graph
or give an agent another handoff tool, the picture changes with it.
"""

from __future__ import annotations

import html
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

from IPython.display import HTML, display
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphRecursionError


# ======================================================================
# Look & feel
# ======================================================================
AGENT_STYLE = {   # node name -> (icon, display name, Italian role)
    "forecast_agent":   ("🔮", "Forecast",   "il previsore"),
    "soup_agent":       ("🍲", "Soup",       "il cuoco"),
    "purchasing_agent": ("🛒", "Purchasing", "il dispensiere"),
    "staffing_agent":   ("👥", "Staffing",   "il caposala"),
    "sommelier_agent":  ("🍷", "Sommelier",  "il sommelier"),
    "supervisor":       ("👨‍🍳", "Robo-Giovanni",   "il capo della brigata"),
    "robo_giovanni":    ("👨‍🍳", "Robo-Giovanni",   "il capo della brigata"),
}
CHEFS = {"supervisor", "robo_giovanni"}   # Robo-Giovanni's node names (dark card in every pattern)

# Every agent has its own colour – used for its card, its arrows,
# its lines in the event log and its result. The same colours are used in the notebook text.
AGENT_COLORS = {
    "robo_giovanni":    "#facc15",   # gold
    "supervisor":       "#facc15",
    "forecast_agent":   "#60a5fa",   # blue
    "soup_agent":       "#fb923c",   # orange
    "purchasing_agent": "#2dd4bf",   # teal
    "staffing_agent":   "#c084fc",   # violet
    "sommelier_agent":  "#f472b6",   # rose
}
_FALLBACK_COLOR = "#94a3b8"          # any agent you add yourself


def _agent_css(names):
    """CSS that gives every [data-ag=<name>] element its colour as --ag."""
    return "<style>" + "".join(f".pz [data-ag='{n}']{{--ag:{AGENT_COLORS.get(n, _FALLBACK_COLOR)}}}"
                               for n in sorted(set(names))) + "</style>"


def _who(n):
    """An agent's name in its colour, for the event log."""
    icon, name, _ = _agent_style(n)
    return f"<span class='pz-who' data-ag='{n}'>{icon} {_esc(name)}</span>"
STEP_STYLE = {    # plain-Python nodes in the sequential graph
    "stock_check": ("🔍", "Stock check"),
    "persist":     ("💾", "Save plan"),
}
PATTERNS = {
    "sequential": ("🏭", "Sequential", "La Mise en Place"),
    "supervisor": ("👨‍🍳", "Supervisor", "Il Capo della Brigata"),
    "network":    ("🕸️", "Network", "Che Casino!"),
}


AGENT_ORDER = {n: i for i, n in enumerate(AGENT_STYLE)}


def _agent_style(name):
    if name in AGENT_STYLE:
        return AGENT_STYLE[name]
    pretty = name.replace("_agent", "").replace("_", " ").title()
    return ("🤖", pretty, "")


def _step_style(name):
    if name in STEP_STYLE:
        return STEP_STYLE[name]
    return ("⚙️", name.replace("_", " ").title())


_CSS = """
<style>
.pz{--bg:#1c1917;--panel:#292524;--ink:#f5f5f4;--muted:#a8a29e;--line:#57534e;--chip:#3a3633;
    --accent:#fb923c;--accent-soft:#3b2416;--ok:#4ade80;--ok-soft:#14281c;--live:#fbbf24;--live-soft:#3a2e12;
    --bad:#f87171;--bad-soft:#3a1717;--boss:#f5f5f4;--boss-ink:#1c1917;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    color-scheme:dark;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:14px;
    padding:14px 16px 12px;margin:4px 0;max-width:920px;box-sizing:border-box;line-height:1.35;}
.pz *{box-sizing:border-box}
.pz-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.pz-title{font-size:16px;font-weight:700}
.pz-sub{font-size:12px;color:var(--muted);font-style:italic}
.pz-chips{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.pz-chip{background:var(--chip);border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;white-space:nowrap}
.pz-chip.run{background:var(--live-soft);color:var(--live)}
.pz-chip.done{background:var(--ok-soft);color:var(--ok)}
.pz-chip.stop{background:var(--bad-soft);color:var(--bad)}
.pz-chip[data-ag]{background:color-mix(in srgb,var(--ag) 14%,var(--panel));color:var(--ag);
    box-shadow:inset 0 0 0 1px var(--ag)}
.pz-chip.now{font-weight:800}
.pz-chip.run::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--live);
    margin-right:6px;vertical-align:1px;animation:pzblink 1s infinite}
.pz-scroll{overflow-x:auto;border-radius:10px;background:var(--panel);border:1px solid var(--line)}
.pz-canvas{position:relative;margin:0 auto}
.pz-canvas svg{position:absolute;inset:0;overflow:visible}
.pz-e{fill:none;stroke:var(--line);stroke-width:1.6}
.pz-e.cond{stroke-dasharray:4 5}
.pz-e.used{stroke:var(--ag,var(--ok));stroke-width:2.6;stroke-dasharray:none}
.pz-e.back.used{stroke:var(--bad)}
.pz-e.live{stroke:var(--ag,var(--live));stroke-width:3.4;stroke-dasharray:7 6;animation:pzflow .7s linear infinite}
.pz-mk path{fill:var(--line)} .pz-pat .pz-mk path{fill:var(--muted)} .pz-mk.used path{fill:var(--ok)} .pz-mk.live path{fill:var(--live)} .pz-mk.bad path{fill:var(--bad)}
.pz-mk.ag path{fill:var(--ag)}
.pz-lbl{position:absolute;transform:translate(-50%,-50%);font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;
    background:var(--panel);border:1px solid var(--line);color:var(--muted);white-space:nowrap;z-index:3}
.pz-lbl.used,.pz-lbl.live{color:var(--ag,var(--ok));border-color:var(--ag,var(--ok))}
.pz-lbl.bad{color:var(--bad);border-color:var(--bad)}
.pz-card{position:absolute;transform:translate(-50%,-50%);background:var(--panel);border:1px solid var(--line);
    border-top:5px solid var(--ag,var(--line));border-radius:12px;padding:8px 10px;z-index:2;overflow:hidden;
    box-shadow:0 1px 4px rgba(0,0,0,.4);display:flex;flex-direction:column;gap:2px;
    transition:opacity .25s,transform .25s}
.pz-card.live{border:2px solid var(--ag,var(--live));border-top-width:5px;z-index:4;
    background:color-mix(in srgb,var(--ag,var(--live)) 13%,var(--panel));
    transform:translate(-50%,-50%) scale(1.07);animation:pzglow 1.2s infinite}
.pz-card.live::after{content:"● LIVE";position:absolute;top:7px;right:9px;font-size:9px;font-weight:800;
    letter-spacing:.5px;color:var(--ag,var(--live));animation:pzblink 1s infinite}
.pz-card.boss{background:var(--boss);color:var(--boss-ink);border-color:var(--boss);border-top-color:var(--ag,var(--accent))}
.pz-card.boss.live{background:var(--boss);border-color:var(--ag,var(--live))}
.pz-card.boss .pz-role,.pz-card.boss .pz-ft{color:var(--boss-ink);opacity:.7}
.pz.running .pz-card:not(.live):not(.done){opacity:.45}
.pz.running .pz-card.done{opacity:.8}
.pz.running .pz-card.boss:not(.live){opacity:.85}
.pz-r{display:flex;align-items:center;gap:7px}
.pz-ico{font-size:18px;line-height:1;width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;flex:none;background:color-mix(in srgb,var(--ag,var(--line)) 18%,transparent);
    box-shadow:inset 0 0 0 2px var(--ag,var(--line))}
.pz-nm{font-weight:800;font-size:13px;color:var(--ag,var(--ink))}
.pz-card.boss .pz-nm{color:var(--boss-ink)}
.pz-role{font-size:10.5px;font-style:italic;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pz-st{font-size:11px;font-weight:700;color:var(--muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pz-card.live .pz-st{color:var(--ag,var(--live))} .pz-card.done .pz-st{color:var(--muted)}
.pz-snip{font-size:10.5px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.pz-ft{margin-top:auto;font-size:10.5px;color:var(--muted);display:flex;gap:8px}
.pz-badge{background:var(--bad);color:#fff;font-size:10px;font-weight:700;border-radius:999px;padding:0 6px;margin-left:4px}
.pz-tag{font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;color:var(--accent)}
.pz-step{position:absolute;transform:translate(-50%,-50%);width:46px;height:46px;border-radius:12px;z-index:2;
    background:var(--panel);border:1.5px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:20px}
.pz-step.done{border-color:var(--ok)} .pz-step.live{border-color:var(--live);background:var(--live-soft)} .pz-step.bad{border-color:var(--bad);background:var(--bad-soft)}
.pz-step span{position:absolute;top:50px;font-size:10px;font-weight:700;color:var(--muted);white-space:nowrap}
.pz-pill{position:absolute;transform:translate(-50%,-50%);font-size:10px;font-weight:800;letter-spacing:.5px;z-index:2;
    background:var(--boss);color:var(--boss-ink);border-radius:999px;padding:5px 9px}
.pz-banner{margin-top:10px;padding:8px 12px;border-radius:10px;font-size:13px;font-weight:600}
.pz-banner.bad{background:var(--bad-soft);color:var(--bad)} .pz-banner.ok{background:var(--ok-soft);color:var(--ok)}
.pz-log{margin-top:10px;font-size:12px}
.pz-log h4,.pz-res h4{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
.pz-ev{display:flex;gap:8px;padding:3px 0;border-bottom:1px dashed var(--line)}
.pz-ev:last-child{border-bottom:none}
.pz-ev .t{color:var(--muted);font-variant-numeric:tabular-nums;min-width:44px}
.pz-ev{border-left:3px solid var(--ag,transparent);padding-left:7px}
.pz-ev.warn{color:var(--bad);font-weight:600}
.pz.running .pz-log .pz-ev:last-child{background:color-mix(in srgb,var(--ag,var(--live)) 12%,transparent);font-weight:600}
.pz-who{color:var(--ag);font-weight:800;white-space:nowrap}
.pz-res{margin-top:12px}
.pz-res details{background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--ag,var(--line));
    border-radius:10px;margin-bottom:6px}
.pz-res summary .pz-who{font-size:13px;color:var(--ag);font-weight:800;margin-left:0}
.pz-res summary{cursor:pointer;padding:7px 12px;font-weight:700;font-size:13px}
.pz-res summary span{font-weight:400;color:var(--muted);font-size:12px;margin-left:6px}
.pz-md{padding:0 14px 10px;font-size:13px}
.pz-md table{border-collapse:collapse;margin:6px 0;font-size:12px}
.pz-md th,.pz-md td{border:1px solid var(--line);padding:3px 8px;text-align:left}
.pz-md code{background:var(--chip);padding:0 4px;border-radius:4px}
.pz-md ul,.pz-md ol{margin:4px 0;padding-left:20px}
.pz-md p{margin:5px 0} .pz-md h1,.pz-md h2,.pz-md h3,.pz-md h4{font-size:13.5px;margin:10px 0 4px}
.pz-cmp{border-collapse:collapse;width:100%;font-size:13px;background:var(--panel);border-radius:10px;overflow:hidden}
.pz-cmp th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--line)}
.pz-cmp td{padding:8px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
.pz-cmp tr:last-child td{border-bottom:none}
.pz-best{color:var(--ok);font-weight:700}
.pz-note{font-size:12px;color:var(--muted);margin-top:8px}
.pz-pat{display:flex;gap:12px;flex-wrap:wrap}
.pz-pat>div{flex:1 1 0;min-width:240px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.pz-pat b{font-size:14px} .pz-pat p{margin:4px 0 0;font-size:12px;color:var(--muted)}
.pz-pat svg{display:block;margin:6px auto 0} .pz-pat svg text{fill:var(--ink);font-size:16px}
.pz-brigade{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;align-items:center}
.pz-brigade>b{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;margin-right:4px}
@keyframes pzpulse{0%,100%{box-shadow:0 0 0 0 rgba(217,119,6,.45)}50%{box-shadow:0 0 0 7px rgba(217,119,6,0)}}
@keyframes pzglow{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,var(--ag,var(--live)) 60%,transparent)}
    50%{box-shadow:0 0 0 10px color-mix(in srgb,var(--ag,var(--live)) 0%,transparent)}}
@keyframes pzflow{to{stroke-dashoffset:-13}}
@keyframes pzblink{50%{opacity:.25}}
</style>
"""

_MARKERS = """<defs>""" + "".join(
    f"<marker id='pz-a-{k}' class='pz-mk {k}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
    f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z'/></marker>"
    for k in ("idle", "used", "live", "bad")) + "</defs>"


# ======================================================================
# Small helpers
# ======================================================================
def _esc(s):
    return html.escape(str(s or ""))


def _snippet(s, n=110):
    s = re.sub(r"[*#`|>]+", "", str(s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n] + "…") if len(s) > n else s


def _fmt_tokens(n):
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def _md(text):
    """Tiny markdown -> HTML (bold, code, headings, lists, tables, paragraphs)."""
    def inline(s):
        s = _esc(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return s

    out, lines, i = [], str(text or "").splitlines(), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f"<table><tr>{head}</tr>{body}</table>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            out.append(f"<h4>{inline(m.group(2))}</h4>")
            i += 1
            continue
        if re.match(r"^\s*([-*•]|\d+[.)])\s+", line):
            ordered = bool(re.match(r"^\s*\d+[.)]", line))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*•]|\d+[.)])\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", lines[i]))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(\s*[-*•|]|\s*\d+[.)]\s|#)", lines[i]):
            para.append(lines[i].rstrip())
            i += 1
        out.append("<p>" + "<br>".join(inline(p) for p in para) + "</p>")
    return "".join(out)


def _clip(box, tx, ty, pad=3):
    """Point where the line from the box centre towards (tx, ty) leaves the box."""
    cx, cy, w, h = box
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2 + pad) / abs(dx) if dx else math.inf
    sy = (h / 2 + pad) / abs(dy) if dy else math.inf
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def _curve(a, b, bend=0.0):
    """Quadratic curve between two boxes, bent sideways by `bend` px. Returns (path, label point)."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    mx, my = (ax + bx) / 2, (ay + by) / 2
    length = math.hypot(bx - ax, by - ay) or 1
    nx, ny = -(by - ay) / length, (bx - ax) / length
    cx, cy = mx + nx * bend, my + ny * bend
    x1, y1 = _clip(a, cx, cy)
    x2, y2 = _clip(b, cx, cy)
    lx, ly = 0.25 * x1 + 0.5 * cx + 0.25 * x2, 0.25 * y1 + 0.5 * cy + 0.25 * y2
    return f"M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}", (lx, ly)


def _edge_svg(path, state, extra="", ag=None):
    marker = {"live": "live", "used": "used", "bad": "bad"}.get(state, "idle")
    if ag and state in ("live", "used"):
        marker = f"ag-{ag}"
    cls = {"live": "live", "used": "used", "bad": "used"}.get(state, "")
    data = f" data-ag='{ag}'" if ag else ""
    return f"<path class='pz-e {cls} {extra}'{data} d='{path}' marker-end='url(#pz-a-{marker})'/>"


def _agent_markers(names):
    """One arrow head per agent, coloured via --ag (markers inherit from their <g>, not from the path)."""
    return "<defs>" + "".join(
        f"<g data-ag='{n}'><marker id='pz-a-ag-{n}' class='pz-mk ag' viewBox='0 0 10 10' refX='9' refY='5' "
        f"markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z'/></marker></g>"
        for n in sorted(set(names))) + "</defs>"


def _label(x, y, text, state="", ag=None):
    data = f" data-ag='{ag}'" if ag else ""
    return f"<div class='pz-lbl {state}'{data} style='left:{x:.0f}px;top:{y:.0f}px'>{text}</div>"


# ======================================================================
# Run bookkeeping
# ======================================================================
@dataclass
class RunResult:
    pattern: str
    date: str
    seconds: float = 0.0
    llm_calls: int = 0
    coordination_calls: int = 0
    tokens: int = 0
    agent_turns: int = 0
    handoffs: int = 0
    loop_backs: int = 0
    outputs: dict = field(default_factory=dict)
    final: str = ""
    state: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def messages(self):
        return self.state.get("messages", [])

    def __repr__(self):
        status = "stopped: " + self.error if self.error else "completed"
        return (f"<RunResult {self.pattern} {self.date}: {status}, {self.seconds:.0f}s, "
                f"{self.llm_calls} LLM calls, {self.tokens} tokens>")


RUNS: list = []


class _Tracker(BaseCallbackHandler):
    """Turns LangChain callbacks into events for the live view."""

    raise_error = False

    def __init__(self, view):
        self.view = view
        self.runs = {}   # run_id -> (agent, tool name or None)
        self.lock = threading.Lock()

    @staticmethod
    def _who(metadata):
        # create_agent(name=...) tags every call with lc_agent_name – also for agents that run
        # INSIDE a tool (supervisor pattern). Otherwise fall back to the graph node name.
        metadata = metadata or {}
        if metadata.get("lc_agent_name"):
            return metadata["lc_agent_name"]
        ns = metadata.get("langgraph_checkpoint_ns", "")
        return (ns.split("|")[0].split(":")[0] if ns else "") or "?"

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        who = self._who(metadata)
        self.runs[run_id] = (who, None)
        with self.lock:
            self.view.llm_start(who)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kw):
        if run_id not in self.runs:
            self.on_chat_model_start(serialized, [], run_id=run_id, metadata=metadata)

    def on_llm_end(self, response, *, run_id, **kw):
        who, _ = self.runs.get(run_id, ("?", None))
        try:
            msg = response.generations[0][0].message
        except Exception:
            return
        usage = getattr(msg, "usage_metadata", None) or {}
        with self.lock:
            self.view.llm_end(who, msg, int(usage.get("total_tokens", 0) or 0))

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, inputs=None, **kw):
        who = self._who(metadata)
        name = (serialized or {}).get("name") or kw.get("name") or "tool"
        self.runs[run_id] = (who, name)
        with self.lock:
            self.view.tool_start(who, name, inputs if inputs is not None else input_str)

    def on_tool_end(self, output, *, run_id, **kw):
        who, name = self.runs.get(run_id, ("?", None))
        with self.lock:
            self.view.tool_end(who, name, getattr(output, "content", output))


# ======================================================================
# Live view
# ======================================================================
class _LiveView:
    W = 880

    def __init__(self, pattern, date, layout, edges, boss=None, entry=None, width=None, tools_mode=False):
        self.pattern, self.date = pattern, date
        self.W = width or _LiveView.W
        self.layout = layout          # node -> dict(kind, x, y, w, h)
        self.edges = edges            # (a, b, kind) -> dict(cond=bool)
        self.boss, self.entry = boss, entry
        self.agents = [n for n, v in layout.items() if v["kind"] in ("agent", "boss")]
        self.status = {n: "idle" for n in layout}
        self.activity, self.snip = {}, {}
        self.calls = {n: 0 for n in layout}
        self.tokens = {n: 0 for n in layout}
        self.visits = {n: 0 for n in layout}
        self.edge_count, self.edge_label = {}, {}
        self.live_edges, self.current = set(), None
        self.tools_mode = tools_mode      # supervisor whose specialists are ask_<agent> tools
        self.working = set()              # specialists currently working (can be several at once)
        self.outputs, self.final = {}, ""
        self.events = []
        self.t0 = time.time()
        self.state, self.banner = "run", None
        self.result = RunResult(pattern, date)
        self._last = 0.0
        self.handle = display(HTML(self.render()), display_id=True)

    # ---------- events ----------
    def _event(self, icon, text, kind="", who=None):
        self.events.append((time.time() - self.t0, icon, text, kind, who))

    def _use_edge(self, a, b, kind="flow", label=None, exclusive=True):
        key = (a, b, kind)
        if key not in self.edges:
            self.edges[key] = {"cond": True}
        self.edge_count[key] = self.edge_count.get(key, 0) + 1
        if exclusive:
            self.live_edges = {key}
        else:
            self.live_edges.add(key)
        if label:
            self.edge_label[key] = label

    def enter(self, node, via=None):
        """A node starts working."""
        if node not in self.layout or node == self.current:
            return
        prev = self.current
        if prev and self.layout[prev]["kind"] != "boss" and self.status.get(prev) == "live":
            self.status[prev] = "done"
        if via is not None:
            self._use_edge(*via)
        self.current = node
        self.status[node] = "live"
        if self.layout[node]["kind"] in ("agent",):
            self.visits[node] += 1
            self.result.agent_turns += 1
            icon, name, _ = _agent_style(node)
            again = f" (visit #{self.visits[node]})" if self.visits[node] > 1 else ""
            if node in CHEFS and self.visits[node] > 1:
                self._event("▶", f"{_who(node)} takes the work back", who=node)
            else:
                self._event("▶", f"{_who(node)} starts working{again}", who=node)

    def finish(self, node):
        if node in self.status:
            self.status[node] = "done"
            self.activity.pop(node, None)

    def llm_start(self, who):
        if who not in self.layout:
            return
        self.calls[who] += 1
        self.result.llm_calls += 1
        if self.layout[who]["kind"] == "boss":
            self.result.coordination_calls += 1
        if self.pattern == "supervisor" and self.tools_mode:
            if who == self.boss:
                self.status[who] = "live"
        elif self.pattern == "supervisor":
            if who == self.boss and self.current not in (None, self.boss):
                came_from = self.current
                self.finish(came_from)
                self._use_edge(came_from, self.boss, "report")
                self._event("📨", f"{_who(came_from)} reports back to {_who(self.boss)}", "hand", who=came_from)
                self.current = None
            if who == self.boss:
                self.status[who] = "live"
                self.current = who
        if self.pattern == "sequential" and who != self.current:
            self.enter(who, via=(self.current or "__start__", who, "flow"))
        elif self.pattern == "network" and self.current is None:
            self.enter(who, via=("__start__", who, "flow") if ("__start__", who, "flow") in self.edges else None)
        self.activity[who] = "🧠 thinking…"
        self.render()

    def llm_end(self, who, msg, tokens):
        if who in self.tokens:
            self.tokens[who] += tokens
        self.result.tokens += tokens
        text = msg.content if isinstance(msg.content, str) else ""
        calls = getattr(msg, "tool_calls", None) or []
        if self.tools_mode and who == self.boss:
            asks = [c["name"][len("ask_"):] for c in calls if c.get("name", "").startswith("ask_")]
            if asks:
                self.status[who] = "wait"
                self.activity[who] = f"⏳ waiting for {len(asks)} report{'s' if len(asks) > 1 else ''}"
        only_handoffs = calls and all(c.get("name", "").startswith("transfer_") for c in calls)
        if text.strip() and (not calls or only_handoffs):
            if who == self.boss or (self.pattern == "network" and who in CHEFS and not calls):
                self.final = text
            else:
                self.outputs[who] = text
                self.snip[who] = text
            self.activity[who] = "✍️ answered"
        self.render()

    def tool_start(self, who, name, args):
        if who not in self.layout:
            return
        if name.startswith("transfer_back_to_"):
            return
        if self.tools_mode and who == self.boss and name.startswith("ask_"):
            target = name[len("ask_"):]
            if target in self.layout:
                task = args.get("task", "") if isinstance(args, dict) else str(args or "")
                self._use_edge(self.boss, target, "task", exclusive=False)
                self.result.handoffs += 1
                self.result.agent_turns += 1
                self.visits[target] += 1
                self.working.add(target)
                self.status[target] = "live"
                self.activity[target] = "📋 got a task"
                again = f" (again, #{self.visits[target]})" if self.visits[target] > 1 else ""
                self._event("📋", f"{_who(self.boss)} → {_who(target)}{again}: <i>{_esc(_snippet(task, 90))}</i>",
                            "warn" if self.visits[target] > 1 else "hand", who=target)
                self.render(force=True)
                return
        if name.startswith("transfer_to_"):
            target = name[len("transfer_to_"):]
            if self.pattern == "supervisor" and who == self.boss:
                self._use_edge(self.boss, target, "task")
                self.result.handoffs += 1
                self.status[self.boss] = "idle"
                self._event("📋", f"{_who(self.boss)} delegates to {_who(target)}", "hand", who=target)
                self.current = None
                self.enter(target)
                self.activity[target] = "📋 got a task"
            else:
                self.result.handoffs += 1
                note = args.get("note") if isinstance(args, dict) else None
                if note:                                   # the handoff note carries the agent's result
                    self.outputs[who] = self.snip[who] = note
                back = (self.visits.get(target, 0) > 0
                        and AGENT_ORDER.get(target, 0) < AGENT_ORDER.get(who, 0))
                if back and (who, target, "flow") in self.edges:
                    self.edges[(who, target, "flow")]["back"] = True
                self._event("🤝", f"{_who(who)} hands off to {_who(target)}"
                            + (" — <b>going back!</b>" if back else ""), "warn" if back else "hand", who=who)
                self.finish(who)
                self.enter(target, via=(who, target, "flow"))
                self.activity[target] = "🤝 took over"
            self.render(force=True)
            return
        args_txt = ""
        if isinstance(args, dict):
            args_txt = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "config")
        self.activity[who] = f"🔧 {name}"
        self._event("🔧", f"{_who(who)} calls <code>{_esc(name)}({_esc(args_txt)[:60]})</code>", who=who)
        self.render()

    def tool_end(self, who, name, output):
        out = str(output or "")
        if self.tools_mode and who == self.boss and name and name.startswith("ask_"):
            target = name[len("ask_"):]
            if target in self.layout:
                self.working.discard(target)
                self.live_edges.discard((self.boss, target, "task"))
                self._use_edge(target, self.boss, "report", exclusive=False)
                self.live_edges.discard((target, self.boss, "report"))
                self.status[target] = "done"
                self.activity.pop(target, None)
                self.outputs[target] = self.snip[target] = out
                self._event("📨", f"{_who(target)} reports back to {_who(self.boss)}", "hand", who=target)
                if not self.working:
                    self.activity.pop(self.boss, None)
                self.render(force=True)
            return
        if name == "check_soup_ingredient" and ("OUT OF STOCK" in out or "FALLBACK REQUIRED" in out):
            m = re.search(r"'([^']+)'", out)
            ing = m.group(1) if m else "an ingredient"
            again = "FALLBACK REQUIRED" in out
            self._event("⚠️", f"{_who(who)}: <b>{_esc(ing)}</b> is out of stock"
                        + (" again — falling back to the plain soup" if again else ""), "warn", who=who)

    def node_update(self, node, update):
        """Sequential graph: a node finished (plain-Python steps have no LLM calls)."""
        if node not in self.layout:
            return
        kind = self.layout[node]["kind"]
        update = update or {}
        if kind == "step":
            prev = self.current
            self.current = node
            self._use_edge(prev or "__start__", node, "flow")
            if node == "stock_check":
                blocked = update.get("avoid") or []
                if update.get("retry"):
                    self.status[node] = "bad"
                    self._event("🔍", f"Stock check: <b>{_esc(', '.join(blocked))}</b> is out of stock → back to {_who('soup_agent')}", "warn")
                    self.edge_label[(node, "soup_agent", "flow")] = f"avoid {', '.join(blocked)}"
                    self.current = node
                elif update.get("fallback"):
                    self.status[node] = "bad"
                    self._event("🔍", f"Stock check: still not possible → fallback to <b>{_esc(update.get('soup'))}</b>", "warn")
                else:
                    self.status[node] = "done"
                    self._event("🔍", "Stock check passed: the soup ingredient is available", "hand")
            elif node == "persist":
                self.status[node] = "done"
                self._event("💾", "Day plan saved to <code>utils/pizzeria_store.json</code>", "hand")
            else:
                self.status[node] = "done"
                self._event("⚙️", f"{_esc(node)} done")
        else:
            if self.current != node:
                self.enter(node, via=(self.current or "__start__", node, "flow"))
            self.finish(node)
            self.activity.pop(node, None)
        self.render(force=True)

    def close(self, error=None, state=None):
        self.result.seconds = time.time() - self.t0
        self.result.state = state or {}
        self.result.outputs = dict(self.outputs)
        self.result.final = self.final
        self.result.loop_backs = sum(max(0, v - 1) for n, v in self.visits.items()
                                     if self.layout[n]["kind"] == "agent" and n not in CHEFS)
        self.live_edges = set()
        if error:
            self.state, self.result.error = "stop", error
            self.banner = ("bad", error)
            if self.current in self.status:
                self.status[self.current] = "bad" if self.layout[self.current]["kind"] == "step" else "done"
        else:
            self.state = "done"
            if self.current in self.status and self.layout[self.current]["kind"] != "boss":
                self.status[self.current] = "done"
            if self.boss:
                self.status[self.boss] = "done"
            last = self.current or "__start__"
            if "__end__" in self.layout:
                self._use_edge(last, "__end__", "flow")
                self.live_edges = set()
            self._event("🏁", "Finished", "hand")
        for n in list(self.activity):
            self.activity.pop(n)
        RUNS.append(self.result)
        self.render(force=True, final=True)
        return self.result

    # ---------- rendering ----------
    def render(self, force=False, final=False):
        now = time.time()
        if not force and now - self._last < 0.25:
            return ""
        self._last = now
        out = self._html(final)
        if hasattr(self, "handle"):
            self.handle.update(HTML(out))
        return out

    def _card(self, n):
        v = self.layout[n]
        icon, name, role = _agent_style(n)
        st = self.status.get(n, "idle")
        boss = v["kind"] == "boss"
        cls = ("boss " if boss or n in CHEFS else "") + ("live" if st == "live" else "done" if st in ("done", "bad") else "")
        if st == "live":
            status = self.activity.get(n, "🧠 thinking…")
        elif st in ("done", "bad"):
            status = "✓ done"
        else:
            status = "waiting"
        if boss:
            status = {"live": self.activity.get(n, "🧠 deciding…"), "wait": self.activity.get(n, "⏳ waiting"),
                      "done": "✓ coordinated"}.get(st, "delegates & reviews")
        badge = f"<span class='pz-badge'>×{self.visits[n]}</span>" if self.visits.get(n, 0) > 1 and n not in CHEFS else ""
        tag = ""
        if self.entry == n and v.get("returns"):
            tag = "<div class='pz-tag'>▶ start · 🏁 end</div>"
        elif self.entry == n:
            tag = "<div class='pz-tag'>▶ starts here</div>"
        elif v.get("exit"):
            tag = "<div class='pz-tag'>🏁 only exit</div>"
        snip = f"<div class='pz-snip'>{_esc(_snippet(self.snip.get(n), 90))}</div>" if self.snip.get(n) and st != "live" else ""
        ft = ""
        if self.calls.get(n):
            ft = f"<div class='pz-ft'><span>🧠 {self.calls[n]}</span><span>🔤 {_fmt_tokens(self.tokens[n])}</span></div>"
        return (f"<div class='pz-card {cls}' data-ag='{n}' style='left:{v['x']}px;top:{v['y']}px;width:{v['w']}px;height:{v['h']}px'>"
                f"{tag}<div class='pz-r'><div class='pz-ico'>{icon}</div><div><div class='pz-nm'>{_esc(name)}{badge}</div>"
                f"<div class='pz-role'>{_esc(role)}</div></div></div><div class='pz-st'>{_esc(status)}</div>{snip}{ft}</div>")

    def _node(self, n):
        v = self.layout[n]
        if v["kind"] in ("agent", "boss"):
            return self._card(n)
        if v["kind"] == "pill":
            return f"<div class='pz-pill' style='left:{v['x']}px;top:{v['y']}px'>{'START' if n == '__start__' else 'END'}</div>"
        icon, name = _step_style(n)
        st = self.status.get(n, "idle")
        cls = {"done": "done", "live": "live", "bad": "bad"}.get(st, "")
        return f"<div class='pz-step {cls}' style='left:{v['x']}px;top:{v['y']}px'>{icon}<span>{_esc(name)}</span></div>"

    def _box(self, n):
        v = self.layout[n]
        if v["kind"] == "pill":
            return (v["x"], v["y"], 46, 22)
        if v["kind"] == "step":
            return (v["x"], v["y"], 46, 46)
        return (v["x"], v["y"], v["w"], v["h"])

    def _edges_html(self):
        svg, labels = [], []
        for key, meta in self.edges.items():
            a, b, kind = key
            if a not in self.layout or b not in self.layout:
                continue
            A, B = self._box(a), self._box(b)
            count = self.edge_count.get(key, 0)
            back = meta.get("back", False)
            state = "live" if key in self.live_edges and self.state == "run" else ("bad" if back and count else "used" if count else "idle")
            ag = self._edge_agent(a, b)
            if meta.get("arc") is not None:            # sequential arcs above / below the row
                h = meta["arc"]
                x1, x2 = A[0], B[0]
                y1 = A[1] + (-1 if h < 0 else 1) * (A[3] / 2 + 3)
                y2 = B[1] + (-1 if h < 0 else 1) * (B[3] / 2 + 3)
                yc = min(y1, y2) + h if h < 0 else max(y1, y2) + h
                path = f"M{x1:.1f},{y1:.1f} C{x1:.1f},{yc:.1f} {x2:.1f},{yc:.1f} {x2:.1f},{y2:.1f}"
                lp = ((x1 + x2) / 2, 0.125 * (y1 + y2) + 0.75 * yc)
            else:
                path, lp = _curve(A, B, meta.get("bend", 0))
                if not meta.get("bend") and abs(A[1] - B[1]) < 2:   # straight horizontal: label above
                    lp = (lp[0], lp[1] - 13)
            extra = ("cond " if meta.get("cond") else "") + ("back" if back else "")
            svg.append(_edge_svg(path, state, extra, ag))
            text = self.edge_label.get(key)
            if count > 1:
                text = f"{text} ×{count}" if text else f"×{count}"
            if text:
                labels.append(_label(lp[0], lp[1], _esc(text),
                                     {"live": "live", "bad": "bad", "used": "used"}.get(state, ""), ag))
        return "".join(svg), "".join(labels)

    def _is_agent(self, n):
        return n in self.layout and self.layout[n]["kind"] in ("agent", "boss")

    def _edge_agent(self, a, b):
        """An arrow takes the colour of the specialist it belongs to (Robo-Giovanni only if no specialist is involved)."""
        for n in (a, b):
            if self._is_agent(n) and n not in CHEFS:
                return n
        return next((n for n in (a, b) if self._is_agent(n)), None)

    def _html(self, final=False):
        icon, title, nick = PATTERNS[self.pattern]
        secs = time.time() - self.t0 if self.state == "run" else self.result.seconds
        status_chip = {"run": "<span class='pz-chip run'>running</span>",
                       "done": "<span class='pz-chip done'>✓ done</span>",
                       "stop": "<span class='pz-chip stop'>🛑 stopped</span>"}[self.state]
        coord = f" ({self.result.coordination_calls} by Robo-Giovanni)" if self.result.coordination_calls else ""
        now = ""
        if self.state == "run":
            live = [n for n in self.layout if self._is_agent(n) and self.status.get(n) == "live"]
            live.sort(key=lambda n: n in CHEFS)           # a working specialist first, then the boss
            if live:
                n = live[0]
                icon, name, _ = _agent_style(n)
                act = self.activity.get(n, "")
                now = f"<span class='pz-chip now' data-ag='{n}'>▶ now: {icon} {_esc(name)}{(' · ' + _esc(act)) if act else ''}</span>"
        chips = (f"{now}{status_chip}<span class='pz-chip'>⏱ {secs:.0f}s</span>"
                 f"<span class='pz-chip'>🧠 {self.result.llm_calls} LLM calls{coord}</span>"
                 f"<span class='pz-chip'>🔤 {_fmt_tokens(self.result.tokens)} tokens</span>")
        head = (f"<div class='pz-head'><span style='font-size:22px'>{icon}</span><div><div class='pz-title'>{title} pattern</div>"
                f"<div class='pz-sub'>„{_esc(nick)}“ · planning {_esc(self.date)}</div></div><div class='pz-chips'>{chips}</div></div>")
        H = max(v["y"] + v.get("h", 46) / 2 for v in self.layout.values()) + 40
        svg, labels = self._edges_html()
        nodes = "".join(self._node(n) for n in self.layout)
        canvas = (f"<div class='pz-scroll'><div class='pz-canvas' style='width:{self.W}px;height:{H:.0f}px'>"
                  f"<svg width='{self.W}' height='{H:.0f}'>{_MARKERS}{_agent_markers(self.agents)}{svg}</svg>"
                  f"{nodes}{labels}</div></div>")
        legend = {
            "sequential": "each agent has its own colour · ● LIVE = working right now · faded = waiting · "
                          "dashed = conditional edge · red = loop-back · ×n = used n times",
            "supervisor": "each agent has its own colour · ● LIVE = working right now · faded = waiting · "
                          "arrows from Robo-Giovanni = tasks · arrows back = reports · ×n = used n times",
            "network": "each agent has its own colour · ● LIVE = working right now · faded = waiting · "
                       "dashed = allowed handoff (not used) · red = sent back · ×n = used n times",
        }[self.pattern]
        canvas += f"<div class='pz-note'>{legend}</div>"
        banner = ""
        if self.banner:
            banner = f"<div class='pz-banner {self.banner[0]}'>{self.banner[1]}</div>"
        log = self._log_html(final)
        res = self._results_html() if final else ""
        running = " running" if self.state == "run" else ""
        wide = f" style='max-width:{self.W + 40}px'" if self.W > _LiveView.W else ""
        return f"{_CSS}{_agent_css(self.agents)}<div class='pz{running}'{wide}>{head}{canvas}{banner}{log}{res}</div>"

    def _log_html(self, final):
        def row(e):
            t, icon, text, kind, who = e
            data = f" data-ag='{who}'" if who else ""
            return (f"<div class='pz-ev {kind}'{data}><span class='t'>+{t:4.1f}s</span><span>{icon}</span>"
                    f"<span>{text}</span></div>")
        if not final:
            rows = "".join(row(e) for e in self.events[-7:])
            return f"<div class='pz-log'><h4>What is happening</h4>{rows or '<div class=pz-ev>…</div>'}</div>"
        rows = "".join(row(e) for e in self.events)
        return (f"<div class='pz-res'><details><summary>📜 Full event log <span>{len(self.events)} events</span></summary>"
                f"<div class='pz-md pz-log'>{rows}</div></details></div>")

    def _results_html(self):
        parts = []
        if self.final:
            chef = self.boss or next((n for n in self.layout if n in CHEFS), None)
            icon, name, _ = _agent_style(chef) if chef else ("📝", "Final answer", "")
            data = f" data-ag='{chef}'" if chef else ""
            parts.append(f"<details open{data}><summary>{_who(chef) if chef else icon + ' ' + name}'s summary</summary>"
                         f"<div class='pz-md'>{_md(self.final)}</div></details>")
        for n in self.agents:
            if n in self.outputs and n != self.boss:
                icon, name, role = _agent_style(n)
                if n in CHEFS and self.final:
                    role = "his brief at the start"
                is_open = " open" if (not self.final and n == self.agents[-1]) else ""
                parts.append(f"<details{is_open} data-ag='{n}'><summary>{_who(n)} <span>{_esc(role)}</span></summary>"
                             f"<div class='pz-md'>{_md(self.outputs[n])}</div></details>")
        if not parts:
            return ""
        return f"<div class='pz-res'><h4>🍽️ Day plan · click to expand</h4>{''.join(parts)}</div>"


# ======================================================================
# Layouts
# ======================================================================
def _ring(names, cx, cy, rx, ry, start_deg=225):
    pos, n = {}, len(names)
    for i, name in enumerate(names):
        a = math.radians(start_deg + 360 * i / max(n, 1))
        pos[name] = (cx + rx * math.cos(a), cy + ry * math.sin(a))
    return pos


def _sequential_layout(graph, agent_names):
    g = graph.get_graph()
    nodes = list(g.nodes)
    edges = [(e.source, e.target, bool(e.conditional)) for e in g.edges]
    succ = {n: [] for n in nodes}
    for a, b, _ in edges:
        succ.setdefault(a, []).append(b)

    # classify back edges with a DFS from START, then rank by longest forward path
    back, state = set(), {}

    def dfs(u):
        state[u] = 1
        for v in succ.get(u, []):
            if state.get(v) == 1:
                back.add((u, v))
            elif v not in state:
                dfs(v)
        state[u] = 2
    dfs("__start__")
    nodes = [n for n in nodes if n in state]                  # hide nodes nothing leads to
    edges = [(a, b, c) for a, b, c in edges if a in state and b in state]
    rank = {"__start__": 0}
    changed = True
    while changed:
        changed = False
        for a, b, _ in edges:
            if (a, b) in back or a not in rank:
                continue
            if rank.get(b, -1) < rank[a] + 1:
                rank[b] = rank[a] + 1
                changed = True
    for n in nodes:
        rank.setdefault(n, max(rank.values()) + 1)

    def kind(n):
        if n in ("__start__", "__end__"):
            return "pill"
        return "agent" if n in agent_names or n.endswith("_agent") else "step"

    width = {"pill": 46, "step": 52, "agent": 132}
    cols = {}
    for n in nodes:
        cols.setdefault(rank[n], []).append(n)
    gap = 20
    total = sum(max(width[kind(n)] for n in col) for col in cols.values()) + gap * (len(cols) - 1)
    W = max(_LiveView.W, total + 40)
    x = (W - total) / 2
    layout, y0 = {}, 175
    for r in sorted(cols):
        col = cols[r]
        cw = max(width[kind(n)] for n in col)
        for j, n in enumerate(col):
            k = kind(n)
            y = y0 + (j - (len(col) - 1) / 2) * 160
            layout[n] = {"kind": k, "x": round(x + cw / 2), "y": round(y),
                         "w": 132 if k == "agent" else 46, "h": 140 if k == "agent" else 46}
        x += cw + gap

    edge_meta = {}
    for a, b, cond in edges:
        meta = {"cond": cond}
        if (a, b) in back:
            meta.update(back=True, arc=-80)
        elif rank[b] - rank[a] > 1:
            meta.update(arc=55)
        edge_meta[(a, b, "flow")] = meta
    return layout, edge_meta, W


def _star_layout(boss, agents):
    cx, cy = _LiveView.W / 2, 215
    layout = {boss: {"kind": "boss", "x": cx, "y": cy, "w": 170, "h": 104}}
    for n, (x, y) in _ring(agents, cx, cy, 310, 150).items():
        layout[n] = {"kind": "agent", "x": round(x), "y": round(y), "w": 156, "h": 128}
    edges = {}
    for n in agents:
        edges[(boss, n, "task")] = {"bend": 16}
        edges[(n, boss, "report")] = {"bend": 16}
    return layout, edges


def _agent_tools(compiled_agent):
    try:
        return list(compiled_agent.nodes["tools"].bound.tools_by_name)
    except Exception:
        return []


def _network_layout(swarm):
    agents = [n for n, node in swarm.nodes.items() if not n.startswith("__")]
    handoff = {n: [t[len("transfer_to_"):] for t in _agent_tools(swarm.nodes[n].bound)
                   if t.startswith("transfer_to_")] for n in agents}
    cx, cy = _LiveView.W / 2, 210
    layout = {}
    for n, (x, y) in _ring(agents, cx, cy, 290, 145).items():
        layout[n] = {"kind": "agent", "x": round(x), "y": round(y), "w": 156, "h": 132,
                     "exit": not handoff[n]}
    edges = {}
    for a, targets in handoff.items():
        for b in targets:
            if b in layout:
                two_way = a in handoff.get(b, [])
                edges[(a, b, "flow")] = {"bend": 22 if two_way else 0, "cond": True}
                layout[b]["returns"] = True
    return layout, edges


# ======================================================================
# Public runners
# ======================================================================
def _request(date, wish):
    from utils.pizzeria_tools import request
    return request(date, wish)


def _drive(view, graph, inputs, config):
    tracker = _Tracker(view)
    config = {**config, "callbacks": [tracker]}
    state = {}
    try:
        for mode, chunk in graph.stream(inputs, config, stream_mode=["updates", "values"]):
            if mode == "values":
                state = chunk
            elif view.pattern == "sequential":
                for node, update in chunk.items():
                    view.node_update(node, update)
    except GraphRecursionError:
        limit = config.get("recursion_limit", 25)
        return view.close(error=(f"🛑 Stopped by the guardrail: recursion_limit={limit} reached. The agents kept "
                                 "handing the work back and forth without finishing."), state=state)
    except KeyboardInterrupt:
        view.close(error="⏹ Interrupted by you.", state=state)
        raise
    except Exception as exc:
        view.close(error=f"💥 {type(exc).__name__}: {_esc(exc)}", state=state)
        raise
    return view.close(state=state)


def run_sequential(graph, date, wish=""):
    """Run a sequential StateGraph (input: date + wish) and animate it."""
    agents = [n for n in graph.nodes if n.endswith("_agent") or n in CHEFS]
    layout, edges, W = _sequential_layout(graph, agents)
    view = _LiveView("sequential", date, layout, edges, width=W)
    view.current = "__start__"
    return _drive(view, graph, {"date": date, "wish": wish}, {"recursion_limit": 40})


def run_supervisor(app, date, wish="", request=None):
    """Run a supervisor and animate the chef delegating to the specialists.

    `app` is an agent whose tools are the specialists (`ask_<agent>`). A langgraph-supervisor app
    (specialists as graph nodes with transfer_to_ handoffs) still works, too."""
    asks = [t[len("ask_"):] for t in _agent_tools(app) if t.startswith("ask_")]
    if asks:
        boss = getattr(app, "name", None) or "supervisor"
        layout, edges = _star_layout(boss, asks)
        view = _LiveView("supervisor", date, layout, edges, boss=boss, tools_mode=True)
    else:
        agents = [n for n in app.nodes if not n.startswith("__") and n != "supervisor"]
        layout, edges = _star_layout("supervisor", agents)
        view = _LiveView("supervisor", date, layout, edges, boss="supervisor")
    msg = request or _request(date, wish)
    return _drive(view, app, {"messages": [{"role": "user", "content": msg}]},
                  {"configurable": {"thread_id": f"sup-{uuid4().hex[:8]}"}, "recursion_limit": 60})


def _swarm_entry(swarm, layout):
    """The default active agent of a swarm (where every new conversation starts)."""
    try:
        branch = next(iter(swarm.builder.branches["__start__"].values()))
        func = getattr(branch.path, "func", None) or branch.path
        for cell in (getattr(func, "__closure__", None) or []):
            if isinstance(cell.cell_contents, str) and cell.cell_contents in layout:
                return cell.cell_contents
    except Exception:
        pass
    return next((n for n in layout if n.startswith("forecast")), next(iter(layout)))


def run_network(swarm, date, wish="", request=None, recursion_limit=16):
    """Run a swarm and animate every handoff, including loop-backs."""
    layout, edges = _network_layout(swarm)
    entry = _swarm_entry(swarm, layout)
    view = _LiveView("network", date, layout, edges, entry=entry)
    msg = request or _request(date, wish)
    return _drive(view, swarm, {"messages": [{"role": "user", "content": msg}]},
                  {"configurable": {"thread_id": f"net-{uuid4().hex[:8]}"}, "recursion_limit": recursion_limit})


# ======================================================================
# Comparison & overview
# ======================================================================
def compare_runs(last_per_pattern=True):
    """Compare the runs of this session: time, LLM calls, tokens, handoffs."""
    runs = RUNS
    if last_per_pattern:
        latest = {}
        for r in RUNS:
            latest[r.pattern] = r
        runs = [latest[p] for p in PATTERNS if p in latest]
    if not runs:
        display(HTML(f"{_CSS}<div class='pz'>No runs yet – run one of the patterns first.</div>"))
        return
    ok = [r for r in runs if not r.error]

    def best(attr, r):
        vals = [getattr(x, attr) for x in ok]
        return " pz-best" if ok and not r.error and getattr(r, attr) == min(vals) and len(ok) > 1 else ""

    rows = ""
    for r in runs:
        icon, title, _ = PATTERNS[r.pattern]
        outcome = "✓ completed" if not r.error else "🛑 stopped"
        rows += (f"<tr><td><b>{icon} {title}</b><br><span class='pz-sub'>{_esc(r.date)}</span></td>"
                 f"<td class='{best('seconds', r)}'>{r.seconds:.0f}s</td>"
                 f"<td class='{best('llm_calls', r)}'>{r.llm_calls}</td>"
                 f"<td>{r.coordination_calls if r.pattern == 'supervisor' else '–'}</td>"
                 f"<td class='{best('tokens', r)}'>{_fmt_tokens(r.tokens)}</td>"
                 f"<td>{r.agent_turns}</td><td>{'fixed' if r.pattern == 'sequential' else r.handoffs}</td>"
                 f"<td>{r.loop_backs}</td><td>{outcome}</td></tr>")
    table = (f"<div class='pz-scroll'><table class='pz-cmp'><tr><th>Pattern</th><th>Time</th><th>LLM calls</th><th>…by the boss</th>"
             f"<th>Tokens</th><th>Agent turns</th><th>Handoffs</th><th>Agent re-runs</th>"
             f"<th>Outcome</th></tr>{rows}</table></div>")
    note = ("<div class='pz-note'>Green = cheapest / fastest. LLM output varies from run to run – "
            "run each pattern a few times before drawing conclusions.</div>")
    display(HTML(f"{_CSS}<div class='pz'><div class='pz-head'><span style='font-size:22px'>📊</span>"
                 f"<div class='pz-title'>Pattern comparison</div></div>{table}{note}</div>"))


def show_patterns():
    """Static overview of the three orchestration patterns."""
    def mini(kind):
        names = {"🔮": "forecast_agent", "🍲": "soup_agent", "🛒": "purchasing_agent", "👥": "staffing_agent",
                 "👨‍🍳": "robo_giovanni"}
        dot = lambda x, y, t: (f"<g data-ag='{names.get(t, '')}'><circle cx='{x}' cy='{y}' r='15' "
                               f"style='fill:color-mix(in srgb,var(--ag,var(--line)) 18%,var(--panel));"
                               f"stroke:var(--ag,var(--line));stroke-width:2.5'/>"
                               f"<text x='{x}' y='{y + 6}' text-anchor='middle'>{t}</text></g>")
        line = lambda x1, y1, x2, y2, dash="": (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
                                               f"style='stroke:var(--muted);stroke-width:1.8' {dash} "
                                               f"marker-end='url(#pz-a-idle)'/>")
        icons = ["🔮", "🍲", "🛒", "👥"]
        if kind == "sequential":
            xs = [22, 72, 122, 172, 222]
            body = "".join(line(xs[i] + 16, 50, xs[i + 1] - 18, 50) for i in range(4))
            body += "".join(dot(x, 50, i) for x, i in zip(xs, ["👨‍🍳", *icons]))
        elif kind == "supervisor":
            pts = [(40, 22), (190, 22), (190, 82), (40, 82)]
            body = "".join(line(115, 52, x + (16 if x < 115 else -16), y + (8 if y < 52 else -8)) for x, y in pts)
            body += dot(115, 52, "👨‍🍳") + "".join(dot(x, y, i) for (x, y), i in zip(pts, icons))
        else:
            pts = [(75, 22), (215, 22), (215, 82), (75, 82)]
            body = (line(91, 22, 197, 22) + line(209, 38, 209, 64) + line(223, 64, 223, 38, "stroke-dasharray='4 4'")
                    + line(198, 82, 93, 82) + line(34, 41, 60, 29) + line(60, 75, 34, 63))
            body += dot(22, 52, "👨‍🍳") + "".join(dot(x, y, i) for (x, y), i in zip(pts, icons))
        return f"<svg width='244' height='104' viewBox='0 0 244 104'>{_MARKERS}{body}</svg>"

    cards = [
        ("sequential", "A fixed assembly line: Robo-Giovanni briefs, then the order is hard-wired in the graph – "
                       "predictable and cheap, but rigid."),
        ("supervisor", "A central chef decides who works next and collects the results – flexible, but every step costs an extra LLM call."),
        ("network", "No boss deciding the order: the agents hand off to each other, Robo-Giovanni only starts and ends. "
                    "Reacts to surprises, but is hard to predict and can loop."),
    ]
    html_cards = "".join(
        f"<div><b>{PATTERNS[k][0]} {PATTERNS[k][1]}</b> <span class='pz-sub'>„{PATTERNS[k][2]}“</span>{mini(k)}<p>{t}</p></div>"
        for k, t in cards)
    brigade = ["robo_giovanni", "forecast_agent", "soup_agent", "purchasing_agent", "staffing_agent"]
    chips = "".join(f"<span class='pz-chip' data-ag='{n}'>{AGENT_STYLE[n][0]} {AGENT_STYLE[n][1]} · "
                    f"<i>{AGENT_STYLE[n][2]}</i></span>" for n in brigade)
    display(HTML(f"{_CSS}{_agent_css(brigade)}<div class='pz'><div class='pz-pat'>{html_cards}</div>"
                 f"<div class='pz-brigade'><b>The brigade</b>{chips}</div></div>"))
