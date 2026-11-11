"""
agent_viz.py
------------
Live visualisations for Step 5 (multi-agent patterns).

    run_sequential(graph, date, wish)   -> animates a LangGraph StateGraph
    run_supervisor(app, date, wish)     -> animates a langgraph-supervisor app
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
    "menu_agent":       ("🍕", "Menu",       "il cuoco"),
    "purchasing_agent": ("🛒", "Purchasing", "il dispensiere"),
    "staffing_agent":   ("👥", "Staffing",   "il caposala"),
    "sommelier_agent":  ("🍷", "Sommelier",  "il sommelier"),
    "supervisor":       ("👨‍🍳", "Robo-Giovanni",   "il capo della brigata"),
}
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
.pz{--bg:#fbf7f0;--panel:#ffffff;--ink:#1c1917;--muted:#78716c;--line:#d6d3d1;--chip:#f1ebe2;
    --accent:#c2410c;--accent-soft:#fff1e6;--ok:#15803d;--ok-soft:#ecfdf3;--live:#d97706;--live-soft:#fff7e0;
    --bad:#b91c1c;--bad-soft:#fef2f2;--boss:#292524;--boss-ink:#fafaf9;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:14px;
    padding:14px 16px 12px;margin:4px 0;max-width:920px;box-sizing:border-box;line-height:1.35;}
body.vscode-dark .pz,body.vscode-high-contrast .pz,body[data-jp-theme-light="false"] .pz{
    --bg:#1c1917;--panel:#292524;--ink:#f5f5f4;--muted:#a8a29e;--line:#44403c;--chip:#33302d;
    --accent:#fb923c;--accent-soft:#3b2416;--ok:#4ade80;--ok-soft:#14281c;--live:#fbbf24;--live-soft:#3a2e12;
    --bad:#f87171;--bad-soft:#3a1717;--boss:#f5f5f4;--boss-ink:#1c1917;}
.pz *{box-sizing:border-box}
.pz-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.pz-title{font-size:16px;font-weight:700}
.pz-sub{font-size:12px;color:var(--muted);font-style:italic}
.pz-chips{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.pz-chip{background:var(--chip);border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;white-space:nowrap}
.pz-chip.run{background:var(--live-soft);color:var(--live)}
.pz-chip.done{background:var(--ok-soft);color:var(--ok)}
.pz-chip.stop{background:var(--bad-soft);color:var(--bad)}
.pz-chip.run::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--live);
    margin-right:6px;vertical-align:1px;animation:pzblink 1s infinite}
.pz-scroll{overflow-x:auto;border-radius:10px;background:var(--panel);border:1px solid var(--line)}
.pz-canvas{position:relative;margin:0 auto}
.pz-canvas svg{position:absolute;inset:0;overflow:visible}
.pz-e{fill:none;stroke:var(--line);stroke-width:1.6}
.pz-e.cond{stroke-dasharray:4 5}
.pz-e.used{stroke:var(--ok);stroke-width:2.4;stroke-dasharray:none}
.pz-e.back.used{stroke:var(--bad)}
.pz-e.live{stroke:var(--live);stroke-width:3;stroke-dasharray:7 6;animation:pzflow .7s linear infinite}
.pz-mk path{fill:var(--line)} .pz-mk.used path{fill:var(--ok)} .pz-mk.live path{fill:var(--live)} .pz-mk.bad path{fill:var(--bad)}
.pz-lbl{position:absolute;transform:translate(-50%,-50%);font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;
    background:var(--panel);border:1px solid var(--line);color:var(--muted);white-space:nowrap;z-index:3}
.pz-lbl.used{color:var(--ok);border-color:var(--ok)} .pz-lbl.bad{color:var(--bad);border-color:var(--bad)}
.pz-lbl.live{color:var(--live);border-color:var(--live)}
.pz-card{position:absolute;transform:translate(-50%,-50%);background:var(--panel);border:1px solid var(--line);
    border-top:4px solid var(--line);border-radius:12px;padding:8px 10px;z-index:2;overflow:hidden;
    box-shadow:0 1px 3px rgba(0,0,0,.08);display:flex;flex-direction:column;gap:2px}
.pz-card.live{border-top-color:var(--live);background:var(--live-soft);animation:pzpulse 1.2s infinite}
.pz-card.done{border-top-color:var(--ok)}
.pz-card.boss{background:var(--boss);color:var(--boss-ink);border-color:var(--boss);border-top-color:var(--accent)}
.pz-card.boss .pz-role,.pz-card.boss .pz-ft{color:var(--boss-ink);opacity:.7}
.pz-card.boss.live{border-top-color:var(--live);animation:pzpulse 1.2s infinite}
.pz-r{display:flex;align-items:center;gap:7px}
.pz-ico{font-size:20px;line-height:1}
.pz-nm{font-weight:700;font-size:13px}
.pz-role{font-size:10.5px;font-style:italic;color:var(--muted)}
.pz-st{font-size:11px;font-weight:700;color:var(--muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pz-card.live .pz-st{color:var(--live)} .pz-card.done .pz-st{color:var(--ok)}
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
.pz-ev.warn{color:var(--bad);font-weight:600} .pz-ev.hand{color:var(--ok)}
.pz-res{margin-top:12px}
.pz-res details{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:6px}
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
@keyframes pzpulse{0%,100%{box-shadow:0 0 0 0 rgba(217,119,6,.45)}50%{box-shadow:0 0 0 7px rgba(217,119,6,0)}}
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


def _edge_svg(path, state, extra=""):
    marker = {"live": "live", "used": "used", "bad": "bad"}.get(state, "idle")
    cls = {"live": "live", "used": "used", "bad": "used"}.get(state, "")
    return f"<path class='pz-e {cls} {extra}' d='{path}' marker-end='url(#pz-a-{marker})'/>"


def _label(x, y, text, state=""):
    return f"<div class='pz-lbl {state}' style='left:{x:.0f}px;top:{y:.0f}px'>{text}</div>"


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
        ns = (metadata or {}).get("langgraph_checkpoint_ns", "")
        first = ns.split("|")[0].split(":")[0] if ns else ""
        return first or (metadata or {}).get("lc_agent_name") or "?"

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

    def __init__(self, pattern, date, layout, edges, boss=None, entry=None, width=None):
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
        self.live_edge, self.current = None, None
        self.outputs, self.final = {}, ""
        self.events = []
        self.t0 = time.time()
        self.state, self.banner = "run", None
        self.result = RunResult(pattern, date)
        self._last = 0.0
        self.handle = display(HTML(self.render()), display_id=True)

    # ---------- events ----------
    def _event(self, icon, text, kind=""):
        self.events.append((time.time() - self.t0, icon, text, kind))

    def _use_edge(self, a, b, kind="flow", label=None):
        key = (a, b, kind)
        if key not in self.edges:
            self.edges[key] = {"cond": True}
        self.edge_count[key] = self.edge_count.get(key, 0) + 1
        self.live_edge = key
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
            self._event(icon, f"<b>{name}</b> starts working{again}")

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
        if self.pattern == "supervisor":
            if who == self.boss and self.current not in (None, self.boss):
                came_from = self.current
                self.finish(came_from)
                self._use_edge(came_from, self.boss, "report")
                icon, name, _ = _agent_style(came_from)
                self._event("📨", f"{icon} <b>{name}</b> reports back to Robo-Giovanni", "hand")
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
        only_handoffs = calls and all(c.get("name", "").startswith("transfer_") for c in calls)
        if text.strip() and (not calls or only_handoffs):
            if who == self.boss:
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
        if name.startswith("transfer_to_"):
            target = name[len("transfer_to_"):]
            icon_a, name_a, _ = _agent_style(who)
            icon_b, name_b, _ = _agent_style(target)
            if self.pattern == "supervisor" and who == self.boss:
                self._use_edge(self.boss, target, "task")
                self.result.handoffs += 1
                self.status[self.boss] = "idle"
                self._event("📋", f"Robo-Giovanni delegates to {icon_b} <b>{name_b}</b>", "hand")
                self.current = None
                self.enter(target)
                self.activity[target] = "📋 got a task"
            else:
                self.result.handoffs += 1
                back = (self.visits.get(target, 0) > 0
                        and AGENT_ORDER.get(target, 0) < AGENT_ORDER.get(who, 0))
                if back and (who, target, "flow") in self.edges:
                    self.edges[(who, target, "flow")]["back"] = True
                self._event("🤝", f"{icon_a} <b>{name_a}</b> hands off to {icon_b} <b>{name_b}</b>"
                            + (" — <b>going back!</b>" if back else ""), "warn" if back else "hand")
                self.finish(who)
                self.enter(target, via=(who, target, "flow"))
                self.activity[target] = "🤝 took over"
            self.render(force=True)
            return
        args_txt = ""
        if isinstance(args, dict):
            args_txt = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "config")
        self.activity[who] = f"🔧 {name}"
        icon, nm, _ = _agent_style(who)
        self._event("🔧", f"{icon} {nm} calls <code>{_esc(name)}({_esc(args_txt)[:60]})</code>")
        self.render()

    def tool_end(self, who, name, output):
        out = str(output or "")
        if name == "check_special_ingredient" and ("OUT OF STOCK" in out or "FALLBACK REQUIRED" in out):
            m = re.search(r"'([^']+)'", out)
            ing = m.group(1) if m else "an ingredient"
            again = "FALLBACK REQUIRED" in out
            self._event("⚠️", f"<b>{_esc(ing)}</b> is out of stock"
                        + (" again — falling back to a classic" if again else ""), "warn")

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
                    self._event("🔍", f"Stock check: <b>{_esc(', '.join(blocked))}</b> is out of stock → back to the menu", "warn")
                    self.edge_label[(node, "menu_agent", "flow")] = f"avoid {', '.join(blocked)}"
                    self.current = node
                elif update.get("fallback"):
                    self.status[node] = "bad"
                    self._event("🔍", f"Stock check: still not possible → fallback to <b>{_esc(update.get('dish'))}</b>", "warn")
                else:
                    self.status[node] = "done"
                    self._event("🔍", "Stock check passed: every key ingredient is available", "hand")
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
                                     if self.layout[n]["kind"] == "agent")
        self.live_edge = None
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
                self.live_edge = None
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
        cls = ("boss " if boss else "") + ("live" if st == "live" else "done" if st in ("done", "bad") else "")
        if st == "live":
            status = self.activity.get(n, "🧠 thinking…")
        elif st in ("done", "bad"):
            status = "✓ done"
        else:
            status = "waiting"
        if boss:
            status = {"live": self.activity.get(n, "🧠 deciding…"), "done": "✓ coordinated"}.get(st, "delegates & reviews")
        badge = f"<span class='pz-badge'>×{self.visits[n]}</span>" if self.visits.get(n, 0) > 1 else ""
        tag = ""
        if self.entry == n:
            tag = "<div class='pz-tag'>▶ starts here</div>"
        elif v.get("exit"):
            tag = "<div class='pz-tag'>🏁 only exit</div>"
        snip = f"<div class='pz-snip'>{_esc(_snippet(self.snip.get(n), 90))}</div>" if self.snip.get(n) and st != "live" else ""
        ft = ""
        if self.calls.get(n):
            ft = f"<div class='pz-ft'><span>🧠 {self.calls[n]}</span><span>🔤 {_fmt_tokens(self.tokens[n])}</span></div>"
        return (f"<div class='pz-card {cls}' style='left:{v['x']}px;top:{v['y']}px;width:{v['w']}px;height:{v['h']}px'>"
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
            state = "live" if key == self.live_edge and self.state == "run" else ("bad" if back and count else "used" if count else "idle")
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
            svg.append(_edge_svg(path, state, extra))
            text = self.edge_label.get(key)
            if count > 1:
                text = f"{text} ×{count}" if text else f"×{count}"
            if text:
                labels.append(_label(lp[0], lp[1], _esc(text),
                                     {"live": "live", "bad": "bad", "used": "used"}.get(state, "")))
        return "".join(svg), "".join(labels)

    def _html(self, final=False):
        icon, title, nick = PATTERNS[self.pattern]
        secs = time.time() - self.t0 if self.state == "run" else self.result.seconds
        status_chip = {"run": "<span class='pz-chip run'>running</span>",
                       "done": "<span class='pz-chip done'>✓ done</span>",
                       "stop": "<span class='pz-chip stop'>🛑 stopped</span>"}[self.state]
        coord = f" ({self.result.coordination_calls} by Robo-Giovanni)" if self.result.coordination_calls else ""
        chips = (f"{status_chip}<span class='pz-chip'>⏱ {secs:.0f}s</span>"
                 f"<span class='pz-chip'>🧠 {self.result.llm_calls} LLM calls{coord}</span>"
                 f"<span class='pz-chip'>🔤 {_fmt_tokens(self.result.tokens)} tokens</span>")
        head = (f"<div class='pz-head'><span style='font-size:22px'>{icon}</span><div><div class='pz-title'>{title} pattern</div>"
                f"<div class='pz-sub'>„{_esc(nick)}“ · planning {_esc(self.date)}</div></div><div class='pz-chips'>{chips}</div></div>")
        H = max(v["y"] + v.get("h", 46) / 2 for v in self.layout.values()) + 40
        svg, labels = self._edges_html()
        nodes = "".join(self._node(n) for n in self.layout)
        canvas = (f"<div class='pz-scroll'><div class='pz-canvas' style='width:{self.W}px;height:{H:.0f}px'>"
                  f"<svg width='{self.W}' height='{H:.0f}'>{_MARKERS}{svg}</svg>{nodes}{labels}</div></div>")
        legend = {
            "sequential": "solid = fixed edge · dashed = conditional edge · red = loop-back · ×n = used n times",
            "supervisor": "arrows from Robo-Giovanni = tasks · arrows to Robo-Giovanni = reports · ×n = used n times",
            "network": "dashed = allowed handoff (not used) · green = used · red = sent back · ×n = used n times",
        }[self.pattern]
        canvas += f"<div class='pz-note'>{legend}</div>"
        banner = ""
        if self.banner:
            banner = f"<div class='pz-banner {self.banner[0]}'>{self.banner[1]}</div>"
        log = self._log_html(final)
        res = self._results_html() if final else ""
        return f"{_CSS}<div class='pz'>{head}{canvas}{banner}{log}{res}</div>"

    def _log_html(self, final):
        def row(e):
            t, icon, text, kind = e
            return f"<div class='pz-ev {kind}'><span class='t'>+{t:4.1f}s</span><span>{icon}</span><span>{text}</span></div>"
        if not final:
            rows = "".join(row(e) for e in self.events[-7:])
            return f"<div class='pz-log'><h4>What is happening</h4>{rows or '<div class=pz-ev>…</div>'}</div>"
        rows = "".join(row(e) for e in self.events)
        return (f"<div class='pz-res'><details><summary>📜 Full event log <span>{len(self.events)} events</span></summary>"
                f"<div class='pz-md pz-log'>{rows}</div></details></div>")

    def _results_html(self):
        parts = []
        if self.final:
            icon, name, _ = _agent_style(self.boss) if self.boss else ("📝", "Final answer", "")
            parts.append(f"<details open><summary>{icon} {name}'s summary</summary><div class='pz-md'>{_md(self.final)}</div></details>")
        for n in self.agents:
            if n in self.outputs and n != self.boss:
                icon, name, role = _agent_style(n)
                is_open = " open" if (not self.final and n == self.agents[-1]) else ""
                parts.append(f"<details{is_open}><summary>{icon} {name} <span>{_esc(role)}</span></summary>"
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

    width = {"pill": 46, "step": 52, "agent": 124}
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
                         "w": 124 if k == "agent" else 46, "h": 128 if k == "agent" else 46}
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
    return layout, edges


# ======================================================================
# Public runners
# ======================================================================
def _request(date, wish):
    from utils.pizzeria_tools import weekday
    return (f"Plan {date} (a {weekday(date)}): how busy will it be, what is the daily special, what do we "
            f"need to buy and who has to work? {wish or ''}".strip())


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
    agents = [n for n in graph.nodes if n.endswith("_agent")]
    layout, edges, W = _sequential_layout(graph, agents)
    view = _LiveView("sequential", date, layout, edges, width=W)
    view.current = "__start__"
    return _drive(view, graph, {"date": date, "wish": wish}, {"recursion_limit": 40})


def run_supervisor(app, date, wish="", request=None):
    """Run a supervisor app and animate the chef delegating to the specialists."""
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


def run_network(swarm, date, wish="", request=None, recursion_limit=12):
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
    table = (f"<table class='pz-cmp'><tr><th>Pattern</th><th>Time</th><th>LLM calls</th><th>…by the boss</th>"
             f"<th>Tokens</th><th>Agent turns</th><th>Handoffs</th><th>Agent re-runs</th><th>Outcome</th></tr>{rows}</table>")
    note = ("<div class='pz-note'>Green = cheapest / fastest. LLM output varies from run to run – "
            "run each pattern a few times before drawing conclusions.</div>")
    display(HTML(f"{_CSS}<div class='pz'><div class='pz-head'><span style='font-size:22px'>📊</span>"
                 f"<div class='pz-title'>Pattern comparison</div></div>{table}{note}</div>"))


def show_patterns():
    """Static overview of the three orchestration patterns."""
    def mini(kind):
        dot = lambda x, y, t: (f"<circle cx='{x}' cy='{y}' r='15' style='fill:var(--chip);stroke:var(--line)'/>"
                               f"<text x='{x}' y='{y + 6}' text-anchor='middle'>{t}</text>")
        line = lambda x1, y1, x2, y2, dash="": (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
                                               f"style='stroke:var(--ok);stroke-width:2' {dash} "
                                               f"marker-end='url(#pz-a-used)'/>")
        icons = ["🔮", "🍕", "🛒", "👥"]
        if kind == "sequential":
            xs = [25, 85, 145, 205]
            body = "".join(line(xs[i] + 16, 50, xs[i + 1] - 18, 50) for i in range(3))
            body += "".join(dot(x, 50, i) for x, i in zip(xs, icons))
        elif kind == "supervisor":
            pts = [(40, 22), (190, 22), (190, 82), (40, 82)]
            body = "".join(line(115, 52, x + (16 if x < 115 else -16), y + (8 if y < 52 else -8)) for x, y in pts)
            body += dot(115, 52, "👨‍🍳") + "".join(dot(x, y, i) for (x, y), i in zip(pts, icons))
        else:
            pts = [(45, 22), (185, 22), (185, 82), (45, 82)]
            body = (line(61, 22, 167, 22) + line(179, 38, 179, 64) + line(193, 64, 193, 38, "stroke-dasharray='4 4'")
                    + line(168, 82, 63, 82))
            body += "".join(dot(x, y, i) for (x, y), i in zip(pts, icons))
        return f"<svg width='230' height='104' viewBox='0 0 230 104'>{_MARKERS}{body}</svg>"

    cards = [
        ("sequential", "A fixed assembly line. The order is hard-wired in the graph – predictable and cheap, but rigid."),
        ("supervisor", "A central chef decides who works next and collects the results – flexible, but every step costs an extra LLM call."),
        ("network", "No boss: the agents hand off to each other. Reacts to surprises, but is hard to predict and can loop."),
    ]
    html_cards = "".join(
        f"<div><b>{PATTERNS[k][0]} {PATTERNS[k][1]}</b> <span class='pz-sub'>„{PATTERNS[k][2]}“</span>{mini(k)}<p>{t}</p></div>"
        for k, t in cards)
    display(HTML(f"{_CSS}<div class='pz'><div class='pz-pat'>{html_cards}</div></div>"))
