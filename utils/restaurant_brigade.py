"""
restaurant_brigade.py
----------
Helpers for Step 5 (multi-agent patterns) that keep the notebook short and easy to read.
Nothing in here is needed to *understand* the three patterns – it is the plumbing around them.

    Sequential:  DayPlan, ask, stock_check, after_stock_check, save_plan
    Supervisor:  as_tool
    Network:     network_agent
"""

from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain.tools import tool

from utils import restaurant_brigade_tools, restaurant_demo as demo
from utils.restaurant_db import save_daily_plan

# ======================================================================
# Toolboxes per specialist (least privilege)
# ======================================================================
FORECAST_TOOLS = [
    restaurant_brigade_tools.get_reservations,
    restaurant_brigade_tools.get_walkin_baseline,
]
SOUP_TOOLS = [
    restaurant_brigade_tools.get_seasonal_ingredients,
    restaurant_brigade_tools.get_soup_base_tool,
]
PURCHASING_TOOLS = [
    restaurant_brigade_tools.check_soup_ingredient,
    restaurant_brigade_tools.check_pantry,
    restaurant_brigade_tools.calculate_order,
]
STAFFING_TOOLS = [restaurant_brigade_tools.get_required_staff, restaurant_brigade_tools.get_staff_status]
SOMMELIER_TOOLS = [restaurant_brigade_tools.get_wine_list]


# ======================================================================
# Pattern 1: Sequential
# ======================================================================
class DayPlan(TypedDict, total=False):
    """The shared state of the assembly line: every step reads from it and adds its result."""
    date: str
    wish: str
    brief: str        # Robo-Giovanni
    guests: str       # forecast
    soup: str         # soup
    order: str        # purchasing
    staffing: str     # staffing
    avoid: list       # stock check: ingredients the cook must not use
    tries: int        # stock check: how often the cook was sent back
    retry: bool       # stock check: send the cook back?
    fallback: bool    # stock check: gave up, plain soup


def ask(agent, task, plan=None):
    """Give one agent a task and return its final answer.

    If the plan already has Robo-Giovanni's brief, it is put in front of the task – so every
    specialist knows the big picture, not just its own little job."""
    if plan and plan.get("brief"):
        task = f"Brief from Robo-Giovanni:\n{plan['brief']}\n\nYour task: {task}"
    return agent.invoke({"messages": [{"role": "user", "content": task}]})["messages"][-1].content


def stock_check(plan):
    """Guardrail in plain Python (no LLM): is the soup's special ingredient out of stock?

    Out of stock -> send the cook back (at most MAX_STOCK_REJECTIONS times), then fall back to the plain soup."""
    blocked = restaurant_brigade_tools.blocked_ingredients(plan["soup"])
    if not blocked:
        return {"retry": False}
    if plan.get("tries", 0) < demo.MAX_STOCK_REJECTIONS:
        return {"retry": True, "tries": plan.get("tries", 0) + 1, "avoid": plan.get("avoid", []) + blocked}
    return {"retry": False, "fallback": True,
            "soup": f"{restaurant_brigade_tools.fallback_soup()} – {', '.join(blocked)} out of stock"}


def after_stock_check(plan) -> Literal["soup_agent", "purchasing_agent"]:
    """Where to go after the stock check: back to the cook, or on to purchasing.
    (The return type tells LangGraph – and the diagram – the two possible next steps.)"""
    return "soup_agent" if plan.get("retry") else "purchasing_agent"


def save_plan(plan):
    """Save the finished day plan to utils/restaurant_store.json (recall it with get_daily_plan)."""
    save_daily_plan(plan["date"], plan.get("guests"), plan.get("soup"), plan.get("order"), plan.get("staffing"))
    return {}


# ======================================================================
# Pattern 2: Supervisor
# ======================================================================
def as_tool(agent):
    """Wrap a specialist as a tool `ask_<name>(task)`: Robo-Giovanni sends a task, the specialist
    works on it and its final answer comes back as the tool result."""
    @tool(f"ask_{agent.name}", description=f"Give {agent.name} a task. It only sees this task text, "
                                            "so include everything it needs (date, guests, soup, wish …).")
    def ask_specialist(task: str) -> str:
        return ask(agent, task)
    return ask_specialist


# ======================================================================
# Pattern 3: Network
# ======================================================================
BRIGADE = ["robo_giovanni", "forecast_agent", "soup_agent", "purchasing_agent", "staffing_agent"]

TEAM = (
    "\n\nYour team (you can hand the work to ANY of them):\n"
    "- robo_giovanni: head chef. Starts the planning and writes the final summary once everything is done.\n"
    "- forecast_agent: guest numbers. Needs only the date.\n"
    "- soup_agent: special ingredient of the soup of the day. Needs the month and Giovanni's wish.\n"
    "- purchasing_agent: shopping list for pizzas and soup. Needs the guest number AND the soup.\n"
    "- staffing_agent: shift plan. Needs the date and the guest number.\n"
)
MUST_HAND_OFF = (
    "When your part is done, YOU decide who should continue: look at what is still missing and call the "
    "transfer tool of that colleague – as the ONLY tool call of that step. Put your result into its `note` – "
    "the next agent needs it. "
    "If everything (guests, soup, shopping list, shift plan) is done, hand back to robo_giovanni. "
    "Never end with a plain answer."
)


def network_agent(model, name, tools, prompt, can_hand_off_to=None):
    """An agent for the network: its own tools plus a handoff tool to colleagues.

    can_hand_off_to: who it may hand the work to – by default EVERY other member of the brigade,
    so the order is not fixed anywhere. Robo-Giovanni is the only one allowed to finish."""
    targets = can_hand_off_to if can_hand_off_to is not None else [n for n in BRIGADE if n != name]
    rules = "" if name == "robo_giovanni" else " " + MUST_HAND_OFF
    return create_agent(model, name=name, tools=[*tools, *restaurant_brigade_tools.handoffs(*targets)],
                        system_prompt=prompt + TEAM + rules)
