"""
brigade.py
----------
Helpers for Step 5 (multi-agent patterns) that keep the notebook short and easy to read.
Nothing in here is needed to *understand* the three patterns – it is the plumbing around them.

    Sequential:  DayPlan, giovannis_request, process_request, check_inventory, save_day_plan
    Supervisor:  nothing here, the notebook turns the specialists into tools itself
    Network:     network_agent
"""

from typing import TypedDict

from langchain.agents import create_agent

from utils import brigade_tools, restaurant_param
from utils.helper_functions import weekday
from utils.restaurant_db import save_daily_plan

# ======================================================================
# Pattern 1: Sequential
# ======================================================================
class DayPlan(TypedDict, total=False):
    """The shared state of the assembly line: every step reads from it and adds its result."""
    date: str
    wish: str
    brief: str        # Robo-Giovanni
    expected_guests: str   # forecast
    soup: str         # soup
    shopping_list: str     # purchasing
    shift_plan: str        # staffing
    avoid: list       # stock check: ingredients the cook must not use
    times_sent_back: int   # stock check: how often the cook was sent back
    send_cook_back: bool   # stock check: send the cook back?
    plain_soup: bool       # stock check: gave up, plain soup


def giovannis_request(date: str, wish: str = "") -> str:
    """Giovanni's request for one day: his four morning questions, plus his wish if he has one.
    The same text starts the planning in all three patterns."""
    return (f"Plan {date} (a {weekday(date)}): how busy will it be, what is the soup of the day, what do we "
            f"need to buy and who has to work? {wish or ''}".strip())


def process_request(agent, request, day_plan=None):
    """Let one agent process a request and return its final answer as text.

    If the day_plan already has Robo-Giovanni's brief, it is put in front of the request, so every
    specialist knows the big picture, not just its own little job."""
    if day_plan and day_plan.get("brief"):
        request = f"Brief from Robo-Giovanni:\n{day_plan['brief']}\n\nYour task: {request}"
    return agent.invoke({"messages": [{"role": "user", "content": request}]})["messages"][-1].content


def check_inventory(day_plan):
    """Guardrail in plain Python (no LLM): is the soup's special ingredient out of stock?

    Out of stock -> send the cook back (at most MAX_STOCK_REJECTIONS times), then fall back to the plain soup."""
    blocked = brigade_tools.blocked_ingredients(day_plan["soup"])
    if not blocked:
        return {"send_cook_back": False}
    if day_plan.get("times_sent_back", 0) < restaurant_param.max_stock_rejections:
        return {"send_cook_back": True, "times_sent_back": day_plan.get("times_sent_back", 0) + 1, "avoid": day_plan.get("avoid", []) + blocked}
    return {"send_cook_back": False, "plain_soup": True,
            "soup": f"{brigade_tools.fallback_soup()} – {', '.join(blocked)} out of stock"}


def save_day_plan(day_plan):
    """Save the finished day plan to utils/restaurant_store.json (recall it with get_daily_plan)."""
    save_daily_plan(day_plan["date"], day_plan.get("expected_guests"), day_plan.get("soup"), day_plan.get("shopping_list"), day_plan.get("shift_plan"))
    return {}


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
    "- staffing_agent: shift day_plan. Needs the date and the guest number.\n"
)
MUST_HAND_OFF = (
    "When your part is done, YOU decide who should continue: look at what is still missing and call the "
    "transfer tool of that colleague – as the ONLY tool call of that step. Put your result into its `note` – "
    "the next agent needs it. "
    "If everything (guests, soup, shopping list, shift day_plan) is done, hand back to robo_giovanni. "
    "Never end with a plain answer."
)


def network_agent(model, definition, can_hand_off_to=None):
    """An agent for the network, built from the same definition as the specialist (a dict with
    `name`, `system_prompt` and `tools`): its own tools plus a handoff tool to colleagues, and
    the team list plus the handoff rule added to its prompt.

    can_hand_off_to: who it may hand the work to – by default EVERY other member of the brigade,
    so the order is not fixed anywhere. Robo-Giovanni is the only one allowed to finish."""
    name = definition["name"]
    targets = can_hand_off_to if can_hand_off_to is not None else [n for n in BRIGADE if n != name]
    rules = "" if name == "robo_giovanni" else " " + MUST_HAND_OFF
    return create_agent(model, name=name, tools=[*definition["tools"], *brigade_tools.handoffs(*targets)],
                        system_prompt=definition["system_prompt"] + TEAM + rules)
