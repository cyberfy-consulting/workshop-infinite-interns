"""
brigade.py
----------
Helpers for Step 5 (multi-agent patterns) that keep the notebook short and easy to read.
Nothing in here is needed to *understand* the three patterns – it is the plumbing around them.

    setup()              -> quiet logs, fresh pantry and staff roster
    prepare_demo_day()   -> fresh booking database with a few bookings for the demo evening
    reset_knobs()        -> all kitchen knobs back to their default values

    Sequential:  DayPlan, ask, stock_check, after_stock_check, save_plan
    Supervisor:  as_tool
    Network:     network_agent
"""

import logging
from datetime import datetime
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain.tools import tool

from utils import pizzeria_tools as kitchen
from utils.pantry_db import save_daily_plan


# ======================================================================
# Setup
# ======================================================================
def setup():
    """Quiet logs (no line for every HTTP request) and a fresh pantry and staff roster."""
    for name in ("httpx", "httpcore", "openai", "azure"):
        logging.getLogger(name).setLevel(logging.WARNING)
    kitchen.reset_kitchen()


DEMO_DAY = datetime(2026, 11, 19)     # Thursday, 19 November 2026 – autumn
DEMO_BOOKINGS = [                     # (name, hour, party size, table) – all inside, it's November
    ("Alice", 19, 2, "TI1"), ("Bob", 19, 4, "TI10"), ("Carol", 20, 4, "TI11"),
    ("Dave", 20, 4, "TI12"), ("Frank", 21, 2, "TI2"),
]


def prepare_demo_day():
    """Rebuild the booking database from Step 3 and add the bookings for the demo evening."""
    from utils.booking_db import create_tables_table, reset_reservations
    from utils.booking_mcp_server import add_reservation

    create_tables_table(rebuild=True)
    reset_reservations()
    for name, hour, size, table in DEMO_BOOKINGS:
        add_reservation(name, DEMO_DAY.replace(hour=hour), size, table)
    guests = sum(size for _, _, size, _ in DEMO_BOOKINGS)
    print(f"Booking database ready: {len(DEMO_BOOKINGS)} bookings ({guests} guests) on {DEMO_DAY:%A, %d %B %Y}.")


_DEFAULT_KNOBS = dict(
    OUT_OF_STOCK={"pumpkin"},
    WALKINS={"Friday": 60, "Saturday": 80, "Sunday": 50},
    SOUPS_PER_GUEST=0.5,
    GUESTS_PER_COOK=30,
    GUESTS_PER_SERVICE=20,
    MAX_STOCK_REJECTIONS=1,
)


def reset_knobs():
    """Put every kitchen knob back to its default value (handy after an experiment)."""
    for name, value in _DEFAULT_KNOBS.items():
        setattr(kitchen, name, type(value)(value))


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
    blocked = kitchen.blocked_ingredients(plan["soup"])
    if not blocked:
        return {"retry": False}
    if plan.get("tries", 0) < kitchen.MAX_STOCK_REJECTIONS:
        return {"retry": True, "tries": plan.get("tries", 0) + 1, "avoid": plan.get("avoid", []) + blocked}
    return {"retry": False, "fallback": True,
            "soup": f"{kitchen.fallback_soup()} – {', '.join(blocked)} out of stock"}


def after_stock_check(plan) -> Literal["soup_agent", "purchasing_agent"]:
    """Where to go after the stock check: back to the cook, or on to purchasing.
    (The return type tells LangGraph – and the diagram – the two possible next steps.)"""
    return "soup_agent" if plan.get("retry") else "purchasing_agent"


def save_plan(plan):
    """Save the finished day plan to utils/pizzeria_store.json (recall it with get_daily_plan)."""
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
    return create_agent(model, name=name, tools=[*tools, *kitchen.handoffs(*targets)],
                        system_prompt=prompt + TEAM + rules)
