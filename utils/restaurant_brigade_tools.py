"""Restaurant-agent tools and orchestration helpers for the brigade."""

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph_swarm.handoff import METADATA_KEY_HANDOFF_DESTINATION

from utils import restaurant_demo as demo
from utils.booking_system_db import RESERVATION_COLUMNS, fetch_reservations
from utils.helper_functions import month_to_season, weekday
from utils.restaurant_db import (
    _on_vacation,
    get_menu,
    get_pantry,
    get_seasonal_ingredients as get_seasonal_ingredient_data,
    get_soup_base,
    get_staff,
    get_wine_list as get_wine_data,
)


def request(date: str, wish: str = "") -> str:
    """Build Giovanni's shared planning request for a day."""
    return (f"Plan {date} (a {weekday(date)}): how busy will it be, what is the soup of the day, what do we "
            f"need to buy and who has to work? {wish or ''}".strip())


def handoffs(*agent_names: str):
    """Create one transfer_to_<name>(note) tool per target agent."""
    return [_handoff_tool(name) for name in agent_names]


def _skipped_calls(messages, handoff_call_id: str, agent_name: str):
    """Answer tool calls skipped when the agent hands work off in the same step."""
    last_ai = next((message for message in reversed(messages) if getattr(message, "tool_calls", None)), None)
    if last_ai is None:
        return []
    answered = {message.tool_call_id for message in messages if isinstance(message, ToolMessage)}
    return [
        ToolMessage(
            content=f"Not executed: the work was handed over to {agent_name} in the same step.",
            name=call["name"],
            tool_call_id=call["id"],
        )
        for call in last_ai.tool_calls
        if call["id"] != handoff_call_id and call["id"] not in answered
    ]


def _handoff_tool(agent_name: str):
    name = f"transfer_to_{agent_name}"

    @tool(
        name,
        description=(
            f"Hand the work over to {agent_name}. Put your result and everything "
            f"{agent_name} needs to know into `note`. Call this tool ALONE – never together "
            "with other tools in the same step."
        ),
    )
    def handoff(
        note: str,
        state: Annotated[Any, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        messages = list(state["messages"])
        message = ToolMessage(
            content=f"Transferred to {agent_name}. Note: {note}",
            name=name,
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": [
                    *messages,
                    message,
                    *_skipped_calls(messages, tool_call_id, agent_name),
                ],
                "active_agent": agent_name,
            },
        )

    handoff.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff


# ======================================================================
# Pantry
# ======================================================================
def _norm(ingredient: str) -> str:
    """'Chestnuts ' -> 'chestnut' (lower case, underscores, no plural s) for lenient matching."""
    ing = ingredient.strip().lower().replace(" ", "_")
    return ing[:-1] if ing.endswith("s") else ing


def _is_out_of_stock(ingredient: str) -> bool:
    return _norm(ingredient) in {_norm(item) for item in demo.OUT_OF_STOCK}


def blocked_ingredients(soup: str) -> list:
    """Out-of-stock ingredients in the soup of the day (plain Python, no LLM).

    Looks at the 'Special ingredient:' line the soup agent is asked to write, so that a
    sentence like 'pumpkin is out of stock' does not count. Falls back to the whole text."""
    text = soup.lower()
    for line in text.splitlines():
        if "special ingredient" in line:
            text = line
            break
    return sorted(item for item in demo.OUT_OF_STOCK if _norm(item).replace("_", " ") in text)


@tool("check_pantry")
def check_pantry() -> str:
    """Current stock in the pantry (kg per ingredient)."""
    return "Current stock (kg): " + ", ".join(f"{key} {value}" for key, value in sorted(get_pantry().items()))


# ======================================================================
# Procurement
# ======================================================================
_REJECTIONS = {}   # thread_id -> how often an ingredient was rejected in that run


def fallback_soup() -> str:
    """The safe choice when no special ingredient works: the plain soup base."""
    return f"{get_soup_base()['name']} (plain, without a special ingredient)"


@tool("check_soup_ingredient")
def check_soup_ingredient(ingredient: str, config: RunnableConfig) -> str:
    """Check whether the special ingredient of the soup of the day can be sourced in time."""
    ing = ingredient.strip().lower().replace(" ", "_")
    if _is_out_of_stock(ing):
        thread = (config or {}).get("configurable", {}).get("thread_id")
        if thread is not None:
            _REJECTIONS[thread] = _REJECTIONS.get(thread, 0) + 1
            if _REJECTIONS[thread] > demo.MAX_STOCK_REJECTIONS:
                return (f"FALLBACK REQUIRED: '{ing}' is still unavailable and the cook was already asked "
                        f"to change it. Do NOT hand off to soup_agent again. Use '{fallback_soup()}' as the soup "
                        "of the day and continue with the shopping list (calculate_order without a special ingredient).")
        return (f"OUT OF STOCK: '{ing}' is unavailable and cannot be reordered in time. "
                "The soup of the day MUST get a different special ingredient.")
    stock = get_pantry().get(ing)
    if stock is not None:
        return f"AVAILABLE: '{ing}' is in stock ({stock} kg)."
    return f"AVAILABLE: '{ing}' is not in the pantry but can be ordered from the supplier in time."


@tool("calculate_order")
def calculate_order(guests: int, special_ingredient: str = "") -> str:
    """Shopping list for the whole day: the classic pizzas AND the soup of the day (fixed base + the
    special ingredient). Estimates pizzas and soups per guest, sums recipe demand, compares it to the
    stock and returns what has to be ordered. Leave special_ingredient empty for the plain soup."""
    menu, pantry, base = get_menu(), get_pantry(), get_soup_base()
    total = round(int(guests) * demo.PIZZAS_PER_GUEST)
    soups = round(int(guests) * demo.SOUPS_PER_GUEST)
    pizza_demand, breakdown = {}, [f"Expected classic pizzas: ~{total}"]
    for name, share, recipe in menu:
        qty = round(total * share)
        breakdown.append(f"  - {name}: ~{qty} pizzas")
        for ingredient, per in recipe.items():
            pizza_demand[ingredient] = pizza_demand.get(ingredient, 0) + per * qty

    special = special_ingredient.strip().lower().replace(" ", "_")
    if special in ("", "none", "plain", "-"):
        special = ""
    soup_demand = {ingredient: per * soups for ingredient, per in base["recipe"].items()}
    if special:
        soup_demand[special] = soup_demand.get(special, 0) + base["special_per_portion"] * soups
    breakdown.append(f"Expected soups: ~{soups} portions of {base['name']}"
                     + (f" with {special.replace('_', ' ')}" if special else " (plain)"))

    def section(title, demand):
        lines = ["", f"{title} — order = demand - stock:"]
        for ingredient in sorted(demand):
            need, have = round(demand[ingredient], 1), round(pantry.get(ingredient, 0), 1)
            order = max(0, round(need - have, 1))
            flag = "  <-- SHORTAGE" if have < need * 0.3 else ""
            lines.append(f"  {ingredient}: need {need}kg, stock {have}kg -> order {order}kg{flag}")
        return lines

    out = breakdown + section("CLASSIC PIZZAS", pizza_demand) + section("SOUP OF THE DAY", soup_demand)
    if special and _is_out_of_stock(special):
        out.append(f"WARNING: '{special}' is OUT OF STOCK and cannot be ordered in time.")
    return "\n".join(out)


# ======================================================================
# Kitchen
# ======================================================================
@tool("get_seasonal_ingredients")
def get_seasonal_ingredients(month: int) -> str:
    """Ingredients in season for a given month (1-12)."""
    season = month_to_season(month)
    ingredients = get_seasonal_ingredient_data()[season]
    return f"In season ({season}): {ingredients}."


@tool("get_soup_base")
def get_soup_base_tool() -> str:
    """The fixed base of the soup of the day. The cook adds exactly ONE special ingredient to it."""
    base = get_soup_base()
    return (f"Soup base '{base['name']}': " + ", ".join(ingredient.replace("_", " ") for ingredient in base["recipe"])
            + ". Add exactly ONE seasonal special ingredient that is not already in the base.")


# ======================================================================
# Service
# ======================================================================
@tool("get_reservations")
def get_reservations(date: str) -> str:
    """Number of already-booked guests for a date (YYYY-MM-DD), summing party sizes."""
    rows = fetch_reservations(date)
    party_size = RESERVATION_COLUMNS.index("party_size")
    booked = sum(row[party_size] for row in rows)
    return f"{booked} guests booked across {len(rows)} reservations on {date}."


@tool("get_walkin_baseline")
def get_walkin_baseline(weekday: str) -> str:
    """Typical number of walk-in guests (no reservation) for a weekday, e.g. 'Saturday'."""
    base = demo.WALKINS.get(weekday.strip().capitalize(), demo.DEFAULT_WALKINS)
    return f"Typical walk-ins on {weekday}: ~{base} guests."


@tool("get_wine_list")
def get_wine_list() -> str:
    """Wines Giovanni has in the cellar and what they pair well with."""
    return "Wine list:\n" + "\n".join(
        f"  - {wine}: {description}" for wine, description in get_wine_data().items()
    )

# ======================================================================
# Generic
# ======================================================================
@tool("get_required_staff")
def get_required_staff(guests: int) -> str:
    """How many cooks and service staff are needed for a number of guests."""
    g = int(guests)
    cooks = max(1, -(-g // demo.GUESTS_PER_COOK))
    service = max(1, -(-g // demo.GUESTS_PER_SERVICE))
    return (f"Required: {cooks} cook(s), {service} service "
            f"(rule: 1 cook per {demo.GUESTS_PER_COOK} guests, "
            f"1 service per {demo.GUESTS_PER_SERVICE} guests).")


@tool("get_staff_status")
def get_staff_status(date: str) -> str:
    """Who is WORKING (rostered & not on vacation), CALLABLE (off that day but not on vacation, could be
    asked to come in) and ON VACATION, split by role, for a date (YYYY-MM-DD)."""
    day = weekday(date)
    working, callable_, vacation = ({"cook": [], "service": []} for _ in range(3))
    for name, emp in get_staff().items():
        role = emp["role"]
        if _on_vacation(emp, date):
            vacation[role].append(name)
        elif day in emp["available"]:
            working[role].append(name)
        else:
            callable_[role].append(name)
    fmt = lambda d: "; ".join(f"{r}: {', '.join(v) or '—'}" for r, v in d.items())
    return (f"Staff status for {day} ({date}):\n"
            f"  WORKING:     {fmt(working)}\n"
            f"  CALLABLE:    {fmt(callable_)}\n"
            f"  ON VACATION: {fmt(vacation)}")
