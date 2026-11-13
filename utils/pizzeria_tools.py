# Giovanni's kitchen toolbox for Step 5 (multi-agent patterns).
import sqlite3
from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph_swarm.handoff import METADATA_KEY_HANDOFF_DESTINATION

from utils.helper_functions import DB_PATH as BOOKING_DB_PATH
from utils.pantry_db import get_menu, get_pantry, get_soup_base, init_store
from utils.staffing_db import get_staff, init_staffing, _on_vacation


# ======================================================================
# settings are read every time a tool is called; can be changed from notebook e.g.
# from utils import pizzeria_tools as kitchen
# kitchen.OUT_OF_STOCK.add("porcini")
# kitchen.WALKINS["Saturday"] = 140
# ======================================================================
WALKINS = {"Friday": 60, "Saturday": 80, "Sunday": 50}   # typical walk-ins per weekday
DEFAULT_WALKINS = 30                                     # any other weekday
PIZZAS_PER_GUEST = 1.2
SOUPS_PER_GUEST = 0.5             # share of guests who order the soup of the day
GUESTS_PER_COOK = 30
GUESTS_PER_SERVICE = 20
OUT_OF_STOCK = {"pumpkin"}        # cannot be bought in time, whatever the agents want
MAX_STOCK_REJECTIONS = 1          # how often purchasing may send the soup ingredient back per run

SEASONS = {
    "spring": "asparagus, peas, spring onions, ricotta",
    "summer": "tomatoes, zucchini, basil, eggplant, peppers",
    "autumn": "porcini, pumpkin, radicchio, chestnuts",
    "winter": "cavolo nero, leeks, potatoes, gorgonzola",
}
_MONTH_TO_SEASON = ("winter", "winter", "spring", "spring", "spring", "summer",
                    "summer", "summer", "autumn", "autumn", "autumn", "winter")

WINE_LIST = {  # used by the bonus sommelier agent
    "Chianti Classico": "red, medium body – tomato-based pizzas, salami",
    "Barbera d'Asti": "red, fresh acidity – mushrooms, ham, cheese-heavy pizzas",
    "Vermentino": "white, crisp – vegetables, seafood, light soups",
    "Lambrusco": "sparkling red – spicy salami, Diavola",
    "Prosecco": "sparkling white – aperitivo, light white pizzas",
}


# ======================================================================
# Setup helpers
# ======================================================================
def reset_kitchen():
    """Fresh pantry and staff roster (seed data). Also makes sure the booking DB from Step 3 exists."""
    import utils.booking_db  # noqa: F401  (creates utils/booking.db on first import if it is missing)
    init_store(reset=True)
    init_staffing(reset=True)


def weekday(date: str) -> str:
    return datetime.strptime(date, "%Y-%m-%d").strftime("%A")


def month(date: str) -> int:
    return datetime.strptime(date, "%Y-%m-%d").month


def request(date: str, wish: str = "") -> str:
    """Giovanni's request for a day – the same text for all three patterns."""
    return (f"Plan {date} (a {weekday(date)}): how busy will it be, what is the soup of the day, what do we "
            f"need to buy and who has to work? {wish or ''}".strip())


def _norm(ingredient: str) -> str:
    """'Chestnuts ' -> 'chestnut' (lower case, underscores, no plural s) for lenient matching."""
    ing = ingredient.strip().lower().replace(" ", "_")
    return ing[:-1] if ing.endswith("s") else ing


def _is_out_of_stock(ingredient: str) -> bool:
    return _norm(ingredient) in {_norm(i) for i in OUT_OF_STOCK}


def blocked_ingredients(soup: str) -> list:
    """Out-of-stock ingredients in the soup of the day (plain Python, no LLM).

    Looks at the 'Special ingredient:' line the soup agent is asked to write, so that a
    sentence like 'pumpkin is out of stock' does not count. Falls back to the whole text."""
    text = soup.lower()
    for line in text.splitlines():
        if "special ingredient" in line:
            text = line
            break
    return sorted(i for i in OUT_OF_STOCK if _norm(i).replace("_", " ") in text)


def fallback_soup() -> str:
    """The safe choice when no special ingredient works: the plain soup base."""
    return f"{get_soup_base()['name']} (plain, without a special ingredient)"


def handoffs(*agent_names):
    """Handoff tools for the network pattern: one transfer_to_<name>(note) tool per target.

    The `note` is required: it carries the agent's result (e.g. the chosen soup ingredient)
    over to the next agent, so nothing gets lost on the way."""
    return [_handoff_tool(n) for n in agent_names]


def _handoff_tool(agent_name: str):
    name = f"transfer_to_{agent_name}"

    @tool(name, description=f"Hand the work over to {agent_name}. Put your result and everything "
                            f"{agent_name} needs to know into `note`.")
    def handoff(note: str, state: Annotated[Any, InjectedState],
                tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        msg = ToolMessage(content=f"Transferred to {agent_name}. Note: {note}", name=name, tool_call_id=tool_call_id)
        return Command(goto=agent_name, graph=Command.PARENT,
                       update={"messages": [*state["messages"], msg], "active_agent": agent_name})

    handoff.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}   # lets create_swarm draw the edge
    return handoff


# ======================================================================
# Forecast tools
# ======================================================================
@tool("get_reservations")
def get_reservations(date: str) -> str:
    """Number of already-booked guests for a date (YYYY-MM-DD), summing party sizes."""
    with sqlite3.connect(BOOKING_DB_PATH) as conn:
        booked, n = conn.execute(
            "SELECT COALESCE(SUM(party_size), 0), COUNT(*) FROM reservations WHERE date(reservation_time) = ?",
            (date,),
        ).fetchone()
    return f"{booked} guests booked across {n} reservations on {date}."


@tool("get_walkin_baseline")
def get_walkin_baseline(weekday: str) -> str:
    """Typical number of walk-in guests (no reservation) for a weekday, e.g. 'Saturday'."""
    base = WALKINS.get(weekday.strip().capitalize(), DEFAULT_WALKINS)
    return f"Typical walk-ins on {weekday}: ~{base} guests."


# ======================================================================
# Soup tools
# ======================================================================
@tool("get_seasonal_ingredients")
def get_seasonal_ingredients(month: int) -> str:
    """Ingredients in season for a given month (1-12)."""
    season = _MONTH_TO_SEASON[(int(month) - 1) % 12]
    return f"In season ({season}): {SEASONS[season]}."


@tool("get_soup_base")
def get_soup_base_tool() -> str:
    """The fixed base of the soup of the day. The cook adds exactly ONE special ingredient to it."""
    base = get_soup_base()
    return (f"Soup base '{base['name']}': " + ", ".join(i.replace("_", " ") for i in base["recipe"])
            + ". Add exactly ONE seasonal special ingredient that is not already in the base.")


# ======================================================================
# Purchasing tools
# ======================================================================
_REJECTIONS = {}   # thread_id -> how often an ingredient was rejected in that run


@tool("check_pantry")
def check_pantry() -> str:
    """Current stock in the pantry (kg per ingredient)."""
    return "Current stock (kg): " + ", ".join(f"{k} {v}" for k, v in sorted(get_pantry().items()))


@tool("check_soup_ingredient")
def check_soup_ingredient(ingredient: str, config: RunnableConfig) -> str:
    """Check whether the special ingredient of the soup of the day can be sourced in time."""
    ing = ingredient.strip().lower().replace(" ", "_")
    if _is_out_of_stock(ing):
        # Guardrail: count rejections per run (thread). After MAX_STOCK_REJECTIONS the
        # tool stops the ping-pong and tells the agent to fall back to the plain soup.
        thread = (config or {}).get("configurable", {}).get("thread_id")
        if thread is not None:
            _REJECTIONS[thread] = _REJECTIONS.get(thread, 0) + 1
            if _REJECTIONS[thread] > MAX_STOCK_REJECTIONS:
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
    total = round(int(guests) * PIZZAS_PER_GUEST)
    soups = round(int(guests) * SOUPS_PER_GUEST)
    pizza_demand, breakdown = {}, [f"Expected classic pizzas: ~{total}"]
    for name, share, recipe in menu:
        qty = round(total * share)
        breakdown.append(f"  - {name}: ~{qty} pizzas")
        for ing, per in recipe.items():
            pizza_demand[ing] = pizza_demand.get(ing, 0) + per * qty

    special = special_ingredient.strip().lower().replace(" ", "_")
    if special in ("", "none", "plain", "-"):
        special = ""
    soup_demand = {ing: per * soups for ing, per in base["recipe"].items()}
    if special:
        soup_demand[special] = soup_demand.get(special, 0) + base["special_per_portion"] * soups
    breakdown.append(f"Expected soups: ~{soups} portions of {base['name']}"
                     + (f" with {special.replace('_', ' ')}" if special else " (plain)"))

    def section(title, demand):
        lines = ["", f"{title} — order = demand - stock:"]
        for ing in sorted(demand):
            need, have = round(demand[ing], 1), round(pantry.get(ing, 0), 1)
            order = max(0, round(need - have, 1))
            flag = "  <-- SHORTAGE" if have < need * 0.3 else ""
            lines.append(f"  {ing}: need {need}kg, stock {have}kg -> order {order}kg{flag}")
        return lines

    out = breakdown + section("CLASSIC PIZZAS", pizza_demand) + section("SOUP OF THE DAY", soup_demand)
    if special and _is_out_of_stock(special):
        out.append(f"WARNING: '{special}' is OUT OF STOCK and cannot be ordered in time.")
    return "\n".join(out)


# ======================================================================
# Staffing tools
# ======================================================================
@tool("get_required_staff")
def get_required_staff(guests: int) -> str:
    """How many cooks and service staff are needed for a number of guests."""
    g = int(guests)
    cooks = max(1, -(-g // GUESTS_PER_COOK))
    service = max(1, -(-g // GUESTS_PER_SERVICE))
    return (f"Required: {cooks} cook(s), {service} service "
            f"(rule: 1 cook per {GUESTS_PER_COOK} guests, 1 service per {GUESTS_PER_SERVICE} guests).")


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


# ======================================================================
# Bonus: sommelier
# ======================================================================
@tool("get_wine_list")
def get_wine_list() -> str:
    """Wines Giovanni has in the cellar and what they pair well with."""
    return "Wine list:\n" + "\n".join(f"  - {w}: {d}" for w, d in WINE_LIST.items())


# ======================================================================
# Toolboxes per specialist (least privilege)
# ======================================================================
FORECAST_TOOLS = [get_reservations, get_walkin_baseline]
SOUP_TOOLS = [get_seasonal_ingredients, get_soup_base_tool]
PURCHASING_TOOLS = [check_soup_ingredient, check_pantry, calculate_order]
STAFFING_TOOLS = [get_required_staff, get_staff_status]
SOMMELIER_TOOLS = [get_wine_list]
