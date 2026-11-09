"""
pizzeria_tools.py
-----------------
Giovanni's kitchen toolbox for Step 5 (multi-agent patterns).

Every tool the four specialists use lives here, so the notebook only has to
show what matters for the workshop: the agents, their prompts and how they are
orchestrated.

The UPPER_CASE settings below are "knobs": they are read every time a tool is
called, so you can change them from the notebook without restarting anything:

    from utils import pizzeria_tools as kitchen
    kitchen.OUT_OF_STOCK.add("zucchini")
    kitchen.WALKINS["Saturday"] = 140
"""

from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph_swarm import create_handoff_tool

from utils.helper_functions import list_reservations_df
from utils.pantry_db import get_menu, get_pantry, init_store
from utils.staffing_db import get_staff, init_staffing, _on_vacation


# ======================================================================
# Knobs
# ======================================================================
WALKINS = {"Friday": 60, "Saturday": 80, "Sunday": 50}   # typical walk-ins per weekday
DEFAULT_WALKINS = 30                                     # any other weekday
PIZZAS_PER_GUEST = 1.2
GUESTS_PER_COOK = 30
GUESTS_PER_SERVICE = 20
OUT_OF_STOCK = {"ricotta"}        # cannot be bought in time, whatever the agents want
MAX_STOCK_REJECTIONS = 1          # how often purchasing may send the menu back per run

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
    "Vermentino": "white, crisp – vegetables, seafood, summer specials",
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


def blocked_ingredients(dish: str) -> list:
    """Out-of-stock ingredients in a dish (plain Python, no LLM).

    Looks at the 'Key ingredients:' line the menu agent is asked to write, so that a
    sentence like 'no ricotta needed' does not count. Falls back to the whole text."""
    text = dish.lower()
    for line in text.splitlines():
        if "key ingredient" in line:
            text = line
            break
    return sorted(i for i in OUT_OF_STOCK if i.replace("_", " ") in text)


def fallback_classic() -> str:
    """The safest choice when nothing creative works: the most popular classic."""
    return max(get_menu(), key=lambda m: m[1])[0]


def handoffs(*agent_names):
    """Handoff tools for the network pattern: one transfer_to_<name> tool per target."""
    return [create_handoff_tool(agent_name=n, description=f"Hand the work over to {n}.")
            for n in agent_names]


# ======================================================================
# Forecast tools
# ======================================================================
@tool("get_reservations")
def get_reservations(date: str) -> str:
    """Number of already-booked guests for a date (YYYY-MM-DD), summing party sizes."""
    df = list_reservations_df()
    mask = df["reservation_time"].astype(str).str.startswith(date)
    booked, n = int(df.loc[mask, "party_size"].sum()), int(mask.sum())
    return f"{booked} guests booked across {n} reservations on {date}."


@tool("get_walkin_baseline")
def get_walkin_baseline(weekday: str) -> str:
    """Typical number of walk-in guests (no reservation) for a weekday, e.g. 'Saturday'."""
    base = WALKINS.get(weekday.strip().capitalize(), DEFAULT_WALKINS)
    return f"Typical walk-ins on {weekday}: ~{base} guests."


# ======================================================================
# Menu tools
# ======================================================================
@tool("get_seasonal_ingredients")
def get_seasonal_ingredients(month: int) -> str:
    """Ingredients in season for a given month (1-12)."""
    season = _MONTH_TO_SEASON[(int(month) - 1) % 12]
    return f"In season ({season}): {SEASONS[season]}."


@tool("get_menu_classics")
def get_menu_classics() -> str:
    """The pizzas already on the regular menu, so the special does not duplicate them."""
    return "Regular classics: " + ", ".join(name for name, _, _ in get_menu())


# ======================================================================
# Purchasing tools
# ======================================================================
_REJECTIONS = {}   # thread_id -> how often an ingredient was rejected in that run


@tool("check_pantry")
def check_pantry() -> str:
    """Current stock in the pantry (kg per ingredient)."""
    return "Current stock (kg): " + ", ".join(f"{k} {v}" for k, v in sorted(get_pantry().items()))


@tool("check_special_ingredient")
def check_special_ingredient(ingredient: str, config: RunnableConfig) -> str:
    """Check whether the KEY ingredient of the daily special can be sourced in time."""
    ing = ingredient.strip().lower().replace(" ", "_")
    if ing in OUT_OF_STOCK:
        # Guardrail: count rejections per run (thread). After MAX_STOCK_REJECTIONS the
        # tool stops the ping-pong and tells the agent to fall back to a classic.
        thread = (config or {}).get("configurable", {}).get("thread_id")
        if thread is not None:
            _REJECTIONS[thread] = _REJECTIONS.get(thread, 0) + 1
            if _REJECTIONS[thread] > MAX_STOCK_REJECTIONS:
                return (f"OUT OF STOCK AGAIN: '{ing}' is still unavailable and the menu was already asked "
                        f"to change. Do NOT send it back again. Use '{fallback_classic()}' as the daily "
                        "special and continue with the shopping list.")
        return (f"OUT OF STOCK: '{ing}' is unavailable and cannot be reordered in time. "
                "The daily special MUST be changed to one that does NOT need it.")
    stock = get_pantry().get(ing)
    if stock is not None:
        return f"AVAILABLE: '{ing}' is in stock ({stock} kg)."
    return f"AVAILABLE: '{ing}' is not in the pantry but can be ordered from the supplier in time."


@tool("calculate_order")
def calculate_order(guests: int) -> str:
    """Shopping list for the popular CLASSICS only: estimates pizzas per guest, splits them by sales
    share, sums recipe demand, compares to stock and returns the shortfall. Excludes the daily special."""
    menu, pantry = get_menu(), get_pantry()
    total = round(int(guests) * PIZZAS_PER_GUEST)
    demand, breakdown = {}, [f"Expected classic pizzas: ~{total}"]
    for name, share, recipe in menu:
        qty = round(total * share)
        breakdown.append(f"  - {name}: ~{qty} pizzas")
        for ing, per in recipe.items():
            demand[ing] = demand.get(ing, 0) + per * qty
    lines = ["", "CLASSICS — order = demand - stock:"]
    for ing in sorted(demand):
        need, have = round(demand[ing], 1), round(pantry.get(ing, 0), 1)
        order = max(0, round(need - have, 1))
        flag = "  <-- SHORTAGE" if have < need * 0.3 else ""
        lines.append(f"  {ing}: need {need}kg, stock {have}kg -> order {order}kg{flag}")
    return "\n".join(breakdown + lines)


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
MENU_TOOLS = [get_seasonal_ingredients, get_menu_classics]
PURCHASING_TOOLS = [check_special_ingredient, check_pantry, calculate_order]
STAFFING_TOOLS = [get_required_staff, get_staff_status]
SOMMELIER_TOOLS = [get_wine_list]
