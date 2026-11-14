"""
restaurant_db.py
------------
A small JSON-backed store for the restaurant workshop.

The store is shared by the menu, pantry, daily plan, and staffing helpers.

Tables (top-level keys in the JSON file):
  - menu        : the classic pizzas (fixed), their sales share and recipe (kg/pizza)
  - soup_base   : the fixed base of the soup of the day and its recipe (kg/portion)
  - pantry      : current stock per ingredient (kg)
  - daily_plan  : the last persisted plan per day (recallable)
  - staff       : employees, availability, and vacation ranges
"""

from copy import deepcopy
import json
import os
from datetime import datetime

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restaurant_store.json")

# Reference data used by the restaurant tools.
_DEFAULT_SEASONAL_INGREDIENTS = {
    "spring": "asparagus, peas, spring onions, ricotta",
    "summer": "tomatoes, zucchini, basil, eggplant, peppers",
    "autumn": "porcini, pumpkin, radicchio, chestnuts",
    "winter": "cavolo nero, leeks, potatoes, gorgonzola",
}
_DEFAULT_WINE_LIST = {
    "Chianti Classico": "red, medium body – tomato-based pizzas, salami",
    "Barbera d'Asti": "red, fresh acidity – mushrooms, ham, cheese-heavy pizzas",
    "Vermentino": "white, crisp – vegetables, seafood, light soups",
    "Lambrusco": "sparkling red – spicy salami, Diavola",
    "Prosecco": "sparkling white – aperitivo, light white pizzas",
}

# --- Seed data used when the store does not exist yet ---
_DEFAULT_MENU = {
    # name: {share (sums to 1.0), recipe in kg per pizza}
    "Margherita":          {"share": 0.35, "recipe": {"flour": 0.25, "tomato_sauce": 0.10, "mozzarella": 0.12, "basil": 0.01, "olive_oil": 0.01}},
    "Diavola":             {"share": 0.20, "recipe": {"flour": 0.25, "tomato_sauce": 0.10, "mozzarella": 0.12, "salami": 0.06, "olive_oil": 0.01}},
    "Prosciutto e Funghi": {"share": 0.15, "recipe": {"flour": 0.25, "tomato_sauce": 0.10, "mozzarella": 0.12, "ham": 0.05, "mushrooms": 0.05}},
    "Quattro Formaggi":    {"share": 0.15, "recipe": {"flour": 0.25, "mozzarella": 0.08, "gorgonzola": 0.05, "parmesan": 0.03, "fontina": 0.05}},
    "Marinara":            {"share": 0.15, "recipe": {"flour": 0.25, "tomato_sauce": 0.12, "garlic": 0.02, "oregano": 0.005, "olive_oil": 0.02}},
}

# The soup of the day always has the same base; the cook only adds ONE seasonal special ingredient.
_DEFAULT_SOUP_BASE = {
    "name": "Minestra della Casa",
    "recipe": {"vegetable_stock": 0.30, "potatoes": 0.10, "onions": 0.04, "cream": 0.03},  # kg per portion
    "special_per_portion": 0.08,                                                            # kg of the special ingredient
}

_DEFAULT_PANTRY = {  # current stock in kg
    "flour": 40, "tomato_sauce": 8, "mozzarella": 12, "basil": 0.5, "olive_oil": 10,
    "salami": 2, "ham": 2, "mushrooms": 3, "gorgonzola": 1.5, "parmesan": 2,
    "fontina": 1, "garlic": 1, "oregano": 0.3,
    "vegetable_stock": 5, "potatoes": 4, "onions": 1, "cream": 0.5,
}

_DEFAULT_STAFF = {
    "Mario":   {"role": "cook",    "weekly_hours": 42, "available": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], "vacations": []},
    "Luca":    {"role": "cook",    "weekly_hours": 40, "available": ["Wednesday","Thursday","Friday","Saturday","Sunday"],           "vacations": [["2026-08-01","2026-08-14"]]},
    "Elena":   {"role": "cook",    "weekly_hours": 32, "available": ["Friday","Saturday","Sunday"],                                   "vacations": []},
    "Giulia":  {"role": "service", "weekly_hours": 38, "available": ["Thursday","Friday","Saturday","Sunday"],                        "vacations": []},
    "Antonio": {"role": "service", "weekly_hours": 40, "available": ["Tuesday","Wednesday","Thursday","Friday","Saturday"],          "vacations": []},
    "Sofia":   {"role": "service", "weekly_hours": 30, "available": ["Friday","Saturday","Sunday"],                                   "vacations": [["2026-07-28","2026-08-03"]]},
    "Marco":   {"role": "service", "weekly_hours": 24, "available": ["Saturday","Sunday"],                                            "vacations": []},
}


def _default_store():
    return {
        "menu": deepcopy(_DEFAULT_MENU),
        "soup_base": deepcopy(_DEFAULT_SOUP_BASE),
        "pantry": deepcopy(_DEFAULT_PANTRY),
        "daily_plan": {},
        "staff": deepcopy(_DEFAULT_STAFF),
        "seasonal_ingredients": deepcopy(_DEFAULT_SEASONAL_INGREDIENTS),
        "wine_list": deepcopy(_DEFAULT_WINE_LIST),
    }


# ----------------------------------------------------------------------
# Low-level load / save
# ----------------------------------------------------------------------
def _load():
    """Read the store from disk, or return a fresh seeded store."""
    store = _default_store()
    if not os.path.exists(STORE_PATH):
        return store
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        saved_store = json.load(f)
    if not isinstance(saved_store, dict):
        raise ValueError(f"Restaurant store must contain a JSON object: {STORE_PATH}")
    for key, value in store.items():
        saved_store.setdefault(key, value)
    store = saved_store
    return store


def _save(store):
    """Write the whole store back to disk."""
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def initialize_restaurant(reset: bool = False):
    """Create the store; pass reset=True to replace it with all seed data."""
    store = _default_store() if reset else _load()
    _save(store)
    return STORE_PATH


def setup():
    """Reset the restaurant store to its seed data."""
    initialize_restaurant(reset=True)


# ----------------------------------------------------------------------
# Read helpers
# ----------------------------------------------------------------------
def get_menu():
    """Return list of (name, share, recipe_dict) for the popular classics."""
    menu = _load()["menu"]
    return [(name, item["share"], item["recipe"]) for name, item in menu.items()]


def get_soup_base():
    """Return the fixed soup base: {name, recipe (kg/portion), special_per_portion}."""
    return _load()["soup_base"]


def get_seasonal_ingredients():
    """Return seasonal ingredients by season."""
    return _load()["seasonal_ingredients"]


def get_wine_list():
    """Return wines and their suggested pairings."""
    return _load()["wine_list"]


def get_pantry():
    """Return the current stock as {ingredient: kg}."""
    return _load()["pantry"]


def get_daily_plan(day):
    """Return the persisted plan dict for a day, or None if not saved yet."""
    return _load()["daily_plan"].get(day)


# ----------------------------------------------------------------------
# Write helpers
# ----------------------------------------------------------------------
def set_stock(ingredient, kg):
    """Overwrite the stock of one ingredient (e.g. after a delivery)."""
    store = _load()
    store["pantry"][ingredient] = kg
    _save(store)


def save_daily_plan(day, guests, soup, order_list, staffing):
    """Persist the finished plan for a day so it can be recalled later."""
    store = _load()
    store["daily_plan"][day] = {
        "day": day,
        "guests": guests,
        "soup": soup,
        "order_list": order_list,
        "staffing": staffing,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save(store)


# ----------------------------------------------------------------------
# Staffing helpers
# ----------------------------------------------------------------------
def get_staff():
    """Return the full roster: {name: {role, weekly_hours, available, vacations}}."""
    return _load()["staff"]


def _on_vacation(emp, date_str):
    """True if date_str falls inside any of the employee's vacation ranges."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    for start, end in emp.get("vacations", []):
        if datetime.strptime(start, "%Y-%m-%d").date() <= d <= datetime.strptime(end, "%Y-%m-%d").date():
            return True
    return False


def available_staff(date_str):
    """Return staff rostered for the weekday and not on vacation."""
    weekday = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
    out = {"cook": [], "service": []}
    for name, emp in get_staff().items():
        if weekday in emp["available"] and not _on_vacation(emp, date_str):
            out[emp["role"]].append(name)
    return out


def add_vacation(name, start, end):
    """Add a vacation range [start, end] (ISO dates) for an employee."""
    store = _load()
    store["staff"][name]["vacations"].append([start, end])
    _save(store)


def add_employee(name, role, available, weekly_hours=40, vacations=None):
    """Add a new employee to the roster."""
    store = _load()
    store["staff"][name] = {
        "role": role,
        "weekly_hours": weekly_hours,
        "available": available,
        "vacations": vacations or [],
    }
    _save(store)
