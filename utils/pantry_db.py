"""
pantry_db.py
------------
A tiny file-based "database" for Giovanni's pizzeria demo.

No real DB engine, just a JSON file (pizzeria_store.json) that is read and
written by simple functions. This keeps the demo runnable and the daily plan
recallable across cells / kernel restarts.

Tables (top-level keys in the JSON file):
  - menu        : the classic pizzas (fixed), their sales share and recipe (kg/pizza)
  - soup_base   : the fixed base of the soup of the day and its recipe (kg/portion)
  - pantry      : current stock per ingredient (kg)
  - daily_plan  : the last persisted plan per day (recallable)
"""

import json
import os
from datetime import datetime

# The JSON file lives next to this module.
STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pizzeria_store.json")

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


# ----------------------------------------------------------------------
# Low-level load / save
# ----------------------------------------------------------------------
def _load():
    """Read the whole store from disk (or return a fresh seeded store)."""
    defaults = {"menu": _DEFAULT_MENU, "soup_base": _DEFAULT_SOUP_BASE, "pantry": _DEFAULT_PANTRY, "daily_plan": {}}
    if not os.path.exists(STORE_PATH):
        return json.loads(json.dumps(defaults))   # deep copy of the seed data
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)
    for key, value in defaults.items():           # older stores may miss a table
        store.setdefault(key, json.loads(json.dumps(value)))
    return store


def _save(store):
    """Write the whole store back to disk."""
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def init_store(reset: bool = False):
    """Create the JSON store with seed data. Pass reset=True to overwrite it."""
    if reset and os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)
    store = _load()
    _save(store)
    return STORE_PATH


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


# Ensure the file exists on import so the tools can read it immediately.
init_store(reset=False)