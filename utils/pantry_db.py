"""
pantry_db.py
------------
A tiny file-based "database" for Giovanni's pizzeria demo.

No real DB engine, just a JSON file (pizzeria_store.json) that is read and
written by simple functions. This keeps the demo runnable and the daily plan
recallable across cells / kernel restarts.

Tables (top-level keys in the JSON file):
  - menu        : the popular classics, their sales share and recipe (kg/pizza)
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

_DEFAULT_PANTRY = {  # current stock in kg
    "flour": 40, "tomato_sauce": 8, "mozzarella": 12, "basil": 0.5, "olive_oil": 10,
    "salami": 2, "ham": 2, "mushrooms": 3, "gorgonzola": 1.5, "parmesan": 2,
    "fontina": 1, "garlic": 1, "oregano": 0.3,
}


# ----------------------------------------------------------------------
# Low-level load / save
# ----------------------------------------------------------------------
def _load():
    """Read the whole store from disk (or return a fresh seeded store)."""
    if not os.path.exists(STORE_PATH):
        return {"menu": _DEFAULT_MENU, "pantry": _DEFAULT_PANTRY, "daily_plan": {}}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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


def save_daily_plan(day, guests, special, order_list, staffing):
    """Persist the finished plan for a day so it can be recalled later."""
    store = _load()
    store["daily_plan"][day] = {
        "day": day,
        "guests": guests,
        "special": special,
        "order_list": order_list,
        "staffing": staffing,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save(store)


# Ensure the file exists on import so the tools can read it immediately.
init_store(reset=False)