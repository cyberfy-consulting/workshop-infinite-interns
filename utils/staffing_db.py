"""
staffing_db.py
--------------
A tiny file-based "database" for the pizzeria's staff, in the same spirit as
pantry_db.py: no real DB engine, just a JSON file (staffing_store.json) that is
read and written by simple functions.

It stores, per employee:
  - role         : "cook" or "service"
  - available    : the weekdays they are generally rostered for
  - weekly_hours : contracted hours (extra context for the agent)
  - vacations    : list of [start, end] ISO date ranges (inclusive)

The staffing tool uses this to decide WHICH concrete employees work on a given
date: someone counts only if the date's weekday is in their `available` list
AND the date is not inside one of their vacation ranges.
"""

import json
import os
from datetime import datetime

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "staffing_store.json")

# --- Seed roster used when the store does not exist yet ---
_DEFAULT_STAFF = {
    "Mario":   {"role": "cook",    "weekly_hours": 42, "available": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], "vacations": []},
    "Luca":    {"role": "cook",    "weekly_hours": 40, "available": ["Wednesday","Thursday","Friday","Saturday","Sunday"],           "vacations": [["2026-08-01","2026-08-14"]]},
    "Elena":   {"role": "cook",    "weekly_hours": 32, "available": ["Friday","Saturday","Sunday"],                                   "vacations": []},
    "Giulia":  {"role": "service", "weekly_hours": 38, "available": ["Thursday","Friday","Saturday","Sunday"],                        "vacations": []},
    "Antonio": {"role": "service", "weekly_hours": 40, "available": ["Tuesday","Wednesday","Thursday","Friday","Saturday"],          "vacations": []},
    "Sofia":   {"role": "service", "weekly_hours": 30, "available": ["Friday","Saturday","Sunday"],                                   "vacations": [["2026-07-28","2026-08-03"]]},
    "Marco":   {"role": "service", "weekly_hours": 24, "available": ["Saturday","Sunday"],                                            "vacations": []},
}


# ----------------------------------------------------------------------
# Low-level load / save
# ----------------------------------------------------------------------
def _load():
    if not os.path.exists(STORE_PATH):
        return {"staff": _DEFAULT_STAFF}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(store):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def init_staffing(reset: bool = False):
    """Create the JSON store with the seed roster. Pass reset=True to overwrite."""
    if reset and os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)
    store = _load()
    _save(store)
    return STORE_PATH


# ----------------------------------------------------------------------
# Read helpers
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
    """Return {'cook': [names], 'service': [names]} available on the given date:
    rostered for that weekday AND not on vacation."""
    weekday = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
    out = {"cook": [], "service": []}
    for name, emp in get_staff().items():
        if weekday in emp["available"] and not _on_vacation(emp, date_str):
            out[emp["role"]].append(name)
    return out


# ----------------------------------------------------------------------
# Write helpers
# ----------------------------------------------------------------------
def add_vacation(name, start, end):
    """Add a vacation range [start, end] (ISO dates) for an employee."""
    store = _load()
    store["staff"][name]["vacations"].append([start, end])
    _save(store)


def add_employee(name, role, available, weekly_hours=40, vacations=None):
    """Add a new employee to the roster."""
    store = _load()
    store["staff"][name] = {"role": role, "weekly_hours": weekly_hours,
                            "available": available, "vacations": vacations or []}
    _save(store)


# Ensure the file exists on import so the tool can read it immediately.
init_staffing(reset=False)