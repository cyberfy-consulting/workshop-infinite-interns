"""Parameters of Giovanni's restaurant for the workshop.

The tools of the brigade read these values on every call, so they can be changed between runs.
reset() puts them back to the values below."""

walkins = {"Friday": 60, "Saturday": 80, "Sunday": 50}   # typical walk-in guests per weekday
default_walkins = 30                                     # walk-ins on any other day
pizzas_per_guest = 1.2
soups_per_guest = 0.5
guests_per_cook = 30
guests_per_service = 20
out_of_stock = {"pumpkin"}                               # ingredients that can't be bought in time
max_stock_rejections = 1                                 # how often the soup ingredient may be sent back

_DEFAULTS = {
    "walkins": dict(walkins),
    "default_walkins": default_walkins,
    "pizzas_per_guest": pizzas_per_guest,
    "soups_per_guest": soups_per_guest,
    "guests_per_cook": guests_per_cook,
    "guests_per_service": guests_per_service,
    "out_of_stock": set(out_of_stock),
    "max_stock_rejections": max_stock_rejections,
}


def reset():
    """Restore all parameters to their defaults."""
    for name, value in _DEFAULTS.items():
        globals()[name] = type(value)(value)
