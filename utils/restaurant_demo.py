"""Demo settings for the restaurant workshop."""

# Settings can be changed for workshop experiments and reset with reset_knobs().
WALKINS = {"Friday": 60, "Saturday": 80, "Sunday": 50}
DEFAULT_WALKINS = 30
PIZZAS_PER_GUEST = 1.2
SOUPS_PER_GUEST = 0.5
GUESTS_PER_COOK = 30
GUESTS_PER_SERVICE = 20
OUT_OF_STOCK = {"pumpkin"}
MAX_STOCK_REJECTIONS = 1

_DEFAULT_KNOBS = {
    "OUT_OF_STOCK": OUT_OF_STOCK.copy(),
    "WALKINS": WALKINS.copy(),
    "SOUPS_PER_GUEST": SOUPS_PER_GUEST,
    "GUESTS_PER_COOK": GUESTS_PER_COOK,
    "GUESTS_PER_SERVICE": GUESTS_PER_SERVICE,
    "MAX_STOCK_REJECTIONS": MAX_STOCK_REJECTIONS,
}


def reset_knobs():
    """Restore the demo's configurable restaurant settings to their defaults."""
    for name, value in _DEFAULT_KNOBS.items():
        globals()[name] = type(value)(value)
