import sqlite3

from utils.booking_db import DB_PATH, TODAY

# Table positions as (column, row) on a grid, following the arrangement of the sketch.
POSITIONS = {
    # inside, along the top wall
    "TI1":  (0, 0), "TI2":  (1, 0), "TI3":  (2, 0), "TI4":  (3, 0),
    "TI5":  (0, 1), "TI6":  (1, 1), "TI7":  (2, 1), "TI8":  (3, 1),
    # inside, along the left wall
    "TI9":  (0, 2), "TI10": (0, 3), "TI11": (0, 4),
    # inside, lower room (next to the bar)
    "TI12": (2, 5), "TI13": (3, 5),
    # outside, next to the building
    "TO1":  (5, 0), "TO2":  (5, 1), "TO3":  (5, 5), "TO4":  (5, 6),
    # outside, waterfront
    "TO5":  (7, 0), "TO6":  (7, 1), "TO7":  (7, 2), "TO8":  (7, 3),
    "TO9":  (7, 5), "TO10": (7, 6),
}

CELL = 100   # grid spacing in pixels
BOX = 76     # table size in pixels
X0, Y0 = 90, 110   # top-left corner of the grid

COLORS = {"free": "#ffffff", "booked": "#f98383", "new": "#98c3f7"}

# Walls, bar and water: everything in the sketch that is not a table
BACKGROUND = f"""
<defs>
  <pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="8" stroke="#555" stroke-width="2"/>
  </pattern>
</defs>
<text x="500" y="40" font-family="sans-serif" font-size="26" fill="#333" text-anchor="middle">Giovanni's Pizzeria, {TODAY:%A %d %B %Y}</text>
<rect x="900" y="60" width="100" height="820" rx="6" fill="#dbeafe"/>
<polyline points="490,470 490,90 90,90 90,860 490,860 490,560" fill="none" stroke="#9ca3af" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
<rect x="90" y="790" width="150" height="70" rx="6" fill="#e5e7eb"/>
<rect x="320" y="740" width="170" height="120" rx="6" fill="#e5e7eb"/>
<g font-family="sans-serif" font-size="14" fill="#6b7280" letter-spacing="2">
  <text x="105" y="112">INSIDE</text>
  <text x="560" y="112">TERRACE</text>
  <text x="915" y="112">WATER</text>
  <text x="105" y="830">BAR</text>
  <text x="335" y="765">KITCHEN</text>
</g>
"""

LEGEND = """
<g font-family="sans-serif" font-size="18" fill="#333">
  <rect x="90" y="905" width="24" height="24" rx="4" fill="#ffffff" stroke="#555" stroke-width="2"/><text x="122" y="924">free</text>
  <rect x="190" y="905" width="24" height="24" rx="4" fill="#f4b6b6" stroke="#555" stroke-width="2"/><text x="222" y="924">booked</text>
  <rect x="320" y="905" width="24" height="24" rx="4" fill="#bfdbfe" stroke="#555" stroke-width="2"/><text x="352" y="924">new reservation</text>
  <rect x="520" y="905" width="24" height="24" rx="4" fill="url(#hatch)" stroke="#555" stroke-width="2"/><text x="552" y="924">shaded (sun cover)</text>
  <text x="90" y="950" font-size="15" fill="#666">Hover a table for details.</text>
</g>
"""


def restaurant_html(booked=None, new=()):
    """Floor plan as an inline SVG.

    booked: {table label: "guest, HH:MM"} shown in red with the text under the table.
    new: table labels of reservations added since the last call, shown in blue.
    """
    booked = booked or {}
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT label, outside, waterfront, shade, seats FROM tables").fetchall()
    conn.close()

    parts = [BACKGROUND]
    for label, outside, waterfront, shade, seats in rows:
        if label not in POSITIONS:
            continue
        status = "new" if label in new else "booked" if label in booked else "free"
        tip = f"{label}: {seats} seats, {'outside' if outside else 'inside'}"
        if outside:
            tip += f", {'waterfront' if waterfront else 'no waterfront'}, {'shaded' if shade else 'no shade'}"
        tip += f" ({status})"

        col, row = POSITIONS[label]
        x = X0 + col * CELL + (CELL - BOX) / 2
        y = Y0 + row * CELL + (CELL - BOX) / 2
        cx = x + BOX / 2
        box = f'<rect x="{x}" y="{y}" width="{BOX}" height="{BOX}" rx="10"'
        parts.append(f"""
<g font-family="sans-serif" text-anchor="middle"><title>{tip}</title>
  {box} fill="{COLORS[status]}" stroke="#555" stroke-width="2"/>
  {box + ' fill="url(#hatch)" opacity="0.35"/>' if shade else ''}
  <text x="{cx}" y="{y + 34}" font-size="20" fill="#222">{label}</text>
  <text x="{cx}" y="{y + 58}" font-size="15" fill="#444">{seats} seats</text>
  <text x="{cx}" y="{y + BOX + 15}" font-size="13" fill="#333">{booked.get(label, "")}</text>
</g>""")
    parts.append(LEGEND)
    return f'<svg viewBox="0 0 1000 970" width="560" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def save_floor_plan(path, booked=None, new=()):
    """Write the floor plan to an .svg file, e.g. to embed it in a markdown cell."""
    with open(path, "w") as f:
        f.write(restaurant_html(booked, new))