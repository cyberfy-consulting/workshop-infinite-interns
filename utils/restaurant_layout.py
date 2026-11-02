import sqlite3

from utils.booking_db import DB_PATH

STYLE = """
<style>
.plan    { display: flex; gap: 16px; font-family: sans-serif; }
.area    { background: #f3f4f6; border-radius: 12px; padding: 14px; }
.water   { background: #dbeafe; }
.area h4 { margin: 0 0 10px; color: #374151; }
.tables  { display: flex; flex-wrap: wrap; gap: 8px; width: 150px; }
.table   { width: 66px; padding: 8px 0; border-radius: 8px; text-align: center; color: #1f2937;
           border: 4px solid transparent; box-sizing: border-box; }
.free    { background: #bbf7d0; }
.booked  { background: #fecaca; }
.new     { background: #fde68a; }
.legend  { margin-top: 12px; font-family: sans-serif; font-size: 14px; color: #374151; }
.swatch  { display: inline-block; width: 16px; height: 16px; margin: 0 4px 0 14px; vertical-align: middle;
           border: 3px solid transparent; border-radius: 4px; box-sizing: border-box; }
.shaded  { border-color: black; }
</style>
"""


def restaurant_html(booked=(), new=()):
    """Floor plan from the `tables` table. Tables whose label is in `booked` turn red,
    new reservations yellow, shaded tables get a black border."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT label, outside, waterfront, shade, seats FROM tables ORDER BY seats, id").fetchall()
    conn.close()

    def table_box(label, seats, shade):
        status = "new" if label in new else "booked" if label in booked else "free"
        shaded = " shaded" if shade else ""
        return f'<div class="table {status}{shaded}">{label}<br><small>{seats} seats</small></div>'

    def area(title, tables, extra_class=""):
        boxes = "".join(table_box(label, seats, shade) for label, seats, shade in tables)
        return f'<div class="area {extra_class}"><h4>{title}</h4><div class="tables">{boxes}</div></div>'

    inside     = [(label, seats, shade) for label, outside, waterfront, shade, seats in rows if not outside]
    outside    = [(label, seats, shade) for label, outside, waterfront, shade, seats in rows if outside and not waterfront]
    waterfront = [(label, seats, shade) for label, outside, waterfront, shade, seats in rows if outside and waterfront]
    legend = ('<div class="legend">'
              '<span class="swatch free"></span>free'
              '<span class="swatch booked"></span>booked'
              '<span class="swatch new"></span>new'
              '<span class="swatch free shaded"></span>shaded'
              '</div>')
    return (STYLE + '<div class="plan">'
            + area("Inside", inside)
            + area("Outside", outside)
            + area("Waterfront", waterfront, "water")
            + "</div>"
            + legend)