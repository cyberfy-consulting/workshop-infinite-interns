import sqlite3

from utils.booking_db import DB_PATH

STYLE = """
<style>
.plan    { display: flex; gap: 16px; font-family: sans-serif; }
.area    { background: #f3f4f6; border-radius: 12px; padding: 14px; }
.water   { background: #dbeafe; }
.area h4 { margin: 0 0 10px; color: #374151; }
.tables  { display: flex; flex-wrap: wrap; gap: 8px; width: 150px; }
.table   { width: 66px; padding: 8px 0; border-radius: 8px; text-align: center; color: #1f2937; }
.free    { background: #bbf7d0; }
.booked  { background: #fecaca; }
</style>
"""


def restaurant_html(booked=()):
    """Floor plan from the `tables` table. Tables whose label is in `booked` turn red."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT label, outside, waterfront, seats FROM tables ORDER BY seats, id").fetchall()
    conn.close()

    def table_box(label, seats):
        status = "booked" if label in booked else "free"
        return f'<div class="table {status}">{label}<br><small>{seats} seats</small></div>'

    def area(title, tables, extra_class=""):
        boxes = "".join(table_box(label, seats) for label, seats in tables)
        return f'<div class="area {extra_class}"><h4>{title}</h4><div class="tables">{boxes}</div></div>'

    inside     = [(label, seats) for label, outside, waterfront, seats in rows if not outside]
    outside    = [(label, seats) for label, outside, waterfront, seats in rows if outside and not waterfront]
    waterfront = [(label, seats) for label, outside, waterfront, seats in rows if outside and waterfront]

    return (STYLE + '<div class="plan">'
            + area("Inside", inside)
            + area("Outside", outside)
            + area("Waterfront", waterfront, "water")
            + "</div>")