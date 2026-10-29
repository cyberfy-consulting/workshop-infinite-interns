import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "booking.db")

# (outside, waterfront, shade, seats, count)
# waterfront/shade only apply to outside tables, so they stay None for inside tables.
TABLE_TYPES = [
    (False, None,  None,  2, 9),   # inside,  9 tables, 2 seats each
    (False, None,  None,  4, 4),   # inside,  4 tables, 4 seats each
    (True,  False, False, 4, 1),   # outside, no waterfront, no shade
    (True,  False, True,  2, 1),   # outside, no waterfront, shaded
    (True,  False, True,  4, 2),   # outside, no waterfront, shaded
    (True,  True,  False, 4, 1),   # outside, waterfront, no shade
    (True,  True,  True,  4, 1),   # outside, waterfront, shaded
    (True,  True,  True,  2, 4),   # outside, waterfront, shaded
]


def _table_rows():
    """Expand TABLE_TYPES into one row per physical table, with a unique label."""
    rows = []
    inside_idx = outside_idx = 1
    for outside, waterfront, shade, seats, count in TABLE_TYPES:
        for _ in range(count):
            if outside:
                label = f"TO{outside_idx}"
                outside_idx += 1
            else:
                label = f"TI{inside_idx}"
                inside_idx += 1
            rows.append((label, outside, waterfront, shade, seats))
    return rows


# --- One-time DB setup ---
if not os.path.exists("utils/booking.db"):
    conn = sqlite3.connect("utils/booking.db")
    cur = conn.cursor()

    # Create a simple reservations table: name, reservation_time, party_size
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reservations (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        reservation_time DATETIME NOT NULL,
        party_size INTEGER NOT NULL,
        outside BOOLEAN DEFAULT FALSE
    )
    """)

    # Sample reservation entries
    cur.executemany(
        "INSERT INTO reservations (name, reservation_time, party_size, outside) VALUES (?, ?, ?, ?)",
        [
            ("Alice", "2026-06-27 18:30", 2, False),
            ("Bob",   "2026-06-27 19:00", 4, True),
            ("Carol", "2026-06-28 12:30", 3, False),
            ("Dave",  "2026-06-26 20:00", 5, True),
        ]
    )

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# -------------------------