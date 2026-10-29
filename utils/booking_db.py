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

def create_tables_table(db_path: str = DB_PATH, rebuild: bool = False):
    """Create the `tables` table from TABLE_TYPES. rebuild=True builds table from scratch to avoid duplicates"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if rebuild:
        cur.execute("DROP TABLE IF EXISTS tables")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tables (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL UNIQUE,
        outside BOOLEAN NOT NULL DEFAULT FALSE,
        waterfront BOOLEAN,
        shade BOOLEAN,
        seats INTEGER NOT NULL,
        CHECK (outside = 1 OR (waterfront IS NULL AND shade IS NULL))
    )
    """)

    table_rows = _table_rows()
    if table_rows:
        cur.executemany(
            "INSERT INTO tables (label, outside, waterfront, shade, seats) VALUES (?, ?, ?, ?, ?)",
            table_rows,
        )

    conn.commit()
    conn.close()
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if rebuild:
        cur.execute("DROP TABLE IF EXISTS tables")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tables (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL UNIQUE,
        outside BOOLEAN NOT NULL DEFAULT FALSE,
        waterfront BOOLEAN,
        shade BOOLEAN,
        seats INTEGER NOT NULL,
        CHECK (outside = 1 OR (waterfront IS NULL AND shade IS NULL))
    )
    """)

    table_rows = _table_rows()
    if table_rows:
        cur.executemany(
            "INSERT INTO tables (label, outside, waterfront, shade, seats) VALUES (?, ?, ?, ?, ?)",
            table_rows,
        )

    conn.commit()
    conn.close()


# --- One-time DB setup ---
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
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

SEED_RESERVATIONS = [
    ("Alice", "2026-06-27 18:30", 2, False),
    ("Bob",   "2026-06-27 19:00", 4, True),
    ("Carol", "2026-06-28 12:30", 3, False),
    ("Dave",  "2026-06-26 20:00", 5, True),
]


def reset_reservations(db_path: str = DB_PATH):
    """Drop and recreate the reservations table, with only the sample entries."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS reservations")
    cur.execute("""
    CREATE TABLE reservations (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        reservation_time DATETIME NOT NULL,
        party_size INTEGER NOT NULL,
        outside BOOLEAN DEFAULT FALSE
    )
    """)
    cur.executemany(
        "INSERT INTO reservations (name, reservation_time, party_size, outside) VALUES (?, ?, ?, ?)",
        SEED_RESERVATIONS,
    )
    conn.commit()
    conn.close()


# --- One-time DB setup ---
if not os.path.exists(DB_PATH):
    reset_reservations()

# -------------------------

if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        create_tables_table(rebuild=True)
        reset_reservations()
    else:
        print("Nothing to do. Use --rebuild to recreate the tables and reservations.")