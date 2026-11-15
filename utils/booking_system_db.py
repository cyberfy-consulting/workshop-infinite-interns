import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date

DB_PATH = os.path.join(os.path.dirname(__file__), "booking_system.db")

TODAY = date(2026, 5, 29)   # the fixed "today" of the demo. Summer day to test differen scenarios
DEMO_DAY = date(2026, 11, 19)
OPENING_TIME = "18:00"
CLOSING_TIME = "22:00"

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


SEED_RESERVATIONS = [
        ("Alice", f"{TODAY} 20:30", 2, False, False, False, "TI-1"),
        ("Bob",   f"{TODAY} 18:00", 4, True, True, True, "TO-6"),
        ("Carol", f"{TODAY} 19:30", 3, False, False, False, "TI-10"),
        ("Dave",  f"{TODAY} 21:00", 4, True, True, False, "TO-5"),
        ("Alice", f"{DEMO_DAY} 19:00", 2, False, False, False, "TI-1"),
        ("Bob",   f"{DEMO_DAY} 19:00", 4, False, False, False, "TI-10"),
        ("Carol", f"{DEMO_DAY} 20:00", 4, False, False, False, "TI-11"),
        ("Dave",  f"{DEMO_DAY} 20:00", 4, False, False, False, "TI-12"),
        ("Frank", f"{DEMO_DAY} 21:00", 2, False, False, False, "TI-2"),
]


@contextmanager
def booking_connection(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Open a short-lived connection to the booking database and always close it."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_reservation_summary(reservation_date: str, db_path: str = DB_PATH) -> tuple[int, int]:
    """Return the booked guest count and reservation count for a date."""
    with booking_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(party_size), 0), COUNT(*) FROM reservations "
            "WHERE date(reservation_time) = ?",
            (reservation_date,),
        ).fetchone()
    if row is None:
        raise RuntimeError("The reservation summary query returned no result.")
    return int(row[0]), int(row[1])


def _expand_table_types():
    """Expand TABLE_TYPES into one row per physical table, with a unique label."""
    rows = []
    inside_idx = outside_idx = 1
    for outside, waterfront, shade, seats, count in TABLE_TYPES:
        for _ in range(count):
            if outside:
                label = f"TO-{outside_idx}"
                outside_idx += 1
            else:
                label = f"TI-{inside_idx}"
                inside_idx += 1
            rows.append((label, outside, waterfront, shade, seats))
    return rows

def initialize_tables(db_path: str = DB_PATH):
    """Recreate the tables table from the static TABLE_TYPES configuration."""
    with booking_connection(db_path) as conn:
        cur = conn.cursor()
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

        table_rows = _expand_table_types()
        if table_rows:
            cur.executemany(
                "INSERT INTO tables (label, outside, waterfront, shade, seats) VALUES (?, ?, ?, ?, ?)",
                table_rows,
            )

        conn.commit()


def initialize_reservations(db_path: str = DB_PATH):
    """Recreate the reservations table with the sample reservations."""
    with booking_connection(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS reservations")
        conn.execute("""
        CREATE TABLE reservations (
            reservation_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            reservation_time DATETIME NOT NULL,
            party_size INTEGER NOT NULL,
            outside BOOLEAN DEFAULT FALSE,
            waterfront BOOLEAN DEFAULT FALSE,
            shade BOOLEAN DEFAULT FALSE,
            table_id STRING
        )
        """)
        conn.executemany(
            "INSERT INTO reservations (name, reservation_time, party_size, outside, waterfront, shade, table_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            SEED_RESERVATIONS,
        )
        conn.commit()


if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        initialize_tables()
        initialize_reservations()
    else:
        print("Nothing to do. Use --rebuild to recreate the tables and reservations.")