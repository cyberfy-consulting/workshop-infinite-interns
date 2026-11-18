import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime, time

DB_PATH = os.path.join(os.path.dirname(__file__), "booking_system.db")

TODAY = date(2026, 5, 29)   # the fixed "today" of the demo. Summer day to test differen scenarios
DEMO_DAY = date(2026, 11, 19)
OPENING_TIME = "18:00"
CLOSING_TIME = "22:00"

RESERVATION_COLUMNS = ("reservation_id", "name", "reservation_time", "party_size", "outside", "waterfront", "shade", "table_id")
TABLE_COLUMNS = ("label", "outside", "waterfront", "shade", "seats")

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


def fetch_reservations(reservation_date: date | str | None = TODAY, db_path: str = DB_PATH) -> list[tuple]:
    """Reservations of one date (all dates for None), as rows in RESERVATION_COLUMNS order."""
    query = f"SELECT {', '.join(RESERVATION_COLUMNS)} FROM reservations"
    params: tuple = ()
    if reservation_date is not None:
        query += " WHERE date(reservation_time) = date(?)"
        params = (str(reservation_date),)
    with booking_connection(db_path) as conn:
        return conn.execute(query + " ORDER BY reservation_id", params).fetchall()


def fetch_tables(db_path: str = DB_PATH) -> list[tuple]:
    """All tables of the restaurant, as rows in TABLE_COLUMNS order."""
    with booking_connection(db_path) as conn:
        return conn.execute(f"SELECT {', '.join(TABLE_COLUMNS)} FROM tables ORDER BY id").fetchall()


def fetch_free_tables(reservation_date: date | str, db_path: str = DB_PATH) -> list[tuple]:
    """Tables without a reservation on that date, as rows in TABLE_COLUMNS order.

    A table with any reservation that day counts as taken for the whole evening.
    """
    with booking_connection(db_path) as conn:
        return conn.execute(
            f"SELECT {', '.join(TABLE_COLUMNS)} FROM tables "
            "WHERE label NOT IN (SELECT table_id FROM reservations WHERE table_id IS NOT NULL AND date(reservation_time) = date(?)) "
            "ORDER BY id",
            (str(reservation_date),),
        ).fetchall()


def add_reservation(
    name: str,
    reservation_time: datetime,
    party_size: int,
    table_id: str,
    outside: bool = False,
    waterfront: bool = False,
    shade: bool = False,
    db_path: str = DB_PATH,
) -> str:
    """Add a reservation and describe the booked table next to the guest's preferences."""
    if not (time.fromisoformat(OPENING_TIME) <= reservation_time.time() <= time.fromisoformat(CLOSING_TIME)):
        return (f"Reservation request is outside of opening hours. "
                f"Please reserve a table between {OPENING_TIME} and {CLOSING_TIME}.")

    with booking_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM reservations WHERE table_id = ? AND date(reservation_time) = date(?)",
            (table_id, reservation_time.strftime("%Y-%m-%d")),
        )
        if cur.fetchone():
            return f"Table {table_id} is already booked on {reservation_time:%Y-%m-%d}. Please choose another table."

        cur.execute(
            "INSERT INTO reservations (name, reservation_time, party_size, outside, waterfront, shade, table_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, reservation_time, party_size, outside, waterfront, shade, table_id),
        )
        conn.commit()

        # What the table really offers, as opposed to what the guest asked for
        cur.execute("SELECT outside, waterfront, shade, seats FROM tables WHERE label = ?", (table_id,))
        table = cur.fetchone()

    preferences = "outside" if outside else "inside"
    if outside:
        preferences += f", {'waterfront' if waterfront else 'no waterfront'}, {'shade' if shade else 'no shade'}"

    where = f"table {table_id}"
    if table:
        table_outside, table_waterfront, table_shade, seats = table
        where += f" ({seats} seats, {'outside' if table_outside else 'inside'}"
        if table_outside:
            where += f", {'waterfront' if table_waterfront else 'no waterfront'}, {'shaded' if table_shade else 'no shade'}"
        where += ")"

    return (
        f"Reservation for {name} at {reservation_time} for {party_size} {'people' if party_size > 1 else 'person'}"
        f" added at {where}. Guest preferences: {preferences}."
    )


def delete_reservation(name: str, reservation_time: datetime, db_path: str = DB_PATH) -> str:
    """Delete the reservation of a guest at a given time."""
    with booking_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM reservations WHERE name = ? AND datetime(reservation_time) = datetime(?)",
            (name, reservation_time),
        )
        if cur.rowcount == 0:
            return f"No reservation found for {name} at {reservation_time}."
        conn.commit()
    return f"Reservation for {name} at {reservation_time} deleted."



if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        initialize_tables()
        initialize_reservations()
    else:
        print("Nothing to do. Use --rebuild to recreate the tables and reservations.")