import os
import sqlite3

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