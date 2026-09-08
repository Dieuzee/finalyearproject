"""
Lightweight SQLite storage for BID CROP.

Every diagnosis is saved as one row in the ``scans`` table so the app can show
a history of past scans and simple statistics. Uses Python's built-in
``sqlite3`` module, so there is no extra dependency to install and the whole
database lives in a single file (``scans.db``) next to the app.
"""

import os
import sqlite3
from datetime import datetime

# The database is a single file in the project directory.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scans.db")


def get_connection():
    """Open a connection whose rows behave like dictionaries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the ``scans`` table on first run (safe to call every startup)."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT    NOT NULL,
                status         TEXT    NOT NULL,   -- ok / not_leaf
                crop           TEXT,
                disease        TEXT,
                healthy        INTEGER,            -- 0 / 1
                confidence     INTEGER,            -- 0-100
                severity       TEXT,
                summary        TEXT,
                treatment      TEXT,
                symptoms       TEXT,
                prevention     TEXT,
                model_used     TEXT,
                created_at     TEXT    NOT NULL
            )
            """
        )


def save_scan(image_filename, result, model_used=""):
    """Insert one diagnosis. Returns the new row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO scans (
                image_filename, status, crop, disease, healthy, confidence,
                severity, summary, treatment, symptoms, prevention,
                model_used, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_filename,
                result.get("status"),
                result.get("crop"),
                result.get("disease"),
                1 if result.get("healthy") else 0,
                result.get("confidence"),
                result.get("severity"),
                # 'ok' results carry a summary; a rejected 'not_leaf' carries a message.
                result.get("summary") or result.get("message"),
                result.get("treatment"),
                result.get("symptoms"),
                result.get("prevention"),
                model_used,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        return cur.lastrowid


def get_all_scans(limit=200):
    """Return recent scans, newest first, as a list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY datetime(created_at) DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_scan(scan_id):
    """Return a single scan by id, or None if it does not exist."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return dict(row) if row else None


def delete_scan(scan_id):
    """Delete one scan by id. Returns the number of rows removed (0 or 1)."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        return cur.rowcount


def clear_scans():
    """Delete every scan. Returns the number of rows removed."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM scans")
        return cur.rowcount


# Sort options exposed to the history page: label -> SQL ORDER BY clause.
_SORTS = {
    "newest": "datetime(created_at) DESC, id DESC",
    "oldest": "datetime(created_at) ASC, id ASC",
    "confidence": "confidence DESC, id DESC",
}


def query_scans(search="", condition="all", sort="newest", limit=12, offset=0):
    """
    Search, filter, sort and paginate scans.

    - search:    matches crop / disease / summary (case-insensitive)
    - condition: 'all' | 'diseased' | 'healthy' | 'rejected'
    - sort:      'newest' | 'oldest' | 'confidence'
    Returns (rows, total_matching_count).
    """
    where, params = [], []

    if search:
        where.append("(crop LIKE ? OR disease LIKE ? OR summary LIKE ?)")
        like = f"%{search}%"
        params += [like, like, like]

    if condition == "diseased":
        where.append("status = 'ok' AND healthy = 0")
    elif condition == "healthy":
        where.append("status = 'ok' AND healthy = 1")
    elif condition == "rejected":
        where.append("status = 'not_leaf'")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    order_sql = _SORTS.get(sort, _SORTS["newest"])

    with get_connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM scans {where_sql}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM scans {where_sql} ORDER BY {order_sql} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    return [dict(row) for row in rows], total


def get_stats():
    """Return simple headline numbers for the history page."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        diseased = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE status = 'ok' AND healthy = 0"
        ).fetchone()[0]
        healthy = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE status = 'ok' AND healthy = 1"
        ).fetchone()[0]
        top = conn.execute(
            """
            SELECT disease, COUNT(*) AS n
            FROM scans
            WHERE status = 'ok' AND healthy = 0 AND disease IS NOT NULL AND disease != ''
            GROUP BY disease
            ORDER BY n DESC
            LIMIT 1
            """
        ).fetchone()
    return {
        "total": total,
        "diseased": diseased,
        "healthy": healthy,
        "top_disease": top["disease"] if top else None,
    }
