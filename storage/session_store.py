# storage/session_store.py
# Simple SQLite-backed session store for persistence across notebooks.
import sqlite3
from contextlib import closing
import json
from pathlib import Path
from typing import Dict, Any

DB_PATH = Path("data/sessions.db")
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            student_profile TEXT,
            memory TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

def save_session(session_id: str, profile: Dict[str,Any], memory: Dict[str,Any]):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()
        c.execute("""
        INSERT INTO sessions(session_id, student_profile, memory)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          student_profile=excluded.student_profile,
          memory=excluded.memory,
          last_updated=CURRENT_TIMESTAMP
        """, (session_id, json.dumps(profile), json.dumps(memory)))
        conn.commit()

def load_session(session_id: str) -> Dict[str,Any]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()
        c.execute("SELECT student_profile, memory FROM sessions WHERE session_id=?", (session_id,))
        row = c.fetchone()
        if not row:
            return {"profile": {}, "memory": {}}
        profile, memory = row
        return {"profile": json.loads(profile), "memory": json.loads(memory)}

# initialize DB on import
init_db()
