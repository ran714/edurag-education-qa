"""Small persistent conversation store used by the dependency-free demo."""

import sqlite3
from pathlib import Path
from threading import RLock


class ConversationStore:
    def __init__(self, path, max_turns=5):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = RLock()
        self.max_turns = max_turns
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversations(session_id, id)"
            )

    def get(self, session_id):
        with self.lock:
            rows = self.connection.execute(
                "SELECT question, answer FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, self.max_turns),
            ).fetchall()
            return [{"question": q, "answer": a} for q, a in reversed(rows)]

    def append(self, session_id, question, answer):
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO conversations(session_id, question, answer) VALUES (?, ?, ?)",
                (session_id, question, answer),
            )
            self.connection.execute(
                "DELETE FROM conversations WHERE session_id=? AND id NOT IN (SELECT id FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, self.max_turns),
            )

    def clear(self, session_id):
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        return True

    def close(self):
        with self.lock:
            self.connection.close()
