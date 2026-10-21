import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.schemas import ChatMessage


class SessionStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True) if str(Path(path).parent) != "." else None
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, title TEXT NOT NULL DEFAULT 'New session', messages TEXT NOT NULL)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            if "title" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN title TEXT NOT NULL DEFAULT 'New session'")

    def create(self, title: str = "New session") -> str:
        sid = str(uuid4())
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO sessions (id, created_at, title, messages) VALUES (?, ?, ?, ?)", (sid, datetime.now(UTC).isoformat(), title, "[]"))
        return sid

    def set_title(self, sid: str, title: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))

    def delete(self, sid: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM sessions WHERE id=?", (sid,))

    @staticmethod
    def _legacy_title(messages: str) -> str:
        try:
            entries = json.loads(messages)
        except json.JSONDecodeError:
            return "New conversation"
        first_user = next((entry.get("content", "") for entry in entries if entry.get("role") == "user"), "")
        words = re.findall(r"\w[\w'/-]*", first_user, re.UNICODE)
        if len(words) >= 3:
            return " ".join(words[:5]).title()
        return "New conversation"

    def list_sessions(self):
        from app.schemas import SessionSummary

        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, created_at, title, messages FROM sessions ORDER BY created_at DESC").fetchall()
        summaries = []
        for sid, created_at, title, messages in rows:
            parsed = json.loads(messages)
            if not parsed:
                self.delete(sid)
                continue
            if title == "New session":
                title = self._legacy_title(messages)
                self.set_title(sid, title)
            summaries.append(SessionSummary(
                id=sid,
                created_at=datetime.fromisoformat(created_at),
                title=title,
                message_count=len(parsed),
            ))
        return summaries

    def messages(self, sid: str) -> list[ChatMessage]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT messages FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row: return []
        return [ChatMessage.model_validate(x) for x in json.loads(row[0])]

    def append(self, sid: str, message: ChatMessage) -> None:
        items = self.messages(sid) + [message]
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE sessions SET messages=? WHERE id=?", (json.dumps([x.model_dump() for x in items]), sid))

    def get(self, sid: str):
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT id, created_at, title, messages FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            return None
        from app.schemas import Session
        title = row[2] if row[2] != "New session" else self._legacy_title(row[3])
        return Session(id=row[0], created_at=datetime.fromisoformat(row[1]), title=title, messages=[ChatMessage.model_validate(x) for x in json.loads(row[3])])

    def exists(self, sid: str) -> bool:
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone() is not None
