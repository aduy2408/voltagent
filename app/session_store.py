import json
import sqlite3
from datetime import UTC
from pathlib import Path
from uuid import uuid4

from app.schemas import ChatMessage


class SessionStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True) if str(Path(path).parent) != "." else None
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, messages TEXT NOT NULL)")

    def create(self) -> str:
        sid = str(uuid4())
        from datetime import datetime
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO sessions VALUES (?, ?, ?)", (sid, datetime.now(UTC).isoformat(), "[]"))
        return sid

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
            row = db.execute("SELECT id, created_at, messages FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            return None
        from datetime import datetime

        from app.schemas import Session
        return Session(id=row[0], created_at=datetime.fromisoformat(row[1]), messages=[ChatMessage.model_validate(x) for x in json.loads(row[2])])

    def exists(self, sid: str) -> bool:
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone() is not None
