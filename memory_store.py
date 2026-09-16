"""SQLite 持久化会话记忆；只保存少量结构化用户状态。"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStoreError(RuntimeError):
    """SQLite 不可用时由 Agent 捕获并优雅降级。"""


class MemoryManager:
    """单表、按 session 隔离的轻量 Memory Manager。"""

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        path = Path(db_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        self.db_path = path.resolve()

    def _connect(self) -> sqlite3.Connection:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.db_path)
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    city TEXT,
                    budget TEXT,
                    purchase_timing TEXT,
                    interested_model TEXT,
                    loan_preference TEXT,
                    vision_summary TEXT,
                    recent_turns TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            return connection
        except sqlite3.Error as error:
            raise MemoryStoreError(f"SQLite 初始化失败：{error}") from error
        except OSError as error:
            raise MemoryStoreError(f"SQLite 文件不可用：{error}") from error

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def memory_exists(self, session_id: str) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is not None
        except sqlite3.Error as error:
            raise MemoryStoreError(f"SQLite 查询失败：{error}") from error

    def load_memory(self, session_id: str) -> dict[str, Any]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT city, budget, purchase_timing, interested_model, loan_preference, vision_summary, recent_turns FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise MemoryStoreError(f"SQLite 读取失败：{error}") from error
        if row is None:
            return {"profile": {}, "recent_turns": [], "vision_summary": None}
        city, budget, purchase_timing, interested_model, loan_preference, vision_summary, recent_turns = row
        profile = {
            key: value for key, value in {
                "city": city,
                "budget": budget,
                "purchase_timeline": purchase_timing,
                "interested_model": interested_model,
                "loan_preference": loan_preference,
            }.items() if value is not None
        }
        try:
            turns = json.loads(recent_turns or "[]")
            turns = turns if isinstance(turns, list) else []
        except json.JSONDecodeError:
            turns = []
        return {"profile": profile, "recent_turns": turns[-10:], "vision_summary": vision_summary}

    def save_memory(self, session_id: str, state: Any) -> None:
        profile = state.profile
        now = self._now()
        values = (
            session_id,
            profile.get("city"),
            profile.get("budget"),
            profile.get("purchase_timeline"),
            profile.get("interested_model"),
            profile.get("loan_preference"),
            getattr(state, "vision_summary", None),
            json.dumps(list(state.recent_turns)[-10:], ensure_ascii=False),
            now,
            now,
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO sessions (
                        session_id, city, budget, purchase_timing, interested_model,
                        loan_preference, vision_summary, recent_turns, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        city=excluded.city, budget=excluded.budget,
                        purchase_timing=excluded.purchase_timing,
                        interested_model=excluded.interested_model,
                        loan_preference=excluded.loan_preference,
                        vision_summary=excluded.vision_summary,
                        recent_turns=excluded.recent_turns,
                        updated_at=excluded.updated_at""",
                    values,
                )
        except sqlite3.Error as error:
            raise MemoryStoreError(f"SQLite 写入失败：{error}") from error

    def clear_memory(self, session_id: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        except sqlite3.Error as error:
            raise MemoryStoreError(f"SQLite 清理失败：{error}") from error
