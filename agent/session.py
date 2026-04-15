"""
session.py — Isolated per-user session state.

Each user (identified by an encrypted user_id) gets their own SessionState.
Sessions expire after SESSION_TTL_SECONDS of inactivity to free memory.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SESSION_TTL_SECONDS, RIASEC_MAX_QUESTIONS


@dataclass
class SessionState:
    user_id: str                                # encrypted identifier
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # RIASEC survey progress
    riasec_answers: list[str] = field(default_factory=list)   # e.g. ["R", "I", "S"]
    riasec_step: int = 0                                       # 0 = not started, 1-5 = in progress
    riasec_result: Optional[str] = None                        # final top-2 types, e.g. "IE"

    # ORT
    ort_score: Optional[int] = None
    ort_program: Optional[str] = None

    # Conversation
    current_topic: Optional[str] = None
    history: list[dict] = field(default_factory=list)          # [{role, content}, …]

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.touch()

    def riasec_in_progress(self) -> bool:
        return 0 < self.riasec_step <= RIASEC_MAX_QUESTIONS

    def riasec_complete(self) -> bool:
        return self.riasec_step >= RIASEC_MAX_QUESTIONS

    def to_summary_dict(self) -> dict:
        """Structured state used by Human_Handoff_Trigger — never summarized by LLM."""
        return {
            "user_id": self.user_id,
            "session_duration_min": round((time.time() - self.created_at) / 60, 1),
            "messages_exchanged": len(self.history),
            "ort_score": self.ort_score,
            "ort_program": self.ort_program,
            "riasec_result": self.riasec_result,
            "riasec_step": self.riasec_step,
            "current_topic": self.current_topic,
        }


# ── Session registry ───────────────────────────────────────────────────────────

_sessions: dict[str, SessionState] = {}


def _encrypt_id(raw_id: str) -> str:
    """One-way hash of raw user identifier (Telegram ID, browser fingerprint, etc.)."""
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


def get_session(raw_user_id: str) -> SessionState:
    """Return existing session or create a new one for this user."""
    uid = _encrypt_id(raw_user_id)
    _purge_expired()
    if uid not in _sessions:
        _sessions[uid] = SessionState(user_id=uid)
    else:
        _sessions[uid].touch()
    return _sessions[uid]


def clear_session(raw_user_id: str) -> None:
    uid = _encrypt_id(raw_user_id)
    _sessions.pop(uid, None)


def _purge_expired() -> None:
    expired = [uid for uid, s in _sessions.items() if s.is_expired()]
    for uid in expired:
        del _sessions[uid]
