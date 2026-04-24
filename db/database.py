"""
AATAS — Database Layer
SQLite via SQLAlchemy for:
  - Multi-user management & Gmail OAuth tokens
  - Persistent automation rules (e.g. "archive anything about Google Classroom")
  - Conversation memory per user
  - Action history / audit log
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DB_PATH = os.environ.get("DB_PATH", "data/aatas.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

# Enable WAL for concurrent reads
@event.listens_for(engine, "connect")
def set_wal(conn, _):
    conn.execute("PRAGMA journal_mode=WAL")

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ── Models ────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True)
    telegram_id     = Column(Integer, unique=True, nullable=False, index=True)
    telegram_name   = Column(String(128))
    gmail_email     = Column(String(256))
    gmail_token_json = Column(Text)          # serialised Credentials JSON
    setup_complete  = Column(Boolean, default=False)
    setup_step      = Column(String(64), default="start")  # onboarding FSM state
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_active     = Column(DateTime, default=datetime.utcnow)

    rules           = relationship("AutomationRule", back_populates="user", cascade="all, delete")
    memories        = relationship("Memory", back_populates="user", cascade="all, delete")
    history         = relationship("ActionLog", back_populates="user", cascade="all, delete")
    conv_turns      = relationship("ConversationTurn", back_populates="user", cascade="all, delete")


class AutomationRule(Base):
    """
    Natural-language rules the user sets via chat.
    e.g. "archive anything about Google Classroom"
         "label emails from boss@company.com as urgent"
         "auto-reply to newsletters with unsubscribe"
    """
    __tablename__ = "automation_rules"

    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    rule_text   = Column(Text, nullable=False)       # original NL instruction
    action      = Column(String(64))                 # archive | label | reply | forward | delete
    target_json = Column(Text, default="{}")         # {"keywords": [...], "from": "...", "label": "..."}
    enabled     = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    trigger_count = Column(Integer, default=0)

    user        = relationship("User", back_populates="rules")

    def get_target(self) -> dict:
        return json.loads(self.target_json or "{}")

    def set_target(self, d: dict):
        self.target_json = json.dumps(d)


class Memory(Base):
    """
    Key facts AATAS remembers about a user or their preferences.
    e.g. "user is a student at NUS", "user's boss is called John"
    """
    __tablename__ = "memories"

    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    key         = Column(String(128))
    value       = Column(Text)
    created_at  = Column(DateTime, default=datetime.utcnow)

    user        = relationship("User", back_populates="memories")


class ConversationTurn(Base):
    """Rolling conversation history for multi-turn chat context."""
    __tablename__ = "conversation_turns"

    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    role        = Column(String(16))        # "user" | "assistant"
    content     = Column(Text)
    created_at  = Column(DateTime, default=datetime.utcnow)

    user        = relationship("User", back_populates="conv_turns")


class ActionLog(Base):
    """Audit trail of every action AATAS takes on emails."""
    __tablename__ = "action_log"

    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    action      = Column(String(64))
    email_id    = Column(String(128))
    email_subject = Column(Text)
    detail      = Column(Text)
    created_at  = Column(DateTime, default=datetime.utcnow)

    user        = relationship("User", back_populates="history")


# ── Create tables ─────────────────────────────────
Base.metadata.create_all(engine)


# ── Repository helpers ────────────────────────────

def get_db() -> Session:
    return SessionLocal()

def get_or_create_user(db: Session, telegram_id: int, name: str = "") -> User:
    user = db.query(User).filter_by(telegram_id=telegram_id).first()
    if not user:
        user = User(telegram_id=telegram_id, telegram_name=name)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.last_active = datetime.utcnow()
        user.telegram_name = name or user.telegram_name
        db.commit()
    return user

def save_conv_turn(db: Session, user_id: int, role: str, content: str, keep_last: int = 20):
    db.add(ConversationTurn(user_id=user_id, role=role, content=content))
    db.commit()
    # trim old turns
    turns = (
        db.query(ConversationTurn)
        .filter_by(user_id=user_id)
        .order_by(ConversationTurn.id.desc())
        .all()
    )
    if len(turns) > keep_last:
        for old in turns[keep_last:]:
            db.delete(old)
        db.commit()

def get_conv_history(db: Session, user_id: int, last_n: int = 10) -> list[dict]:
    turns = (
        db.query(ConversationTurn)
        .filter_by(user_id=user_id)
        .order_by(ConversationTurn.id.asc())
        .all()
    )
    return [{"role": t.role, "content": t.content} for t in turns[-last_n:]]

def add_rule(db: Session, user_id: int, rule_text: str, action: str, target: dict) -> AutomationRule:
    rule = AutomationRule(user_id=user_id, rule_text=rule_text, action=action)
    rule.set_target(target)
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule

def get_rules(db: Session, user_id: int) -> list[AutomationRule]:
    return db.query(AutomationRule).filter_by(user_id=user_id, enabled=True).all()

def log_action(db: Session, user_id: int, action: str, email_id: str, subject: str, detail: str = ""):
    db.add(ActionLog(
        user_id=user_id, action=action,
        email_id=email_id, email_subject=subject, detail=detail
    ))
    db.commit()

def upsert_memory(db: Session, user_id: int, key: str, value: str):
    mem = db.query(Memory).filter_by(user_id=user_id, key=key).first()
    if mem:
        mem.value = value
    else:
        db.add(Memory(user_id=user_id, key=key, value=value))
    db.commit()

def get_memories(db: Session, user_id: int) -> dict[str, str]:
    mems = db.query(Memory).filter_by(user_id=user_id).all()
    return {m.key: m.value for m in mems}
