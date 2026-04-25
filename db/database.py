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
from datetime import datetime, timedelta
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


class SearchCache(Base):
    """
    Persistent cache for web search results.

    Each row stores a normalised query string → JSON-serialised result list,
    plus an *expires_at* timestamp so stale entries are never served.

    TTL strategy (configurable via SEARCH_CACHE_TTL_SECONDS env var):
      - Weather / price queries  → short TTL (default 10 min)
      - General queries          → longer TTL (default 1 hour)
    """
    __tablename__ = "search_cache"

    id          = Column(Integer, primary_key=True)
    query_key   = Column(String(512), unique=True, nullable=False, index=True)
    results_json = Column(Text, nullable=False)   # JSON list of {title, link, snippet}
    hit_count   = Column(Integer, default=0)      # how many times this cache entry was reused
    cached_at   = Column(DateTime, default=datetime.utcnow)
    expires_at  = Column(DateTime, nullable=False, index=True)


class KnowledgeEntry(Base):
    """
    Math and science knowledge base entries.
    Each entry has a topic, a question/trigger phrase, and a full explanation.
    Searched via keyword matching when math_query or science_query intent fires.
    """
    __tablename__ = "knowledge"

    id          = Column(Integer, primary_key=True)
    domain      = Column(String(32), index=True)   # "math" | "science"
    topic       = Column(String(128), index=True)  # e.g. "quadratic equation"
    keywords    = Column(Text)                     # space-separated trigger words
    explanation = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


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

def search_knowledge(db: Session, query: str, domain: str | None = None, top_k: int = 3) -> list["KnowledgeEntry"]:
    """
    TF-IDF-style keyword search over the knowledge base.
    Scores each entry by how many query words appear in its topic + keywords + explanation.
    Returns the top_k most relevant entries, optionally filtered by domain.
    """
    q_words = set(query.lower().split())
    entries = db.query(KnowledgeEntry)
    if domain:
        entries = entries.filter_by(domain=domain)
    entries = entries.all()

    scored = []
    for e in entries:
        haystack = f"{e.topic} {e.keywords} {e.explanation}".lower()
        score = sum(1 for w in q_words if w in haystack)
        if score > 0:
            scored.append((score, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]

def add_knowledge(db: Session, domain: str, topic: str, keywords: str, explanation: str) -> "KnowledgeEntry":
    entry = KnowledgeEntry(domain=domain, topic=topic, keywords=keywords, explanation=explanation)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


# ── Search Cache ──────────────────────────────────

# Default TTLs (override via env vars)
_DEFAULT_TTL_SHORT   = int(os.environ.get("SEARCH_CACHE_TTL_SHORT_SECONDS",   600))   # 10 min — weather, prices
_DEFAULT_TTL_GENERAL = int(os.environ.get("SEARCH_CACHE_TTL_SECONDS",         3600))  # 1 hour — general queries

# Keywords that indicate time-sensitive queries deserving a shorter TTL
_SHORT_TTL_KEYWORDS = {
    "weather", "temperature", "rain", "forecast", "humidity", "wind",
    "price", "stock", "crypto", "bitcoin", "ethereum", "rate", "exchange",
    "news", "breaking", "today", "tonight", "now", "live", "score", "result",
}


def _normalise_query(query: str) -> str:
    """Lowercase + collapse whitespace so minor phrasing differences share a cache entry."""
    return " ".join(query.lower().split())


def _ttl_for_query(query: str) -> int:
    """Return a TTL (seconds) appropriate for the query's volatility."""
    words = set(query.lower().split())
    if words & _SHORT_TTL_KEYWORDS:
        return _DEFAULT_TTL_SHORT
    return _DEFAULT_TTL_GENERAL


def get_cached_search(db: Session, query: str) -> list[dict] | None:
    """
    Return cached results for *query* if a fresh entry exists, else None.
    Increments the hit_count so you can see how effective caching is.
    """
    key = _normalise_query(query)
    entry = db.query(SearchCache).filter_by(query_key=key).first()
    if entry is None:
        return None
    if entry.expires_at < datetime.utcnow():
        # Stale — delete and signal a miss so the caller re-fetches
        db.delete(entry)
        db.commit()
        return None
    # Cache hit — bump the counter and return results
    entry.hit_count += 1
    db.commit()
    return json.loads(entry.results_json)


def set_cached_search(db: Session, query: str, results: list[dict]) -> None:
    """
    Persist *results* for *query*.  Upserts so repeated searches overwrite
    the old entry rather than creating duplicates.
    """
    if not results:
        return  # Don't cache empty result sets

    key      = _normalise_query(query)
    ttl      = _ttl_for_query(query)
    expires  = datetime.utcnow() + timedelta(seconds=ttl)
    payload  = json.dumps(results, ensure_ascii=False)

    entry = db.query(SearchCache).filter_by(query_key=key).first()
    if entry:
        entry.results_json = payload
        entry.cached_at    = datetime.utcnow()
        entry.expires_at   = expires
        # Reset hit count on refresh so stats reflect the new cache window
        entry.hit_count    = 0
    else:
        db.add(SearchCache(
            query_key=key,
            results_json=payload,
            expires_at=expires,
        ))
    db.commit()


def purge_expired_cache(db: Session) -> int:
    """
    Delete all expired cache rows.  Call this periodically (e.g. on bot start
    or in a background task) to keep the DB tidy.
    Returns the number of rows deleted.
    """
    deleted = (
        db.query(SearchCache)
        .filter(SearchCache.expires_at < datetime.utcnow())
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted


def get_cache_stats(db: Session) -> dict:
    """Return a quick summary of the cache table (useful for /admin or debugging)."""
    total   = db.query(SearchCache).count()
    live    = db.query(SearchCache).filter(SearchCache.expires_at >= datetime.utcnow()).count()
    expired = total - live
    top_hits = (
        db.query(SearchCache)
        .filter(SearchCache.expires_at >= datetime.utcnow())
        .order_by(SearchCache.hit_count.desc())
        .limit(5)
        .all()
    )
    return {
        "total_entries":   total,
        "live_entries":    live,
        "expired_entries": expired,
        "top_queries": [
            {"query": e.query_key, "hits": e.hit_count}
            for e in top_hits
        ],
    }