"""
AATAS — An AI-Assisted Technological Arrangement System
Telegram Bot — Main Entry Point

Features:
  - Multi-user with self-service Gmail OAuth onboarding
  - Self-trained ML brain (TF-IDF + MLP) — NO external AI API
  - Secret trainer mode: /train 24426 — teach the bot directly from Telegram
  - Persistent automation rules ("archive anything about Google Classroom")
  - Full email management: archive, label, reply, prioritise, summarise
  - Auto-fallback: low confidence → bot asks if you want to train it
"""

import asyncio
import logging
import os
import sys
from typing import Optional

# Ensure project root is on sys.path so 'api', 'db', etc. resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup,
    Update, BotCommand,
)
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters,
)
from telegram.helpers import escape_markdown

from api.brain import AATASBrain
from api.gmail_auth import get_auth_url, get_gmail_service
from api.gmail_ops import (
    archive_email, apply_label, apply_rules, fetch_emails,
    mark_read, send_reply, trash_email, send_new_email, search_emails,
)
from api.web_ops import search_web, scrape_page
from db.database import (
    ActionLog, AutomationRule, SessionLocal,
    add_rule, get_or_create_user, get_rules, log_action, upsert_memory, get_memories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TRAINER_CODE   = os.environ.get("TRAINER_SECRET_CODE", "24426")

brain = AATASBrain()

# ── Terms of service message ──────────────────────────────────
_TOS = r"""🤖 *Welcome to AATAS (Beta)*
_AI Assisted Technological Arrangement System_

Before we start, please read and accept:

*⚠️ Beta Notice*
AATAS is still in development. It may make mistakes, break, or change at any time.

*📊 Data Usage*
• Messages, email content, and commands you send may be collected
• Your data may be used to *improve and train the AI*
• Do NOT send passwords, OTPs, or sensitive info

*📬 Gmail Access*
By continuing, you allow AATAS to access and organise your Gmail content.

*⚖️ Disclaimer*
Use at your own risk. AATAS is not responsible for errors, data loss, or issues.

\-\-\-
Type */accept* to continue
Type */decline* to exit"""

# ── In-memory caches ────────────────────────────────────────
# Email cache per user
_cache: dict[int, list[dict]] = {}

# Trainer mode state per user
# {telegram_id: {"active": bool, "mode": "intent"|"priority",
#                "pending": str|None, "added": int}}
_trainer: dict[int, dict] = {}

# Compose drafts per user
_drafts: dict[int, dict] = {}

PRIORITY_ICON = {"urgent": "🔴", "important": "🟡", "normal": "🟢", "low": "⚪"}

# Intent labels for trainer buttons
_INTENT_LABELS = [
    ("📬 Fetch Inbox",    "fetch_inbox"),
    ("⭐ Fetch Priority", "fetch_priority"),
    ("🔍 Analyse",        "analyse"),
    ("🔍 Search",         "search"),
    ("📦 Archive",        "archive"),
    ("🏷️ Label",          "label"),
    ("🗑️ Trash",          "trash"),
    ("✍️ Reply",          "reply"),
    ("📝 Compose",        "compose"),
    ("⚙️ Create Rule",   "create_rule"),
    ("📋 List Rules",     "list_rules"),
    ("📜 History",        "list_history"),
    ("🗑️ Delete Rule",   "delete_rule"),
    ("❓ Other/None",     "none"),
]

_PRIORITY_LABELS = [
    ("🔴 Urgent",    "urgent"),
    ("🟡 Important", "important"),
    ("🟢 Normal",    "normal"),
    ("⚪ Low",       "low"),
]


# ── Helpers ───────────────────────────────────────────────────

def _db():
    return SessionLocal()

def _get_user(update: Update):
    db   = _db()
    user = get_or_create_user(
        db,
        update.effective_user.id,
        update.effective_user.full_name,
    )
    return db, user

def _cached(tg_id: int, n: int) -> Optional[dict]:
    emails = _cache.get(tg_id, [])
    return emails[n - 1] if 1 <= n <= len(emails) else None

async def _typing(update: Update):
    if update.effective_chat:
        await update.effective_chat.send_action("typing")

def _intent_keyboard() -> InlineKeyboardMarkup:
    rows = []
    row  = []
    for label, cb in _INTENT_LABELS:
        row.append(InlineKeyboardButton(label, callback_data=f"train_intent:{cb}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⏭ Skip this one", callback_data="train_intent:skip")])
    return InlineKeyboardMarkup(rows)

def _priority_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(label, callback_data=f"train_priority:{cb}")
        for label, cb in _PRIORITY_LABELS
    ]])

def _esc(text) -> str:
    """Escape text for Telegram legacy Markdown (v1)."""
    if text is None: return ""
    return escape_markdown(str(text), version=1)


# ── Helpers ───────────────────────────────────────────────────

def _has_accepted_tos(db, user) -> bool:
    from db.database import get_memories
    mems = get_memories(db, user.id)
    return mems.get("tos_accepted") == "true"


# ── /start ────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    name = update.effective_user.first_name or "there"

    if user.setup_complete:
        db.close()
        await update.message.reply_text(
            f"👋 Welcome back, *{_esc(name)}*!\n\n"
            "I'm AATAS — your self-trained AI Gmail manager.\n"
            "Just talk to me naturally — I understand what you need.\n\n"
            "Try: _\"show my inbox\"_ or _\"archive anything about promotions\"_\n\n"
            "🎓 Want to train me? Type `/train [SECRET CODE]`",
            parse_mode="Markdown",
        )
    elif _has_accepted_tos(db, user):
        db.close()
        await cmd_setup(update, ctx)
    else:
        db.close()
        await update.message.reply_text(_TOS, parse_mode="Markdown")


# ── /accept ───────────────────────────────────────────────────

async def cmd_accept(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    upsert_memory(db, user.id, "tos_accepted", "true")
    db.close()
    await update.message.reply_text(
        "✅ *Terms accepted!* Welcome to AATAS.\n\n"
        "Let's connect your Gmail so I can get to work. 📬",
        parse_mode="Markdown",
    )
    await cmd_setup(update, ctx)


# ── /decline ──────────────────────────────────────────────────

async def cmd_decline(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 No worries! You've declined the terms.\n\n"
        "AATAS won't collect or process any of your data.\n"
        "If you change your mind, just type /start anytime."
    )


# ── /setup ────────────────────────────────────────────────────

async def cmd_setup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    tg_id    = update.effective_user.id

    # Gate behind TOS acceptance
    if not _has_accepted_tos(db, user):
        db.close()
        await update.message.reply_text(
            "⚠️ You need to accept the terms first.\n\n" + _TOS,
            parse_mode="Markdown",
        )
        return

    db.close()
    auth_url = get_auth_url(state=str(tg_id))
    kbd = InlineKeyboardMarkup([[
        InlineKeyboardButton("🔗 Connect Gmail", url=auth_url)
    ]])

    await update.message.reply_text(
        "🤖 *Connect your Gmail*\n\n"
        "Tap the button below to authorise Gmail access.\n"
        "• 📬 Read & manage your emails\n"
        "• 🤖 Apply your automation rules\n"
        "• 📊 Prioritise and summarise\n\n"
        "_Your Gmail token is stored securely and never shared._",
        parse_mode="Markdown",
        reply_markup=kbd,
    )


# ── /help ─────────────────────────────────────────────────────

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *AATAS — How to use me*\n\n"
        "*Just talk to me naturally!* For example:\n"
        '• _"Show my inbox"_\n'
        '• _"Archive anything about Google Classroom"_\n'
        '• _"Analyse email 3"_\n'
        '• _"Draft a reply to email 2"_\n'
        '• _"What automation rules do I have?"_\n'
        '• _"Delete the rule about newsletters"_\n\n'
        "*Commands:*\n"
        "/setup — Connect or reconnect Gmail\n"
        "/inbox — Quick inbox view\n"
        "/check — Apply your rules to new emails\n"
        "/rules — List your automation rules\n"
        "/history — Recent actions AATAS took\n"
        "/stats — Show AI model stats\n"
        "/help — This message\n\n"
        "🎓 *Train me to understand your style:*\n"
        "Type `/train [secret code]` to enter teacher mode!\n"
        "The more you teach me, the smarter I get.",
        parse_mode="Markdown",
    )


# ── /inbox ────────────────────────────────────────────────────

async def cmd_inbox(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    if not user.setup_complete:
        await update.message.reply_text("Please run /setup first to connect Gmail.")
        db.close(); return

    await _typing(update)
    msg = await update.message.reply_text("⏳ Fetching inbox...")
    try:
        svc    = get_gmail_service(user)
        emails = fetch_emails(svc, max_results=20, query="is:unread")
        _cache[user.telegram_id] = emails  # force refresh cache

        if not emails:
            await msg.edit_text("📭 No unread emails!")
            db.close(); return

        lines = ["📬 *Your unread inbox:*\n"]
        for e in emails:
            lines.append(f"`{e['idx']}.` *{_esc(e['subject'][:45])}*\n   👤 {_esc(e['sender'][:35])}\n")
        lines.append("\n_Say \"analyse 3\" or \"archive 2\" to take action._")
        await msg.edit_text("\n".join(lines), parse_mode="Markdown")

    except Exception as ex:
        await msg.edit_text(f"❌ {ex}")
    finally:
        db.close()


# ── /search ───────────────────────────────────────────────────

async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)

    if not user.setup_complete:
        await update.message.reply_text("Run /setup first.")
        db.close()
        return

    if not ctx.args:
        await update.message.reply_text("Usage: /search <keyword>")
        db.close()
        return

    keyword = " ".join(ctx.args)

    await _typing(update)
    msg = await update.message.reply_text(f"🔍 Searching for '{keyword}'...")

    try:
        svc = get_gmail_service(user)
        # Check for date filters in keywords
        query = keyword
        emails = search_emails(svc, query, max_results=20)
        
        if not emails:
            await msg.edit_text("No results found.")
            return

        # Perform semantic ranking if it's a natural language query
        if len(keyword.split()) > 1:
            ranked = brain.semantic_search(keyword, emails, top_n=10)
            if ranked:
                emails = ranked

        _cache[user.telegram_id] = emails

        lines = [f"🔍 *Results for '{_esc(keyword)}':*\n"]
        for e in emails:
            lines.append(f"`{e['idx']}.` *{_esc(e['subject'][:45])}*\n   👤 {_esc(e['sender'][:35])}")
        await msg.edit_text("\n".join(lines), parse_mode="Markdown")

    except Exception as ex:
        await msg.edit_text(f"❌ {ex}")
    finally:
        db.close()


# ── /mute ───────────────────────────────────────────────────

async def cmd_mute(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    upsert_memory(db, user.id, "mute_automations", "true")
    db.close()
    await update.message.reply_text("🔇 *Automations muted.* I'll still run rules, but I won't notify you.", parse_mode="Markdown")

async def cmd_unmute(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    upsert_memory(db, user.id, "mute_automations", "false")
    db.close()
    await update.message.reply_text("🔊 *Automations unmuted.* I'll notify you of automated actions.", parse_mode="Markdown")


# ── Auto-Refresh Job ──────────────────────────────────────────

async def job_auto_check(ctx: ContextTypes.DEFAULT_TYPE):
    """Background task to check inboxes and apply rules."""
    db = SessionLocal()
    from db.database import User
    users = db.query(User).filter_by(setup_complete=True).all()
    
    for user in users:
        try:
            svc = get_gmail_service(user)
            # Fetch unread emails
            emails = fetch_emails(svc, max_results=20, query="is:unread")
            if not emails:
                continue
                
            rules = get_rules(db, user.id)
            if not rules:
                continue
                
            applied = apply_rules(svc, db, user.id, emails, rules, brain)
            
            if applied:
                mems = get_memories(db, user.id)
                if mems.get("mute_automations") == "true":
                    continue

                count = len(applied)
                text = f"🤖 *Auto-Automation Applied*\n\nI processed {count} emails based on your rules.\n"
                for a in applied[:5]:
                    text += f"• {a['action'].title()}: _{_esc(a['email']['subject'][:40])}_\n"
                
                await ctx.bot.send_message(user.telegram_id, text, parse_mode="Markdown")
                
        except Exception as e:
            log.error(f"Auto-check failed for user {user.id}: {e}")
            
    db.close()


# ── /check ────────────────────────────────────────────────────

async def cmd_check(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    if not user.setup_complete:
        await update.message.reply_text("Please run /setup first."); db.close(); return

    rules = get_rules(db, user.id)
    if not rules:
        await update.message.reply_text(
            "You have no automation rules yet.\n"
            'Try: _"archive anything about Google Classroom"_',
            parse_mode="Markdown"
        )
        db.close(); return

    await _typing(update)
    msg = await update.message.reply_text(f"⏳ Checking inbox against {len(rules)} rules...")
    try:
        svc     = get_gmail_service(user)
        emails  = fetch_emails(svc, max_results=50, query="is:unread")
        applied = apply_rules(svc, db, user.id, emails, rules, brain)

        if not applied:
            await msg.edit_text("✅ Checked inbox — no rules triggered.")
        else:
            lines = [f"✅ *AATAS applied {len(applied)} actions:*\n"]
            for a in applied[:10]:
                lines.append(f"• {a['action'].title()}: _{_esc(a['email']['subject'][:45])}_")
            if len(applied) > 10:
                lines.append(f"...and {len(applied)-10} more.")
            await msg.edit_text("\n".join(lines), parse_mode="Markdown")

    except Exception as ex:
        await msg.edit_text(f"❌ {ex}")
    finally:
        db.close()


# ── /rules ────────────────────────────────────────────────────

async def cmd_rules(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    rules    = get_rules(db, user.id)
    db.close()

    if not rules:
        await update.message.reply_text(
            "You have no rules yet.\n\n"
            "Tell me what to automate, e.g.:\n"
            '_"Archive anything about Google Classroom"_\n'
            '_"Label emails from boss@work.com as important"_',
            parse_mode="Markdown",
        )
        return

    lines = ["⚙️ *Your automation rules:*\n"]
    for r in rules:
        lines.append(
            f"`[{r.id}]` {r.action.upper()} — _{_esc(r.rule_text)}_\n"
            f"   Triggered {r.trigger_count}× since {r.created_at.strftime('%d %b')}\n"
        )
    lines.append("\n_Say \"delete rule 3\" to remove a rule._")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── /history ──────────────────────────────────────────────────

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db, user = _get_user(update)
    logs     = (
        db.query(ActionLog)
        .filter_by(user_id=user.id)
        .order_by(ActionLog.created_at.desc())
        .limit(15)
        .all()
    )
    db.close()

    if not logs:
        await update.message.reply_text("No actions taken yet."); return

    lines = ["📋 *Recent AATAS actions:*\n"]
    for l in logs:
        ts = l.created_at.strftime("%d %b %H:%M")
        lines.append(f"`{ts}` {l.action.upper()} — _{_esc(l.email_subject[:45])}_")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── /stats — model stats ──────────────────────────────────────

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats = brain.model_stats()
    await update.message.reply_text(
        "🧠 *AATAS ML Model Stats*\n\n"
        f"*Intent Model*\n"
        f"• Training examples: `{stats['intent_examples']}`\n"
        f"• Trained: {'✅ Yes' if stats['intent_trained'] else '⚠️ Bootstrapped from seed data'}\n\n"
        f"*Priority Model*\n"
        f"• Training examples: `{stats['priority_examples']}`\n"
        f"• Trained: {'✅ Yes' if stats['priority_trained'] else '⚠️ Using keyword heuristics'}\n\n"
        "🎓 Train me more at any time with `/train 24426`!",
        parse_mode="Markdown",
    )


# ── /train — secret trainer mode ─────────────────────────────

async def cmd_train(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tg_id = update.effective_user.id
    args  = ctx.args or []
    code  = args[0] if args else ""

    if code != TRAINER_CODE:
        await update.message.reply_text(
            "❌ Wrong code. Ask the owner for the trainer code!"
        )
        return

    _trainer[tg_id] = {
        "active":  True,
        "mode":    "intent",
        "pending": None,
        "added":   0,
    }

    stats = brain.model_stats()
    await update.message.reply_text(
        "🎓 *Trainer Mode Activated!*\n\n"
        f"I currently know *{stats['intent_examples']}* intent examples.\n\n"
        "I'm ready to learn from you. Just *type any command* the way a user would "
        "say it, and I'll ask you what it means.\n\n"
        "The more you teach me, the smarter I get! 🧠\n\n"
        "*Commands in trainer mode:*\n"
        "/done — Retrain model & exit\n"
        "/skip — Skip the current example\n"
        "/stats — Show model stats\n"
        "/priority — Switch to email priority training mode",
        parse_mode="Markdown",
    )


# ── /done — exit trainer mode & retrain ──────────────────────

async def cmd_done(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tg_id   = update.effective_user.id
    state   = _trainer.get(tg_id, {})

    if not state.get("active"):
        await update.message.reply_text("You're not in trainer mode. Use `/train [secret code]` to start.")
        return

    added = state.get("added", 0)
    _trainer.pop(tg_id, None)

    if added == 0:
        await update.message.reply_text(
            "🎓 Exited trainer mode — no new examples were added so no retraining needed."
        )
        return

    await _typing(update)
    msg = await update.message.reply_text(f"🧠 Retraining on {added} new examples... please wait...")
    try:
        stats = brain.retrain()
        i     = stats["intent"]
        p     = stats["priority"]

        intent_line = (
            f"Intent Model: `{i['accuracy']}%` accuracy on `{i['examples']}` examples"
            if "accuracy" in i else
            f"Intent Model: `{i.get('error', 'Not enough data yet')}`"
        )
        priority_line = (
            f"Priority Model: trained on `{_esc(p['examples'])}` examples"
            if "examples" in p and "error" not in p else
            f"Priority Model: {_esc(p.get('error', 'Using heuristics'))}"
        )

        await msg.edit_text(
            f"✅ *Retraining complete!*\n\n"
            f"📊 {intent_line}\n"
            f"📊 {priority_line}\n\n"
            "AATAS is now smarter. You can keep training any time with `/train [secret code]`! 🎓",
            parse_mode="Markdown",
        )
    except Exception as ex:
        await msg.edit_text(f"❌ Retraining failed: {ex}")


# ── /skip — skip current training example ────────────────────

async def cmd_skip(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tg_id = update.effective_user.id
    if _trainer.get(tg_id, {}).get("active"):
        _trainer[tg_id]["pending"] = None
        await update.message.reply_text("⏭ Skipped! Send another example to continue teaching me.")
    else:
        await update.message.reply_text("You're not in trainer mode.")


# ── /priority — switch to priority training ───────────────────

async def cmd_priority_train(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    tg_id = update.effective_user.id
    if not _trainer.get(tg_id, {}).get("active"):
        await update.message.reply_text(
            "You need to be in trainer mode first. Use `/train [secret code]`."
        )
        return

    db, user = _get_user(update)
    if not user.setup_complete:
        await update.message.reply_text("Connect Gmail first (/setup) to do priority training.")
        db.close(); return

    _trainer[tg_id]["mode"] = "priority"

    await _typing(update)
    msg = await update.message.reply_text("⏳ Fetching emails for priority training...")
    try:
        svc    = get_gmail_service(user)
        emails = fetch_emails(svc, max_results=5)
        _cache[user.telegram_id] = emails
        db.close()

        if not emails:
            await msg.edit_text("📭 No emails to rate right now. Send me commands to train intent instead!")
            return

        await msg.edit_text(
            "📧 *Priority Training Mode*\n\n"
            "I'll show you each email and you rate its priority.\n"
            "Tap the buttons to teach me what's urgent vs low priority.\n\n"
            "_Type /done when finished._",
            parse_mode="Markdown",
        )

        for e in emails:
            await update.message.reply_text(
                f"📧 *{_esc(e['subject'][:50])}*\n"
                f"👤 {_esc(e['sender'][:40])}\n\n"
                "_How urgent is this email?_",
                parse_mode="Markdown",
                reply_markup=_priority_keyboard(),
            )
            # store email in state for the callback to pick up
            _trainer[tg_id]["pending_email"] = e

    except Exception as ex:
        await msg.edit_text(f"❌ {ex}")
        db.close()


# ── Trainer message handler ────────────────────────────────────

async def _handle_trainer_message(update: Update, ctx, tg_id: int, text: str):
    """Handle a message sent while in intent trainer mode."""
    state = _trainer[tg_id]

    # Store the example and show intent buttons
    state["pending"] = text
    await _typing(update)
    await update.message.reply_text(
        f"📝 *New example received:*\n`{_esc(text)}`\n\n"
        "What should this command do? Tap the correct intent below:",
        parse_mode="Markdown",
        reply_markup=_intent_keyboard(),
    )


# ── Main message handler ──────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Route messages — trainer mode or normal AI conversation."""
    db, user = _get_user(update)
    tg_id    = user.telegram_id
    text     = update.message.text.strip()

    # ── Trainer mode ──────────────────────────────────────────
    if _trainer.get(tg_id, {}).get("active"):
        db.close()
        await _handle_trainer_message(update, ctx, tg_id, text)
        return

    # ── Normal mode: require Gmail setup ─────────────────────
    if not user.setup_complete:
        await update.message.reply_text(
            "I need to connect to your Gmail first. Run /setup to get started!"
        )
        db.close(); return

    # ── Disclaimer toggle detection ───────────────────────────
    tl = text.lower().strip()
    if any(p in tl for p in ["hide disclaimer", "no disclaimer", "stop disclaimer",
                              "turn off disclaimer", "don't show disclaimer"]):
        upsert_memory(db, user.id, "show_disclaimer", "false")
        db.close()
        await update.message.reply_text(
            "✅ Got it! I won't show the AI disclaimer on future replies."
        )
        return
    if any(p in tl for p in ["show disclaimer", "turn on disclaimer", "enable disclaimer"]):
        upsert_memory(db, user.id, "show_disclaimer", "true")
        db.close()
        await update.message.reply_text(
            "✅ The AI disclaimer will now appear on my replies again."
        )
        return

    # ── Load disclaimer preference ────────────────────────────
    memories         = get_memories(db, user.id)
    show_disclaimer  = memories.get("show_disclaimer", "true") != "false"

    await _typing(update)

    try:
        svc = None
        try:
            svc = get_gmail_service(user)
        except Exception:
            pass

        # Build email context hint
        cached   = _cache.get(tg_id, [])
        ctx_hint = ""
        if cached:
            ctx_hint = "Cached emails:\n" + "\n".join(
                f"{e['idx']}. {e['subject']} | {e['sender']}" for e in cached[:10]
            )

        reply_texts, intent_dicts = brain.chat(
            db, user, text,
            context_hint=ctx_hint,
            cached_emails=cached,
            show_disclaimer=show_disclaimer,
        )

        if not intent_dicts:
            for reply_text in reply_texts:
                await update.message.reply_text(
                    reply_text or "I didn't quite catch that.", parse_mode="Markdown"
                )
        else:
            for intent_dict, reply_text in zip(intent_dicts, reply_texts):
                await _execute_intent(update, ctx, db, user, svc, intent_dict, reply_text, text)

    except Exception as ex:
        log.exception("handle_message error")
        await update.message.reply_text(f"❌ Something went wrong: {_esc(str(ex))}")
    finally:
        db.close()


# ── Intent executor ───────────────────────────────────────────

async def _execute_intent(update, ctx, db, user, svc, intent: dict, ai_reply: str, original_text: str = ""):
    """Dispatch Gmail actions based on parsed intent."""
    action = intent.get("intent", "none")
    idx    = intent.get("email_idx")
    tg_id  = user.telegram_id

    if ai_reply:
        await update.message.reply_text(ai_reply, parse_mode="Markdown")

    if action == "fetch_inbox":
        brain.record_passive_example(original_text, action)
        if svc:
            # If user explicitly said "unread", use that filter.
            # Otherwise, use the new permissive default from gmail_ops.
            query = "-in:trash -in:spam"
            if "unread" in original_text.lower():
                query = "is:unread"
            
            emails = fetch_emails(svc, max_results=20, query=query)
            _cache[tg_id] = emails
            if emails:
                title = "Unread Inbox" if "unread" in query else "Latest Emails"
                lines = [f"📬 *{title} (last {len(emails)}):*\n"]
                for e in emails:
                    lines.append(f"`{e['idx']}.` *{_esc(e['subject'][:45])}*\n   👤 {_esc(e['sender'][:35])}\n")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    elif action == "search":
        brain.record_passive_example(original_text, action)
        if svc:
            query = original_text
            # Remove "search for" or "find" from query
            import re
            query = re.sub(r"^(?:search(?:\s+for)?|find(?:\s+emails?\s+about)?)\s+", "", query, flags=re.IGNORECASE).strip()
            
            emails = search_emails(svc, query, max_results=20)
            
            if not query or query == "?":
                await update.message.reply_text("What would you like me to search for, Sir? (e.g. _'search for invoice'_)")
                return

            if not emails:
                await update.message.reply_text("No results found.")
            else:
                # Semantic search integration
                if len(query.split()) > 1:
                    ranked = brain.semantic_search(query, emails, top_n=10)
                    if ranked: emails = ranked
                
                _cache[tg_id] = emails
                lines = [f"🔍 *Results for '{query}':*\n"]
                for e in emails:
                    lines.append(f"`{e['idx']}.` *{e['subject'][:45]}*\n   👤 {e['sender'][:35]}")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    elif action == "web_search":
        brain.record_passive_example(original_text, action)
        import re
        query = original_text
        query = re.sub(r"^(?:search(?:\s+for)?|look\s+up|find(?:\s+info(?:rmation)?\s+on)?)\s+", "", query, flags=re.IGNORECASE).strip()
        query = query.rstrip("?")
        
        if not query:
            await update.message.reply_text("What would you like me to search for, Sir?")
            return

        msg = await update.message.reply_text(f"🌐 Searching for '{query}'...")
        results = search_web(query, max_results=5)
        
        if not results:
            await msg.edit_text("I'm sorry, Sir. My search returned no relevant results.")
        else:
            lines = [f"🌐 *Web Search Results for '{query}':*\n"]
            cache_data = []
            for i, r in enumerate(results, 1):
                lines.append(f"`{i}.` *{r['title']}*\n   _{r['snippet'][:150]}..._")
                cache_data.append({"idx": i, "url": r["link"], "title": r["title"], "subject": r["title"]})
            
            lines.append("\n_Shall I research one of these in detail? Say \"research 1\"._")
            await msg.edit_text("\n".join(lines), parse_mode="Markdown", disable_web_page_preview=True)
            _cache[tg_id] = cache_data

    elif action == "research":
        brain.record_passive_example(original_text, action)
        import re
        m = re.search(r"research\s+(\d+)", original_text.lower())
        url = ""
        title = ""
        if m:
            target_idx = int(m.group(1))
            cached = _cache.get(tg_id, [])
            target = next((c for c in cached if c.get("idx") == target_idx), None)
            if target:
                url = target.get("url")
                title = target.get("title")
        
        if not url:
            # Try to extract URL directly
            m = re.search(r"https?://[^\s]+", original_text)
            if m: url = m.group(0)
            
        if not url:
            await update.message.reply_text("Please specify which result to research (e.g., 'research 1') or provide a URL, Sir.")
            return

        msg = await update.message.reply_text(f"🧠 Researching: {title or url}...")
        content = scrape_page(url)
        if not content or len(content) < 100:
            await msg.edit_text("I'm afraid I couldn't access significant data from that page, Sir. It might be protected or empty.")
            return
            
        summary = brain.summarise_email(title or "Research", content)
        await msg.edit_text(f"📑 *Research Summary: {_esc(title or 'Report')}*\n\n{_esc(summary)}\n\n🔗 {url}", parse_mode="Markdown")

    elif action == "fetch_priority":
        brain.record_passive_example(original_text, action)
        if svc:
            limit = intent.get("email_idx") or 15
            limit = min(limit, 50)
            await _typing(update)
            emails = fetch_emails(svc, max_results=limit)
            _cache[tg_id] = emails
            scored = []
            for e in emails:
                priority, conf = brain.classify_priority(e["subject"], e["body"])
                scored.append((e, priority, conf))
            order = {"urgent": 0, "important": 1, "normal": 2, "low": 3}
            scored.sort(key=lambda x: order.get(x[1], 9))
            lines = [f"📊 *Inbox by priority (last {len(emails)}):*\n"]
            for e, p, c in scored:
                icon = PRIORITY_ICON.get(p, "•")
                lines.append(f"`{e['idx']}.` {icon} `{p}` — {e['subject'][:40]}")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    elif action == "analyse" and idx:
        brain.record_passive_example(original_text, action)
        e = _cached(tg_id, idx)
        if not e:
            await update.message.reply_text(
                f"Email #{idx} not in cache. Say _'show my inbox'_ first.",
                parse_mode="Markdown",
            ); return
        await _typing(update)
        msg      = await update.message.reply_text("🧠 Analysing...")
        priority, conf = brain.classify_priority(e["subject"], e["body"])
        summary  = brain.summarise_email(e["subject"], e["body"])
        actions  = brain.extract_actions(e["subject"], e["body"])
        draft    = brain.draft_reply(e["subject"], e["body"])
        icon     = PRIORITY_ICON.get(priority, "•")
        text = (
            f"📧 *{_esc(e['subject'][:50])}*\n"
            f"👤 {_esc(e['sender'])}\n\n"
            f"Priority: {icon} `{_esc(priority)}` ({conf:.0%})\n\n"
            f"📝 *Summary:*\n{_esc(summary)}\n\n"
            f"✅ *Action items:*\n" + ("\n".join(f"• {_esc(a)}" for a in actions) or "None") +
            f"\n\n💬 *Draft reply:*\n_{_esc(draft)}_"
        )
        kbd = InlineKeyboardMarkup([
            [InlineKeyboardButton("📤 Send Draft", callback_data=f"sendreply:{e['id']}:{idx}")],
            [
                InlineKeyboardButton("📤 Archive", callback_data=f"archive:{e['id']}"),
                InlineKeyboardButton("✅ Mark read", callback_data=f"markread:{e['id']}"),
            ]
        ])
        await msg.edit_text(text, parse_mode="Markdown", reply_markup=kbd)

    elif action == "archive" and idx:
        brain.record_passive_example(original_text, action)
        e = _cached(tg_id, idx)
        if not e or not svc: return
        archive_email(svc, e["id"])
        brain.record_passive_priority_example(e["subject"], e["body"], action)
        log_action(db, user.id, "archive", e["id"], e["subject"], "manual")
        await update.message.reply_text(f"📦 Archived: _{_esc(e['subject'])}_", parse_mode="Markdown")

    elif action == "trash" and idx:
        brain.record_passive_example(original_text, action)
        e = _cached(tg_id, idx)
        if not e or not svc: return
        trash_email(svc, e["id"])
        brain.record_passive_priority_example(e["subject"], e["body"], action)
        log_action(db, user.id, "trash", e["id"], e["subject"], "manual")
        await update.message.reply_text(f"🗑 Trashed: _{_esc(e['subject'])}_", parse_mode="Markdown")

    elif action == "label" and idx:
        brain.record_passive_example(original_text, action)
        e = _cached(tg_id, idx)
        label_name = intent.get("action_params", {}).get("label", "labelled")
        if not e or not svc: return
        apply_label(svc, e["id"], label_name)
        brain.record_passive_priority_example(e["subject"], e["body"], action)
        log_action(db, user.id, "label", e["id"], e["subject"], label_name)
        await update.message.reply_text(
            f"🏷️ Labelled email #{idx} as *{_esc(label_name)}*!", parse_mode="Markdown"
        )

    elif action == "reply" and idx:
        brain.record_passive_example(original_text, action)
        e    = _cached(tg_id, idx)
        tone = intent.get("action_params", {}).get("tone", "professional")
        if not e: return
        brain.record_passive_priority_example(e["subject"], e["body"], action)
        draft = brain.draft_reply(e["subject"], e["body"], tone)
        kbd = InlineKeyboardMarkup([[
            InlineKeyboardButton("📤 Send this reply", callback_data=f"sendreply:{e['id']}:{idx}"),
            InlineKeyboardButton("❌ Cancel",           callback_data="cancel"),
        ]])
        await update.message.reply_text(
            f"💬 *Draft reply ({tone}):*\n\n_{draft}_",
            parse_mode="Markdown", reply_markup=kbd,
        )

    elif action == "compose":
        brain.record_passive_example(original_text, action)
        params = intent.get("action_params", {})
        to_address = params.get("to")
        message = params.get("message")
        subject = params.get("subject")
        tone = params.get("tone", "professional")
        
        # Pronoun resolution: resolve "him/her" if we have a pending draft
        if to_address and to_address.lower() in ["him", "her", "them", "it"] and tg_id in _drafts:
            to_address = _drafts[tg_id]["to"]
            if not subject:
                subject = _drafts[tg_id].get("subject")

        if not to_address:
            await update.message.reply_text("I couldn't figure out who to send this to. Try specifying an email address or name.")
            return
            
        final_subject, final_body = brain.draft_new_email(to_address, subject, message, tone)
        
        _drafts[tg_id] = {"to": to_address, "subject": final_subject, "body": final_body}
        
        kbd = InlineKeyboardMarkup([[
            InlineKeyboardButton("📤 Send Email", callback_data="sendnew"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel"),
        ]])
        
        await update.message.reply_text(
            f"📝 *Draft new email ({tone}):*\n\n"
            f"*To:* {to_address}\n"
            f"*Subject:* {final_subject}\n"
            f"*Message:*\n_{final_body}_\n",
            parse_mode="Markdown", reply_markup=kbd,
        )

    elif action == "create_rule":
        brain.record_passive_example(original_text, action)
        rule_data = intent.get("rule", {})
        if rule_data:
            target = {
                "keywords": rule_data.get("keywords", []),
                "from":     rule_data.get("from", ""),
                "label":    rule_data.get("label", ""),
            }
            rule = add_rule(
                db, user.id,
                rule_text = rule_data.get("rule_text", "User rule"),
                action    = rule_data.get("action", "archive"),
                target    = target,
            )
            await update.message.reply_text(
                f"⚙️ *Rule created!*\n\n"
                f"ID: `{rule.id}`\n"
                f"Action: `{rule.action}`\n"
                f"Keywords: `{', '.join(target['keywords']) or 'any'}`\n\n"
                "AATAS will apply this every time you run /check.\n"
                "Want me to run it now? Say _\"yes, check now\"_",
                parse_mode="Markdown",
            )

    elif action == "delete_rule":
        brain.record_passive_example(original_text, action)
        rule_id = intent.get("action_params", {}).get("rule_id")
        if rule_id:
            rule = db.query(AutomationRule).filter_by(id=rule_id, user_id=user.id).first()
            if rule:
                rule.enabled = False
                db.commit()
                await update.message.reply_text(f"🗑 Rule `{_esc(rule_id)}` deleted.", parse_mode="Markdown")

    elif action == "list_rules":
        brain.record_passive_example(original_text, action)
        rules = get_rules(db, user.id)
        if not rules:
            await update.message.reply_text("No rules set. Tell me what to automate!")
        else:
            lines = ["⚙️ *Active rules:*\n"]
            for r in rules:
                lines.append(f"`[{r.id}]` {r.action.upper()} — _{_esc(r.rule_text)}_ ({r.trigger_count}×)")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    elif action == "list_history":
        brain.record_passive_example(original_text, action)
        logs = (
            db.query(ActionLog).filter_by(user_id=user.id)
            .order_by(ActionLog.created_at.desc()).limit(10).all()
        )
        if not logs:
            await update.message.reply_text("No actions yet.")
        else:
            lines = ["📋 *Recent actions:*\n"]
            for l in logs:
                lines.append(f"• {l.action.upper()}: _{_esc(l.email_subject[:40])}_")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── Inline button callbacks ───────────────────────────────────

async def button_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q     = update.callback_query
    await q.answer()
    data  = q.data
    tg_id = update.effective_user.id
    db, user = _get_user(update)

    try:
        await _typing(update)
        # ── Trainer intent labelling ──────────────────────────
        if data.startswith("train_intent:"):
            intent_label = data.split(":", 1)[1]
            state = _trainer.get(tg_id, {})

            if not state.get("active"):
                await q.edit_message_text("❌ You're no longer in trainer mode.")
                return

            pending = state.get("pending")
            if not pending:
                await q.edit_message_text("⚠️ No pending example. Type a command first!")
                return

            if intent_label == "skip":
                state["pending"] = None
                await q.edit_message_text("⏭ Skipped! Type another command to keep teaching me.")
                return

            brain.add_intent_example(pending, intent_label)
            state["added"]  += 1
            state["pending"] = None
            await q.edit_message_text(
                f"✅ *Learned!*\n\n"
                f"`{_esc(pending)}` → *{_esc(intent_label)}*\n\n"
                f"Total new examples this session: *{state['added']}*\n\n"
                "Keep teaching me! Send another example, or\n"
                "/done — save & retrain | /skip — skip | /stats — model info",
                parse_mode="Markdown",
            )
            return

        # ── Trainer priority labelling ────────────────────────
        if data.startswith("train_priority:"):
            priority = data.split(":", 1)[1]
            state    = _trainer.get(tg_id, {})
            email    = state.get("pending_email")

            if email:
                brain.learn_priority(email["subject"], email["body"], priority)
                state["added"] += 1
                state["pending_email"] = None
                await q.edit_message_text(
                    f"✅ Rated *{_esc(email['subject'][:40])}* as `{_esc(priority)}`\n"
                    f"Total ratings this session: *{state['added']}*",
                    parse_mode="Markdown",
                )
            else:
                await q.edit_message_text("⚠️ No email found for rating.")
            return

        # ── Standard Gmail actions ────────────────────────────
        svc = get_gmail_service(user)

        if data.startswith("archive:"):
            msg_id = data.split(":")[1]
            archive_email(svc, msg_id)
            log_action(db, user.id, "archive", msg_id, "email", "inline button")
            await q.edit_message_text("📦 Archived!")

        elif data.startswith("markread:"):
            msg_id = data.split(":")[1]
            mark_read(svc, msg_id)
            await q.edit_message_text("✅ Marked as read.")

        elif data.startswith("sendreply:"):
            _, msg_id, idx = data.split(":")
            e = _cached(user.telegram_id, int(idx))
            if e:
                draft = brain.draft_reply(e["subject"], e["body"])
                send_reply(svc, e, draft)
                log_action(db, user.id, "reply", msg_id, e["subject"])
                await q.edit_message_text("📤 Reply sent!")

        elif data == "sendnew":
            draft = _drafts.get(tg_id)
            if draft:
                send_new_email(svc, draft["to"], draft["subject"], draft["body"])
                log_action(db, user.id, "compose", "", draft["subject"])
                await q.edit_message_text("📤 Email sent!")
                _drafts.pop(tg_id, None)
            else:
                await q.edit_message_text("⚠️ Draft expired or not found.")

        elif data == "cancel":
            await q.edit_message_text("Cancelled.")

    except Exception as ex:
        await q.edit_message_text(f"❌ {_esc(str(ex))}")
    finally:
        db.close()


# ── App wiring ────────────────────────────────────────────────

def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("accept",   cmd_accept))
    app.add_handler(CommandHandler("decline",  cmd_decline))
    app.add_handler(CommandHandler("setup",    cmd_setup))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("search",   cmd_search))
    app.add_handler(CommandHandler("inbox",    cmd_inbox))
    app.add_handler(CommandHandler("check",    cmd_check))
    app.add_handler(CommandHandler("rules",    cmd_rules))
    app.add_handler(CommandHandler("history",  cmd_history))
    app.add_handler(CommandHandler("mute",     cmd_mute))
    app.add_handler(CommandHandler("unmute",   cmd_unmute))
    app.add_handler(CommandHandler("stats",    cmd_stats))
    app.add_handler(CommandHandler("train",    cmd_train))
    app.add_handler(CommandHandler("done",     cmd_done))
    app.add_handler(CommandHandler("skip",     cmd_skip))
    app.add_handler(CommandHandler("priority", cmd_priority_train))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button_cb))

    # Schedule auto-refresh job every 2 minutes
    app.job_queue.run_repeating(job_auto_check, interval=120, first=10)

    return app


async def _set_commands(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start",    "Start AATAS"),
        BotCommand("accept",   "Accept terms and continue setup"),
        BotCommand("decline",  "Decline terms and exit"),
        BotCommand("setup",    "Connect Gmail"),
        BotCommand("search",   "Search emails"),
        BotCommand("inbox",    "View inbox"),
        BotCommand("check",    "Apply automation rules"),
        BotCommand("rules",    "Manage rules"),
        BotCommand("mute",     "Mute automation alerts"),
        BotCommand("unmute",   "Unmute automation alerts"),
        BotCommand("history",  "View action history"),
        BotCommand("stats",    "View AI model stats"),
        BotCommand("train",    "Enter trainer mode (secret code required)"),
        BotCommand("help",     "Help"),
    ])


if __name__ == "__main__":
    log.info("Starting AATAS bot (Self-Trained ML Edition)...")
    application = build_app()

    async def post_init(app):
        await _set_commands(app)

    application.post_init = post_init
    application.run_polling(drop_pending_updates=True)