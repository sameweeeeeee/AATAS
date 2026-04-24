"""
AATAS — AI Conversation Brain (Self-Trained ML Edition)
No external APIs. No Anthropic. No OpenAI.
Uses YOUR locally-trained TF-IDF + MLP model to understand intent.
Passively learns from every successful interaction.
"""

import os
import re
import random
from typing import Optional

from database import Session, User, get_conv_history, save_conv_turn
from ml.trainer import get_intent_model, get_priority_model
from ml.rule_parser import parse_rule

# Confidence threshold — below this, AATAS asks you to teach it
CONFIDENCE_THRESHOLD = 0.60

# Passive learning — auto-retrain silently every N successful interactions
RETRAIN_EVERY = 10
_passive_count = 0

# Disclaimer appended to every AI reply (user can turn off)
DISCLAIMER = (
    "\n\n"
    "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
    "_\u26a0\ufe0f AI-generated \u00b7 may not be correct._\n"
    "_Reply_ `hide disclaimer` _to stop showing this._"
)

# ── Response templates ────────────────────────────────────────
# Multiple options per intent so the bot doesn't sound like a robot.

_RESPONSES = {
    "fetch_inbox": [
        "Sure! Pulling up your inbox now... 📬",
        "On it! Fetching your latest emails... 📬",
        "Let me grab your inbox for you! 📬",
    ],
    "fetch_priority": [
        "Let me rank your emails by priority for you! ⭐",
        "Checking what needs your attention first... ⭐",
        "Sorting your inbox by importance... ⭐",
    ],
    "analyse": [
        "Analysing that email for you... 🔍",
        "Let me dig into that one! 🔍",
        "Taking a closer look at that email... 🔍",
    ],
    "archive": [
        "Got it, archiving that right away! 📦",
        "Done! That email is archived. 📦",
        "Archived! ✅",
    ],
    "label": [
        "I'll label that for you! 🏷️",
        "Tagging it now! 🏷️",
    ],
    "trash": [
        "Sending it to the bin... 🗑️",
        "Deleted! Gone for good. 🗑️",
    ],
    "reply": [
        "Let me draft a reply for you... ✍️",
        "Writing up a reply now! ✍️",
        "Here's a draft reply for that email... ✍️",
    ],
    "create_rule": [
        "Got it! Creating that automation rule now. ⚙️",
        "Rule saved! I'll apply it every time you run /check. ⚙️",
        "Done! AATAS will remember that and act on it automatically. ⚙️",
    ],
    "delete_rule": [
        "Removing that rule now... 🗑️",
        "Rule deleted! ✅",
    ],
    "list_rules": [
        "Here are all your active automation rules! 📋",
        "Here's what I'm set up to do automatically for you: 📋",
    ],
    "list_history": [
        "Here's what I've been up to recently: 📜",
        "Here's my recent action log: 📜",
    ],
    "none": [
        "Hmm, I'm not quite sure what you mean — could you rephrase that?",
        "I didn't catch that one. Try something like _\"show my inbox\"_ or _\"archive email 2\"_.",
        "Not sure I understood that. Try _\"show my rules\"_ or _\"check my emails\"_.",
    ],
}


def _pick(intent: str) -> str:
    return random.choice(_RESPONSES.get(intent, _RESPONSES["none"]))


def _parse_index(text: str) -> Optional[int]:
    """Extract an email number like 'email 3' or 'analyse 2'."""
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None


def _parse_rule_id(text: str) -> Optional[int]:
    """Extract a rule ID like 'delete rule 3'."""
    m = re.search(r"rule\s+(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_tone(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["casual", "friendly", "informal", "chill"]):
        return "casual"
    if any(w in t for w in ["formal", "official", "business"]):
        return "formal"
    return "professional"


# ── Brain ──────────────────────────────────────────────────────

class AATASBrain:
    """AATAS self-trained ML brain. No external API calls."""

    def __init__(self):
        # Models are singletons loaded by the trainer
        self.intent_model   = get_intent_model()
        self.priority_model = get_priority_model()

    # ── Core chat ────────────────────────────────────────────

    def chat(
        self,
        db:             Session,
        user:           User,
        user_message:   str,
        context_hint:   str  = "",
        show_disclaimer: bool = True,
    ) -> tuple[str, Optional[dict]]:
        """
        Process a user message. Returns (reply_text, intent_dict | None).
        Low-confidence inputs prompt the user to enter trainer mode.
        """
        text   = user_message.strip()
        intent, confidence = self.intent_model.predict(text)

        # ── Low confidence → suggest training ────────────────
        if confidence < CONFIDENCE_THRESHOLD:
            reply = (
                f"🤔 I'm not quite sure what you meant by that "
                f"*(confidence: {confidence:.0%})*.\n\n"
                "Want to teach me? Type `/train 24426` to enter trainer mode — "
                "you can show me exactly what this kind of message should do! 🎓"
            )
            save_conv_turn(db, user.id, "user",      user_message)
            save_conv_turn(db, user.id, "assistant", reply)
            return reply, None

        # ── Build intent dict ─────────────────────────────────
        intent_dict: dict = {"intent": intent, "email_idx": None, "action_params": {}}

        if intent in ("analyse", "archive", "label", "trash", "reply"):
            intent_dict["email_idx"] = _parse_index(text)
            if intent == "label":
                intent_dict["action_params"]["label"] = self._parse_label(text)
            if intent == "reply":
                intent_dict["action_params"]["tone"] = _parse_tone(text)

        elif intent == "create_rule":
            rule_data = parse_rule(text)
            intent_dict["rule"] = {
                "rule_text": text,
                "action":    rule_data["action"],
                "keywords":  rule_data["keywords"],
                "from":      rule_data["from"],
                "label":     rule_data["label"],
            }

        elif intent == "delete_rule":
            intent_dict["action_params"]["rule_id"] = _parse_rule_id(text)

        reply = _pick(intent)
        if show_disclaimer:
            reply += DISCLAIMER

        save_conv_turn(db, user.id, "user",      user_message)
        save_conv_turn(db, user.id, "assistant", reply)

        return reply, intent_dict

    # ── Email analysis helpers ────────────────────────────────

    def classify_priority(self, subject: str, body: str) -> tuple[str, float]:
        """Score email priority using the trained priority model."""
        return self.priority_model.score(subject, body)

    def summarise_email(self, subject: str, body: str) -> str:
        """Extract a brief summary from the email body."""
        sentences = re.split(r"[.!?]\s+", body.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if sentences:
            summary = ". ".join(sentences[:2]) + "."
            return summary[:250] + "..." if len(summary) > 250 else summary
        return f"Email regarding: {subject}"

    def extract_actions(self, subject: str, body: str) -> list[str]:
        """Extract action items using pattern matching."""
        patterns = [
            r"please\s+(.+?)(?:[.!?]|$)",
            r"could you\s+(.+?)(?:[.!?]|$)",
            r"need(?:s)? to\s+(.+?)(?:[.!?]|$)",
            r"(?:action|todo|task):\s*(.+?)(?:[.!?]|$)",
            r"by\s+(?:monday|tuesday|wednesday|thursday|friday|eod|tomorrow)[,: ]+(.+?)(?:[.!?]|$)",
        ]
        actions = []
        combined = f"{subject} {body}"
        for p in patterns:
            for m in re.findall(p, combined, re.IGNORECASE):
                m = m.strip()
                if 10 < len(m) < 120:
                    actions.append(m[0].upper() + m[1:])
        return list(dict.fromkeys(actions))[:5]

    def draft_reply(self, subject: str, body: str, tone: str = "professional") -> str:
        """Generate a template-based email reply."""
        subj = subject[:50]
        if tone == "casual":
            return (
                f"Hey! Thanks for getting in touch about \"{subj}\". "
                "I'll have a look and get back to you soon. Cheers! 😊"
            )
        elif tone == "formal":
            return (
                f"Dear Sir/Madam,\n\n"
                f"Thank you for your correspondence regarding \"{subj}\". "
                "I will review this matter and respond in due course.\n\n"
                "Yours sincerely"
            )
        else:  # professional
            return (
                f"Hi,\n\n"
                f"Thank you for reaching out regarding \"{subj}\". "
                "I'll review this and get back to you shortly.\n\n"
                "Best regards"
            )

    # ── Training helpers ──────────────────────────────────────

    def add_intent_example(self, text: str, intent: str):
        """Add a labelled example to the intent model."""
        self.intent_model.add_example(text, intent)

    def record_passive_example(self, text: str, intent: str):
        """
        Passively record a successful interaction as a training example.
        Silently retrains the model every RETRAIN_EVERY examples.
        Call this after any successful intent execution.
        """
        global _passive_count
        self.intent_model.add_example(text, intent)
        _passive_count += 1
        if _passive_count >= RETRAIN_EVERY:
            _passive_count = 0
            try:
                self.intent_model.retrain()
            except Exception:
                pass  # retrain is best-effort, never crash the bot

    def learn_priority(self, subject: str, body: str, priority: str):
        """Record your priority rating for an email."""
        self.priority_model.learn(subject, body, priority)

    def retrain(self) -> dict:
        """Retrain all models. Returns accuracy stats."""
        intent_stats   = self.intent_model.retrain()
        priority_stats = self.priority_model.retrain()
        return {"intent": intent_stats, "priority": priority_stats}

    def model_stats(self) -> dict:
        return {
            "intent_examples":   self.intent_model.example_count,
            "priority_examples": self.priority_model.example_count,
            "intent_trained":    self.intent_model.is_trained,
            "priority_trained":  self.priority_model.is_trained,
        }

    # ── Internals ─────────────────────────────────────────────

    @staticmethod
    def _parse_label(text: str) -> str:
        m = re.search(
            r"(?:label|tag|mark|flag)[^a-z]*(?:as|:)?\s*['\"]?([a-z][a-z0-9_\- ]{0,25})['\"]?",
            text, re.IGNORECASE,
        )
        return m.group(1).strip() if m else "labelled"
