"""
AATAS — AI Conversation Brain (Self-Trained ML Edition)
No external APIs. No cloud. Pure local ML + NLP.

Intelligence features:
- Reference resolution: understands "it", "that", "the second one", "the last email"
- Context-aware intent: uses conversation history to fill in missing email index
- TF-IDF extractive summarisation: picks most informative sentences (no extra deps)
- Style mirroring: detects casual / formal / terse and replies to match
- Passive style learning: builds a per-user style profile in the DB over time
"""

import re
import random
import math
from collections import Counter
from typing import Optional

from db.database import (
    Session, User,
    get_conv_history, save_conv_turn,
    get_memories, upsert_memory,
)
from files.ml.trainer import get_intent_model, get_priority_model
from files.ml.rule_parser import parse_rule

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.60
RETRAIN_EVERY        = 10
_passive_count       = 0

DISCLAIMER = (
    "\n\n"
    "\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\u2015\n"
    "_\u26a0\ufe0f AI-generated \u00b7 may not be correct._\n"
    "_Reply_ `hide disclaimer` _to stop showing this._"
)
EMAIL_DISCLAIMER = "\n\n---\nSent via AATAS AI (Local Self-Trained ML)"


# ══════════════════════════════════════════════════════════════════════════════
# STYLE ENGINE — detects & mirrors the user's communication style
# ══════════════════════════════════════════════════════════════════════════════

_SLANG = re.compile(
    r"\b(yo|bro|dude|pls|plz|thx|tnx|lol|bruh|ngl|omg|asap|rn|tbh|"
    r"smh|imo|ikr|idk|wtf|gonna|wanna|gotta|kinda|ya|yup|nope|"
    r"u\b|r\b|ur\b|lemme|gimme|dunno|kk|np\b)\b",
    re.IGNORECASE,
)

_FORMAL = re.compile(
    r"\b(please|kindly|could you|would you|I would like|regarding|"
    r"concerning|with respect to|sincerely|herewith|pursuant)\b",
    re.IGNORECASE,
)


def _detect_style(text: str) -> str:
    wc = len(text.split())
    if wc <= 3 and not _FORMAL.search(text):
        return "terse"
    if _SLANG.search(text) or (text == text.lower() and wc > 2 and not _FORMAL.search(text)):
        return "casual"
    if _FORMAL.search(text):
        return "formal"
    return "casual"


def _blend_style(db: Session, user_id: int, current: str) -> str:
    """Update the per-user style count in Memory and return the dominant style."""
    mems    = get_memories(db, user_id)
    history = mems.get("style_history", "")
    counts: dict[str, int] = {}
    for entry in history.split(","):
        if ":" in entry:
            s, n = entry.rsplit(":", 1)
            counts[s] = int(n) if n.isdigit() else 0
    counts[current] = counts.get(current, 0) + 1
    upsert_memory(
        db, user_id, "style_history",
        ",".join(f"{s}:{n}" for s, n in counts.items()),
    )
    return max(counts, key=counts.get)


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE RESOLVER — "it", "that one", "the second one", subject keywords
# ══════════════════════════════════════════════════════════════════════════════

_ORDINALS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "last": -1, "latest": 1, "newest": 1, "top": 1, "bottom": -1, "oldest": -1,
}

_PRONOUN_RE = re.compile(
    r"\b(it|that|this|that one|this one|the email|the message|"
    r"that email|that message|this email|this message)\b",
    re.IGNORECASE,
)

_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(" + "|".join(re.escape(k) for k in _ORDINALS) + r")\s*(?:one|email|message|mail)?\b",
    re.IGNORECASE,
)

_REF_STOP = {
    "email", "mail", "message", "show", "get", "tell", "about",
    "the", "that", "this", "please", "can", "could", "you", "me", "my",
}


def resolve_email_index(
    text: str,
    cached_emails: list[dict],
    conv_history: list[dict],
) -> Optional[int]:
    """
    Intelligently work out which email the user means.
    Priority: explicit digit → ordinal word → pronoun → subject keyword.
    """
    # 1. Explicit digit
    m = re.search(r"(?:^|[^a-z])(\d+)(?:[^a-z]|$)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 2. Ordinal word
    m = _ORDINAL_RE.search(text)
    if m:
        n = _ORDINALS.get(m.group(1).lower())
        if n == -1 and cached_emails:
            return len(cached_emails)
        if n:
            return n

    # 3. Pronoun → scan assistant turns for last mentioned index "`N.`"
    if _PRONOUN_RE.search(text):
        for turn in reversed(conv_history):
            if turn["role"] == "assistant":
                m2 = re.search(r"[`#](\d+)[`.)]", turn["content"])
                if m2:
                    return int(m2.group(1))
        if cached_emails:
            return 1  # default: most-recent

    # 4. Subject keyword match
    qwords = [
        w.lower() for w in re.findall(r"\b[a-z]{4,}\b", text.lower())
        if w.lower() not in _REF_STOP
    ]
    if qwords and cached_emails:
        best_idx, best_score = None, 0
        for e in cached_emails:
            subj = e.get("subject", "").lower()
            sc   = sum(1 for w in qwords if w in subj)
            if sc > best_score:
                best_score, best_idx = sc, e["idx"]
        if best_score > 0:
            return best_idx

    return None


# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF EXTRACTIVE SUMMARISER — pure Python
# ══════════════════════════════════════════════════════════════════════════════

_BOILERPLATE_RE = re.compile(
    r"(please find attached|hope this (?:email |message )?finds you well|"
    r"i hope you(?:'re| are) doing well|dear (?:sir|madam|all)|"
    r"best regards.*|kind regards.*|yours (?:sincerely|faithfully).*|"
    r"thank you for your (?:email|message|time)\.?|"
    r"this email (?:was sent|is intended)|if you (?:received|are not))",
    re.IGNORECASE | re.DOTALL,
)

_SENT_SPLIT = re.compile(r"(?<![A-Z][a-z])(?<![Mm]r)(?<![Dd]r)[.!?]\s+")

_TFIDF_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "have", "will",
    "been", "they", "your", "our", "its", "are", "was", "has", "had",
    "not", "but", "can", "you", "all", "one", "any", "his", "her", "their",
    "into", "just", "also", "more", "when", "where", "which", "there",
}


def _tfidf_summary(body: str, subject: str, max_chars: int = 280) -> str:
    cleaned = _BOILERPLATE_RE.sub("", body)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    sentences = [s.strip() for s in _SENT_SPLIT.split(cleaned) if len(s.strip()) > 15]

    if not sentences:
        return f"Email regarding: {subject}"
    if len(sentences) == 1:
        s = sentences[0]
        return (s[:max_chars - 1] + "...") if len(s) > max_chars else s

    def tokenise(s: str) -> list[str]:
        return [w for w in re.findall(r"\b[a-z]{3,}\b", s.lower()) if w not in _TFIDF_STOP]

    tf_per_sent = [Counter(tokenise(s)) for s in sentences]
    N  = len(sentences)
    df: Counter = Counter()
    for tf in tf_per_sent:
        df.update(tf.keys())

    def score(tf: Counter) -> float:
        total = sum(tf.values()) or 1
        return sum(
            (count / total) * math.log((N + 1) / (df[term] + 1))
            for term, count in tf.items()
        )

    scores = [score(tf) for tf in tf_per_sent]
    subj_tokens = set(tokenise(subject))
    scores = [
        sc + 0.35 * sum(1 for t in subj_tokens if t in sentences[i].lower())
        for i, sc in enumerate(scores)
    ]

    ranked      = sorted(range(len(sentences)), key=lambda i: -scores[i])
    top_indices = sorted(ranked[:2])
    summary     = ". ".join(sentences[i] for i in top_indices)
    if not summary.endswith("."):
        summary += "."

    return (summary[:max_chars - 1] + "...") if len(summary) > max_chars else summary


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE POOL — casual / formal / terse variants per intent
# ══════════════════════════════════════════════════════════════════════════════

_RESPONSES: dict[str, dict[str, list[str]]] = {

    "fetch_inbox": {
        "casual": [
            "Pulling up your emails now 📬",
            "On it, grabbing your inbox!",
            "Let me fetch those for you real quick.",
            "Checking your mail...",
            "Loading your messages now.",
        ],
        "formal": [
            "Retrieving your inbox messages now.",
            "I'll pull up your latest emails right away.",
            "Fetching your inbox — just a moment, please.",
        ],
        "terse": ["Fetching inbox... 📬", "On it.", "Loading."],
    },

    "fetch_priority": {
        "casual": [
            "Let me figure out what actually needs your attention first!",
            "Sorting by what matters most — give me a sec.",
            "Ranking your inbox by priority now ⭐",
            "Finding what you should look at first.",
        ],
        "formal": [
            "Organising your emails by priority now.",
            "I'll rank your inbox by urgency momentarily.",
        ],
        "terse": ["Prioritising... ⭐", "Ranking now.", "On it."],
    },

    "analyse": {
        "casual": [
            "Let me have a look at that one for you 🔍",
            "Digging into that email now.",
            "Taking a closer look...",
            "Let me break that down for you.",
        ],
        "formal": [
            "Analysing that email now.",
            "I'll review the contents of that message for you.",
        ],
        "terse": ["Analysing... 🔍", "Looking into it.", "On it."],
    },

    "archive": {
        "casual": [
            "Archiving that — out of your way! 📦",
            "Gone from the inbox, archived.",
            "Done, tucked it away for you.",
            "Archived! ✅",
        ],
        "formal": [
            "Archiving that email now.",
            "I'll move that message to your archive.",
        ],
        "terse": ["Archiving. 📦", "Done. ✅", "Archived."],
    },

    "label": {
        "casual": ["Tagging it now! 🏷️", "Slapping a label on that.", "Labelling it for you."],
        "formal": ["Applying the label now.", "I'll categorise that email accordingly."],
        "terse": ["Labelling. 🏷️", "Tagged.", "Done."],
    },

    "trash": {
        "casual": [
            "Binning it! 🗑️",
            "Gone — deleted.",
            "Trashed.",
            "Into the bin it goes.",
            "Bye bye, email.",
        ],
        "formal": [
            "Deleting that email now.",
            "That message will be moved to trash.",
        ],
        "terse": ["Deleted. 🗑️", "Trashed.", "Gone."],
    },

    "reply": {
        "casual": [
            "Drafting a reply for you now ✍️",
            "Let me write something back for you.",
            "Getting a reply ready.",
            "On it — drafting a response.",
        ],
        "formal": [
            "Composing a reply to that email.",
            "I'll draft a response for your review.",
        ],
        "terse": ["Drafting... ✍️", "On it.", "Writing."],
    },

    "compose": {
        "casual": [
            "Let me put that email together for you 📝",
            "Drafting it now!",
            "Getting that email ready.",
        ],
        "formal": [
            "Composing a new email for you now.",
            "I'll prepare that message for your review.",
        ],
        "terse": ["Composing... 📝", "Drafting.", "On it."],
    },

    "create_rule": {
        "casual": [
            "Got it — setting that rule up! ⚙️",
            "Rule saved! I'll remember that every time you check.",
            "Done, I'll handle that automatically from now on.",
        ],
        "formal": [
            "Creating that automation rule now.",
            "The rule will be saved and applied on every inbox check.",
        ],
        "terse": ["Rule saved. ⚙️", "Done.", "Set."],
    },

    "delete_rule": {
        "casual": ["Removing that rule now 🗑️", "Done — rule deleted.", "Gone."],
        "formal": ["Deleting that automation rule.", "Removing the selected rule now."],
        "terse": ["Rule deleted. 🗑️", "Removed.", "Done."],
    },

    "list_rules": {
        "casual": [
            "Here's what I'm set up to do automatically 📋",
            "Your automation rules — coming right up!",
        ],
        "formal": ["Displaying your active automation rules."],
        "terse": ["Rules: 📋", "Here they are."],
    },

    "list_history": {
        "casual": [
            "Here's what I've been up to recently 📜",
            "Your recent action log — let me show you.",
        ],
        "formal": ["Displaying your recent action history."],
        "terse": ["Recent actions: 📜", "Here's the log."],
    },

    "none": {
        "casual": [
            "Hmm, not sure what you mean — try rephrasing?",
            "Didn't quite catch that. Something like _'show my inbox'_ or _'archive email 2'_?",
            "I'm lost on that one — can you say it differently?",
        ],
        "formal": [
            "I'm sorry, I didn't understand that request. Could you please rephrase?",
            "That wasn't clear to me — perhaps try _'show my inbox'_ or _'reply to email 2'_.",
        ],
        "terse": ["Didn't catch that.", "Try rephrasing.", "Rephrase?"],
    },

    "chat_greeting": {
        "casual": [
            "Hey! What's up? 👋",
            "Yo! Need help with your inbox?",
            "Hi there! Ready when you are.",
            "Hey 👋 What can I do for you?",
        ],
        "formal": [
            "Hello! How can I assist you today?",
            "Good day — how may I help you?",
        ],
        "terse": ["Hey 👋", "Hi!", "Hello."],
    },

    "chat_how_are_you": {
        "casual": [
            "Doing great, just sorting emails as usual! How about you? 🤖",
            "All good! What do you need?",
            "Running smoothly — ready to tackle your inbox.",
        ],
        "formal": [
            "I'm functioning well, thank you. How may I assist you?",
            "All systems operational. How can I help?",
        ],
        "terse": ["Good, thanks! You?", "All good. 💻"],
    },

    "chat_identity": {
        "casual": [
            "I'm AATAS — your private email assistant! Runs entirely on your machine 🧠",
            "I'm your self-trained AI. Everything stays local — no cloud needed.",
        ],
        "formal": [
            "I am AATAS — an AI-assisted email management system running locally on your device.",
        ],
        "terse": ["AATAS — your local AI. 🧠", "I'm AATAS, your email bot."],
    },

    "chat_thanks": {
        "casual": [
            "No worries! Let me know if you need anything else 😊",
            "Anytime! 🚀",
            "Happy to help!",
        ],
        "formal": [
            "You're very welcome. Please don't hesitate to ask.",
            "Not at all — glad I could help.",
        ],
        "terse": ["No problem! 🌟", "Anytime.", "Sure."],
    },

    "chat_goodbye": {
        "casual": [
            "Catch you later! 👋",
            "See ya! Have a good one 🌟",
            "Bye for now — I'll be here when you need me.",
        ],
        "formal": ["Goodbye. Have a productive day."],
        "terse": ["Bye! 👋", "See ya.", "Later."],
    },
}


def pick_response(intent: str, style: str) -> str:
    bucket  = _RESPONSES.get(intent, _RESPONSES["none"])
    options = bucket.get(style) or bucket.get("casual") or list(bucket.values())[0]
    return random.choice(options)


# ══════════════════════════════════════════════════════════════════════════════
# PARSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_tone(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["casual", "friendly", "informal", "chill"]):
        return "casual"
    if any(w in t for w in ["formal", "official", "business", "professional"]):
        return "formal"
    return "professional"


def _parse_rule_id(text: str) -> Optional[int]:
    m = re.search(r"rule\s+(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_compose_details(text: str) -> dict:
    details: dict = {"to": "", "subject": "", "message": ""}
    m = re.search(r"\b[\w.+\-]+@[\w\-]+\.\w+\b", text)
    if m:
        details["to"] = m.group(0)
    else:
        m2 = re.search(
            r"\b(?:to|email|tell)\s+([a-zA-Z0-9_ ]{2,20}?)(?:\s+(?:saying|about|that|with)|$)",
            text, re.IGNORECASE,
        )
        if m2:
            details["to"] = m2.group(1).strip()
    m = re.search(
        r"\b(?:about|regarding|re:?)\s+([a-zA-Z0-9_ ]{2,50}?)(\s+(?:saying|with|that)|$)",
        text, re.IGNORECASE,
    )
    if m:
        details["subject"] = m.group(1).strip()
    m = re.search(
        r"\b(?:saying|that|with message|say|tell (?:him|her|them))\s+(.+)",
        text, re.IGNORECASE,
    )
    if m:
        details["message"] = m.group(1).strip()
    return details


# ══════════════════════════════════════════════════════════════════════════════
# BRAIN
# ══════════════════════════════════════════════════════════════════════════════

class AATASBrain:
    """AATAS self-trained ML brain. No external API calls."""

    def __init__(self):
        self.intent_model   = get_intent_model()
        self.priority_model = get_priority_model()

    # ── Core chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        db:              Session,
        user:            User,
        user_message:    str,
        context_hint:    str        = "",
        cached_emails:   list[dict] = None,
        show_disclaimer: bool       = True,
    ) -> tuple[list[str], list[dict]]:
        """
        Process a user message with full contextual intelligence.
        - Resolves pronouns/ordinals using conversation history + email cache
        - Mirrors the user's communication style
        - Returns ([reply_texts], [intent_dicts])
        """
        text   = user_message.strip()
        cached = cached_emails or []

        conv_history   = get_conv_history(db, user.id, last_n=10)
        current_style  = _detect_style(text)
        dominant_style = _blend_style(db, user.id, current_style)

        predictions = self.intent_model.predict_multi(text, threshold=0.25)

        reply_texts:  list[str]  = []
        intent_dicts: list[dict] = []

        email_idx  = resolve_email_index(text, cached, conv_history)
        label_name = self._parse_label(text)
        tone_name  = _parse_tone(text)

        for intent, confidence in predictions:

            if confidence < CONFIDENCE_THRESHOLD and len(predictions) == 1:
                reply = self._low_confidence_reply(dominant_style)
                save_conv_turn(db, user.id, "user",      user_message)
                save_conv_turn(db, user.id, "assistant", reply)
                return [reply], []

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            intent_dict: dict = {
                "intent":        intent,
                "email_idx":     None,
                "action_params": {},
            }

            if intent in ("analyse", "archive", "label", "trash", "reply",
                          "fetch_inbox", "fetch_priority", "compose"):
                intent_dict["email_idx"] = email_idx
                if intent == "label":
                    intent_dict["action_params"]["label"] = label_name
                if intent == "reply":
                    intent_dict["action_params"]["tone"] = tone_name
                if intent == "compose":
                    intent_dict["action_params"] = _parse_compose_details(text)
                    intent_dict["action_params"]["tone"] = tone_name

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

            reply = pick_response(intent, dominant_style)
            reply_texts.append(reply)
            intent_dicts.append(intent_dict)

        if not reply_texts:
            reply = pick_response("none", dominant_style)
            save_conv_turn(db, user.id, "user",      user_message)
            save_conv_turn(db, user.id, "assistant", reply)
            return [reply], []

        if show_disclaimer:
            reply_texts[-1] += DISCLAIMER

        save_conv_turn(db, user.id, "user",      user_message)
        save_conv_turn(db, user.id, "assistant", "\n\n".join(reply_texts))

        return reply_texts, intent_dicts

    # ── Low-confidence reply ───────────────────────────────────────────────

    def _low_confidence_reply(self, style: str) -> str:
        if style == "formal":
            return random.choice([
                "I'm sorry, I wasn't able to determine your intent. Could you rephrase? "
                "For example: _'show my inbox'_ or _'archive email 2'_.",
                "I didn't quite understand that. Perhaps try _'check priority'_ or _'trash email 3'_.",
            ])
        if style == "terse":
            return random.choice([
                "Didn't get that — try _'inbox'_ or _'archive 2'_.",
                "Not sure — rephrase? e.g. _'trash 1'_.",
            ])
        return random.choice([
            "Hmm, not sure what you're after — something like _'show my inbox'_ or _'archive email 2'_?",
            "Didn't quite catch that. Try _'trash email 3'_ or _'check priority'_.",
            "I'm not following — can you rephrase? e.g. _'check my emails'_ or _'reply to email 1'_.",
        ])

    # ── Email analysis helpers ─────────────────────────────────────────────

    def classify_priority(self, subject: str, body: str) -> tuple[str, float]:
        return self.priority_model.score(subject, body)

    def summarise_email(self, subject: str, body: str) -> str:
        """TF-IDF extractive summary — picks the most informative sentences."""
        return _tfidf_summary(body, subject, max_chars=280)

    def extract_actions(self, subject: str, body: str) -> list[str]:
        patterns = [
            r"please\s+(.+?)(?:[.!?]|$)",
            r"could you\s+(.+?)(?:[.!?]|$)",
            r"can you\s+(.+?)(?:[.!?]|$)",
            r"need(?:s)? to\s+(.+?)(?:[.!?]|$)",
            r"(?:action|todo|task)[:\s]+(.+?)(?:[.!?]|$)",
            r"by\s+(?:monday|tuesday|wednesday|thursday|friday|eod|today|tomorrow)[,: ]+(.+?)(?:[.!?]|$)",
            r"deadline[:\s]+(.+?)(?:[.!?]|$)",
        ]
        seen:    set[str]  = set()
        actions: list[str] = []
        for p in patterns:
            for m in re.findall(p, f"{subject} {body}", re.IGNORECASE):
                m = m.strip()
                if 8 < len(m) < 120:
                    key = m.lower()[:40]
                    if key not in seen:
                        seen.add(key)
                        actions.append(m[0].upper() + m[1:])
        return actions[:5]

    def draft_reply(self, subject: str, body: str, tone: str = "professional") -> str:
        subj = subject[:60]
        sender_name = ""
        m = re.search(r"(?:hi|hello|hey|dear)\s+([A-Z][a-z]{1,20})\b", body)
        if m:
            sender_name = m.group(1)
        is_question = bool(re.search(r"\?", body))
        follow_up   = (
            "I'll look into this and get back to you."
            if is_question else
            "I've noted the details and will follow up accordingly."
        )
        greeting = sender_name or "there"
        if tone == "casual":
            text = (
                f"Hey {greeting}! Thanks for the message about \"{subj}\". "
                f"{follow_up} Cheers! 😊"
            )
        elif tone == "formal":
            salutation = f"Dear {sender_name}," if sender_name else "Dear Sir/Madam,"
            text = (
                f"{salutation}\n\n"
                f"Thank you for your correspondence regarding \"{subj}\". "
                f"{follow_up}\n\n"
                "Yours sincerely,"
            )
        else:
            text = (
                f"Hi {greeting},\n\n"
                f"Thanks for getting in touch about \"{subj}\". "
                f"{follow_up}\n\n"
                "Best regards,"
            )
        return text + EMAIL_DISCLAIMER

    def draft_new_email(
        self,
        recipient: str,
        subject:   str,
        message:   str,
        tone:      str = "professional",
    ) -> tuple[str, str]:
        final_subject = subject or "Message from AATAS"
        final_subject = final_subject[0].upper() + final_subject[1:]
        content = message or "I wanted to reach out — please let me know if you have any questions."
        r = recipient or "there"
        if tone == "casual":
            body = f"Hey {r},\n\n{content}\n\nCheers! 😊"
        elif tone == "formal":
            body = (
                f"Dear {r},\n\n"
                f"I am writing regarding {final_subject}.\n\n{content}\n\n"
                "Yours sincerely,"
            )
        else:
            body = f"Hi {r},\n\nHope you're doing well. {content}\n\nBest regards,"
        return final_subject, body + EMAIL_DISCLAIMER

    # ── Training helpers ───────────────────────────────────────────────────

    def add_intent_example(self, text: str, intent: str):
        self.intent_model.add_example(text, intent)

    def record_passive_example(self, text: str, intent: str):
        global _passive_count
        self.intent_model.add_example(text, intent)
        _passive_count += 1
        if _passive_count >= RETRAIN_EVERY:
            _passive_count = 0
            try:
                self.intent_model.retrain()
            except Exception:
                pass

    def learn_priority(self, subject: str, body: str, priority: str):
        self.priority_model.learn(subject, body, priority)

    def retrain(self) -> dict:
        return {
            "intent":   self.intent_model.retrain(),
            "priority": self.priority_model.retrain(),
        }

    def model_stats(self) -> dict:
        return {
            "intent_examples":   self.intent_model.example_count,
            "priority_examples": self.priority_model.example_count,
            "intent_trained":    self.intent_model.is_trained,
            "priority_trained":  self.priority_model.is_trained,
        }

    @staticmethod
    def _parse_label(text: str) -> str:
        m = re.search(
            r"(?:label|tag|mark|flag)[^a-z]*(?:as|:)?\s*['\"]?([a-z][a-z0-9_\- ]{0,25})['\"]?",
            text, re.IGNORECASE,
        )
        return m.group(1).strip() if m else "labelled"