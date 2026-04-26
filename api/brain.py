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
import logging
from api.memory_guesser import MemoryGuesser
import math
from collections import Counter
from typing import Optional

log = logging.getLogger(__name__)

from db.database import (
    Session, User,
    get_conv_history, save_conv_turn,
    get_memories, upsert_memory,
    search_knowledge,
)
from files.ml.trainer import get_intent_model, get_priority_model
from files.ml.neural_fallback import NeuralFallback
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


_STYLE_COMMANDS = {
    "friendly": ["be friendly", "be more friendly", "speak casually", "be chill", "be nice", "casual mode", "casual"],
    "formal":   ["be formal", "speak formally", "professional tone", "be official", "formal mode", "formal"],
    "terse":    ["be terse", "be brief", "short answers only", "stay brief"],
    "jarvis":   ["be jarvis", "jarvis mode", "iron man", "speak like jarvis", "jarvis", "activate jarvis"],
    # Revert commands clear any saved style preference and return to auto-detection
    "normal":   [
        "revert to normal", "stop jarvis", "exit jarvis", "disable jarvis",
        "normal mode", "default mode", "reset style", "stop being jarvis",
        "be normal", "turn off jarvis", "deactivate jarvis",
    ],
}


def _detect_style(text: str, memories: dict[str, str] = None) -> str:
    t = text.lower()
    
    # Check for explicit style commands
    for style, commands in _STYLE_COMMANDS.items():
        if any(cmd in t for cmd in commands):
            return "FORCE:" + style

    wc = len(text.split())
    
    # Jarvis is only activated via explicit commands ("be jarvis", "jarvis mode")
    # handled above by _STYLE_COMMANDS → FORCE: path.
    # Auto-detecting on "sir"/"jarvis" in any message caused it to permanently
    # stick in the DB via _blend_style even during normal searches.

    # Polite words trigger friendly/casual assistant tone
    if any(w in t for w in ["please", "thanks", "thank you", "could you", "would you", "hey", "hi", "hello"]):
        return "casual"
        
    if wc <= 3 and not _FORMAL.search(text):
        return "terse"
    if _SLANG.search(text) or (text == text.lower() and wc > 2 and not _FORMAL.search(text)):
        return "casual"
    if _FORMAL.search(text):
        return "formal"
    return "casual"


def _normalize_text(text: str) -> str:
    """Collapse repeated characters (3+ consecutive) down to 2 to improve ML matching."""
    # "hiii" -> "hii", "hellooo" -> "helloo"
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def _blend_style(db: Session, user_id: int, current: str) -> str:
    """Update the per-user style count in Memory and return the dominant style."""
    mems    = get_memories(db, user_id)

    # If user explicitly set a style in this message, save it as their preference
    if current.startswith("FORCE:"):
        pref = current.split(":")[1]

        # "normal" is a special revert token — clear the saved preference entirely
        # so auto-detection takes over again for future messages.
        if pref == "normal":
            if "preferred_style" in mems:
                from db.database import Memory
                db.query(Memory).filter_by(
                    user_id=user_id, key="preferred_style"
                ).delete()
                db.commit()
            return "casual"  # sensible default immediately after revert

        upsert_memory(db, user_id, "preferred_style", pref)
        return pref

    # If they have a saved preference, always use that
    if "preferred_style" in mems:
        return mems["preferred_style"]

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
    r"that email|that message|this email|this message|him|her|them)\b",
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

def _esc(text) -> str:
    """Escape text for Telegram legacy Markdown (v1)."""
    if text is None: return ""
    import re
    return re.sub(r"([*_`\[])", r"\\\1", str(text))


def _extract_email_entities(db: Session, user_id: int, emails: list[dict]):
    """Scan emails for recurring entities or important info."""
    # This is a passive background task. We look for names and subjects.
    for e in emails:
        # Example: if an email is from "John Doe <john@example.com>"
        # we can save john@example.com -> John Doe
        sender = e.get("sender", "")
        if "<" in sender and ">" in sender:
            name, email = sender.split("<")
            name = name.strip()
            email = email.replace(">", "").strip()
            if name and email:
                upsert_memory(db, user_id, f"contact:{email}", name)


def _search_history(db: Session, user_id: int, query: str) -> str:
    """Keyword search through past conversation turns."""
    from db.database import ConversationTurn
    
    q_words = [w for w in query.lower().split() if len(w) > 3]
    if not q_words: return ""
    
    turns = db.query(ConversationTurn).filter_by(user_id=user_id).all()
    matches = []
    for t in turns:
        content = t.content.lower()
        score = sum(1 for w in q_words if w in content)
        if score > 0:
            matches.append((score, t))
    
    if not matches: return ""
    
    # Return the most relevant turn
    matches.sort(key=lambda x: x[0], reverse=True)
    best = matches[0][1]
    return f"On {best.created_at.strftime('%Y-%m-%d')}, you said: \"{best.content}\""


def resolve_email_index(
    text: str,
    cached_emails: list[dict],
    conv_history: list[dict],
) -> Optional[int]:
    """
    Intelligently work out which email the user means.
    Priority: explicit digit → ordinal word → pronoun → subject keyword.
    """
    t_low = text.lower()

    # 1. Explicit digit
    m = re.search(r"(?:^|[^a-z])(\d+)(?:[^a-z]|$)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 2. Ordinal word (handle "latest", "newest", "last")
    m = _ORDINAL_RE.search(text)
    if m:
        word = m.group(1).lower()
        n = _ORDINALS.get(word)
        if n == -1 and cached_emails:
            return len(cached_emails)
        if n == 1 and ("latest" in word or "newest" in word or "top" in word):
            return 1
        if n:
            return n

    # 3. Handle phrases like "the email I just sent" or "the one I sent myself"
    if any(w in t_low for w in ["i just sent", "sent myself", "my own email"]):
        if cached_emails:
            # Look for an email where the sender is "me" or matches the user's name if possible
            # For now, just default to 1 (the most recent)
            return 1

    # 4. Pronoun → scan assistant turns for last mentioned index "`N.`"
    if _PRONOUN_RE.search(text):
        for turn in reversed(conv_history):
            if turn["role"] == "assistant":
                m2 = re.search(r"[`#](\d+)[`.)]", turn["content"])
                if m2:
                    return int(m2.group(1))
        if cached_emails:
            return 1  # default: most-recent

    # 5. Subject keyword match
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
            "I'm on it! Pulling up your recent emails now. 📬",
            "Sure thing! Let me grab your latest messages for you.",
            "Checking your inbox now. Just a second!",
            "I'll fetch those for you right away. 📬",
            "Looking into your mail... here's what I found.",
        ],
        "formal": [
            "Retrieving your inbox messages now.",
            "I'll pull up your latest emails for your review.",
            "Fetching your inbox — just a moment, please.",
        ],
        "terse": ["Fetching inbox... 📬", "On it.", "Loading."],
        "jarvis": [
            "Scanning the frequencies now, Sir. Your inbox is displayed.",
            "Accessing the Gmail servers... Here is the latest data, Sir.",
            "Right away, Sir. Initialising inbox sweep.",
            "Everything seems to be in order. Here are your recent messages, Sir.",
        ],
    },

    "fetch_priority": {
        "casual": [
            "Let me see what needs your attention most. 🔍",
            "Sorting through your mail to find the important stuff!",
            "Ranking your emails by priority now. ⭐",
            "I'll find the most urgent messages for you.",
        ],
        "formal": [
            "Organising your emails by priority now.",
            "I'll rank your inbox by urgency momentarily.",
        ],
        "terse": ["Prioritising... ⭐", "Ranking now.", "On it."],
        "jarvis": [
            "Analysing urgency levels now, Sir. These require your attention.",
            "Prioritising your communications, Sir. The critical items are at the top.",
            "Ranking the data stream... Most urgent items are ready for review, Sir.",
        ],
    },

    "analyse": {
        "casual": [
            "I'll take a look at that one for you! 🔍",
            "Let me break down the details of that email.",
            "Digging into the contents for you now.",
            "I'll check that message and give you a summary.",
        ],
        "formal": [
            "Analysing that email now.",
            "I'll review the contents of that message for you.",
        ],
        "terse": ["Analysing... 🔍", "Looking into it.", "On it."],
        "jarvis": [
            "Running a full diagnostic on that message, Sir.",
            "Analysing the data packets now... Here is a summary of the contents, Sir.",
            "Parsing the details for you as we speak, Sir.",
        ],
    },

    "search": {
        "casual": [
            "Searching your emails now! 🔍",
            "Let me find those for you. Just a moment.",
            "I'm looking through your messages... 🔍",
            "I'll search for that right away.",
        ],
        "formal": [
            "Searching your email records now.",
            "I'll retrieve the relevant messages for you.",
        ],
        "terse": ["Searching... 🔍", "Looking it up.", "Scanning."],
        "jarvis": [
            "Accessing the local archives, Sir. Searching now.",
            "Scanning the message database... I'll have the results for you momentarily, Sir.",
            "Retrieving the requested data from your mail, Sir.",
        ],
    },

    "archive": {
        "casual": [
            "Done! I've archived that for you. 📦",
            "No problem, that's moved to your archive now.",
            "Tucked that one away! It's out of your inbox. ✅",
            "Archived! Let me know if you need anything else.",
        ],
        "formal": [
            "Archiving that email now.",
            "I'll move that message to your archive as requested.",
        ],
        "terse": ["Archiving. 📦", "Done. ✅", "Archived."],
        "jarvis": [
            "Message filed away, Sir. It's safe in the archives.",
            "Relocating to the long-term storage, Sir. Handled.",
            "Archive protocol complete, Sir.",
        ],
    },

    "label": {
        "casual": [
            "I've added that label for you! 🏷️",
            "Sure thing, tagging it now.",
            "All set! I've labelled that email.",
        ],
        "formal": ["Applying the label now.", "I'll categorise that email accordingly."],
        "terse": ["Labelling. 🏷️", "Tagged.", "Done."],
        "jarvis": [
            "Tagging that data point now, Sir.",
            "Label applied. The database has been updated, Sir.",
            "Categorising according to your instructions, Sir.",
        ],
    },

    "trash": {
        "casual": [
            "I've moved that to the trash for you. 🗑️",
            "No problem, that email is gone.",
            "Done! I've deleted that message.",
            "Into the bin it goes! 🗑️",
        ],
        "formal": [
            "Deleting that email now.",
            "That message will be moved to trash.",
        ],
        "terse": ["Deleted. 🗑️", "Trashed.", "Gone."],
        "jarvis": [
            "Redundant data deleted, Sir.",
            "Incinerating the message as we speak, Sir. It's gone.",
            "Trash protocol executed, Sir.",
        ],
    },

    "reply": {
        "casual": [
            "I'm drafting a reply for you right now! ✍️",
            "Sure thing! Let me write something back for you.",
            "Working on a response for you now.",
            "I'll get a draft ready for that email.",
        ],
        "formal": [
            "Composing a reply to that email.",
            "I'll draft a response for your review.",
        ],
        "terse": ["Drafting... ✍️", "On it.", "Writing."],
        "jarvis": [
            "Initialising the response protocol, Sir. Drafting now.",
            "Composing a suitable reply, Sir. I'll have it ready for your approval shortly.",
            "Writing the draft as we speak, Sir.",
        ],
    },

    "compose": {
        "casual": [
            "I'll put that email together for you! 📝",
            "Drafting it now — let me know if it looks good.",
            "Getting your new email ready.",
        ],
        "formal": [
            "Composing a new email for you now.",
            "I'll prepare that message for your review.",
        ],
        "terse": ["Composing... 📝", "Drafting.", "On it."],
        "jarvis": [
            "Preparing a new outgoing transmission, Sir.",
            "Composing the message now, Sir. Please review it before I send it off.",
            "Drafting your communication, Sir.",
        ],
    },

    "create_rule": {
        "casual": [
            "I've got it! That rule is all set up. ⚙️",
            "Sure thing, I'll handle that automatically from now on.",
            "Done! I've saved that rule for you.",
        ],
        "formal": [
            "Creating that automation rule now.",
            "The rule will be saved and applied on every inbox check.",
        ],
        "terse": ["Rule saved. ⚙️", "Done.", "Set."],
        "jarvis": [
            "New protocol established, Sir. I'll monitor for matches automatically.",
            "Automation rule uploaded to the core, Sir.",
            "Understood, Sir. I have set a permanent watch for those conditions.",
        ],
    },

    "delete_rule": {
        "casual": [
            "No problem, I've removed that rule for you. 🗑️",
            "Done — that rule is now deleted.",
            "I've cleared that automation for you.",
        ],
        "formal": ["Deleting that automation rule.", "Removing the selected rule now."],
        "terse": ["Rule deleted. 🗑️", "Removed.", "Done."],
        "jarvis": [
            "Protocol deactivated, Sir.",
            "The automation has been purged from the system, Sir.",
            "Rule removed, Sir. Back to manual monitoring for that category.",
        ],
    },

    "list_rules": {
        "casual": [
            "Here's a list of everything I'm doing for you automatically: 📋",
            "Sure! Here are your active automation rules.",
        ],
        "formal": ["Displaying your active automation rules."],
        "terse": ["Rules: 📋", "Here they are."],
        "jarvis": [
            "Displaying your current active protocols, Sir.",
            "Here is the list of automations currently running in the background, Sir.",
        ],
    },

    "list_history": {
        "casual": [
            "Here's what I've been up to lately: 📜",
            "I'll show you my recent actions right now.",
        ],
        "formal": ["Displaying your recent action history."],
        "terse": ["Recent actions: 📜", "Here's the log."],
        "jarvis": [
            "Retrieving the mission logs now, Sir.",
            "Here is the record of my recent operations, Sir.",
        ],
    },

    "recall": {
        "casual": [
            "Let me check my memory... here's what I found: 🧠",
            "Searching my notes for you... ah, here it is!",
            "I remember you mentioning that! Here's the info:",
            "Checking your history... here's what I recall.",
        ],
        "formal": [
            "Searching historical records for the requested information.",
            "I have retrieved the following information from our previous correspondence.",
        ],
        "terse": ["Searching... 🧠", "Recall: ", "Found it."],
        "jarvis": [
            "Accessing the memory banks now, Sir. Here is what I recall.",
            "Searching the data logs... I have found the relevant information, Sir.",
            "My records indicate the following, Sir.",
        ],
    },

    "none": {
        "casual": [
            "I'm sorry, I'm not quite sure what you mean. Could you try rephrasing that?",
            "I didn't quite catch that. Would you like me to show your inbox or archive an email?",
            "I'm a little lost — could you say that a different way? 😊",
        ],
        "formal": [
            "I'm sorry, I didn't understand that request. Could you please rephrase?",
            "That wasn't clear to me — perhaps try _'show my inbox'_ or _'reply to email 2'_.",
        ],
        "terse": ["Didn't catch that.", "Try rephrasing.", "Rephrase?"],
        "jarvis": [
            "I'm afraid I didn't quite catch that, Sir. Shall I run a scan for new messages instead?",
            "The request was unclear, Sir. Perhaps a different phrasing?",
            "I'm not following, Sir. Could you clarify?",
        ],
    },

    "chat_greeting": {
        "casual": [
            "Hello! How can I help you today? 👋",
            "Hi there! Ready to tackle your inbox whenever you are.",
            "Hey! Need a hand?",
            "Hello! I'm here to help you today.",
        ],
        "formal": [
            "Hello! How can I assist you today?",
            "Good day — how may I help you?",
        ],
        "terse": ["Hey 👋", "Hi!", "Hello."],
        "jarvis": [
            "At your service, Sir. How may I assist you today?",
            "Online and ready, Sir. What's the mission for today?",
            "Good day, Sir. I am standing by for your instructions.",
        ],
    },

    "chat_how_are_you": {
        "casual": [
            "I'm doing great! Just keeping your emails organized. How about you? 🤖",
            "All systems go! Ready to help you out.",
            "I'm running smoothly and ready for action!",
        ],
        "formal": [
            "I'm functioning well, thank you. How may I assist you?",
            "All systems operational. How can I help?",
        ],
        "terse": ["Good, thanks! You?", "All good. 💻"],
        "jarvis": [
            "All systems are green, Sir. I am functioning at peak capacity.",
            "Optimal performance levels detected, Sir. Thank you for asking.",
        ],
    },

    "chat_identity": {
        "casual": [
            "I'm AATAS, your personal AI email assistant! I run entirely on your machine and learn about your preferences over time. 🧠",
            "I'm your self-trained AI helper. Everything stays private and local, and the more we talk, the better I understand how you work.",
        ],
        "formal": [
            "I am AATAS — an AI-assisted email management system running locally on your device.",
        ],
        "terse": ["AATAS — your local AI. 🧠", "I'm AATAS, your email bot."],
        "jarvis": [
            "I am AATAS, Sir. Think of me as your personal digital butler, running locally and securely on your own hardware.",
        ],
    },

    "chat_thanks": {
        "casual": [
            "You're very welcome! Let me know if there's anything else I can do. 😊",
            "Anytime! I'm happy to help.",
            "No problem at all!",
        ],
        "formal": [
            "You're very welcome. Please don't hesitate to ask if you need further assistance.",
            "Not at all — I am glad I could be of help.",
        ],
        "terse": ["No problem! 🌟", "Anytime.", "Sure."],
        "jarvis": [
            "It is my pleasure, Sir.",
            "Always happy to assist, Sir.",
            "Don't mention it, Sir.",
        ],
    },

    "web_search": {
        "casual": [
            "I'll search the web for that right away! 🌐",
            "Searching for info now... let me see what I can find.",
            "I'll look that up for you. Just a second!",
        ],
        "formal": [
            "Searching the internet for the requested information.",
            "I'll retrieve the relevant search results for your review.",
        ],
        "terse": ["Searching... 🌐", "Looking it up.", "Scanning."],
        "jarvis": [
            "Scanning the global data stream now, Sir.",
            "Accessing the internet protocols, Sir. Searching for the requested data.",
            "Initialising a web-wide search, Sir. One moment please.",
            "Retrieving information from the external networks, Sir.",
        ],
    },

    "research": {
        "casual": [
            "I'll take a deep dive into that page for you! 🧠",
            "Researching the details now. I'll give you a summary.",
            "Let me read through that and summarize the important parts.",
        ],
        "formal": [
            "Analysing the specified web content now.",
            "I will provide a comprehensive summary of the page shortly.",
        ],
        "terse": ["Researching... 🧠", "Analysing.", "Summarising."],
        "jarvis": [
            "Analysing the data from that source now, Sir.",
            "Running a deep-scan on the webpage, Sir. I'll have a summary for you in a moment.",
            "Compiling a research report on that URL, Sir.",
        ],
    },

    "math_query": {
        "casual": [
            "On it! Let me crunch that for you. 🔢",
            "Great question — here's what I know about that! 📐",
            "Let's work through this together. 🧮",
        ],
        "formal": [
            "Retrieving the relevant mathematical explanation.",
            "Here is the pertinent mathematical information.",
        ],
        "terse": ["Calculating... 🔢", "Math answer:", "Here:"],
        "jarvis": [
            "Running the mathematical analysis now, Sir.",
            "Accessing the knowledge matrix for that formula, Sir.",
            "Retrieving the requested mathematical data, Sir.",
        ],
    },

    "science_query": {
        "casual": [
            "Great science question! Here's what I've got. 🔬",
            "Let me pull that up from my science knowledge. ⚗️",
            "Science time! Here's what I know. 🧪",
        ],
        "formal": [
            "Retrieving the relevant scientific explanation.",
            "Here is the pertinent scientific information.",
        ],
        "terse": ["Science answer:", "Here:", "Got it:"],
        "jarvis": [
            "Accessing the scientific database, Sir.",
            "Running the scientific analysis, Sir.",
            "Retrieving the requested scientific data, Sir.",
        ],
    },

    "chat_goodbye": {
        "casual": [
            "Goodbye! Have a wonderful day. 👋",
            "See you later! I'll be here if you need me.",
            "Take care! Talk to you soon. 🌟",
        ],
        "formal": ["Goodbye. Have a productive day."],
        "terse": ["Bye! 👋", "See ya.", "Later."],
        "jarvis": [
            "Goodbye, Sir. I'll be right here if you need me.",
            "Signing off for now, Sir. Have a productive afternoon.",
        ],
    },
    "revert_style": {
        "casual": [
            "Done! I'm back to my normal self. 😊",
            "Jarvis mode off — talking normally again!",
            "Style reset! Back to regular mode.",
        ],
        "formal": [
            "Understood. Reverting to the default communication style.",
            "Style preference cleared. Standard mode is now active.",
        ],
        "terse": ["Normal mode. ✅", "Style reset.", "Done."],
        "jarvis": [
            "As you wish, Sir. Reverting to standard operating mode.",
            "Jarvis Protocol suspended, Sir. Defaulting to casual mode.",
        ],
    },
    "code_generate": {
        "casual": [
            "Sure! Let me write that for you. 💻",
            "On it! Here's the code:",
            "Here you go! 🧑‍💻",
        ],
        "terse": ["Here:", "Code below:", "💻"],
        "formal": ["Certainly. Here is the requested code:", "As requested:"],
        "jarvis": [
            "Compiling your request now, Sir.",
            "Writing the code as instructed, Sir.",
        ],
    },
    "code_explain": {
        "casual": [
            "Let me break that down for you! 🔍",
            "Sure, here's what that code does:",
            "Happy to explain! 👇",
        ],
        "terse": ["Here's the breakdown:", "Explanation:"],
        "formal": ["I will now explain the provided code:", "Here is the explanation:"],
        "jarvis": [
            "Analysing the code now, Sir.",
            "Allow me to break this down for you, Sir.",
        ],
    },
    "code_debug": {
        "casual": [
            "Let me take a look at what's wrong! 🐛",
            "I'll find the bug for you!",
            "On it — let me debug this. 🔧",
        ],
        "terse": ["Debugging:", "Found it:"],
        "formal": ["I will now analyse the code for errors:", "Reviewing for bugs:"],
        "jarvis": [
            "Running diagnostics on your code, Sir.",
            "Scanning for errors now, Sir.",
        ],
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
        self.priority_model.load_or_init()
        self.neural_fallback = NeuralFallback()
        self.neural_fallback.load_or_init(self.intent_model)
        if not self.neural_fallback.is_trained:
            self.neural_fallback.train(self.intent_model.training_data)
        self.guesser        = MemoryGuesser()

    def _smart_memory_guesser(self, db: Session, user_id: int, text: str) -> list[str]:
        """Use the Guessing Engine to learn about the user."""
        mems = get_memories(db, user_id)
        facts = self.guesser.guess_facts(text, existing_memories=mems)
        confirmations = []
        
        for f in facts:
            upsert_memory(db, user_id, f["key"], f["value"])
            if f["confidence"] >= 0.7:
                key_display = f["key"].replace("fact:", "").replace("_", " ")
                # Jarvis-style confirmation if applicable
                style_pref = mems.get("preferred_style", "casual")
                if style_pref == "jarvis":
                    confirmations.append(f"Memory updated, Sir. I've logged your *{_esc(key_display)}* as *{_esc(f['value'])}*.")
                else:
                    confirmations.append(f"I've noted that your *{_esc(key_display)}* is *{_esc(f['value'])}*.")
        
        return confirmations

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
        # Strip leading filler words that confuse intent detection
        # e.g. "anyways who am i" → "who am i", "so like what is photosynthesis" → "what is photosynthesis"
        import re as _re_filler
        text = _re_filler.sub(
            r"^(?:anyways?|so\s+like|so|btw|by\s+the\s+way|like|okay\s+so|ok\s+so|right\s+so|also|also\s+like)[,\s]+",
            "", text, flags=_re_filler.IGNORECASE
        ).strip() or text
        cached = cached_emails or []

        # Normalise common typos and shorthand before intent detection
        _TYPO_MAP = [
            (r"todays",     "today's"),
            (r"tomorrows",  "tomorrow's"),
            (r"weathers?",  "weather"),
            (r"wats",       "what is"),
            (r"wut",        "what"),
            (r"u",          "you"),
            (r"ur",         "your"),
            (r"r",          "are"),
            (r"pls",        "please"),
            (r"plz",        "please"),
            (r"thx",        "thanks"),
            (r"tnx",        "thanks"),
            (r"gonna",      "going to"),
            (r"wanna",      "want to"),
            (r"gotta",      "got to"),
            (r"dunno",      "don't know"),
            (r"lemme",      "let me"),
            (r"gimme",      "give me"),
            (r"cya",        "goodbye"),
            (r"idk",        "i don't know"),
            (r"ngl",        "not gonna lie"),
            (r"tbh",        "to be honest"),
        ]
        import re as _re_typo
        _text_normalised = text
        for pattern, replacement in _TYPO_MAP:
            _text_normalised = _re_typo.sub(pattern, replacement, _text_normalised, flags=_re_typo.IGNORECASE)
        if _text_normalised != text:
            text = _text_normalised

        conv_history   = get_conv_history(db, user.id, last_n=10)
        current_style  = _detect_style(text)
        dominant_style = _blend_style(db, user.id, current_style)
        
        # ── Style-revert short-circuit ────────────────────────────────────
        # If the user just asked to revert style, confirm and return immediately.
        if current_style == "FORCE:normal":
            reply = pick_response("revert_style", dominant_style)
            save_conv_turn(db, user.id, "user",      text)
            save_conv_turn(db, user.id, "assistant", reply)
            return [reply], []

        # ── Other explicit style-switch short-circuit ─────────────────────
        # "be formal", "jarvis mode", "speak casually" etc. — just confirm the switch.
        if current_style.startswith("FORCE:"):
            style_name = dominant_style  # _blend_style already saved the new pref
            _STYLE_CONFIRM = {
                "jarvis":   "Jarvis Protocol activated, Sir. How may I assist you?",
                "formal":   "Understood. Switching to formal mode.",
                "friendly": "Sure! I'll keep things friendly and casual. 😊",
                "terse":    "Got it. Short answers from here on.",
                "casual":   "No problem! Keeping it casual. 😊",
            }
            reply = _STYLE_CONFIRM.get(style_name, f"Style set to *{style_name}*. ✅")
            save_conv_turn(db, user.id, "user",      text)
            save_conv_turn(db, user.id, "assistant", reply)
            return [reply], []

        # Smart Passive Learning
        memory_notes = self._smart_memory_guesser(db, user.id, text)
        if cached:
            _extract_email_entities(db, user.id, cached)

        # 1. Split text into sentences for better recognition
        # Also split on conjunction patterns that signal a topic shift:
        # "hello and tell me about X", "hi, what's the weather", "hey search for X"
        _CONJ_SPLIT = re.compile(
            r"(?<=[a-z])\s+(?:and\s+(?:also\s+)?|also\s+|then\s+|but\s+(?:also\s+)?)"
            r"(?=(?:search|find|tell|show|get|check|what|who|how|when|where|why|look|help|can you|could you))",
            re.IGNORECASE,
        )
        # First split on punctuation, then on conjunctions within each piece
        _raw_sentences = _SENT_SPLIT.split(text)
        sentences = []
        for piece in _raw_sentences:
            piece = piece.strip()
            if not piece:
                continue
            sub = [s.strip() for s in _CONJ_SPLIT.split(piece) if s.strip()]
            sentences.extend(sub if sub else [piece])
        if not sentences:
            sentences = [text]

        # Split greeting prefix from a following request in the same sentence
        # e.g. "hello tell me about weather" → ["hello", "tell me about weather"]
        _GREETING_PREFIX = re.compile(
            r"^(hi+|hey+|hello+|hiya|yo|sup|greetings?)[,!]?\s+(?=\w)",
            re.IGNORECASE,
        )
        expanded = []
        for s in sentences:
            m = _GREETING_PREFIX.match(s)
            if m and len(s) > len(m.group(0)) + 2:
                expanded.append(m.group(1))          # just the greeting
                expanded.append(s[m.end():].strip()) # the rest
            else:
                expanded.append(s)
        sentences = expanded

        reply_texts:  list[str]  = []
        intent_dicts: list[dict] = []
        
        # Track email index context across sentences
        current_email_idx = resolve_email_index(text, cached, conv_history)

        for s_text in sentences:
            s_norm = _normalize_text(s_text)
            predictions = self.intent_model.predict_multi(s_norm, threshold=0.25)
            
            # ── Apply heuristics to each sentence ───────────────────────────

            # Heuristic for 'chat_greeting'
            if not any(p[0] == "chat_greeting" for p in predictions):
                # Standard and collapsed forms (e.g. hello -> helo, wassup -> wasup)
                greetings = {
                    "hi", "hello", "helo", "hey", "wassup", "wasup", "sup", "yo", "hiya", "greetings", "greet"
                }
                s_low = s_norm.lower().strip("!? ")
                # Even more aggressive collapse for checking vs greeting list
                s_collapsed = re.sub(r'(.)\1+', r'\1', s_low)
                if s_collapsed in greetings or any(k in s_low for k in ["good morning", "good afternoon", "good evening"]):
                    predictions.append(("chat_greeting", 0.9))
            
            # ── Time query shortcut ─────────────────────────────────────────
            _TIME_PATTERNS = re.compile(
                r"\b(what(?:'s|\s+is)?\s+the\s+time|what\s+time\s+is\s+it|current\s+time|"
                r"time\s+now|time\s+in\s+\w+|what\s+time\s+is\s+it\s+in)\b",
                re.IGNORECASE,
            )
            if _TIME_PATTERNS.search(s_text):
                import re as _re
                _loc_match = _re.search(r"time\s+(?:is\s+it\s+)?in\s+([\w\s]+?)(?:\?|$)", s_text, _re.IGNORECASE)
                if _loc_match:
                    location = _loc_match.group(1).strip()
                    # For specific locations, let the web search catchall handle it
                    predictions.append(("web_search", 0.8))
                else:
                    # Local time — answer directly with datetime
                    from datetime import datetime
                    import zoneinfo
                    try:
                        sgt = zoneinfo.ZoneInfo("Asia/Singapore")
                        now = datetime.now(sgt)
                        time_reply = f"🕐 The current time is *{now.strftime('%I:%M %p')}* (SGT, UTC+8)."
                    except Exception:
                        now = datetime.now()
                        time_reply = f"🕐 The current local time is *{now.strftime('%I:%M %p')}*."
                    reply_texts.append(time_reply)
                    intent_dicts.append({"intent": "chat_general", "confidence": 1.0, "source": "time_shortcut", "action_params": {}})
                    save_conv_turn(db, user.id, "user",      user_message)
                    save_conv_turn(db, user.id, "assistant", time_reply)
                    return reply_texts, intent_dicts

            # Heuristic for 'recall' intent
            if not any(p[0] == "recall" for p in predictions):
                recall_keywords = ["what was", "who is", "remember", "when is", "recall", "what did i say", "who am i", "what do you know about me", "what do you remember", "what is my", "whats my", "what's my", "do you know my", "tell me my"]
                # Exclude time queries from recall
                if any(k in s_text.lower() for k in recall_keywords) and not _TIME_PATTERNS.search(s_text):
                    predictions.append(("recall", 0.6))

            # Heuristic for 'search' email intent
            _SEARCH_EMAIL_PATTERNS = re.compile(
                r"^(?:search(?:\s+(?:for|emails?|my\s+emails?|inbox))?|"
                r"find(?:\s+(?:emails?|messages?|mail))?(?:\s+(?:about|from|with|containing|where))?|"
                r"look\s+up\s+(?:emails?|messages?)|"
                r"any\s+(?:emails?|messages?)\s+(?:about|from|with|containing))\b",
                re.IGNORECASE,
            )
            if _SEARCH_EMAIL_PATTERNS.match(s_text.strip()):
                # Force search — remove any fetch_inbox the ML may have added
                predictions = [p for p in predictions if p[0] != "fetch_inbox"]
                if not any(p[0] == "search" for p in predictions):
                    predictions.append(("search", 0.95))
            elif not any(p[0] == "search" for p in predictions):
                if s_text.lower().strip("?") in ["search", "find", "results", "any results"]:
                    predictions.append(("search", 0.7))

            # Short acknowledgements — map to chat_thanks
            ack_words = {"ok", "okay", "alright", "aight", "got it", "sure", "cool",
                        "noted", "ight", "k", "kk", "roger", "understood", "nice",
                        "sounds good", "makes sense", "perfect", "great", "awesome"}
            if s_text.lower().strip().rstrip("!.~ ") in ack_words:
                predictions.append(("chat_thanks", 0.85))

            # Code generation intent
            code_generate_keywords = [
                "write me", "write a", "create a", "make a", "generate",
                "code for", "script for", "function for", "function that",
                "program that", "class that", "how do i code", "how to code",
                "give me code", "can you code",
            ]
            code_explain_keywords = [
                "explain this code", "what does this code do", "explain this",
                "what does this do", "walk me through", "what is this code",
                "break this down", "what does this mean",
            ]
            code_debug_keywords = [
                "debug", "fix this", "fix my code", "whats wrong", "what's wrong",
                "find the bug", "error in", "not working", "broken code",
                "why is this failing", "why doesn't this work", "why isnt this working",
            ]

            if any(k in s_text.lower() for k in code_generate_keywords):
                predictions.append(("code_generate", 0.85))
            elif any(k in s_text.lower() for k in code_explain_keywords):
                predictions.append(("code_explain", 0.85))
            elif any(k in s_text.lower() for k in code_debug_keywords):
                predictions.append(("code_debug", 0.85))

            # Heuristic for 'web_search' intent
            if not any(p[0] == "web_search" for p in predictions):
                # Only add web_search if we don't already have a strong chat intent
                has_chat_intent = any(p[0].startswith("chat_") and p[1] >= 0.4 for p in predictions)
                if not has_chat_intent:
                    web_keywords = [
                        "search for", "search the web", "search the internet",
                        "help me search", "look up", "google",
                        "find information on", "find me", "find out",
                        "who is", "what is", "tell me about",
                        "today's news", "latest news", "news about",
                        "what happened", "current events",
                    ]
                    internal_keywords = ["inbox", "rule", "email", "archive", "trash", "label", "reply", "compose", "your name", "who are you", "who am i"]
                    if any(k in s_text.lower() for k in web_keywords) and not any(k in s_text.lower() for k in internal_keywords):
                        predictions.append(("web_search", 0.6))
                    elif s_text.endswith("?") and not any(p[0] == "recall" for p in predictions):
                        # Don't trigger on very short questions or bot-identity questions
                        if len(s_text.split()) > 3 and not any(k in s_text.lower() for k in ["your name", "who are you", "what are you", "who am i"]):
                            predictions.append(("web_search", 0.55))

            # Heuristic for 'research' intent
            if not any(p[0] == "research" for p in predictions):
                research_keywords = ["research", "summarize that page", "what's on this site", "analyze this url"]
                if any(k in s_text.lower() for k in research_keywords):
                    predictions.append(("research", 0.7))

            # Heuristic for 'math_query' intent
            if not any(p[0] == "math_query" for p in predictions):
                import re as _re
                _arith_words = [
                    "plus", "minus", "times", "divided by", "multiplied by",
                    "squared", "cubed", "square root", "percent", "remainder",
                    "added to", "subtracted from", "modulo",
                ]
                _math_topic_words = [
                    "solve", "calculate", "equation", "formula", "derivative",
                    "integral", "quadratic", "pythagoras", "factor", "simplify",
                    "algebra", "calculus", "trigonometry", "sin", "cos", "tan",
                    "logarithm", "exponent", "matrix", "vector", "probability",
                    "statistics", "mean", "median", "variance", "polynomial",
                    "how do i solve", "what is the formula", "how to calculate",
                    "what does dx mean", "what is pi",
                ]
                _s_lower = s_text.lower()
                _has_operator_expr = bool(_re.search(r'\d+\s*[\+\-\*\/\^x]\s*\d+', _s_lower))
                _has_arith_word    = any(k in _s_lower for k in _arith_words)
                _has_topic_word    = any(k in _s_lower for k in _math_topic_words)
                _has_number        = bool(_re.search(r'\d', _s_lower))
                if _has_operator_expr or _has_topic_word or (_has_arith_word and _has_number):
                    predictions.append(("math_query", 0.75))

            # Heuristic for 'science_query' intent
            if not any(p[0] == "science_query" for p in predictions):
                science_keywords = [
                    "atom", "molecule", "electron", "proton", "neutron", "nucleus",
                    "photosynthesis", "mitosis", "dna", "rna", "protein", "enzyme",
                    "newton", "gravity", "force", "mass", "acceleration", "velocity",
                    "energy", "kinetic", "potential", "thermodynamics", "entropy",
                    "chemical", "reaction", "element", "periodic table", "bond",
                    "cell", "organism", "evolution", "species", "ecosystem",
                    "electromagnetic", "wavelength", "frequency", "ohm", "voltage",
                    "how does", "what is", "explain", "what causes", "why does",
                ]
                # Only trigger science if the query looks knowledge-seeking
                knowledge_triggers = ["how does", "what is", "explain", "what causes", "why does", "describe"]
                is_knowledge_q = any(k in s_text.lower() for k in knowledge_triggers)
                has_sci_term   = any(k in s_text.lower() for k in science_keywords[:-5])
                if is_knowledge_q and has_sci_term:
                    predictions.append(("science_query", 0.75))

            # ── Process predictions for this sentence ───────────────────────
            
            # Local resolution for THIS sentence (might find a specific number)
            s_idx = resolve_email_index(s_text, cached, conv_history)
            if s_idx: current_email_idx = s_idx

            label_name = self._parse_label(s_text)
            tone_name  = _parse_tone(s_text)

            # If web_search is present, suppress knowledge intents that would
            # double-respond with "I don't have that in my knowledge base"
            if any(p[0] == "web_search" for p in predictions):
                predictions = [p for p in predictions if p[0] not in ("science_query", "math_query", "recall")]

            # Filter out 'none' if we have other specific intents
            if len(predictions) > 1:
                has_specific = any(p[0] != "none" and p[1] >= 0.4 for p in predictions)
                if has_specific:
                    predictions = [p for p in predictions if p[0] != "none"]

            for intent, confidence in predictions:
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Let 'none' fall through to the web search catchall
                if intent == "none":
                    continue

                # Deduplicate intents (don't execute the same intent twice in one turn if it's the same index)
                if any(d["intent"] == intent and d.get("email_idx") == current_email_idx for d in intent_dicts):
                    continue

                intent_dict: dict = {
                    "intent":        intent,
                    "email_idx":     None,
                    "action_params": {},
                }

                if intent in ("analyse", "archive", "label", "trash", "reply",
                              "fetch_inbox", "fetch_priority", "compose"):
                    intent_dict["email_idx"] = current_email_idx
                    if intent == "label":
                        intent_dict["action_params"]["label"] = label_name
                    if intent == "reply":
                        intent_dict["action_params"]["tone"] = tone_name
                    if intent == "compose":
                        intent_dict["action_params"] = _parse_compose_details(s_text)
                        intent_dict["action_params"]["tone"] = tone_name

                elif intent == "create_rule":
                    rule_data = parse_rule(s_text)
                    intent_dict["rule"] = {
                        "rule_text": s_text,
                        "action":    rule_data["action"],
                        "keywords":  rule_data["keywords"],
                        "from":      rule_data["from"],
                        "label":     rule_data["label"],
                    }

                elif intent == "delete_rule":
                    intent_dict["action_params"]["rule_id"] = _parse_rule_id(s_text)

                reply = pick_response(intent, dominant_style)
                
                if intent == "recall":
                    # Handle Knowledge Recall
                    mems = get_memories(db, user.id)
                    fact_found = False
                    fact_replies = []

                    # Special case: "who am i" — dump everything known about the user
                    if any(k in s_text.lower() for k in ["who am i", "what do you know about me", "what do you remember about me"]):
                        known = []
                        for k, v in mems.items():
                            if k.startswith("fact:"):
                                label = k.replace("fact:", "").replace("_", " ")
                                known.append(f"• Your *{_esc(label)}* is *{_esc(v)}*")
                            elif k.startswith("contact:"):
                                email_addr = k.replace("contact:", "")
                                known.append(f"• I know *{_esc(v)}* at `{_esc(email_addr)}`")
                            elif k == "preferred_style":
                                known.append(f"• Your preferred style is *{_esc(v)}*")
                        if known:
                            reply = "Here's what I know about you so far:\n" + "\n".join(known)
                        else:
                            reply = "I don't know much about you yet! The more we talk, the more I'll learn. 🧠"
                        reply_texts.append(reply)
                        intent_dicts.append(intent_dict)
                        continue
                    
                    # 1. Check direct fact memory
                    for k, v in mems.items():
                        if k.startswith("fact:"):
                            key_name = k.replace("fact:", "").replace("_", " ")
                            if key_name in s_text.lower():
                                fact_replies.append(f"• I remember you told me that your *{_esc(key_name)}* is *{_esc(v)}*.")
                                fact_found = True
                    
                    # 2. Check contacts
                    if "contact" in s_text.lower() or "who is" in s_text.lower():
                        for k, v in mems.items():
                            if k.startswith("contact:"):
                                email_addr = k.replace("contact:", "")
                                if email_addr in s_text.lower() or v.lower() in s_text.lower():
                                    fact_replies.append(f"• I know that *{_esc(v)}* is the contact for `{_esc(email_addr)}`.")
                                    fact_found = True

                    # 3. Check history if no direct fact found
                    if not fact_found:
                        history_match = _search_history(db, user.id, s_text)
                        if history_match:
                            fact_replies.append(f"• {history_match}")
                            fact_found = True
                    
                    if fact_found:
                        reply += "\n" + "\n".join(fact_replies)
                    else:
                        reply = "I'm sorry, I don't have any specific records of that in my memory yet. 🧠"

                elif intent in ("math_query", "science_query"):
                    # ── Step 1: try to evaluate arithmetic directly ───────
                    import re as _re
                    _calc_result = None
                    if intent == "math_query":
                        _expr = s_text.lower().strip()
                        # Normalise word operators
                        _expr = _re.sub(r'\bplus\b',           '+',  _expr)
                        _expr = _re.sub(r'\bminus\b',          '-',  _expr)
                        _expr = _re.sub(r'\btimes\b',          '*',  _expr)
                        _expr = _re.sub(r'\bmultiplied by\b',  '*',  _expr)
                        _expr = _re.sub(r'\bdivided by\b',     '/',  _expr)
                        _expr = _re.sub(r'\bsquared\b',        '**2',_expr)
                        _expr = _re.sub(r'\bcubed\b',          '**3',_expr)
                        _expr = _re.sub(r'\bto the power of\b','**', _expr)
                        _expr = _re.sub(r'\bsquare root of\b', 'sqrt', _expr)
                        _expr = _re.sub(r'\bmod\b|\bmodulo\b', '%',  _expr)
                        # Strip everything except digits and operators
                        _clean = _re.sub(r'[^\d\.\+\-\*\/\(\)\%\^\s]', '', _expr).strip()
                        _clean = _clean.replace('^', '**')
                        if _clean and _re.search(r'\d', _clean):
                            try:
                                _calc_result = eval(  # noqa: S307
                                    _clean,
                                    {"__builtins__": {}},
                                    {"sqrt": lambda x: x ** 0.5},
                                )
                            except Exception:
                                _calc_result = None

                    if _calc_result is not None:
                        # Format nicely — drop .0 for whole numbers
                        _fmt = int(_calc_result) if isinstance(_calc_result, float) and _calc_result == int(_calc_result) else _calc_result
                        reply = f"That's *{_fmt}*. 🔢"
                    else:
                        # ── Step 2: search the knowledge base ────────────
                        domain = "math" if intent == "math_query" else "science"
                        hits   = search_knowledge(db, s_text, domain=domain, top_k=2)
                        if hits:
                            parts = []
                            for h in hits:
                                parts.append(f"*{_esc(h.topic)}*\n{_esc(h.explanation)}")
                            reply += "\n\n" + "\n\n─────\n\n".join(parts)
                        else:
                            # Fallback: try searching without domain filter
                            hits = search_knowledge(db, s_text, domain=None, top_k=1)
                            if hits:
                                h = hits[0]
                                reply += f"\n\n*{_esc(h.topic)}*\n{_esc(h.explanation)}"
                            else:
                                reply = (
                                    "I don't have that in my knowledge base yet. 📚\n"
                                    "Try rephrasing, or use `/wrong` if I misunderstood your question.\n"
                                    "You can also ask me to search the web for it!"
                                )

                reply_texts.append(reply)
                intent_dicts.append(intent_dict)

        # ── Cross-sentence deduplication ────────────────────────────────────
        # If web_search was triggered anywhere, drop science/math/recall intents
        # that would produce a redundant "not in knowledge base" reply.
        if any(d["intent"] == "web_search" for d in intent_dicts):
            keep_indices = [
                i for i, d in enumerate(intent_dicts)
                if d["intent"] not in ("science_query", "math_query", "recall")
            ]
            intent_dicts = [intent_dicts[i] for i in keep_indices]
            reply_texts  = [reply_texts[i]  for i in keep_indices]

        # ── Normal return (sklearn layer succeeded) ─────────────────────────
        if reply_texts:
            save_conv_turn(db, user.id, "user",      user_message)
            save_conv_turn(db, user.id, "assistant", reply_texts[-1])
            return reply_texts, intent_dicts

        # ── Final Fallback ──────────────────────────────────────────────────
        # If absolutely NO high-confidence intent was found across ALL sentences
        if not reply_texts:
 
            # <<< NEURAL LAYER: try the PyTorch net before giving up >>>
            neural_result = self.neural_fallback.predict(text)
            if neural_result is not None:
                n_intent, n_conf = neural_result
                # Re-enter the normal intent handling pipeline for this intent
                intent_dict = {
                    "intent":        n_intent,
                    "confidence":    n_conf,
                    "source":        "neural_fallback",
                    "action_params": {},
                }
                reply = pick_response(n_intent, dominant_style)
                reply_texts  = [reply]
                intent_dicts = [intent_dict]
                log.info(
                    f"NeuralFallback rescued intent '{n_intent}' "
                    f"@ {n_conf:.2f} for text: '{text[:80]}'"
                )
                save_conv_turn(db, user.id, "user",      user_message)
                save_conv_turn(db, user.id, "assistant", reply)
                return [reply], [intent_dict]
            else:
                # <<< END NEURAL LAYER — web search catchall >>>
                # Neither sklearn nor neural fallback was confident enough.
                # Try a web search before giving up entirely.
                try:
                    from api.web_ops import search_web
                    results = search_web(text, max_results=5, db=db)
                    if results:
                        # Build a conversational answer from the top snippets
                        snippets = [
                            r.get("snippet", "").strip()
                            for r in results
                            if r.get("snippet", "").strip()
                        ]
                        best = snippets[:3]

                        # Merge snippets into a flowing answer
                        combined = " ".join(best)
                        # Trim to ~400 chars at a sentence boundary
                        if len(combined) > 400:
                            cutoff = combined.rfind(".", 0, 400)
                            combined = combined[:cutoff + 1] if cutoff > 100 else combined[:400] + "…"

                        top_link  = results[0].get("link", "").strip()
                        top_title = results[0].get("title", "").strip()

                        reply = f"{combined}"
                        if top_link:
                            reply += f"\n\n📎 [{top_title}]({top_link})"
                    else:
                        reply = self._low_confidence_reply(dominant_style)
                except Exception as exc:
                    log.warning(f"Web search catchall failed: {exc}")
                    reply = self._low_confidence_reply(dominant_style)

                save_conv_turn(db, user.id, "user",      user_message)
                save_conv_turn(db, user.id, "assistant", reply)
                intent_dict = {
                    "intent":        "web_search",
                    "confidence":    0.0,
                    "source":        "catchall",
                    "action_params": {},
                }
                return [reply], [intent_dict]
 
 


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

    def semantic_search(self, query: str, emails: list[dict], top_n: int = 5) -> list[dict]:
        """
        Find emails semantically similar to the query using TF-IDF cosine similarity.
        """
        if not emails:
            return []

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Combine subject and body for search
        documents = [f"{e['subject']} {e['body']}" for e in emails]
        
        # Add query to the list of documents to vectorize together
        all_texts = [query] + documents
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarity between query (index 0) and all documents (index 1+)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Rank and filter
        ranked_indices = similarities.argsort()[::-1]
        results = []
        for idx in ranked_indices:
            if similarities[idx] > 0.1: # Minimum similarity threshold
                results.append(emails[idx])
            if len(results) >= top_n:
                break
        
        return results

    def check_semantic_match(self, rule_description: str, email: dict, threshold: float = 0.35) -> bool:
        """
        Check if an email matches a rule description semantically.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        doc   = f"{email['subject']} {email['body']}"
        texts = [rule_description.lower(), doc.lower()]
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf      = vectorizer.fit_transform(texts)
            sim        = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return sim >= threshold
        except Exception:
            return False

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

    def record_correction(self, text: str, wrong_intent: str, correct_intent: str):
        for _ in range(5):
            self.intent_model.add_example(text, correct_intent)
        log.info(f"Correction: '{text}' was '{wrong_intent}' → now '{correct_intent}'")
        try:
            self.intent_model.retrain()
        except Exception as e:
            log.error(f"Retrain after correction failed: {e}")

    def record_passive_priority_example(self, subject: str, body: str, action: str):
        """Learn priority passively from user actions (archive, trash, reply)."""
        mapping = {
            "archive": "low",
            "trash":   "low",
            "reply":   "important",
            "compose": "important",
            "label":   "important", # simple heuristic: if they label it, it matters
        }
        priority = mapping.get(action)
        if priority:
            self.learn_priority(subject, body, priority)
            # Retrain priority model passively too
            # We use a separate counter or just check example count
            if self.priority_model.example_count % 10 == 0:
                try:
                    self.priority_model.retrain()
                except Exception:
                    pass

    def learn_priority(self, subject: str, body: str, priority: str):
        self.priority_model.learn(subject, body, priority)

    def retrain(self) -> dict:
        self.neural_fallback.train(self.intent_model.training_data)
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