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
from api.memory_guesser import MemoryGuesser
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


_STYLE_COMMANDS = {
    "friendly": ["be friendly", "be more friendly", "speak casually", "be chill", "be nice"],
    "formal":   ["be formal", "speak formally", "professional tone", "be official"],
    "terse":    ["be terse", "be brief", "short answers only", "stay brief"],
    "jarvis":   ["be jarvis", "jarvis mode", "iron man", "speak like jarvis", "sir"],
}


def _detect_style(text: str, memories: dict[str, str] = None) -> str:
    t = text.lower()
    
    # Check for explicit style commands
    for style, commands in _STYLE_COMMANDS.items():
        if any(cmd in t for cmd in commands):
            return "FORCE:" + style

    wc = len(text.split())
    
    # Jarvis mode detection (heuristic: frequent use of Sir/Jarvis)
    if "sir" in t or "jarvis" in t:
        return "jarvis"

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


def _blend_style(db: Session, user_id: int, current: str) -> str:
    """Update the per-user style count in Memory and return the dominant style."""
    mems    = get_memories(db, user_id)
    
    # If user explicitly set a style in this message, save it as their preference
    if current.startswith("FORCE:"):
        pref = current.split(":")[1]
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
            "Hey! Need a hand with your emails?",
            "Hello! I'm here to help you manage your mail.",
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
        cached = cached_emails or []

        conv_history   = get_conv_history(db, user.id, last_n=10)
        current_style  = _detect_style(text)
        dominant_style = _blend_style(db, user.id, current_style)
        
        # Smart Passive Learning
        memory_notes = self._smart_memory_guesser(db, user.id, text)
        if cached:
            _extract_email_entities(db, user.id, cached)

        # 1. Split text into sentences for better recognition
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 2]
        if not sentences: sentences = [text]

        reply_texts:  list[str]  = []
        intent_dicts: list[dict] = []
        
        # Track email index context across sentences
        current_email_idx = resolve_email_index(text, cached, conv_history)

        for s_text in sentences:
            predictions = self.intent_model.predict_multi(s_text, threshold=0.25)
            
            # ── Apply heuristics to each sentence ───────────────────────────
            
            # Heuristic for 'recall' intent
            if not any(p[0] == "recall" for p in predictions):
                recall_keywords = ["what was", "who is", "remember", "when is", "recall", "what did i say"]
                if any(k in s_text.lower() for k in recall_keywords):
                    predictions.append(("recall", 0.6))

            # Heuristic for 'search' intent follow-up
            if not any(p[0] == "search" for p in predictions):
                if s_text.lower().strip("?") in ["search", "find", "results", "any results"]:
                    predictions.append(("search", 0.7))

            # Heuristic for 'web_search' intent
            if not any(p[0] == "web_search" for p in predictions):
                # Only add web_search if we don't already have a strong chat intent
                has_chat_intent = any(p[0].startswith("chat_") and p[1] >= 0.4 for p in predictions)
                if not has_chat_intent:
                    web_keywords = ["search for", "look up", "google", "find information on", "who is", "what is"]
                    internal_keywords = ["inbox", "rule", "email", "archive", "trash", "label", "reply", "compose", "your name", "who are you"]
                    if any(k in s_text.lower() for k in web_keywords) and not any(k in s_text.lower() for k in internal_keywords):
                        predictions.append(("web_search", 0.6))
                    elif s_text.endswith("?") and not any(p[0] == "recall" for p in predictions):
                        # Don't trigger on very short questions or bot-identity questions
                        if len(s_text.split()) > 3 and not any(k in s_text.lower() for k in ["your name", "who are you", "what are you"]):
                            predictions.append(("web_search", 0.55))

            # Heuristic for 'research' intent
            if not any(p[0] == "research" for p in predictions):
                research_keywords = ["research", "summarize that page", "what's on this site", "analyze this url"]
                if any(k in s_text.lower() for k in research_keywords):
                    predictions.append(("research", 0.7))

            # ── Process predictions for this sentence ───────────────────────
            
            # Local resolution for THIS sentence (might find a specific number)
            s_idx = resolve_email_index(s_text, cached, conv_history)
            if s_idx: current_email_idx = s_idx

            label_name = self._parse_label(s_text)
            tone_name  = _parse_tone(s_text)

            # Filter out 'none' if we have other specific intents
            if len(predictions) > 1:
                has_specific = any(p[0] != "none" and p[1] >= 0.4 for p in predictions)
                if has_specific:
                    predictions = [p for p in predictions if p[0] != "none"]

            for intent, confidence in predictions:
                if confidence < CONFIDENCE_THRESHOLD:
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

                reply_texts.append(reply)
                intent_dicts.append(intent_dict)

        # ── Final Fallback ──────────────────────────────────────────────────

        # If absolutely NO high-confidence intent was found across ALL sentences
        if not reply_texts:
            # Fallback to the best overall prediction if it's somewhat okay
            overall_preds = self.intent_model.predict_multi(text, threshold=0.1)
            if overall_preds and overall_preds[0][1] > 0.4:
                # Still a bit low, but maybe we can just use the best one
                # Actually, let's just use the low confidence reply logic
                reply = self._low_confidence_reply(dominant_style)
            else:
                reply = pick_response("none", dominant_style)
            
            save_conv_turn(db, user.id, "user",      user_message)
            save_conv_turn(db, user.id, "assistant", reply)
            return [reply], []

        # Add memory confirmations to the first reply
        if memory_notes and reply_texts:
            note_str = "\n✨ " + "\n✨ ".join(memory_notes)
            reply_texts[0] += note_str

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