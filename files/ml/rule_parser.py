"""
AATAS — Rule Parser
Extracts action, keywords, and sender filters from natural language commands.
Pure Python regex + heuristics. No external API needed.
"""

import re

# ── Action detection ─────────────────────────────────────────

_ACTION_PATTERNS = [
    ("trash",     [r"\btrash\b", r"\bdelete\b", r"\bremove\b", r"\bget rid of\b"]),
    ("label",     [r"\blabel\b", r"\btag\b", r"\bmark as\b", r"\bcategoris[e]?\b",
                   r"\bcategoriz[e]?\b", r"\bflag as\b", r"\bput under\b"]),
    ("mark_read", [r"\bmark.{0,10}read\b", r"\bmark read\b"]),
    ("archive",   [r"\barchive\b", r"\bmove to archive\b", r"\bput in archive\b",
                   r"\bremove from inbox\b", r"\bfile away\b"]),
]

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "i", "me", "my", "you", "your", "it", "its", "all", "any",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
    "by", "about", "from", "into", "through", "during", "after", "before",
    "emails", "email", "messages", "message", "mail", "automatically",
    "auto", "always", "whenever", "every", "time", "get",
}

# Words to strip when extracting keywords
_ACTION_WORDS = r"""
    \b(?:archive|label|trash|delete|remove|mark|tag|categorise|categorize|
    flag|emails?|messages?|anything|everything|all|about|containing|with|that|
    mention|mentions|related|regarding|from|sent\sby|labell?ed?|as|them|
    automatically|auto|always|whenever|every(?:time)?|put|in|archive|
    move|to|file|away|get|rid|of)\b
"""


def _detect_action(text: str) -> str:
    t = text.lower()
    for action, patterns in _ACTION_PATTERNS:
        for p in patterns:
            if re.search(p, t):
                return action
    return "archive"  # sensible default


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful content words from a rule instruction."""
    # First: pull anything in quotes
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    if quoted:
        return [q[0] or q[1] for q in quoted][:5]

    # Remove action vocabulary
    clean = re.sub(_ACTION_WORDS, " ", text, flags=re.IGNORECASE | re.VERBOSE)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Extract meaningful words
    words = re.findall(r"\b[a-z][a-z]{2,}\b", clean.lower())
    keywords = [w for w in words if w not in _STOP_WORDS]

    # Also try bigrams
    bigrams = []
    for i in range(len(keywords) - 1):
        bigrams.append(f"{keywords[i]} {keywords[i+1]}")

    # Prefer bigrams when available, else single words
    result = list(dict.fromkeys(bigrams + keywords))
    return result[:5]


def _extract_sender(text: str) -> str:
    """Extract an email or name-based sender filter."""
    # Full email address
    m = re.search(r"\b[\w.+\-]+@[\w\-]+\.\w+\b", text)
    if m:
        return m.group(0)
    # "from <name>" pattern
    m = re.search(r"\bfrom\s+([A-Za-z][\w ]{1,25}?)(?:\s+(?:as|with|label|tag|to|archive|trash)|$)", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if name.lower() not in _STOP_WORDS and len(name) > 2:
            return name
    return ""


def _extract_label_name(text: str, action: str) -> str:
    """Extract label name for label actions."""
    if action != "label":
        return ""
    m = re.search(
        r'(?:label|tag|mark|flag|categoris[e]?|categoriz[e]?)\s+(?:as|:)?\s*["\']?([a-z][a-z0-9_\- ]{0,25})["\']?',
        text, re.IGNORECASE,
    )
    return m.group(1).strip() if m else "labelled"


def parse_rule(text: str) -> dict:
    """
    Parse a natural language rule into structured components.

    Returns:
        {
            "action":   "archive | label | trash | mark_read",
            "keywords": ["list", "of", "keywords"],
            "from":     "sender filter or empty string",
            "label":    "label name if action=label, else empty",
        }
    """
    action   = _detect_action(text)
    keywords = _extract_keywords(text)
    sender   = _extract_sender(text)
    label    = _extract_label_name(text, action)

    return {
        "action":   action,
        "keywords": keywords,
        "from":     sender,
        "label":    label,
    }
