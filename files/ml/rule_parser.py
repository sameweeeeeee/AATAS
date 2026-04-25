"""
AATAS — Rule Parser
Extracts action, keywords, and sender filters from natural language commands.
Pure Python regex + heuristics. No external API needed.
"""

import re

# ── Action detection ───────────────────────────────────────────────────────

_ACTION_PATTERNS = [
    ("trash",     [r"\btrash\b", r"\bdelete\b", r"\bget rid of\b"]),
    ("label",     [r"\blabel\b", r"\btag\b", r"\bmark\s+as\b", r"\bcategoris[e]?\b",
                   r"\bcategoriz[e]?\b", r"\bflag\s+as\b", r"\bflag\b", r"\bput under\b"]),
    ("mark_read", [r"\bmark\s+read\b", r"\bmark.{1,10}as\s+read\b"]),
    ("archive",   [r"\barchive\b", r"\bmove to archive\b", r"\bfile away\b",
                   r"\bremove from inbox\b"]),
]

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "i", "me", "my", "you", "your", "it", "its", "all", "any",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
    "by", "about", "from", "into", "through", "during", "after", "before",
    "emails", "email", "messages", "message", "mail", "automatically",
    "auto", "always", "whenever", "every", "time", "get", "them",
}

# Verbs/function-words to strip when pulling out content keywords
_STRIP_RE = re.compile(
    r"\b(?:archive|label|tag|flag|trash|delete|remove|mark|categorise|categorize|"
    r"put|move|file|get|rid|away|automatically|auto|always|whenever|everytime?|"
    r"emails?|messages?|mail|anything|everything|all|about|containing|with|that|"
    r"mention|mentions?|related|regarding|sent\s+by|labell?ed?|them|read)\b",
    re.IGNORECASE,
)


def _detect_action(text: str) -> str:
    t = text.lower()
    # mark_read must be checked before generic "mark as" label detection
    for action, patterns in _ACTION_PATTERNS:
        for p in patterns:
            if re.search(p, t):
                return action
    return "archive"


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful content words from a rule instruction."""
    # Pull anything in quotes first
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    if quoted:
        return [q[0] or q[1] for q in quoted][:5]

    # Remove the label phrase (everything from 'as <word>' at end)
    clean = re.sub(r'\bas\s+\S[\w\s]{0,30}$', "", text, flags=re.IGNORECASE)
    # Remove action verbs and function words
    clean = _STRIP_RE.sub(" ", clean)
    # Remove email addresses (handled separately by sender extractor)
    clean = re.sub(r'\b[\w.+\-]+@[\w\-]+\.\w+\b', " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    words    = re.findall(r"\b[a-z][a-z]{2,}\b", clean.lower())
    keywords = [w for w in words if w not in _STOP_WORDS]

    bigrams = [f"{keywords[i]} {keywords[i+1]}" for i in range(len(keywords) - 1)]
    result  = list(dict.fromkeys(bigrams + keywords))
    return result[:5]


def _extract_sender(text: str) -> str:
    """Extract an email address or name after 'from'."""
    # Full email address anywhere in the text
    m = re.search(r"\b[\w.+\-]+@[\w\-]+\.\w+\b", text)
    if m:
        return m.group(0)
    # "from <name>" pattern — stop before action keywords
    m = re.search(
        r"\bfrom\s+([A-Za-z][\w ]{1,30}?)(?=\s+(?:as|label|tag|archive|trash|delete|mark|and|$)|\s*$)",
        text, re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip()
        if name.lower() not in _STOP_WORDS and len(name) > 1:
            return name
    return ""


def _extract_label_name(text: str, action: str) -> str:
    """
    Extract the label name for label/tag/flag actions.
    Always looks for what comes AFTER the word 'as'.

    Examples:
      'label emails from bob as work'          -> 'work'
      'label anything about invoice as bills'  -> 'bills'
      'flag emails about meetings as important'-> 'important'
      'mark as urgent'                         -> 'urgent'
      'tag emails from hr as projects'         -> 'projects'
    """
    if action not in ("label",):
        return ""

    # The label name is whatever comes after the last 'as' in the sentence
    m = re.search(r'\bas\s+([a-z][a-z0-9_\-]{0,30})\s*$', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    # Fallback: 'mark as <label>' where 'as' might be mid-sentence
    m = re.search(r'\bmark\s+as\s+([a-z][a-z0-9_\-]{0,30})\b', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()

    return "labelled"


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