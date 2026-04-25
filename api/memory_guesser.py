import re

class MemoryGuesser:
    """
    Intelligently guesses if a sentence contains a fact worth remembering.
    Uses heuristics, keyword density, and pattern recognition.
    """

    # Indicators that a sentence is declarative and contains a fact
    _FACT_INDICATORS = [
        "is", "was", "are", "were", "call me", "am a", "works at",
        "working on", "located in", "lives in", "deadline is", "code is",
        "email is", "phone is", "address is", "remember that"
    ]

    # Words that suggest a low-importance sentence (filler, questions, etc.)
    _IGNORE_WORDS = [
        "can you", "could you", "please", "maybe", "i think", "i guess",
        "how", "why", "what", "when", "where", "if"
    ]

    def __init__(self, config: dict = None):
        self.config = config or {}
        # Allow overriding indicators and ignore words via config
        self.indicators = self.config.get("indicators", self._FACT_INDICATORS)
        self.ignore_words = self.config.get("ignore", self._IGNORE_WORDS)
        self.custom_patterns = self.config.get("patterns", [])

    def guess_facts(self, text: str, existing_memories: dict = None) -> list[dict]:
        """
        Analyze text and return a list of guessed facts.
        Each fact: {"key": str, "value": str, "confidence": float}
        """
        facts = []
        t = text.strip()
        t_low = t.lower()

        # 1. Importance Scoring
        score = self._calculate_importance(t_low)
        if score < 0.3:
            return []

        # 2. Pattern Matching (High Confidence)
        patterns = [
            (r"\bmy\s+([a-z]{3,15})\s+is\s+([a-z0-9 ]{2,30})\b", "fact:{}", 0.9),
            (r"\bi\s+am\s+(?:a|an)\s+([a-z0-9 ]{3,30})\b", "fact:role", 0.85),
            (r"\b(?:call me|my name is)\s+([a-z]{2,20})\b", "fact:name", 0.95),
            (r"\bi\s+live\s+in\s+([a-z ]{3,30})\b", "fact:location", 0.85),
            (r"\bi(?:\'m| am)\s+working\s+on\s+(?:the\s+)?([a-z0-9 ]{3,30})\b", "fact:project", 0.8),
            (r"\bremember\s+that\s+(?:the\s+)?([a-z0-9 ]{3,20})\s+is\s+([a-z0-9 ]{1,50})\b", "fact:{}", 0.9),
        ]
        
        # Add custom patterns from config
        patterns.extend(self.custom_patterns)

        for pattern, key_fmt, conf in patterns:
            m = re.search(pattern, t_low)
            if m:
                if "{}" in key_fmt:
                    key_name = m.group(1).strip().replace(" ", "_")
                    key = key_fmt.format(key_name)
                    val = m.group(2).strip()
                else:
                    key = key_fmt
                    val = m.group(1).strip()
                
                # Check if we already know this exact fact
                if existing_memories and existing_memories.get(key) == val:
                    continue

                if len(val.split()) < 8:
                    facts.append({"key": key, "value": val, "confidence": conf})

        # 3. Entity Guessing (Medium Confidence - "The Guessing Engine")
        if not facts:
            # Flexible: "The [noun phrase] is [Value]"
            m = re.search(r"\bthe\s+([a-z ]{3,40})\s+is\s+([a-zA-Z0-9\-_ ]{1,40})", t, re.IGNORECASE)
            if m:
                key = f"fact:{m.group(1).strip().lower().replace(' ', '_')}"
                val = m.group(2).strip()
                
                if not (existing_memories and existing_memories.get(key) == val):
                    # If value is capitalized or numeric, it's a good guess
                    conf = 0.6 if (val[0].isupper() or any(c.isdigit() for c in val)) else 0.4
                    facts.append({"key": key, "value": val, "confidence": conf})
            
            # Catch-all: "My [noun] is [value]"
            if not facts:
                m = re.search(r"\bmy\s+([a-z ]{2,20})\s+is\s+([a-zA-Z0-9 ]{1,40})", t, re.IGNORECASE)
                if m:
                    key = f"fact:{m.group(1).strip().lower().replace(' ', '_')}"
                    val = m.group(2).strip()
                    if not (existing_memories and existing_memories.get(key) == val):
                        facts.append({"key": key, "value": val, "confidence": 0.5})

        return facts

    def _calculate_importance(self, text: str) -> float:
        """Score how likely this sentence contains a persistent fact."""
        score = 0.0
        
        # Declarative indicators (using self.indicators instead of hardcoded)
        if any(ind in text for ind in self.indicators):
            score += 0.4
            
        # Proper noun detection (simple heuristic: capitalized words not at start)
        words = text.split()
        if len(words) > 1:
            for w in words[1:]:
                if w and w[0].isupper() and w.lower() not in ["i", "a", "the", "an"]:
                    score += 0.2
                    break
        
        # Penalty for questions or requests (using self.ignore_words)
        if text.endswith("?") or any(w in text for w in self.ignore_words):
            score -= 0.3
            
        return max(0.0, min(1.0, score))

