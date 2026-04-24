"""
AATAS — Email Priority Model
Learns YOUR email priority habits from feedback you give via Telegram.

Starts with keyword heuristics (instant, no training needed).
Upgrades to a trained Logistic Regression classifier as you rate emails.
Saved to disk in data/priority_model.pkl.
"""

import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "data/priority_model.pkl"
DATA_PATH  = "data/priority_training_data.pkl"

PRIORITIES = ["urgent", "important", "normal", "low"]

# ── Keyword heuristics (used before ML model is ready) ──────

_URGENT_KW    = [
    "urgent", "asap", "immediately", "critical", "emergency",
    "deadline", "overdue", "action required", "respond now",
    "time sensitive", "by today", "by tomorrow",
]
_IMPORTANT_KW = [
    "meeting", "interview", "offer", "invoice", "payment",
    "confirm", "approval", "review", "schedule", "appointment",
    "important", "requested", "reminder",
]
_LOW_KW       = [
    "newsletter", "promotion", "offer", "sale", "discount",
    "unsubscribe", "marketing", "no-reply", "noreply",
    "subscribe", "weekly digest", "deal", "coupon",
]


def _heuristic(subject: str, body: str) -> tuple[str, float]:
    text = f"{subject} {body[:500]}".lower()
    for kw in _URGENT_KW:
        if kw in text:
            return "urgent", 0.75
    for kw in _LOW_KW:
        if kw in text:
            return "low", 0.70
    for kw in _IMPORTANT_KW:
        if kw in text:
            return "important", 0.65
    return "normal", 0.60


class PriorityModel:
    """
    Email priority scorer.
    Uses heuristics until you've rated enough emails, then uses ML.
    """

    def __init__(self):
        self._make_fresh()
        self.training_data: list[tuple[str, str]] = []
        self.is_trained = False

    def _make_fresh(self):
        self.vectorizer    = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        self.classifier    = LogisticRegression(max_iter=300, random_state=42, C=1.0)
        self.label_encoder = LabelEncoder()

    # ── Persistence ──────────────────────────────────────────

    def load_or_init(self):
        """Load model from disk, or stay in heuristic mode."""
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "rb") as f:
                self.training_data = pickle.load(f)

        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    state = pickle.load(f)
                self.vectorizer    = state["vectorizer"]
                self.classifier    = state["classifier"]
                self.label_encoder = state["label_encoder"]
                self.is_trained    = True
            except Exception:
                pass

    def _save_data(self):
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "wb") as f:
            pickle.dump(self.training_data, f)

    # ── Public API ───────────────────────────────────────────

    def score(self, subject: str, body: str) -> tuple[str, float]:
        """
        Score email priority.
        Returns (priority_label, confidence).
        """
        if not self.is_trained or len(self.training_data) < 10:
            return _heuristic(subject, body)

        text = f"{subject} {body[:300]}".lower()
        X    = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        idx   = int(np.argmax(probs))
        return self.label_encoder.inverse_transform([idx])[0], float(probs[idx])

    def learn(self, subject: str, body: str, priority: str):
        """
        Record that YOU rated this email as the given priority.
        Call retrain() to apply the new data.
        """
        text = f"{subject} {body[:300]}".lower()
        self.training_data.append((text, priority))
        self._save_data()

    def retrain(self) -> dict:
        """Retrain on all user feedback. Returns stats dict."""
        if len(self.training_data) < 8:
            return {
                "examples": len(self.training_data),
                "error": "Need at least 8 rated emails to train priority model.",
            }

        texts, labels = zip(*self.training_data)
        self._make_fresh()
        self.label_encoder.fit(PRIORITIES)
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.transform(labels)
        self.classifier.fit(X, y)
        self.is_trained = True

        os.makedirs("data", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(
                {
                    "vectorizer":    self.vectorizer,
                    "classifier":    self.classifier,
                    "label_encoder": self.label_encoder,
                },
                f,
            )

        return {"examples": len(self.training_data)}

    @property
    def example_count(self) -> int:
        return len(self.training_data)
