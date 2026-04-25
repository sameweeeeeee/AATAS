"""
AATAS — Intent Classifier
TF-IDF vectorizer + MLP neural network.

Pure math — no pretrained weights from any company.
Trained on YOUR labelled examples via the Telegram /train command.
Saved to disk in data/intent_model.pkl so it persists across restarts.
"""

import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from files.ml.seed_data import SEED_DATA

MODEL_PATH = "data/intent_model.pkl"
DATA_PATH  = "data/intent_training_data.pkl"

# All supported intents
INTENTS = [
    "fetch_inbox", "fetch_priority", "analyse", "search", "archive", "label",
    "trash", "reply", "compose", "create_rule", "delete_rule", "list_rules",
    "list_history", "recall", "none", "web_search", "research",
    "chat_greeting", "chat_how_are_you", "chat_identity", "chat_thanks", "chat_goodbye"
]


class IntentModel:
    """
    TF-IDF + MLP intent classifier.
    Learns from YOUR examples. Gets smarter with every /train session.
    """

    def __init__(self):
        self._make_fresh_components()
        self.training_data: list[tuple[str, str]] = []
        self.is_trained = False

    def _make_fresh_components(self):
        """Create fresh sklearn components (used on init and retrain)."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_features=5000,
            analyzer="word",
            sublinear_tf=True,
        )
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.label_encoder = LabelEncoder()

    # ── Persistence ──────────────────────────────────────────

    def load_or_init(self):
        """Load saved model from disk, or bootstrap from seed data."""
        # Load any previously saved training examples
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "rb") as f:
                self.training_data = pickle.load(f)
        else:
            self.training_data = list(SEED_DATA)

        # Try to load a saved trained model
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    state = pickle.load(f)
                self.vectorizer    = state["vectorizer"]
                self.classifier    = state["classifier"]
                self.label_encoder = state["label_encoder"]
                self.is_trained    = True
                return
            except Exception:
                pass  # fall through to bootstrap

        # Bootstrap: train on seed data
        self._fit(self.training_data)

    def _save_model(self):
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

    def _save_data(self):
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "wb") as f:
            pickle.dump(self.training_data, f)

    # ── Internal fit ────────────────────────────────────────

    def _fit(self, data: list[tuple[str, str]]):
        """Fit the model on a dataset. Used for bootstrap only."""
        if len(data) < 5:
            return
        texts, labels = zip(*data)
        self.label_encoder.fit(labels)
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.transform(labels)
        self.classifier.fit(X, y)
        self.is_trained = True

    # ── Public API ───────────────────────────────────────────

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict single intent from text (legacy).
        Returns (intent_label, confidence_0_to_1).
        """
        if not self.is_trained:
            return "none", 0.0
        X     = self.vectorizer.transform([text.lower()])
        probs = self.classifier.predict_proba(X)[0]
        idx   = int(np.argmax(probs))
        intent     = self.label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        return intent, confidence

    def predict_multi(self, text: str, threshold: float = 0.25) -> list[tuple[str, float]]:
        """
        Predict multiple intents from text.
        Returns a list of (intent_label, confidence) for all probabilities above threshold.
        """
        if not self.is_trained:
            return [("none", 0.0)]
        X     = self.vectorizer.transform([text.lower()])
        probs = self.classifier.predict_proba(X)[0]
        
        results = []
        for idx, prob in enumerate(probs):
            if prob >= threshold:
                intent = self.label_encoder.inverse_transform([idx])[0]
                results.append((intent, float(prob)))
                
        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        if not results:
            # If nothing crossed threshold, return the max one so fallback kicks in
            return [self.predict(text)]
            
        return results

    def add_example(self, text: str, intent: str):
        """
        Add a user-labelled training example.
        Call retrain() afterward to apply it.
        """
        self.training_data.append((text.lower().strip(), intent))
        self._save_data()

    def retrain(self) -> dict:
        """
        Retrain the model on all accumulated examples.
        Returns a stats dict: {"accuracy": float, "examples": int}.
        """
        if len(self.training_data) < 10:
            return {
                "accuracy": 0.0,
                "examples": len(self.training_data),
                "error": "Need at least 10 examples to train.",
            }

        os.makedirs("data", exist_ok=True)
        texts, labels = zip(*self.training_data)

        # Rebuild fresh components for a clean retrain
        self._make_fresh_components()
        self.label_encoder.fit(labels)
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.transform(labels)

        # Measure accuracy with a holdout split if we have enough data
        accuracy = 1.0
        if len(self.training_data) >= 20:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            self.classifier.fit(X_tr, y_tr)
            accuracy = float(accuracy_score(y_te, self.classifier.predict(X_te)))
        else:
            self.classifier.fit(X, y)

        self.is_trained = True
        self._save_model()

        return {
            "accuracy": round(accuracy * 100, 1),
            "examples": len(self.training_data),
        }

    # ── Info ─────────────────────────────────────────────────

    @property
    def example_count(self) -> int:
        return len(self.training_data)
