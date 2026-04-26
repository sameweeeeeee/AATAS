"""
AATAS — Neural Fallback Layer (sentence-transformers + sklearn MLP)
Uses semantic embeddings instead of TF-IDF — understands paraphrasing,
synonyms, and typos. No PyTorch training loop, no OpenMP conflicts.

Install once:
    pip install sentence-transformers
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier

log = logging.getLogger(__name__)

NEURAL_MODEL_PATH   = "data/neural_fallback.pkl"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # ~90 MB, downloads once
NEURAL_THRESHOLD    = 0.55


def _load_embedder():
    """Lazy-load the sentence-transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )


class NeuralFallback:

    def __init__(self):
        self._clf       = None
        self._label_enc = None
        self._embedder  = None
        self.is_trained = False

    def load_or_init(self, intent_model) -> None:
        """
        Borrow the LabelEncoder from the existing IntentModel.
        (We no longer need word_vec / char_vec — embeddings replace them.)
        """
        self._label_enc = intent_model.label_enc

        # Lazy-load embedder (downloads on first run, cached afterward)
        if self._embedder is None:
            log.info("NeuralFallback: loading sentence-transformer model…")
            self._embedder = _load_embedder()
            log.info("NeuralFallback: sentence-transformer ready.")

        if os.path.exists(NEURAL_MODEL_PATH):
            try:
                with open(NEURAL_MODEL_PATH, "rb") as f:
                    self._clf = pickle.load(f)
                self.is_trained = True
                log.info("NeuralFallback: loaded saved MLP from disk.")
                return
            except Exception as exc:
                log.warning(f"NeuralFallback: could not load ({exc}) — retraining.")

        log.info("NeuralFallback: no saved model; call .train() to build one.")

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Convert texts → dense 384-dim semantic vectors."""
        return self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def train(self, training_data: list[tuple[str, str]]) -> dict:
        if len(training_data) < 10:
            return {"error": "Need at least 10 examples."}

        texts  = [t for t, _ in training_data]
        labels = [l for _, l in training_data]

        log.info(f"NeuralFallback: encoding {len(texts)} examples…")
        X = self._embed(texts)
        y = self._label_enc.transform(labels)

        self._clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        self._clf.fit(X, y)
        self.is_trained = True

        os.makedirs("data", exist_ok=True)
        with open(NEURAL_MODEL_PATH, "wb") as f:
            pickle.dump(self._clf, f)

        preds    = self._clf.predict(X)
        accuracy = float((preds == y).mean())
        n_classes = len(set(labels))
        log.info(
            f"NeuralFallback trained: {len(training_data)} examples, "
            f"{n_classes} intents, acc={accuracy:.1%}"
        )
        return {
            "accuracy": round(accuracy * 100, 1),
            "examples": len(training_data),
            "intents":  n_classes,
        }

    def predict(self, text: str) -> Optional[tuple[str, float]]:
        """
        Returns (intent_label, confidence) if confident, else None.
        Called by brain.py when the sklearn layer fails (conf < 0.60).
        """
        if not self.is_trained or self._clf is None:
            return None
        X     = self._embed([text])
        probs = self._clf.predict_proba(X)[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        if conf < NEURAL_THRESHOLD:
            return None
        label = self._label_enc.inverse_transform([idx])[0]
        return label, conf

    def predict_multi(self, text: str, threshold: float = 0.20) -> list[tuple[str, float]]:
        """Returns all intents above threshold, sorted by confidence."""
        if not self.is_trained or self._clf is None:
            return [("none", 0.0)]
        X     = self._embed([text])
        probs = self._clf.predict_proba(X)[0]
        results = [
            (self._label_enc.inverse_transform([i])[0], float(p))
            for i, p in enumerate(probs) if p >= threshold
        ]
        return sorted(results, key=lambda x: x[1], reverse=True) or [("none", 0.0)]