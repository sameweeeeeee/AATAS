"""
AATAS — Central Training Manager
Singleton wrapper that loads/initializes all ML models and exposes
them as a single object for brain.py and aatas_bot.py to use.
"""

from files.ml.intent_model   import IntentModel
from files.ml.priority_model import PriorityModel

_intent_model:   IntentModel   | None = None
_priority_model: PriorityModel | None = None


def get_intent_model() -> IntentModel:
    global _intent_model
    if _intent_model is None:
        _intent_model = IntentModel()
        _intent_model.load_or_init()
    return _intent_model


def get_priority_model() -> PriorityModel:
    global _priority_model
    if _priority_model is None:
        _priority_model = PriorityModel()
        _priority_model.load_or_init()
    return _priority_model


def retrain_all() -> dict:
    intent_model   = get_intent_model()
    intent_stats   = intent_model.retrain()
    priority_stats = get_priority_model().retrain()

    # Retrain neural fallback with the updated intent model
    from files.ml.neural_fallback import NeuralFallback
    nf = NeuralFallback()
    nf.load_or_init(intent_model)
    nf.train(intent_model.training_data)

    return {
        "intent":   intent_stats,
        "priority": priority_stats,
    }
