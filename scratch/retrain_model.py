import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from files.ml.intent_model import IntentModel

def retrain():
    print("🧠 Retraining intent model with new search data...")
    model = IntentModel()
    # Force a fresh train on seed data (which now includes search)
    model.training_data = [] # Clear existing data if any to force seed load
    model.load_or_init()
    stats = model.retrain()
    print(f"✅ Model retrained: {stats}")

if __name__ == "__main__":
    retrain()
