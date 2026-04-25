import pickle
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from files.ml.trainer import retrain_all

DATA_PATH = "data/intent_training_data.pkl"

new_examples = [
    ("help me reply to him", "reply"),
    ("help me write an reply to him", "reply"),
    ("can you help me respond to this?", "reply"),
    ("help me write back to that email", "reply"),
    ("write a response for me", "reply"),
    ("draft a reply for me", "reply"),
    ("help me answer him", "reply"),
]

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        training_data = pickle.load(f)
    print(f"Loaded {len(training_data)} existing examples.")
else:
    from files.ml.seed_data import SEED_DATA
    training_data = list(SEED_DATA)
    print(f"Starting with {len(training_data)} seed examples.")

# Add new examples
for text, intent in new_examples:
    if (text, intent) not in training_data:
        training_data.append((text, intent))

# Save back
os.makedirs("data", exist_ok=True)
with open(DATA_PATH, "wb") as f:
    pickle.dump(training_data, f)

print(f"Added new examples. Total: {len(training_data)}")

# Retrain
print("Retraining models... this might take a few seconds.")
stats = retrain_all()
print("Retraining complete!")
print(f"Stats: {stats}")
