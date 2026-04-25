import sys
import os
import pickle

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from files.ml.intent_model import IntentModel, DATA_PATH, MODEL_PATH
from files.ml.seed_data import SEED_DATA

def train_compose():
    model = IntentModel()
    
    # Load existing training data
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as f:
            model.training_data = pickle.load(f)
        print(f"Loaded {len(model.training_data)} existing training examples.")
    else:
        model.training_data = list(SEED_DATA)
        print("No existing training data found, starting with seed data.")

    # Generate additional diverse compose examples
    extra_compose = [
        ("write an email to client about the update", "compose"),
        ("send a message to sarah@example.com", "compose"),
        ("compose new mail to boss", "compose"),
        ("email support@company.com saying my order is missing", "compose"),
        ("write a message to team regarding the deadline", "compose"),
        ("send an email to john.doe@gmail.com", "compose"),
        ("compose a draft to marketing", "compose"),
        ("write an email to recruitment saying i accept the offer", "compose"),
        ("send email to sales about the pricing", "compose"),
        ("new email to finance@business.org", "compose"),
        ("compose a message to the manager", "compose"),
        ("send an email to someone@somewhere.com saying hello", "compose"),
        ("write to info@website.com", "compose"),
        ("draft an email to the board", "compose"),
        ("send a quick note to david", "compose"),
        ("write a formal email to professor", "compose"),
        ("compose a reply to the customer", "compose"), # Wait, compose might be used for reply too, but here we want new email
        ("send a new email to the client", "compose"),
        ("message hr@company.com", "compose"),
        ("write an email to the developer", "compose"),
    ]

    # Combine seed data compose examples + extra ones
    compose_examples = [ex for ex in SEED_DATA if ex[1] == "compose"] + extra_compose

    # Filter out duplicates
    existing_texts = {ex[0] for ex in model.training_data}
    added_count = 0
    for text, intent in compose_examples:
        if text.lower().strip() not in existing_texts:
            model.training_data.append((text.lower().strip(), intent))
            existing_texts.add(text.lower().strip())
            added_count += 1

    print(f"Added {added_count} new compose examples. Total: {len(model.training_data)}")

    # Save and retrain
    model._save_data()
    print("Retraining MLP model...")
    stats = model.retrain()
    print(f"Success! Model accuracy: {stats.get('accuracy')}% on {stats.get('examples')} examples.")

if __name__ == "__main__":
    train_compose()
