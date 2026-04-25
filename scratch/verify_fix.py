import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.brain import AATASBrain
from db.database import SessionLocal, get_or_create_user

brain = AATASBrain()
db = SessionLocal()
user = get_or_create_user(db, 1, "Erian")

# Mock cached emails
cached = [
    {"idx": 1, "id": "msg1", "subject": "MATH PT SCRIPT", "sender": "Bennett", "body": "Hi, I shared the doc."}
]

# Test intent prediction
text = "help me reply to him"
reply_texts, intent_dicts = brain.chat(db, user, text, cached_emails=cached)

print(f"User: {text}")
print(f"Reply: {reply_texts}")
print(f"Raw Predictions: {brain.intent_model.predict_multi(text)}")
print(f"Intents: {intent_dicts}")

db.close()
