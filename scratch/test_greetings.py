
import sys
import os
sys.path.append(os.getcwd())

from api.brain import AATASBrain
from db.database import SessionLocal, User, get_or_create_user

# Mock DB session and user
db = SessionLocal()
user = get_or_create_user(db, telegram_id=9999, name="Test User")

brain = AATASBrain()

test_phrases = ["hi", "hello", "hiii", "HIIIIIII", "hellooo", "hey", "wassup", "yoooooo", "good morning"]

print(f"{'Phrase':<15} | {'Intent':<15} | {'Response Snippet'}")
print("-" * 60)

for p in test_phrases:
    replies, intents = brain.chat(db, user, p, show_disclaimer=False)
    intent = intents[0]["intent"] if intents else "none"
    response = replies[0][:30].replace("\n", " ") + "..."
    print(f"{p:<15} | {intent:<15} | {response}")
