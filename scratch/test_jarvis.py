import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.brain import AATASBrain
from db.database import SessionLocal, get_or_create_user

def test_jarvis():
    brain = AATASBrain()
    db = SessionLocal()
    
    # Use a dummy user
    user = get_or_create_user(db, 9999, "Tony Stark")
    
    print("--- Testing JARVIS Style Detection ---")
    messages = [
        "Hello Jarvis",
        "Show my inbox, Sir",
        "My birthday is May 29th",
        "Who am I?",
        "Thanks Jarvis"
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        replies, intents = brain.chat(db, user, msg)
        for r in replies:
            print(f"AATAS: {r}")

    db.close()

if __name__ == "__main__":
    test_jarvis()
