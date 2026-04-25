import random

chat_examples = [
    # greeting
    ("hello", "chat_greeting"), ("hi", "chat_greeting"), ("hey", "chat_greeting"),
    ("good morning", "chat_greeting"), ("good afternoon", "chat_greeting"),
    ("good evening", "chat_greeting"), ("greetings", "chat_greeting"),
    ("hi there", "chat_greeting"), ("hello there", "chat_greeting"),
    ("hiya", "chat_greeting"), ("hey bot", "chat_greeting"),
    ("hi bot", "chat_greeting"), ("hello bot", "chat_greeting"),
    ("yo", "chat_greeting"), ("hey aatas", "chat_greeting"),

    # how are you
    ("how are you", "chat_how_are_you"), ("how are you doing", "chat_how_are_you"),
    ("how's it going", "chat_how_are_you"), ("what's up", "chat_how_are_you"),
    ("how have you been", "chat_how_are_you"), ("are you doing well", "chat_how_are_you"),
    ("how are things", "chat_how_are_you"), ("how do you do", "chat_how_are_you"),
    ("what is up", "chat_how_are_you"), ("how is your day", "chat_how_are_you"),

    # identity
    ("who are you", "chat_identity"), ("what are you", "chat_identity"),
    ("are you a robot", "chat_identity"), ("are you an ai", "chat_identity"),
    ("what is your name", "chat_identity"), ("who made you", "chat_identity"),
    ("tell me about yourself", "chat_identity"), ("what can you do", "chat_identity"),
    ("are you human", "chat_identity"), ("identify yourself", "chat_identity"),
    ("what do you do", "chat_identity"), ("who am i talking to", "chat_identity"),

    # thanks
    ("thank you", "chat_thanks"), ("thanks", "chat_thanks"),
    ("thanks a lot", "chat_thanks"), ("appreciate it", "chat_thanks"),
    ("thank u", "chat_thanks"), ("ty", "chat_thanks"),
    ("thanks so much", "chat_thanks"), ("you are the best", "chat_thanks"),
    ("thanks for the help", "chat_thanks"), ("thank you very much", "chat_thanks"),
    ("cheers", "chat_thanks"), ("thanks buddy", "chat_thanks"),

    # goodbye
    ("bye", "chat_goodbye"), ("goodbye", "chat_goodbye"),
    ("see you", "chat_goodbye"), ("see you later", "chat_goodbye"),
    ("cya", "chat_goodbye"), ("catch you later", "chat_goodbye"),
    ("talk to you later", "chat_goodbye"), ("have a good one", "chat_goodbye"),
    ("farewell", "chat_goodbye"), ("bye bot", "chat_goodbye"),
    ("gotta go", "chat_goodbye"), ("im leaving", "chat_goodbye"),
]

# Write to files/ml/seed_data.py
with open("files/ml/seed_data.py", "r") as f:
    lines = f.readlines()

# Remove the last line assuming it's ']'
if lines[-1].strip() == "]":
    lines.pop()

# Add our examples
new_lines = []
for text, intent in chat_examples:
    new_lines.append(f'    ("{text}", "{intent}"),\n')

new_lines.append("]\n")

with open("files/ml/seed_data.py", "w") as f:
    f.writelines(lines + new_lines)

print(f"Added {len(chat_examples)} chat examples to seed_data.py")
