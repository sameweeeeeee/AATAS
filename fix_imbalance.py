import sys
sys.path.append('.')

from files.ml.seed_data import SEED_DATA

# Find how many we currently have
chat_data = [item for item in SEED_DATA if item[1].startswith("chat_")]
other_data = [item for item in SEED_DATA if not item[1].startswith("chat_")]

print(f"Original chat examples: {len(chat_data)}")
print(f"Original other examples: {len(other_data)}")

# Duplicate chat data to reach ~800 per chat class, or just duplicate them 100 times
balanced_chat = chat_data * 100

new_seed = other_data + balanced_chat

with open("files/ml/seed_data.py", "w") as f:
    f.write("SEED_DATA = (\n")
    for text, intent in new_seed:
        f.write(f'    ("{text}", "{intent}"),\n')
    f.write(")\n")

print(f"New total examples: {len(new_seed)}")
