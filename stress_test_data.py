import sys

tricky_cases = [
    # Multi-intent testing
    ("Can you reply saying I'll attend the meeting, and also archive this email?", "reply"),
    ("Can you reply saying I'll attend the meeting, and also archive this email?", "archive"),
    ("archive this and then reply saying thanks", "archive"),
    ("archive this and then reply saying thanks", "reply"),
    ("mark this as done and archive it", "label"),
    ("mark this as done and archive it", "archive"),
    ("trash this junk and delete rule 2", "trash"),
    ("trash this junk and delete rule 2", "delete_rule"),
    ("show my inbox and tell me what the latest email says", "fetch_inbox"),
    ("show my inbox and tell me what the latest email says", "analyse"),
    ("reply to email 2 and label it done", "reply"),
    ("reply to email 2 and label it done", "label"),
    
    # Slang & Noisy input
    ("yo can u send rpt asap thx bro", "reply"),
    ("pls chk inbox rn", "fetch_inbox"),
    ("gimme the latest mails", "fetch_inbox"),
    ("delete dis", "trash"),
    ("label it work n archive", "label"),
    ("label it work n archive", "archive"),
    ("whats the tldr for email 3", "analyse"),
    ("sum it up for me", "analyse"),
    ("dump this in the bin", "trash"),
    
    # No context / implicit action
    ("i think we should keep this one", "archive"),
    ("not sure about this, just label it maybe", "label"),
    ("i don't need to see this again", "archive"),
    ("get rid of this", "trash"),
    ("this is spam right?", "trash"),
    ("what is this even about?", "analyse"),
    ("can you explain what they want?", "analyse"),
    
    # Ambiguous
    ("reply something nice i guess", "reply"),
    ("tell them no but politely", "reply"),
    ("just say ok", "reply"),
    
    # Passive aggressive
    ("It would’ve been nice if this was done on time, but I guess I’ll just fix it myself.", "reply"),
    ("whatever, just archive it", "archive"),
]

# Balance by duplicating them 100 times
amplified_cases = tricky_cases * 100

with open("files/ml/seed_data.py", "r") as f:
    lines = f.readlines()

if lines[-1].strip() == "]":
    lines.pop()
elif lines[-1].strip() == ")":
    lines.pop()

new_lines = []
for text, intent in amplified_cases:
    new_lines.append(f'    ("{text}", "{intent}"),\n')
new_lines.append(")\n")

with open("files/ml/seed_data.py", "w") as f:
    f.writelines(lines + new_lines)

print(f"Added {len(amplified_cases)} tricky cases!")
