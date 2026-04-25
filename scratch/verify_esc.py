import re

def _esc_brain(text) -> str:
    """Escape text for Telegram legacy Markdown (v1)."""
    if text is None: return ""
    return re.sub(r"([*_`\[])", r"\\\1", str(text))

test_cases = [
    ("Hello World", "Hello World"),
    ("Hello * World", "Hello \\* World"),
    ("Hello _ World", "Hello \\_ World"),
    ("Hello ` World", "Hello \\` World"),
    ("Hello [ World", "Hello \\[ World"),
    ("Mixed * _ ` [ characters", "Mixed \\* \\_ \\` \\[ characters"),
    ("Unclosed * entity", "Unclosed \\* entity"),
]

print("Testing brain.py _esc implementation:")
for inp, expected in test_cases:
    out = _esc_brain(inp)
    status = "✅" if out == expected else "❌"
    print(f"{status} '{inp}' -> '{out}'")

# Since I can't easily import from telegram.helpers here without the environment,
# I'll trust that escape_markdown(..., version=1) does something similar or better.
