# AATAS — An AI-Assisted Technological Arrangement System

> **Now with JARVIS Protocol.** Your personal, private, and local AI digital butler.
> Smart Gmail management and global research via Telegram.

---

## What AATAS does

- 🧠 **Local ML Brain** — 100% private. No external AI APIs (no Claude, no OpenAI). Runs entirely on your hardware.
- 🤵 **JARVIS Personality** — Professional, articulate, and efficient. Talk to it like Tony Stark's assistant.
- 🌐 **Research Module** — Scans the web (DuckDuckGo) and summarizes articles locally using built-in NLP.
- 💬 **Conversational Memory** — Remembers facts about you ("My name is Tony", "I live in Malibu") and uses them in context.
- 🤖 **Persistent Rules** — Set automation rules once (e.g., "archive anything about Google Classroom").
- 📊 **Email Analysis** — Priority classification, TF-IDF summarisation, and draft generation.
- ✅ **Inline Actions** — One-tap archive, label, and mark-read directly from Telegram.

---

## Architecture

```
User on Telegram
     │
     ▼
bot/aatas_bot.py          ← Telegram bot (Python)
     │
     ├── api/brain.py     ← Local ML Brain (TF-IDF + MLP) — understands intent
     │    ├── memory_guesser.py ← Passive learning & fact extraction
     │    └── web_ops.py        ← Web searching & scraping
     ├── api/gmail_ops.py ← Gmail read/write (Google API)
     ├── api/gmail_auth.py← Per-user OAuth flow
     └── db/database.py   ← SQLite: users, rules, memory, history
     
api/server.py             ← FastAPI (OAuth callback endpoint)
```

---

## JARVIS Mode — How to Use

AATAS is now configured with the **JARVIS Protocol**.

### 1. Activating Tone
The bot automatically mirrors your style, but you can force JARVIS mode by:
- Addressing it as **"Jarvis"** or **"Sir"**.
- Saying **"Jarvis mode"** or **"Be Jarvis"**.

### 2. Web Research
You can now ask AATAS to find information outside your inbox:
- **Search**: `"Jarvis, look up the latest SpaceX news"` or `"Find information on Python 3.12 features"`
- **Analyse**: After a search, say `"research 1"` to have Jarvis read and summarize the first result.
- **Direct Link**: `"research https://example.com"` to get a summary of any URL.

### 3. Passive Learning
AATAS listens to what you say and remembers facts:
- `"Jarvis, remember that my office address is 123 Stark Tower."`
- `"Who am I?"` → *"You are Tony, Sir. And I recall you work at Stark Industries."*

---

## Setup Guide

### Step 1 — Prerequisites
- Python 3.11+
- Public domain or IP address (for Gmail OAuth callback)
- Telegram Bot Token (from @BotFather)

### Step 2 — Google Cloud / Gmail
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project "AATAS" and enable **Gmail API**.
3. Configure **OAuth consent screen** (External, add scopes: `gmail.modify`, `userinfo.email`).
4. Create **OAuth 2.0 Client ID** (Web application).
5. Add Redirect URI: `https://YOUR_DOMAIN/oauth/callback`
6. Download JSON → rename to `credentials.json` → place in project root.

### Step 3 — Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and fill in `TELEGRAM_BOT_TOKEN`, `OAUTH_REDIRECT_URI`, etc. **Note: No Anthropic/OpenAI keys are required.**

### Step 4 — Run
```bash
# Install dependencies
pip install -r requirements.txt

# Launch AATAS
./files/launch_aatas.sh
```

---

## Example Conversations

```
You: Jarvis, show my inbox
AATAS: Scanning the frequencies now, Sir. Your inbox is displayed.

---

You: look up the current price of Bitcoin
AATAS: Accessing the internet protocols, Sir. Searching for the requested data...
       [1] Bitcoin Price Index — $65,000...
       [2] BTC Live Market Cap...
       Shall I research one of these in detail? Say "research 1".

---

You: archive anything about promotions
AATAS: Protocol established, Sir. I have set a permanent watch for those conditions.
```

---

## Security & Privacy
- **100% Local**: Your messages never leave your server for "AI processing."
- **Encrypted Tokens**: Gmail OAuth tokens are stored securely.
- **Private Butler**: Jarvis only learns from *your* conversations.

---

## File Structure
- `bot/aatas_bot.py`: Main Telegram interface.
- `api/brain.py`: Intent recognition & JARVIS persona.
- `api/memory_guesser.py`: The "Guessing Engine" for passive learning.
- `api/web_ops.py`: Web searching and scraping.
- `api/gmail_ops.py`: Gmail integration and automation rules.
- `files/ml/`: Training data and model logic.
