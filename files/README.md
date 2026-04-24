# AATAS — An AI-Assisted Technological Arrangement System

> Smart Gmail management via Telegram. Talk naturally. AATAS remembers, learns, and acts.

---

## What AATAS does

- 💬 **Conversational AI** — just talk to it. "Archive anything about Google Classroom", "show my urgent emails", "draft a reply to email 3"
- 🤖 **Persistent rules** — set rules once, they run forever. AATAS remembers every instruction.
- 👥 **Multi-user** — anyone can connect their own Gmail via a self-service /setup flow
- 📊 **AI analysis** — priority classification, summarisation, action item extraction, reply drafting
- 📋 **Action history** — audit log of everything AATAS has done to your inbox
- ✅ **Inline buttons** — one-tap archive, label, and mark-read from Telegram

---

## Architecture

```
User on Telegram
     │
     ▼
bot/aatas_bot.py          ← Telegram bot (Python, python-telegram-bot)
     │
     ├── api/brain.py     ← Claude AI (Anthropic API) — understands intent
     ├── api/gmail_ops.py ← Gmail read/write (Google API)
     ├── api/gmail_auth.py← Per-user OAuth flow
     └── db/database.py   ← SQLite: users, rules, memory, history
     
api/server.py             ← FastAPI (OAuth callback endpoint)
```

---

## Setup Guide

### Step 1 — Prerequisites

- Python 3.11+ or Docker
- A **DigitalOcean Droplet** (2GB RAM minimum) — not Vercel (see note below)
- Public domain or IP address for OAuth redirect

> ⚠️ **Why not Vercel?** Gmail OAuth requires a persistent redirect URI, and the
> Telegram bot needs long-polling or a persistent process. Vercel's serverless
> functions timeout in 10s and have no persistent state. Use DigitalOcean, Railway,
> Render, or any VPS.

---

### Step 2 — Anthropic API Key

1. Go to https://console.anthropic.com/
2. Create an API key
3. Save it — you'll need it in your `.env`

---

### Step 3 — Google Cloud / Gmail

1. Go to https://console.cloud.google.com/
2. Create a new project → name it "AATAS"
3. Enable **Gmail API** (search APIs & Services → Library)
4. Go to **APIs & Services → OAuth consent screen**
   - User type: External
   - App name: AATAS
   - Add scopes: `gmail.readonly`, `gmail.modify`, `gmail.labels`, `userinfo.email`
5. Go to **Credentials → Create Credentials → OAuth 2.0 Client ID**
   - Application type: **Web application**
   - Authorised redirect URIs: `https://YOUR_DOMAIN/oauth/callback`
6. Download JSON → rename to `credentials.json` → place in project root

---

### Step 4 — Telegram Bot

1. Message **@BotFather** on Telegram
2. `/newbot` → choose a name → choose username (e.g. `aatas_bot`)
3. Copy the bot token
4. (Optional) Set bot description: `/setdescription` → "AATAS — AI Gmail manager"
5. Set commands: `/setcommands` then paste:
   ```
   start - Start AATAS
   setup - Connect your Gmail
   inbox - View inbox
   check - Apply automation rules
   rules - Manage rules
   history - Recent actions
   help - Help
   ```

---

### Step 5 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=7123456789:AAF...
GOOGLE_CREDENTIALS_FILE=credentials.json
OAUTH_REDIRECT_URI=https://your-server.com/oauth/callback
DB_PATH=data/aatas.db
```

---

### Step 6 — Deploy on DigitalOcean

```bash
# On your Droplet (Ubuntu 22.04+)

# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone repo
git clone https://github.com/you/aatas
cd aatas

# Add your credentials
nano .env                   # fill in values
nano credentials.json       # paste your Google OAuth JSON

# Run
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

For HTTPS (required for OAuth), use nginx + certbot:
```bash
apt install nginx certbot python3-certbot-nginx
certbot --nginx -d your-domain.com
# Edit nginx config to proxy port 80/443 → 8000
```

---

### Step 7 — First Use

1. Open Telegram → find your bot → `/start`
2. Tap `/setup` → tap "Connect Gmail" button → authorise on Google
3. You'll see "✅ Gmail Connected!" in Telegram
4. Start talking:
   - `"show my inbox"`
   - `"archive anything about Google Classroom"`
   - `"what are my most urgent emails?"`
   - `"analyse email 2"`

---

## Example Conversations

```
You: archive anything about promotions or newsletters
AATAS: Done! I've created a rule to automatically archive emails that mention 
       promotions, newsletters, or unsubscribe links. It'll run every time 
       you do /check. Want me to apply it to your inbox now?

You: yes
AATAS: Applied! Archived 8 emails matching your rule.

---

You: what rules do i have?
AATAS: You have 2 active rules:
       [1] ARCHIVE — archive anything about promotions or newsletters (triggered 8×)
       [2] LABEL — label emails from boss@work.com as "urgent" (triggered 3×)

---

You: draft a casual reply to email 4
AATAS: Here's a casual draft:
       "Hey! Thanks for reaching out — I'll take a look and get back to you 
       by end of week. Cheers!"
       [Send] [Cancel]
```

---

## File Structure

```
aatas/
├── bot/
│   └── aatas_bot.py          # Telegram bot — main entry point
├── api/
│   ├── server.py             # FastAPI — OAuth callback
│   ├── brain.py              # Claude AI conversation layer
│   ├── gmail_auth.py         # Per-user Gmail OAuth
│   └── gmail_ops.py          # Gmail read/write + rule engine
├── db/
│   └── database.py           # SQLite: users, rules, memory, history
├── data/                     # DB and exports (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Security

- Each user's Gmail token is stored encrypted in the DB — never shared
- The bot responds to anyone by default; to restrict to specific users,
  add a `TELEGRAM_USER_ID=12345` env var (comma-separated for multiple)
- All OAuth flows use Google's secure PKCE flow
- `credentials.json` and `.env` are gitignored — never commit them
