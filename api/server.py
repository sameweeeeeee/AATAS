"""
AATAS — FastAPI Server
Handles:
  - Gmail OAuth callback (redirect from Google)
  - REST API for the Telegram bot
  - Health check
"""

import json
import os
import sys

# Ensure project root is on sys.path so 'api', 'db', etc. resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from api.gmail_auth import exchange_code, get_auth_url, get_user_email, save_creds
from db.database import SessionLocal, get_or_create_user

app = FastAPI(title="AATAS API", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")


@app.get("/health")
def health():
    return {"status": "ok", "service": "AATAS"}


@app.get("/oauth/callback")
async def oauth_callback(code: str = Query(...), state: str = Query(...)):
    """
    Google redirects here after user authorises Gmail access.
    state = telegram_user_id
    """
    db = SessionLocal()
    try:
        telegram_id = int(state)
        creds       = exchange_code(code)
        user        = get_or_create_user(db, telegram_id)
        save_creds(db, user, creds)

        # fetch their email address
        from api.gmail_auth import get_gmail_service
        svc                = get_gmail_service(user)
        profile            = svc.users().getProfile(userId="me").execute()
        user.gmail_email   = profile.get("emailAddress", "")
        user.setup_step    = "complete"
        user.setup_complete = True
        db.commit()

        # Notify via Telegram
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": telegram_id,
                    "text":    (
                        f"✅ *Gmail connected!*\n\n"
                        f"📧 {user.gmail_email}\n\n"
                        f"AATAS is ready. Try saying:\n"
                        f'• _"archive anything about Google Classroom"_\n'
                        f'• _"show my inbox"_\n'
                        f'• _"what are my unread emails?"_'
                    ),
                    "parse_mode": "Markdown",
                },
            )

        return HTMLResponse("""
        <html><body style="font-family:sans-serif;text-align:center;padding:60px;background:#0f1117;color:#fff">
        <h1 style="color:#4ade80">✅ Gmail Connected!</h1>
        <p style="color:#94a3b8;font-size:1.2em">You can close this tab and return to Telegram.</p>
        <p style="color:#64748b">AATAS is now set up and ready to manage your inbox.</p>
        </body></html>
        """)

    except Exception as e:
        return HTMLResponse(f"""
        <html><body style="font-family:sans-serif;text-align:center;padding:60px;background:#0f1117;color:#fff">
        <h1 style="color:#f87171">❌ Connection Failed</h1>
        <p style="color:#94a3b8">{str(e)}</p>
        <p style="color:#64748b">Please go back to Telegram and try /setup again.</p>
        </body></html>
        """)
    finally:
        db.close()


@app.get("/auth/url/{telegram_id}")
def get_oauth_url(telegram_id: int):
    url = get_auth_url(state=str(telegram_id))
    return {"url": url}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))