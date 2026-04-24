"""
AATAS — Gmail OAuth Manager
Handles per-user Gmail credentials stored in the DB.
Each user authenticates via a web link during /setup.
"""

import json
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# to:
from db.database import Session, User

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
REDIRECT_URI     = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8000/oauth/callback")


def make_flow() -> Flow:
    return Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

def get_auth_url(state: str) -> str:
    """Generate OAuth URL. state = str(telegram_user_id)."""
    flow = make_flow()
    url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return url

def exchange_code(code: str) -> Credentials:
    """Exchange auth code for credentials."""
    flow = make_flow()
    flow.fetch_token(code=code)
    return flow.credentials

def creds_from_user(user: User) -> Credentials | None:
    """Load + refresh credentials for a user."""
    if not user.gmail_token_json:
        return None
    creds = Credentials.from_authorized_user_info(
        json.loads(user.gmail_token_json), SCOPES
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds

def save_creds(db: Session, user: User, creds: Credentials):
    user.gmail_token_json = creds.to_json()
    db.commit()

def get_gmail_service(user: User):
    creds = creds_from_user(user)
    if not creds:
        raise ValueError("User has not connected Gmail")
    return build("gmail", "v1", credentials=creds)

def get_user_email(user: User) -> str:
    """Fetch the authenticated Gmail address."""
    svc    = get_gmail_service(user)
    profile = svc.users().getProfile(userId="me").execute()
    return profile.get("emailAddress", "")
