"""
AATAS — Gmail Operations
All Gmail read/write operations + rule engine.
"""

import base64
import re
from typing import Optional

from googleapiclient.discovery import Resource

from db.database import AutomationRule, Session, ActionLog, log_action


# ── Parsing ────────────────────────────────────────

def _decode_part(part: dict) -> str:
    data = part.get("body", {}).get("data", "")
    if not data: return ""
    return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

def _extract_body(payload: dict) -> str:
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        return _decode_part(payload)
    if mime.startswith("multipart"):
        for part in payload.get("parts", []):
            t = _extract_body(part)
            if t: return t
    return ""

def _header(headers: list, name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""

def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()[:3000]


# ── Fetch ──────────────────────────────────────────

def fetch_emails(svc: Resource, max_results: int = 10, query: str = "is:unread") -> list[dict]:
    resp = svc.users().messages().list(
        userId="me", maxResults=max_results, q=query
    ).execute()
    msgs = resp.get("messages", [])
    out  = []
    for i, ref in enumerate(msgs):
        msg     = svc.users().messages().get(userId="me", id=ref["id"], format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        out.append({
            "idx":     i + 1,
            "id":      ref["id"],
            "subject": _header(headers, "Subject") or "(no subject)",
            "sender":  _header(headers, "From"),
            "date":    _header(headers, "Date"),
            "body":    _clean(_extract_body(payload)),
            "labels":  msg.get("labelIds", []),
        })
    return out


# ── Actions ────────────────────────────────────────

def archive_email(svc: Resource, msg_id: str):
    svc.users().messages().modify(
        userId="me", id=msg_id,
        body={"removeLabelIds": ["INBOX"]}
    ).execute()

def trash_email(svc: Resource, msg_id: str):
    svc.users().messages().trash(userId="me", id=msg_id).execute()

def mark_read(svc: Resource, msg_id: str):
    svc.users().messages().modify(
        userId="me", id=msg_id,
        body={"removeLabelIds": ["UNREAD"]}
    ).execute()

def mark_unread(svc: Resource, msg_id: str):
    svc.users().messages().modify(
        userId="me", id=msg_id,
        body={"addLabelIds": ["UNREAD"]}
    ).execute()

def _ensure_label(svc: Resource, name: str) -> str:
    labels = svc.users().labels().list(userId="me").execute().get("labels", [])
    for l in labels:
        if l["name"].lower() == name.lower():
            return l["id"]
    new = svc.users().labels().create(userId="me", body={"name": name}).execute()
    return new["id"]

def apply_label(svc: Resource, msg_id: str, label_name: str):
    label_id = _ensure_label(svc, label_name)
    svc.users().messages().modify(
        userId="me", id=msg_id,
        body={"addLabelIds": [label_id]}
    ).execute()

def send_reply(svc: Resource, original: dict, reply_body: str):
    import email.mime.text
    msg = email.mime.text.MIMEText(reply_body)
    msg["To"]      = original["sender"]
    msg["Subject"] = "Re: " + original["subject"]
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    svc.users().messages().send(
        userId="me",
        body={"raw": raw, "threadId": original.get("threadId", "")}
    ).execute()


# ── Rule engine ────────────────────────────────────

def _email_matches_rule(email: dict, rule: AutomationRule) -> bool:
    """Check if an email satisfies a rule's target conditions."""
    target = rule.get_target()
    text   = (email["subject"] + " " + email["body"] + " " + email["sender"]).lower()

    # keyword matching
    keywords = target.get("keywords", [])
    if keywords and not any(kw.lower() in text for kw in keywords):
        return False

    # sender matching
    sender_filter = target.get("from", "")
    if sender_filter and sender_filter.lower() not in email["sender"].lower():
        return False

    # label matching (Gmail system label)
    label_filter = target.get("has_label", "")
    if label_filter and label_filter.upper() not in email["labels"]:
        return False

    return True

def apply_rules(
    svc:     Resource,
    db:      Session,
    user_id: int,
    emails:  list[dict],
    rules:   list[AutomationRule],
) -> list[dict]:
    """
    Run all user rules against a list of emails.
    Returns list of applied actions: [{email, rule, action}, ...]
    """
    applied = []
    for email in emails:
        for rule in rules:
            if not _email_matches_rule(email, rule):
                continue

            action = rule.action
            try:
                if action == "archive":
                    archive_email(svc, email["id"])
                elif action == "trash":
                    trash_email(svc, email["id"])
                elif action == "label":
                    lbl = rule.get_target().get("label", "AATAS")
                    apply_label(svc, email["id"], lbl)
                elif action == "mark_read":
                    mark_read(svc, email["id"])

                rule.trigger_count += 1
                db.commit()
                log_action(db, user_id, action, email["id"], email["subject"],
                           f"Rule: {rule.rule_text}")
                applied.append({"email": email, "rule": rule, "action": action})

            except Exception as e:
                log_action(db, user_id, f"ERROR:{action}", email["id"],
                           email["subject"], str(e))

    return applied
