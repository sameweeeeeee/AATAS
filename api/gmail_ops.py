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

def fetch_emails(svc: Resource, max_results: int = 10, query: str = "-in:trash -in:spam") -> list[dict]:
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
            "threadId": msg.get("threadId", ""),
            "labels":  msg.get("labelIds", []),
        })
    return out


def search_emails(service, query, max_results=10):
    # Handle time-based filters like "last 7 days"
    # Gmail API query format for time: "newer_than:7d"
    q = query
    import re
    m = re.search(r"last (\d+) days", q, re.IGNORECASE)
    if m:
        days = m.group(1)
        q = q.replace(m.group(0), "").strip()
        q = f"{q} newer_than:{days}d"
    
    results = service.users().messages().list(
        userId="me",
        q=q,
        maxResults=max_results
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for i, msg in enumerate(messages, start=1):
        full = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = full.get("payload", {})
        headers = payload.get("headers", [])

        subject = _header(headers, "Subject") or "(no subject)"
        sender = _header(headers, "From")

        emails.append({
            "id": msg["id"],
            "idx": i,
            "subject": subject,
            "sender": sender,
            "body": _clean(_extract_body(payload)),
            "threadId": full.get("threadId", ""),
            "date": _header(headers, "Date")
        })

    return emails


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

def send_new_email(svc: Resource, to_address: str, subject: str, body: str):
    import email.mime.text
    msg = email.mime.text.MIMEText(body)
    msg["To"]      = to_address
    msg["Subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    svc.users().messages().send(
        userId="me",
        body={"raw": raw}
    ).execute()



# ── Rule engine ────────────────────────────────────

def _email_matches_rule(email: dict, rule: AutomationRule, brain: Optional[object] = None) -> bool:
    """
    Robust rule matching engine.
    Supports:
    - Multiple keywords (OR matching)
    - Regex patterns (e.g. "regex:.*invoice.*")
    - Case-insensitive sender contains
    """
    target = rule.get_target()
    text   = (email["subject"] + " " + email["body"]).lower()
    sender = email["sender"].lower()

    # 1. Sender matching (priority)
    sender_filter = target.get("from", "").strip().lower()
    if sender_filter and sender_filter not in sender:
        return False

    # 2. Keyword/Regex matching
    keywords = [k.strip() for k in target.get("keywords", []) if k.strip()]
    matched = False
    if keywords:
        for kw in keywords:
            if kw.lower().startswith("regex:"):
                pattern = kw[6:].strip()
                try:
                    if re.search(pattern, text, re.IGNORECASE) or re.search(pattern, sender, re.IGNORECASE):
                        matched = True; break
                except Exception:
                    pass # ignore invalid regex
            elif kw.lower() in text:
                matched = True; break
    
    # 3. Semantic matching (fallback if keywords didn't match or were empty)
    if not matched and brain and hasattr(brain, "check_semantic_match"):
        # We only do semantic matching if it's a reasonably long rule description
        if len(rule.rule_text.split()) > 3:
            matched = brain.check_semantic_match(rule.rule_text, email)

    # safety: require at least one condition to have been met (keywords, sender, or semantic)
    # If we have no keywords and no sender filter, semantic is the ONLY way to match.
    if not matched and not sender_filter:
        return False

    return matched or (sender_filter != "")

def apply_rules(
    svc:     Resource,
    db:      Session,
    user_id: int,
    emails:  list[dict],
    rules:   list[AutomationRule],
    brain:   Optional[object] = None,
) -> list[dict]:
    """
    Run all user rules against a list of emails.
    Returns list of applied actions: [{email, rule, action}, ...]
    """
    applied = []
    for email in emails:
        for rule in rules:
            if not _email_matches_rule(email, rule, brain):
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