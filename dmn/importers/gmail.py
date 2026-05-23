"""Gmail Sent importer from a Google Takeout MBox: parse subjects + first ~500 chars of body."""
from __future__ import annotations

import mailbox
from pathlib import Path


def import_sent(takeout_dir: str | Path, limit: int = 500) -> list[dict]:
    """Parse a Sent.mbox (or All Mail) file from a Takeout dir."""
    base = Path(takeout_dir).expanduser()
    if not base.exists():
        return []
    candidates = (
        list(base.rglob("Sent.mbox"))
        + list(base.rglob("Sent Mail.mbox"))
        + list(base.rglob("All mail Including Spam and Trash.mbox"))
    )
    if not candidates:
        return []
    p = candidates[0]
    out: list[dict] = []
    try:
        mbox = mailbox.mbox(str(p))
    except Exception:
        return []
    for i, msg in enumerate(mbox):
        if i >= limit:
            break
        subj = msg.get("Subject") or ""
        body = _first_body(msg)
        text = f"{subj} :: {body}".strip(" :")
        if text:
            out.append({"text": text[:500], "source": "gmail"})
    return out


def _first_body(msg) -> str:
    """Extract a best-effort first ~500 chars of plaintext body from an email message."""
    try:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="ignore")[:500]
        if isinstance(payload, list) and payload:
            first = payload[0]
            if hasattr(first, "get_payload"):
                p2 = first.get_payload(decode=True)
                if isinstance(p2, bytes):
                    return p2.decode("utf-8", errors="ignore")[:500]
    except Exception:
        return ""
    return ""
