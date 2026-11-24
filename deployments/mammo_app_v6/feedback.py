# feedback.py
import os
from google.cloud import firestore

COLLECTION = "feedback"
_db = None


def _client():
    """Creates the Firestore client once and reuses it."""
    global _db
    if _db is None:
        _db = firestore.Client(project=os.getenv("GCP_PROJECT", "mammograph"))
    return _db


def save_feedback(record: dict) -> str:
    """Saves one feedback record and returns its document ID."""
    record = dict(record)
    record["created_at"] = firestore.SERVER_TIMESTAMP
    ref = _client().collection(COLLECTION).document()
    ref.set(record)
    return ref.id


def export_rows(limit: int = 10000):
    """Yields feedback records, newest first, for CSV export."""
    query = (_client().collection(COLLECTION)
             .order_by("created_at", direction=firestore.Query.DESCENDING)
             .limit(limit))
    for doc in query.stream():
        row = doc.to_dict()
        row["id"] = doc.id
        created = row.get("created_at")
        if created is not None:
            row["created_at"] = created.isoformat()
        yield row