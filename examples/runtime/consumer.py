"""Durable example deduplication/revision handling shared by the three consumers."""

import sqlite3
from pathlib import Path

from litmusai.runtime.models import CloudEvent


class ConsumerState:
    """Record seen event IDs and keep the newest revision even when delivery is reordered."""

    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS latest (
                    project TEXT, alert TEXT, revision INTEGER, body TEXT,
                    PRIMARY KEY(project,alert));
            """)

    def consume(self, body: bytes) -> bool:
        """Accept once; return true only for a newer logical alert revision."""
        event = CloudEvent.model_validate_json(body)
        alert = event.data
        with sqlite3.connect(self.path) as db:
            inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (event.id,)).rowcount
            if not inserted:
                return False
            changed = db.execute(
                "INSERT INTO latest VALUES (?,?,?,?) ON CONFLICT(project,alert) DO UPDATE SET "
                "revision=excluded.revision,body=excluded.body "
                "WHERE excluded.revision>latest.revision",
                (alert.project_id, alert.alert_id, alert.revision, event.model_dump_json()),
            ).rowcount
        if changed:
            # Replace with a durable work queue for your idempotent response handler.
            print(
                f"{alert.category}: {alert.severity}; "
                f"alert={alert.alert_id}; revision={alert.revision}"
            )
        return bool(changed)
