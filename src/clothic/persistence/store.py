"""SQLite-backed event store for Clothic AI.

Privacy by design: we persist the structured *evidence vector* (attributes,
scores, matched rules) rather than raw frames. ``frame_ref`` is an optional
opaque key into encrypted object storage, never the image itself. Retention is
the caller's responsibility (see ``purge_older_than``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from clothic.schemas import FrameResult, PersonDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_versions (
    profile_id TEXT NOT NULL,
    version    TEXT NOT NULL,
    document   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, version)
);
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id         TEXT,
    frame_id          TEXT,
    track_id          INTEGER NOT NULL,
    profile_id        TEXT NOT NULL,
    policy_version    TEXT NOT NULL,
    decision          TEXT NOT NULL,
    exposure_score    REAL,
    formality_score   REAL,
    compliance_score  REAL,
    uncertainty_score REAL,
    overall_violation REAL,
    explanation       TEXT,
    evidence_json     TEXT,
    frame_ref         TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_decision ON events(decision);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id, created_at);
CREATE TABLE IF NOT EXISTS reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   INTEGER NOT NULL REFERENCES events(id),
    reviewer   TEXT NOT NULL,
    verdict    TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS label_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_ref  TEXT,
    event_id   INTEGER,
    reason     TEXT,
    status     TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL
);
"""

VALID_VERDICTS = {"confirm", "override_compliant", "override_violation"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventStore:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock lets the store back a threaded web
        # server (FastAPI runs sync routes in a threadpool).
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # -- policy provenance -------------------------------------------------

    def record_policy_version(self, profile: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO policy_versions VALUES (?,?,?,?)",
            (profile.get("profile_id"), profile.get("version"),
             json.dumps(profile), _now()),
        )
        self.conn.commit()

    # -- event logging -----------------------------------------------------

    def log_decision(
        self,
        decision: PersonDecision,
        *,
        camera_id: str,
        frame_id: str,
        profile_id: str,
        policy_version: str,
        frame_ref: str | None = None,
    ) -> int:
        s = decision.scores
        cur = self.conn.execute(
            """INSERT INTO events (
                camera_id, frame_id, track_id, profile_id, policy_version, decision,
                exposure_score, formality_score, compliance_score, uncertainty_score,
                overall_violation, explanation, evidence_json, frame_ref, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                camera_id, frame_id, decision.track_id, profile_id, policy_version,
                decision.decision.value, s.exposure_score, s.formality_score,
                s.compliance_score, s.uncertainty_score, s.overall_violation,
                decision.explanation,
                json.dumps({
                    "observation": decision.observation.model_dump(mode="json"),
                    "matched_rules": [r.model_dump(mode="json") for r in decision.matched_rules],
                    "remediation": decision.remediation.model_dump(mode="json")
                    if decision.remediation else None,
                }),
                frame_ref, _now(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_frame(self, result: FrameResult, *, frame_ref: str | None = None) -> list[int]:
        """Log every person decision in a FrameResult; return event ids."""
        return [
            self.log_decision(
                p, camera_id=result.camera_id, frame_id=result.frame_id,
                profile_id=result.profile_id, policy_version=result.policy_version,
                frame_ref=frame_ref,
            )
            for p in result.persons
        ]

    # -- reviews / appeals -------------------------------------------------

    def add_review(self, event_id: int, reviewer: str, verdict: str, note: str = "") -> int:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}")
        if self.get_event(event_id) is None:
            raise ValueError(f"no such event: {event_id}")
        cur = self.conn.execute(
            "INSERT INTO reviews (event_id, reviewer, verdict, note, created_at) VALUES (?,?,?,?,?)",
            (event_id, reviewer, verdict, note, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # -- active-learning queue --------------------------------------------

    def enqueue_label(self, reason: str, frame_ref: str | None = None,
                      event_id: int | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO label_queue (frame_ref, event_id, reason, status, created_at) "
            "VALUES (?,?,?,'pending',?)",
            (frame_ref, event_id, reason, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def pending_labels(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM label_queue WHERE status='pending' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- queries -----------------------------------------------------------

    def get_event(self, event_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            return None
        event = dict(row)
        event["evidence"] = json.loads(event.pop("evidence_json") or "null")
        event["reviews"] = [
            dict(r) for r in self.conn.execute(
                "SELECT * FROM reviews WHERE event_id=? ORDER BY id", (event_id,)
            ).fetchall()
        ]
        return event

    def query_events(
        self, *, decision: str | None = None, camera_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        clauses, params = [], []
        if decision:
            clauses.append("decision=?")
            params.append(decision)
        if camera_id:
            clauses.append("camera_id=?")
            params.append(camera_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT id, camera_id, frame_id, track_id, decision, overall_violation, "
            f"explanation, created_at FROM events {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT decision, COUNT(*) c FROM events GROUP BY decision"
        ).fetchall()
        return {r["decision"]: r["c"] for r in rows}

    # -- retention (privacy) ----------------------------------------------

    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self.conn.execute("DELETE FROM events WHERE created_at < ?", (cutoff,))
        self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *exc: Iterable) -> None:
        self.close()
