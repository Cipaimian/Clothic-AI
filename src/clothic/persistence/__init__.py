"""Local persistence: SQLite event store + human-review/appeal workflow.

Deliberately stdlib-only (``sqlite3``) so it runs anywhere with no server.
Implements the accountability loop from ``deploy/db/schema.sql``: every
automated decision is logged with its evidence and the policy version that
judged it, and a human can confirm or override it (appeals).
"""

from clothic.persistence.store import EventStore

__all__ = ["EventStore"]
