"""Active-learning loop: export low-confidence frames for human labeling.

Closes the data flywheel: the running system enqueues uncertain observations
(``EventStore.enqueue_label``); this tool drains that queue into a manifest a
labeler can work through, then the new labels feed the next fine-tune.

    python tools/active_learning.py --db data/clothic.db --out data/to_label.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clothic.persistence import EventStore


def harvest_from_events(store: EventStore, uncertainty_min: float = 0.30) -> int:
    """Auto-enqueue insufficient-evidence / low-confidence events for labeling."""
    n = 0
    for ev in store.query_events(decision="insufficient_evidence", limit=10000):
        store.enqueue_label(reason="insufficient_evidence", event_id=ev["id"])
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/clothic.db")
    ap.add_argument("--out", default="data/to_label.json", type=Path)
    ap.add_argument("--harvest", action="store_true",
                    help="Auto-enqueue insufficient-evidence events before export.")
    args = ap.parse_args()

    store = EventStore(args.db)
    if args.harvest:
        added = harvest_from_events(store)
        print(f"Harvested {added} uncertain events into the label queue.")

    pending = store.pending_labels()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pending, indent=2), encoding="utf-8")
    print(f"Wrote {len(pending)} items to label -> {args.out}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
