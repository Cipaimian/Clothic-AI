"""Test the active-learning harvest loop."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from active_learning import harvest_from_events  # noqa: E402

from clothic.persistence import EventStore
from clothic.pipeline import ClothicPipeline


def test_harvest_enqueues_insufficient_events():
    store = EventStore(":memory:")
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    result = pipe.process_frame(None)
    store.log_frame(result)
    # The occluded persona logs an insufficient_evidence event.
    n = harvest_from_events(store)
    assert n >= 1
    pending = store.pending_labels()
    assert any(p["reason"] == "insufficient_evidence" for p in pending)
    store.close()
