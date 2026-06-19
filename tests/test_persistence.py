"""Tests for the SQLite event store + review/appeal workflow."""

from __future__ import annotations

import pytest

from clothic.persistence import EventStore
from clothic.pipeline import ClothicPipeline


@pytest.fixture
def store():
    s = EventStore(":memory:")
    yield s
    s.close()


def test_log_and_query_frame(store):
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    for _ in range(5):
        result = pipe.process_frame(None)
    ids = store.log_frame(result)
    assert len(ids) == 4
    events = store.query_events(limit=10)
    assert len(events) == 4
    # The major-violation persona should be retrievable by decision filter.
    majors = store.query_events(decision="major_violation")
    assert len(majors) >= 1
    full = store.get_event(majors[0]["id"])
    assert full["evidence"]["observation"]["track_id"] is not None
    assert full["evidence"]["matched_rules"]


def test_review_workflow(store):
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    for _ in range(5):
        result = pipe.process_frame(None)
    ids = store.log_frame(result)
    rid = store.add_review(ids[0], reviewer="dosen_irvan", verdict="confirm", note="ok")
    assert rid > 0
    event = store.get_event(ids[0])
    assert event["reviews"][0]["verdict"] == "confirm"
    assert event["reviews"][0]["reviewer"] == "dosen_irvan"


def test_review_validation(store):
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    result = pipe.process_frame(None)
    ids = store.log_frame(result)
    with pytest.raises(ValueError):
        store.add_review(ids[0], reviewer="x", verdict="not_a_verdict")
    with pytest.raises(ValueError):
        store.add_review(999999, reviewer="x", verdict="confirm")


def test_label_queue(store):
    qid = store.enqueue_label(reason="low_conf", frame_ref="obj://abc")
    assert qid > 0
    pending = store.pending_labels()
    assert len(pending) == 1 and pending[0]["reason"] == "low_conf"


def test_stats_and_purge(store):
    pipe = ClothicPipeline(profile_id="default", backend="mock")
    for _ in range(5):
        result = pipe.process_frame(None)
    store.log_frame(result)
    stats = store.stats()
    assert sum(stats.values()) == 4
    # Nothing is older than 0 days into the future, so purge removes nothing.
    assert store.purge_older_than(days=1) == 0
