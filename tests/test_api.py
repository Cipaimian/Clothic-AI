"""API integration tests (FastAPI TestClient + persistence + dashboard)."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Use an isolated temp DB per test run so logs don't accumulate on disk.
    monkeypatch.setenv("CLOTHIC_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("CLOTHIC_BACKEND", "mock")
    import clothic.api.app as app_module

    importlib.reload(app_module)  # rebuild the store against the temp DB
    with TestClient(app_module.app) as c:
        yield c


def test_health_and_profiles(client):
    h = client.get("/v1/health").json()
    assert h["status"] == "ok"
    assert "default" in client.get("/v1/profiles").json()["profiles"]


def test_infer_logs_events_and_review_flow(client):
    r = client.post("/v1/infer", json={"profile_id": "default", "camera_id": "dashboard"})
    assert r.status_code == 200
    frame = r.json()
    assert len(frame["persons"]) == 4

    events = client.get("/v1/events?camera_id=dashboard").json()
    assert len(events["events"]) == 4
    assert sum(events["stats"].values()) == 4

    eid = events["events"][0]["id"]
    full = client.get(f"/v1/events/{eid}").json()
    assert "evidence" in full

    rv = client.post(f"/v1/events/{eid}/review",
                     json={"reviewer": "operator", "verdict": "confirm"})
    assert rv.status_code == 200
    assert client.get(f"/v1/events/{eid}").json()["reviews"][0]["verdict"] == "confirm"


def test_review_rejects_bad_verdict(client):
    client.post("/v1/infer", json={"camera_id": "dashboard"})
    eid = client.get("/v1/events").json()["events"][0]["id"]
    bad = client.post(f"/v1/events/{eid}/review", json={"reviewer": "x", "verdict": "nope"})
    assert bad.status_code == 400


def test_web_frontend_served(client):
    # The API serves the web/ SPA at the root, so a single process runs both.
    r = client.get("/")
    assert r.status_code == 200
    assert "Clothic AI" in r.text and "text/html" in r.headers["content-type"]


def test_operator_dashboard_served(client):
    # The operator dashboard moved to /ops once / started serving the SPA.
    r = client.get("/ops")
    assert r.status_code == 200
    assert "Clothic AI" in r.text and "text/html" in r.headers["content-type"]
