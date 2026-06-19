-- Clothic AI operational schema (PostgreSQL).
-- Append-only policy versioning + human-review audit trail make every
-- automated decision reproducible and contestable.

CREATE TABLE IF NOT EXISTS policy_versions (
  profile_id    TEXT NOT NULL,
  version       TEXT NOT NULL,
  document      JSONB NOT NULL,            -- full profile JSON in force
  created_by    TEXT,
  created_at    TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (profile_id, version)
);

CREATE TABLE IF NOT EXISTS cameras (
  id             TEXT PRIMARY KEY,
  location       TEXT NOT NULL,
  zone           TEXT,                      -- lab | library | entrance ...
  active_profile TEXT,
  created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
  id                BIGSERIAL PRIMARY KEY,
  camera_id         TEXT REFERENCES cameras(id),
  track_id          INTEGER NOT NULL,
  profile_id        TEXT NOT NULL,
  policy_version    TEXT NOT NULL,
  decision          TEXT NOT NULL,          -- compliant|minor|major|insufficient
  exposure_score    REAL,
  formality_score   REAL,
  compliance_score  REAL,
  uncertainty_score REAL,
  overall_violation REAL,
  evidence_quality  REAL,
  explanation       TEXT,
  frame_ref         TEXT,                   -- encrypted object-store key
  created_at        TIMESTAMPTZ DEFAULT now(),
  FOREIGN KEY (profile_id, policy_version)
      REFERENCES policy_versions(profile_id, version)
);
CREATE INDEX IF NOT EXISTS idx_events_cam_time ON events (camera_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_decision ON events (decision);

-- Full structured evidence (attributes + matched rules) for audit/analytics.
CREATE TABLE IF NOT EXISTS event_evidence (
  event_id  BIGINT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
  payload   JSONB NOT NULL
);

-- Human review / appeals: a person makes any consequential decision.
CREATE TABLE IF NOT EXISTS reviews (
  id         BIGSERIAL PRIMARY KEY,
  event_id   BIGINT REFERENCES events(id),
  reviewer   TEXT NOT NULL,
  verdict    TEXT NOT NULL,                 -- confirm|override_compliant|override_violation
  note       TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Active-learning queue: low-confidence frames to label and retrain on.
CREATE TABLE IF NOT EXISTS label_queue (
  id         BIGSERIAL PRIMARY KEY,
  frame_ref  TEXT NOT NULL,
  reason     TEXT,                          -- low_conf | model_disagreement
  status     TEXT DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Which model artifact produced a decision (provenance).
CREATE TABLE IF NOT EXISTS model_registry (
  id           TEXT PRIMARY KEY,            -- e.g. segformer-b2@campus-v3
  role         TEXT NOT NULL,
  artifact_uri TEXT NOT NULL,
  metrics      JSONB,
  created_at   TIMESTAMPTZ DEFAULT now()
);
