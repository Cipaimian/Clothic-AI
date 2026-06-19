"""Typed data contracts shared between every pipeline stage.

These pydantic models are the *only* coupling between stages. Any perception
backend may be swapped as long as it emits ``PersonObservation`` objects, and
the reasoning core only ever consumes them. This keeps the system modular and
makes every field that influences a decision explicit and auditable.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Bounding box in absolute pixel coords: (x, y, width, height).
BBox = tuple[float, float, float, float]


class Decision(str, Enum):
    """Final per-person verdict bands."""

    COMPLIANT = "compliant"
    MINOR_VIOLATION = "minor_violation"
    MAJOR_VIOLATION = "major_violation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GarmentEvidence(BaseModel):
    """A single detected garment (or footwear) instance."""

    type: str = Field(..., description="Canonical garment type, e.g. 'tank_top'.")
    conf: float = Field(0.0, ge=0.0, le=1.0, description="Detection confidence.")
    attributes: dict[str, float] = Field(
        default_factory=dict,
        description="Multi-label attribute -> probability (0..1), e.g. {'sleeveless': 0.93}.",
    )

    def attr(self, name: str) -> float:
        return float(self.attributes.get(name, 0.0))


class PersonObservation(BaseModel):
    """Everything perception knows about one tracked person in a frame.

    This is the hand-off object from the *perception core* to the
    *reasoning core*. Exposure values are derived geometrically (parsing +
    pose), never guessed by a single model -- see ``perception.exposure``.
    """

    track_id: int
    bbox: BBox
    upper: Optional[GarmentEvidence] = None
    lower: Optional[GarmentEvidence] = None
    footwear: Optional[GarmentEvidence] = None
    # Skin-exposure ratio per body region, 0..1 (e.g. {'shoulder': 0.78}).
    exposure: dict[str, float] = Field(default_factory=dict)
    # 0..1 reliability of this observation (occlusion / blur / resolution / pose).
    evidence_quality: float = Field(1.0, ge=0.0, le=1.0)
    # Frames this track has been observed (used by temporal fusion / debounce).
    frames_seen: int = 1


class Scores(BaseModel):
    """The four-score methodology that replaces a binary sopan/tidak label."""

    exposure_score: float = Field(0.0, ge=0.0, le=1.0)
    formality_score: float = Field(0.0, ge=0.0, le=1.0)
    compliance_score: float = Field(1.0, ge=0.0, le=1.0)
    uncertainty_score: float = Field(0.0, ge=0.0, le=1.0)
    overall_violation: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="None when evidence is insufficient."
    )


class MatchedRule(BaseModel):
    """A policy rule that fired for a person, kept for the audit trail."""

    id: str
    description: str
    category: str
    weight: float
    severity: float
    citation: Optional[str] = None
    advisory_only: bool = False


class GarmentCoverage(BaseModel):
    """How well one garment covers the body regions it is responsible for.

    ``regions`` maps a body region to its coverage ratio (1.0 = fully covered by
    this garment, 0.0 = that region is bare). Derived from Model 1's exposure.
    """

    slot: str
    garment_type: str
    regions: dict[str, float] = Field(default_factory=dict)


class Remediation(BaseModel):
    """A verified counterfactual: the minimal changes to become compliant.

    Each step is human-readable and tied to a rule that fired. ``verified`` is
    True only if re-running the engine on the modified observation actually
    yields a compliant decision -- so the advice is never hand-wavy.
    """

    steps: list[str] = Field(default_factory=list)
    addresses_rules: list[str] = Field(default_factory=list)
    resulting_decision: Optional[Decision] = None
    verified: bool = False


class PersonDecision(BaseModel):
    """The full, explainable verdict for one person."""

    track_id: int
    bbox: BBox
    decision: Decision
    scores: Scores
    observation: PersonObservation
    matched_rules: list[MatchedRule] = Field(default_factory=list)
    coverage: list[GarmentCoverage] = Field(default_factory=list)
    explanation: str = ""
    remediation: Optional[Remediation] = None
    action: str = "log"


class FrameResult(BaseModel):
    """Top-level API/output object for one processed frame."""

    frame_id: str
    camera_id: str
    profile_id: str
    policy_version: str
    persons: list[PersonDecision] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
