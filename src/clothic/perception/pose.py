"""Pose estimation → anatomical anchor lines for exposure geometry.

Pose gives the landmarks that turn "is skin showing" into a *measurable,
defensible* statement: the knee y-coordinate defines the above/below-knee line;
the shoulder/elbow define sleeve coverage. Hemline coverage is normalised by
limb length so it is invariant to camera distance and subject scale.

Lazy-imported: the base package works without ``ultralytics``/``torch``. Install
with ``pip install "clothic[perception]"``. Default model is YOLO-pose (COCO 17
keypoints); swap for RTMPose/ViTPose by changing ``weights``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# COCO-17 keypoint indices used for our coverage lines.
KP = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


@dataclass
class PoseAnchors:
    """Normalised anatomical reference lines for one person (image coords)."""

    shoulder_y: float | None = None
    hip_y: float | None = None
    knee_y: float | None = None
    ankle_y: float | None = None
    torso_len: float | None = None   # shoulder->hip, the normaliser for coverage
    visibility: float = 0.0          # mean keypoint confidence, 0..1
    keypoints: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def leg_coverage_ratio(self, garment_lowest_y: float) -> float | None:
        """1.0 = covers to/past the knee; <1 = above-knee (more thigh shown).

        Measures how far down the leg a garment reaches, as a fraction of the
        hip->knee segment. Returns None if the needed landmarks are missing.
        """
        if self.hip_y is None or self.knee_y is None or self.knee_y <= self.hip_y:
            return None
        span = self.knee_y - self.hip_y
        return float(max(0.0, min(1.0, (garment_lowest_y - self.hip_y) / span)))


class PoseEstimator:
    def __init__(self, weights: str = "yolo11n-pose.pt", device: str | None = None,
                 conf: float = 0.3):
        from ultralytics import YOLO  # lazy

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def estimate(self, frame, person_box=None) -> PoseAnchors:
        res = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return PoseAnchors()
        # Use the first/most-confident detection (or the one matching person_box).
        kpts = res.keypoints.data[0]  # (17, 3): x, y, conf
        pts = {name: (float(kpts[i][0]), float(kpts[i][1]), float(kpts[i][2]))
               for name, i in KP.items()}

        def _avg_y(a, b):
            ya, ca = pts[a][1], pts[a][2]
            yb, cb = pts[b][1], pts[b][2]
            if ca + cb == 0:
                return None
            return (ya * ca + yb * cb) / (ca + cb)

        shoulder_y = _avg_y("left_shoulder", "right_shoulder")
        hip_y = _avg_y("left_hip", "right_hip")
        knee_y = _avg_y("left_knee", "right_knee")
        ankle_y = _avg_y("left_ankle", "right_ankle")
        torso = (hip_y - shoulder_y) if (shoulder_y and hip_y and hip_y > shoulder_y) else None
        vis = float(sum(p[2] for p in pts.values()) / len(pts))

        return PoseAnchors(shoulder_y=shoulder_y, hip_y=hip_y, knee_y=knee_y,
                           ankle_y=ankle_y, torso_len=torso, visibility=vis, keypoints=pts)
