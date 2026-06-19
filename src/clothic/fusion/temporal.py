"""Per-track temporal smoothing and decision debouncing.

Frame-by-frame detection flickers; a single bad frame must not trigger an
audio warning. This fuser keeps an exponential moving average of each garment
attribute and exposure value per ``track_id``, and only lets a non-compliant
decision "stick" after it holds for K of the last M frames.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque

from clothic.schemas import Decision, GarmentEvidence, PersonObservation


def _ema_garment(
    prev: GarmentEvidence | None, obs: GarmentEvidence | None, alpha: float
) -> GarmentEvidence | None:
    if obs is None:
        return prev
    if prev is None or prev.type != obs.type:
        # Type changed (or first sight): adopt the new observation outright.
        return obs
    attrs = dict(prev.attributes)
    for name, val in obs.attributes.items():
        attrs[name] = alpha * val + (1 - alpha) * attrs.get(name, val)
    conf = alpha * obs.conf + (1 - alpha) * prev.conf
    return GarmentEvidence(type=obs.type, conf=conf, attributes=attrs)


class TemporalFuser:
    def __init__(self, alpha: float = 0.4, k: int = 3, m: int = 5):
        self.alpha = alpha
        self.k = k
        self.m = m
        self._state: dict[int, PersonObservation] = {}
        self._history: dict[int, Deque[Decision]] = defaultdict(lambda: deque(maxlen=m))

    def smooth(self, obs: PersonObservation) -> PersonObservation:
        """Return a temporally smoothed copy of an observation for its track."""
        prev = self._state.get(obs.track_id)
        if prev is None:
            self._state[obs.track_id] = obs
            return obs

        a = self.alpha
        exposure = dict(prev.exposure)
        for region, val in obs.exposure.items():
            exposure[region] = a * val + (1 - a) * exposure.get(region, val)
        quality = a * obs.evidence_quality + (1 - a) * prev.evidence_quality

        fused = PersonObservation(
            track_id=obs.track_id,
            bbox=obs.bbox,
            upper=_ema_garment(prev.upper, obs.upper, a),
            lower=_ema_garment(prev.lower, obs.lower, a),
            footwear=_ema_garment(prev.footwear, obs.footwear, a),
            exposure=exposure,
            evidence_quality=quality,
            frames_seen=prev.frames_seen + 1,
        )
        self._state[obs.track_id] = fused
        return fused

    def debounce(self, track_id: int, decision: Decision) -> Decision:
        """Stabilise a per-frame decision using K-of-M agreement.

        A violation only becomes the reported decision once it appears in at
        least K of the last M frames; otherwise the track is reported as its
        last stable state (or compliant while warming up).
        """
        hist = self._history[track_id]
        hist.append(decision)
        if decision in (Decision.MAJOR_VIOLATION, Decision.MINOR_VIOLATION):
            count = sum(
                1 for d in hist if d in (Decision.MAJOR_VIOLATION, Decision.MINOR_VIOLATION)
            )
            if count >= self.k:
                return decision
            # Not enough agreement yet: hold back the alarm.
            return Decision.COMPLIANT if len(hist) < self.m else decision
        return decision

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._state.clear()
            self._history.clear()
        else:
            self._state.pop(track_id, None)
            self._history.pop(track_id, None)
