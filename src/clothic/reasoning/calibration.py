"""Confidence calibration via temperature scaling.

A reported confidence of 0.87 should mean a true 87% likelihood. This matters
because the rule engine thresholds on these numbers and the four-score logic
multiplies them. Temperature scaling fits a single scalar T per model head on a
held-out set, dividing the logits by T before softmax/sigmoid -- it never
changes which class wins (so accuracy is unchanged) but corrects over/under-
confidence.

Pure-numpy, no training framework needed. Fit once offline, persist T, and
apply at inference. Always re-fit after quantization or domain shift.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE for binary/confidence scores: gap between confidence and accuracy.

    ``probs`` and ``labels`` are 1-D arrays in [0,1] / {0,1}. Lower is better.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray, lr: float = 0.01, iters: int = 500
) -> float:
    """Fit a temperature T>0 minimising binary cross-entropy on a held-out set.

    Returns T. Apply with ``apply_temperature(probs, T)``. T>1 softens
    (reduces) over-confidence; T<1 sharpens under-confidence.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    logits = _logit(probs)
    log_t = 0.0  # optimise log T to keep T strictly positive

    for _ in range(iters):
        t = np.exp(log_t)
        p = _sigmoid(logits / t)
        # d(BCE)/d(logT): chain rule through z = logits / T, T = exp(logT).
        grad = float(np.mean((p - labels) * (-logits / t)))
        log_t -= lr * grad
    return float(np.exp(log_t))


def apply_temperature(probs: np.ndarray | float, temperature: float) -> np.ndarray | float:
    """Recalibrate probabilities with a fitted temperature."""
    scalar = np.isscalar(probs)
    arr = np.asarray(probs, dtype=float)
    out = _sigmoid(_logit(arr) / max(temperature, _EPS))
    return float(out) if scalar else out


class Calibrator:
    """Holds per-head temperatures and applies them at inference.

    Head naming convention used by the pipeline:
      * ``garment_conf``      -- garment detection confidence
      * ``attr.<name>``       -- a specific attribute probability
      * ``exposure.<region>`` -- a body-region exposure ratio
    Unknown heads default to T=1 (identity), so an empty calibrator is a no-op.
    """

    def __init__(self, temperatures: dict[str, float] | None = None):
        self.temperatures = temperatures or {}

    def fit(self, head: str, probs: np.ndarray, labels: np.ndarray) -> float:
        t = fit_temperature(probs, labels)
        self.temperatures[head] = t
        return t

    def calibrate(self, head: str, prob: float) -> float:
        t = self.temperatures.get(head, 1.0)
        if t == 1.0:
            return float(prob)
        return float(apply_temperature(prob, t))

    def is_identity(self) -> bool:
        return all(t == 1.0 for t in self.temperatures.values())


def calibrate_observation(obs, calibrator: "Calibrator"):
    """Return a copy of an observation with calibrated confidences/attributes.

    Recalibrating before reasoning means the rule engine thresholds and the
    four-score logic operate on confidences that mean what they say.
    """
    if calibrator is None or calibrator.is_identity():
        return obs
    from clothic.schemas import GarmentEvidence  # local import avoids a cycle

    trial = obs.model_copy(deep=True)
    for slot in ("upper", "lower", "footwear"):
        g = getattr(trial, slot)
        if g is None:
            continue
        attrs = {n: calibrator.calibrate(f"attr.{n}", v) for n, v in g.attributes.items()}
        setattr(trial, slot, GarmentEvidence(
            type=g.type, conf=calibrator.calibrate("garment_conf", g.conf), attributes=attrs,
        ))
    trial.exposure = {
        r: calibrator.calibrate(f"exposure.{r}", v) for r, v in trial.exposure.items()
    }
    return trial
