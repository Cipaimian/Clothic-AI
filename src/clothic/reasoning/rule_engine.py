"""Transparent predicate evaluation engine.

Policies are *data, not code*. A rule's ``when`` clause is a small boolean
expression over the person's attribute vector. The engine resolves dotted
attribute paths (e.g. ``upper.sleeveless``, ``exposure.thigh``,
``footwear.type``) against a ``PersonObservation`` and reports exactly which
rules matched -- the basis for both scoring and the human-readable explanation.

Predicate grammar (JSON)::

    {"all": [<pred>, ...]}          # logical AND
    {"any": [<pred>, ...]}          # logical OR
    {"not": <pred>}                 # negation
    {"attr": "<path>", "op": "<op>", "value": <v>}   # leaf comparison

Supported leaf operators: >, >=, <, <=, ==, !=, in, not_in, exists.
"""

from __future__ import annotations

from typing import Any

from clothic.schemas import MatchedRule, PersonObservation

_NUMERIC_OPS = {">", ">=", "<", "<=", "==", "!="}


def build_attribute_vector(obs: PersonObservation) -> dict[str, Any]:
    """Flatten an observation into a dotted-path lookup table.

    String slots (``upper.type``) resolve to the garment type; attribute and
    exposure paths resolve to floats. Missing numeric paths default to 0.0 so
    a rule never crashes on absent evidence -- it simply does not fire.
    """
    vec: dict[str, Any] = {}
    for slot in ("upper", "lower", "footwear"):
        garment = getattr(obs, slot)
        if garment is not None:
            vec[f"{slot}.type"] = garment.type
            vec[f"{slot}.conf"] = garment.conf
            for name, val in garment.attributes.items():
                vec[f"{slot}.{name}"] = val
    for region, val in obs.exposure.items():
        vec[f"exposure.{region}"] = val
    vec["evidence_quality"] = obs.evidence_quality
    return vec


def _resolve(vec: dict[str, Any], path: str) -> Any:
    if path in vec:
        return vec[path]
    # Absent numeric attributes behave as 0.0; absent string slots as None.
    return None if path.endswith(".type") else 0.0


def _eval_leaf(vec: dict[str, Any], pred: dict[str, Any]) -> bool:
    path = pred["attr"]
    op = pred["op"]
    expected = pred.get("value")
    actual = _resolve(vec, path)

    if op == "exists":
        present = actual is not None and actual != 0.0
        return present if expected in (None, True) else not present
    if op == "in":
        return actual in (expected or [])
    if op == "not_in":
        return actual not in (expected or [])

    if op in _NUMERIC_OPS:
        # Comparisons require a value on both sides; a missing slot can't match.
        if actual is None or expected is None:
            return op in {"!=", "<", "<="} and actual is None and op == "!="
        a, b = float(actual), float(expected)
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
    raise ValueError(f"Unsupported operator: {op!r}")


def evaluate_predicate(vec: dict[str, Any], pred: dict[str, Any]) -> bool:
    """Recursively evaluate a predicate node against the attribute vector."""
    if "all" in pred:
        return all(evaluate_predicate(vec, p) for p in pred["all"])
    if "any" in pred:
        return any(evaluate_predicate(vec, p) for p in pred["any"])
    if "not" in pred:
        return not evaluate_predicate(vec, pred["not"])
    if "attr" in pred:
        return _eval_leaf(vec, pred)
    raise ValueError(f"Malformed predicate node: {pred!r}")


class RuleEngine:
    """Evaluates a campus policy profile against a person observation."""

    def __init__(self, profile: dict[str, Any]):
        self.profile = profile
        self.profile_id: str = profile.get("profile_id", "unknown")
        self.version: str = profile.get("version", "0")
        self.rules: list[dict[str, Any]] = profile.get("rules", [])

    def match(self, obs: PersonObservation, zone: str | None = None) -> list[MatchedRule]:
        """Return every rule whose predicate is satisfied for this observation.

        Rules can be scoped to specific zones via ``enabled_in_zones``; a rule
        with that key only applies when the camera's ``zone`` is listed.
        """
        vec = build_attribute_vector(obs)
        matched: list[MatchedRule] = []
        for rule in self.rules:
            zones = rule.get("enabled_in_zones")
            if zones is not None and zone not in zones:
                continue
            try:
                fired = evaluate_predicate(vec, rule["when"])
            except (KeyError, ValueError):
                # A malformed rule must never crash the pipeline; skip it.
                continue
            if fired:
                matched.append(
                    MatchedRule(
                        id=rule["id"],
                        description=rule.get("description", ""),
                        category=rule.get("category", "general"),
                        weight=float(rule.get("weight", 1.0)),
                        severity=float(rule.get("severity", 0.5)),
                        citation=rule.get("citation"),
                        advisory_only=bool(rule.get("advisory_only", False)),
                    )
                )
        return matched
