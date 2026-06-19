"""End-to-end verification of the FULL hybrid backend (the redesign centerpiece).

Runs person-detect -> pose -> Sapiens body parsing (PIXEL-MEASURED exposure)
-> garment detect -> FashionSigLIP attributes, then through the real reasoning
pipeline, on two real dataset images:

  * singlet + shorts  (sleeveless upper + bare thighs)  -> expect a violation
  * t-shirt + long pants                                -> expect compliant

The point is to prove that exposure = skin_px / (skin_px + clothing_px) from
Sapiens parsing produces a sane, explainable verdict -- not that any single
number is exact. ASCII-only stdout (console is cp1252).
"""

from __future__ import annotations

import glob
import sys

import cv2

from clothic.pipeline import ClothicPipeline

CHECKPOINT = "models/sapiens/sapiens_0.3b_goliath.pt2"
GARMENT = "runs/detect/runs_clothic/garment/stage_c-7/weights/best.pt"  # 13-class merged-data model

# NOTE on fixtures: both sample photos are actually non-compliant outfits.
#   * example_violation_sleeveless.jpg : short-sleeve tee + heavily RIPPED jeans.
#   * example_compliant_shirt_pants.jpg: tied CROP TOP (bare midriff) + RIPPED jeans.
# So both must come back as violations. We still lack a genuinely-compliant photo
# to prove the no-false-accusation path -- drop one in examples/ and add it here.
CASES = [
    ("VIOLATION expected (ripped jeans + tee)", "examples/example_violation_sleeveless.jpg"),
    ("VIOLATION expected (crop top + ripped jeans)", "examples/example_compliant_shirt_pants.jpg"),
]


def first(glob_pat: str) -> str:
    hits = sorted(glob.glob(glob_pat))
    if not hits:
        sys.exit(f"no images match {glob_pat}")
    return hits[0]


def main() -> None:
    print("Building FULL backend (yolo person+garment, yolo-pose, Sapiens 0.3B, FashionSigLIP)...")
    pipe = ClothicPipeline(
        profile_id="default",
        backend="full",
        backend_kwargs=dict(
            person_weights="yolo11n.pt",
            garment_weights=GARMENT,
            pose_weights="yolo11n-pose.pt",
            parser_type="sapiens",
            sapiens_checkpoint=CHECKPOINT,
            device="cpu",
        ),
        enable_temporal=False,  # single-image: don't debounce a lone frame
    )
    print(f"  parser in use: {type(pipe.backend.parser).__name__}\n")

    for label, pattern in CASES:
        path = first(pattern)
        frame = cv2.imread(path)
        if frame is None:
            sys.exit(f"cv2 could not read {path}")
        result = pipe.process_frame(frame)
        print("=" * 72)
        print(f"{label}")
        print(f"  image    : {path.split('/')[-1]}")
        print(f"  persons  : {len(result.persons)}   latency: {result.latency_ms}")
        for i, p in enumerate(result.persons):
            s = p.scores
            print(f"  --- person {i} ---")
            print(f"    decision   : {p.decision}")
            print(f"    scores     : exposure={s.exposure_score:.2f} formality={s.formality_score:.2f} "
                  f"compliance={s.compliance_score:.2f} uncertainty={s.uncertainty_score:.2f}")
            ov = "null" if p.scores.overall_violation is None else f"{p.scores.overall_violation:.2f}"
            print(f"    overall    : {ov}")
            print(f"    exposure   : " + ", ".join(f"{k}={v:.2f}" for k, v in p.observation.exposure.items()))
            up = p.observation.upper.type if p.observation.upper else None
            lo = p.observation.lower.type if p.observation.lower else None
            print(f"    garments   : upper={up} lower={lo}  evidence_q={p.observation.evidence_quality:.2f}")
            if p.matched_rules:
                print(f"    fired rules: " + ", ".join(r.id for r in p.matched_rules))
            if p.remediation:
                print(f"    to comply  : {'; '.join(p.remediation.steps)} "
                      f"(verified={p.remediation.verified})")
    print("=" * 72)
    pipe.close()


if __name__ == "__main__":
    main()
