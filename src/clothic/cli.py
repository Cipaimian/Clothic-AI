"""Clothic AI command-line interface.

Examples::

    clothic demo                       # run the mock pipeline, print decisions
    clothic demo --profile lab_safety --zone lab
    clothic profiles                   # list available policy profiles
    clothic explain --profile default  # show one full explainable verdict
"""

from __future__ import annotations

import argparse
import json
import sys

from clothic.config import list_profiles
from clothic.pipeline import ClothicPipeline


def _print_decisions(result, as_json: bool) -> None:
    if as_json:
        print(result.model_dump_json(indent=2))
        return
    print(f"\nFrame {result.frame_id}")
    print(f"  profile={result.profile_id}@{result.policy_version}  "
          f"latency={result.latency_ms.get('total')}ms")
    for p in result.persons:
        s = p.scores
        print(f"\n  -- track {p.track_id} | {p.decision.value.upper()} | action={p.action}")
        ov = "n/a" if s.overall_violation is None else f"{s.overall_violation:.2f}"
        print(f"     scores: violation={ov} compliance={s.compliance_score:.2f} "
              f"exposure={s.exposure_score:.2f} formality={s.formality_score:.2f} "
              f"uncertainty={s.uncertainty_score:.2f}")
        if p.matched_rules:
            print(f"     rules : {', '.join(r.id for r in p.matched_rules)}")
        print(f"     why   : {p.explanation}")


def cmd_demo(args: argparse.Namespace) -> int:
    pipe = ClothicPipeline(
        profile_id=args.profile, backend=args.backend, camera_id=args.camera, zone=args.zone
    )
    for _ in range(args.frames):
        result = pipe.process_frame(None)
        _print_decisions(result, args.json)
    pipe.close()
    return 0


def cmd_profiles(_: argparse.Namespace) -> int:
    profiles = list_profiles()
    print("Available campus policy profiles:")
    for p in profiles:
        print(f"  - {p}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    pipe = ClothicPipeline(profile_id=args.profile, backend="mock", zone=args.zone)
    result = pipe.process_frame(None)
    target = next((p for p in result.persons if p.decision.value != "compliant"), result.persons[0])
    print(json.dumps(target.model_dump(mode="json"), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clothic", description="Clothic AI - Clothing Vision")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="Run the pipeline and print decisions")
    d.add_argument("--profile", default="default")
    d.add_argument("--backend", default="mock", choices=["mock", "ultralytics"])
    d.add_argument("--camera", default="cam0")
    d.add_argument("--zone", default=None)
    d.add_argument("--frames", type=int, default=1)
    d.add_argument("--json", action="store_true", help="Emit raw JSON")
    d.set_defaults(func=cmd_demo)

    p = sub.add_parser("profiles", help="List policy profiles")
    p.set_defaults(func=cmd_profiles)

    e = sub.add_parser("explain", help="Show one full explainable verdict as JSON")
    e.add_argument("--profile", default="default")
    e.add_argument("--zone", default=None)
    e.set_defaults(func=cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
